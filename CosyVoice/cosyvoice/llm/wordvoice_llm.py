# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#               2025 Alibaba Inc (authors: Xiang Lyu, Yabin Li, Qihua, Shengqiang Li)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os, queue
import random
import time
import threading
from typing import Dict, Optional, Callable, List, Generator
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
from transformers import Qwen2ForCausalLM
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from cosyvoice.utils.common import IGNORE_ID
from cosyvoice.transformer.label_smoothing_loss import LabelSmoothingLoss
from cosyvoice.utils.common import th_accuracy
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.mask import make_pad_mask
from cosyvoice.utils.onnx import SpeechTokenExtractor, online_feature, onnx_path
from cosyvoice.utils.losses import DynamicStyleLoss

class TransformerLM(torch.nn.Module):
    def __init__(
            self,
            text_encoder_input_size: int,
            llm_input_size: int,
            llm_output_size: int,
            text_token_size: int,
            speech_token_size: int,
            text_encoder: torch.nn.Module,
            llm: torch.nn.Module,
            sampling: Callable,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
            spk_embed_dim: int = 192,
    ):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.speech_token_size = speech_token_size
        # 1. build text token inputs related modules
        self.text_embedding = torch.nn.Embedding(text_token_size, text_encoder_input_size)
        self.text_encoder = text_encoder
        self.text_encoder_affine_layer = nn.Linear(
            self.text_encoder.output_size(),
            llm_input_size
        )

        # 2. build speech token language model related modules
        self.sos = 0
        self.task_id = 1
        self.eos_token = self.speech_token_size
        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 1)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 1,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size, llm_input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, llm_input_size)

        # 4. sampling method
        self.sampling = sampling

    def encode(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
    ):
        encoder_out, encoder_mask = self.text_encoder(text, text_lengths, decoding_chunk_size=1, num_decoding_left_chunks=-1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        encoder_out = self.text_encoder_affine_layer(encoder_out)
        return encoder_out, encoder_out_lens

    def pad_unpad_sequence(self, sos_emb, embedding, text_token, text_token_len, task_id_emb, speech_token, speech_token_len):
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech_token = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True)
        lm_input = [torch.concat([sos_emb.squeeze(dim=0), embedding[i], text_token[i], task_id_emb.squeeze(dim=0), speech_token[i]], dim=0)
                    for i in range(len(text_token))]
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.long)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)
        return lm_input, lm_input_len

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text: (B, L, D)
            text_lengths: (B,)
            audio: (B, T, N) or (B, T)
            audio_lengths: (B,)
        """
        text_token = batch['text_token'].to(device)
        text_token_len = batch['text_token_len'].to(device)
        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)
        embedding = batch['embedding'].to(device)

        # 1. prepare llm_target
        lm_target = [torch.tensor([IGNORE_ID] * (2 + text_token_len[i]) + speech_token[i, :speech_token_len[i]].tolist() +
                                  [self.speech_token_size]) for i in range(text_token.size(0))]
        lm_target = pad_sequence(lm_target, batch_first=True, padding_value=IGNORE_ID).to(device)

        # 1. encode text_token
        text_token = self.text_embedding(text_token)
        text_token, text_token_len = self.encode(text_token, text_token_len)

        # 2. embedding projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)
        embedding = embedding.unsqueeze(1)

        # 3. sos and task_id
        sos_emb = self.llm_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 4. encode speech_token
        speech_token = self.speech_embedding(speech_token)

        # 5. unpad and pad
        lm_input, lm_input_len = self.pad_unpad_sequence(sos_emb, embedding, text_token, text_token_len,
                                                         task_id_emb, speech_token, speech_token_len)

        # 6. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output)
        loss = self.criterion_ce(logits, lm_target)
        acc = th_accuracy(logits.view(-1, self.speech_token_size + 1), lm_target, ignore_label=IGNORE_ID)
        return {'loss': loss, 'acc': acc}

    def sampling_ids(
            self,
            weighted_scores: torch.Tensor,
            decoded_tokens: List,
            sampling: int,
            ignore_eos: bool = True,
    ):
        if ignore_eos is True:
            weighted_scores[self.speech_token_size] = -float('inf')
        top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
        return top_ids

    @torch.inference_mode()
    def inference(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
            uuid: str = '',
    ) -> Generator[torch.Tensor, None, None]:
        device = text.device
        text = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        text = self.text_embedding(text)

        # 1. encode text
        text, text_len = self.encode(text, text_len)

        # 2. encode embedding
        if embedding.shape[0] != 0:
            embedding = F.normalize(embedding, dim=1)
            embedding = self.spk_embed_affine_layer(embedding)
            embedding = embedding.unsqueeze(dim=1)
        else:
            embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device).to(text.dtype)

        # 3. concat llm_input
        sos_emb = self.llm_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)
        lm_input = torch.concat([sos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1)

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # 5. step by step decode
        out_tokens = []
        offset = 0
        att_cache, cnn_cache = torch.zeros((0, 0, 0, 0), device=lm_input.device), torch.zeros((0, 0, 0, 0), device=lm_input.device)
        for i in range(max_len):
            y_pred, att_cache, cnn_cache = self.llm.forward_chunk(lm_input, offset=offset, required_cache_size=-1,
                                                                  att_cache=att_cache, cnn_cache=cnn_cache,
                                                                  att_mask=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]),
                                                                                                 device=lm_input.device)).to(torch.bool))
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False)
            if top_ids == self.eos_token:
                break
            # in stream mode, yield token one by one
            yield top_ids
            out_tokens.append(top_ids)
            offset += lm_input.size(1)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)


class Qwen2Encoder(torch.nn.Module):
    def __init__(self, pretrain_path):
        super().__init__()
        self.model = Qwen2ForCausalLM.from_pretrained(pretrain_path)

    def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor):
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T)
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=masks,
            output_hidden_states=True,
            return_dict=True,
        )
        return outs.hidden_states[-1], masks.unsqueeze(1)

    def forward_one_step(self, xs, masks, cache=None):
        input_masks = masks[:, -1, :]
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=input_masks,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
            past_key_values=cache,
        )
        xs = outs.hidden_states[-1]
        new_cache = outs.past_key_values
        return xs, new_cache


class Qwen2LM(TransformerLM):
    def __init__(
            self,
            llm_input_size: int,
            llm_output_size: int,
            speech_token_size: int,
            llm: torch.nn.Module,
            sampling: Callable,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
            mix_ratio: List[int] = [5, 15],
    ):
        torch.nn.Module.__init__(self)
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size
        # 2. build speech token language model related modules
        self.sos = 0
        self.task_id = 1
        self.eos_token = speech_token_size
        self.fill_token = speech_token_size + 2

        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 3)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 3,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size + 3, llm_input_size)

        # 4. sampling method
        self.sampling = sampling
        self.mix_ratio = mix_ratio

        # 5. vllm related
        self.stop_token_ids = [speech_token_size + i for i in range(3)]
        self.vllm_output_queue = {}
        if online_feature is True:
            self.speech_token_extractor = SpeechTokenExtractor(model_path=os.path.join(onnx_path, 'speech_tokenizer_v2.batch.onnx'))

    def prepare_lm_input_target(self, sos_emb, text_token, text_token_emb, text_token_len, task_id_emb,
                                speech_token, speech_token_emb, speech_token_len, words_token_emb, 
                                words_len, start_times, end_times, boundary_list, tone_list, f0_list, energy_list,
                                instruct_token=None, instruct_token_emb=None, instruct_token_len=None):
        lm_target, lm_input = [], []
        device = text_token.device  # 获取当前所在设备
        
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech_token = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True)
        text_token_emb = unpad_sequence(text_token_emb, text_token_len.cpu(), batch_first=True)
        speech_token_emb = unpad_sequence(speech_token_emb, speech_token_len.cpu(), batch_first=True)
        words_token_emb = unpad_sequence(words_token_emb, words_len.cpu(), batch_first=True)
        
        # NOTE add instruct_token in CosyVoice3
        # if instruct_token is not None and instruct_token_emb is not None and instruct_token_len is not None:
        instruct_token = unpad_sequence(instruct_token, instruct_token_len.cpu(), batch_first=True)
        instruct_token_emb = unpad_sequence(instruct_token_emb, instruct_token_len.cpu(), batch_first=True)

        dur_target_list, pau_target_list, boundary_target_list, tone_target_list, word_lm_positions_list = [], [], [], [], []
        
        for i in range(len(text_token)):
            # 加上 device=device，确保 Target 生成在正确的设备上
            this_lm_target = torch.tensor([IGNORE_ID] * (1 + instruct_token_len[i].item() + text_token_len[i].item()) + speech_token[i].tolist() + [self.eos_token], dtype=torch.long, device=device)
            this_lm_input = torch.concat([sos_emb.squeeze(dim=0), instruct_token_emb[i], text_token_emb[i], task_id_emb.squeeze(dim=0), speech_token_emb[i]], dim=0)
            
            # 【修复点】：增加 device=device，防止过 embedding 时 CPU/GPU 张量冲突
            durations = torch.tensor([end_times[i][k] - start_times[i][k] for k in range(len(start_times[i]))], dtype=torch.long, device=device)
            durations = torch.clamp(durations, 0, self.max_duration - 1)
            pauses = torch.tensor([
                start_times[i][k + 1] - end_times[i][k] if k < len(end_times[i]) - 1 else speech_token_len[i] - end_times[i][k]
                for k in range(len(start_times[i]))
            ], dtype=torch.long, device=device)
            pauses = torch.clamp(pauses, 0, self.max_pause - 1)

            boundary = torch.tensor(boundary_list[i], dtype=torch.long, device=device)
            boundary = torch.clamp(boundary, 0, self.max_boundary - 1)
            tone = torch.tensor(tone_list[i], dtype=torch.long, device=device)
            tone = torch.clamp(tone, 0, self.max_tone - 1)
            f0 = f0_list[i].to(device=device, dtype=torch.long)
            energy = energy_list[i].to(device=device, dtype=torch.long)

            boundary_target_list.append(boundary.clone())
            tone_target_list.append(tone.clone())
            dur_target_list.append(durations.clone())
            pau_target_list.append(pauses.clone())
            durations_input = durations.clone()
            pauses_input = pauses.clone()

            if self.training:
                # 对各个字级输入做随机掩码处理
                mask_prob = 0.3
                total_mask = torch.rand(durations_input.shape, device=device) < 0.1  # 10% 的概率全都掩码掉，增加训练时的鲁棒性
                dur_mask = (torch.rand(durations_input.shape, device=device) < mask_prob) | total_mask
                pau_mask = (torch.rand(pauses_input.shape, device=device) < mask_prob) | total_mask
                bnd_mask = (torch.rand(boundary.shape, device=device) < mask_prob) | total_mask
                tone_mask = (torch.rand(tone.shape, device=device) < mask_prob) | total_mask
                f0_mask = (torch.rand(f0.shape, device=device) < mask_prob) | total_mask
                eng_mask = (torch.rand(energy.shape, device=device) < mask_prob) | total_mask
                # 将掩码位置替换为原先设定的最大长度索引，即代表 <MASK>
                durations_input[dur_mask] = self.max_duration
                pauses_input[pau_mask] = self.max_pause
                boundary[bnd_mask] = self.max_boundary
                tone[tone_mask] = self.max_tone
                f0[f0_mask] = self.max_f0
                energy[eng_mask] = self.max_energy
            
            # 计算嵌入
            dur_emb = self.duration_embedding(durations_input)
            pau_emb = self.pause_embedding(pauses_input)
            bnd_emb = self.boundary_embedding(boundary)
            tone_emb = self.tone_embedding(tone)
            f0_emb = self.f0_embedding(f0)
            eng_emb = self.energy_embedding(energy)
            # style_emb = torch.stack([words_token_emb[i], dur_emb, pau_emb], dim=0).mean(dim=0) # xxh: 停顿时间对比
            style_emb = torch.stack([words_token_emb[i], dur_emb, bnd_emb, tone_emb, f0_emb, eng_emb], dim=0).mean(dim=0)

            # insert word-level duration tokens
            offset = 0
            word_positions = []
            for k, start_idx in enumerate(start_times[i]):
                idx = 1 + instruct_token_len[i].item() + text_token_len[i].item() + start_idx + offset
                
                # target序列
                this_lm_target = torch.cat([
                    this_lm_target[:idx],
                    torch.tensor([self.bound_token, IGNORE_ID], dtype=this_lm_target.dtype, device=device),
                    this_lm_target[idx:]
                ])
                # input序列
                this_lm_input = torch.cat([
                    this_lm_input[:idx+1],
                    self.speech_embedding.weight[self.bound_token].unsqueeze(0),
                    style_emb[k].unsqueeze(0),
                    this_lm_input[idx+1:]
                ], dim=0)
                word_positions.append(idx + 1)
                offset += 2
                
            lm_target.append(this_lm_target)
            lm_input.append(this_lm_input)
            word_lm_positions_list.append(torch.tensor(word_positions, dtype=torch.long, device=device))
            
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.long, device=device)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID).to(device)
        lm_target = pad_sequence(lm_target, batch_first=True, padding_value=IGNORE_ID).to(device)
        dur_target = pad_sequence(dur_target_list, batch_first=True, padding_value=IGNORE_ID).to(device)
        pau_target = pad_sequence(pau_target_list, batch_first=True, padding_value=IGNORE_ID).to(device)
        bnd_target = pad_sequence(boundary_target_list, batch_first=True, padding_value=IGNORE_ID).to(device)
        tone_target = pad_sequence(tone_target_list, batch_first=True, padding_value=IGNORE_ID).to(device)
        f0_target = pad_sequence(f0_list, batch_first=True, padding_value=IGNORE_ID).to(device)
        eng_target = pad_sequence(energy_list, batch_first=True, padding_value=IGNORE_ID).to(device)
        word_lm_positions = pad_sequence(word_lm_positions_list, batch_first=True, padding_value=IGNORE_ID).to(device)

        return lm_target, lm_input, lm_input_len, dur_target, pau_target, bnd_target, tone_target, f0_target, eng_target, word_lm_positions

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text: (B, L, D)
            text_lengths: (B,)
            audio: (B, T, N) or (B, T)
            audio_lengths: (B,)
        """
        # 1. encode text_token
        text_token = batch['text_token'].to(device)
        text_token_len = batch['text_token_len'].to(device)
        text_token_emb = self.llm.model.model.embed_tokens(text_token)
        words_token = batch['words'].to(device)
        words_token_emb = self.llm.model.model.embed_tokens(words_token)
        words_len = batch['word_len'].to(device)
        start_times = batch['start']
        end_times = batch['end']
        boundary_list = batch['boundary']
        tone_list = batch['tone']
        f0_list = batch['f0']
        f0_list = [torch.clamp(torch.floor((f0 + 1) / 2 * self.max_f0), min=0, max=self.max_f0 - 1).to(torch.long) for f0 in f0_list]  # 量化为20个区间，并限制最大值为 self.max_f0
        energy_list = batch['energy']
        energy_list = [torch.clamp(torch.floor(energy * self.max_energy), min=0, max=self.max_energy - 1).to(torch.long) for energy in energy_list]  # 量化为20个区间，并限制最大值为 self.max_energy

        # 2. encode speech_token
        if 'speech_token' not in batch:
            speech_token, speech_token_len = self.speech_token_extractor.inference(batch['whisper_feat'], batch['whisper_feat_len'], device)
        else:
            speech_token = batch['speech_token'].to(device)
            speech_token_len = batch['speech_token_len'].to(device)
        speech_token_emb = self.speech_embedding(speech_token)

        # 3. sos and task_id
        sos_emb = self.speech_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.speech_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 4. prepare llm_input/target
        instruct_token = batch['instruct_token'].to(device)
        instruct_token_len = batch['instruct_token_len'].to(device)
        instruct_token_emb = self.llm.model.model.embed_tokens(instruct_token)
        (lm_target, lm_input, lm_input_len, dur_target, pau_target, 
         bnd_target, tone_target, f0_target, eng_target, word_lm_positions) = self.prepare_lm_input_target(
                                                                            sos_emb, text_token, text_token_emb, text_token_len, 
                                                                            task_id_emb, speech_token, speech_token_emb, speech_token_len,
                                                                            words_token_emb, words_len, start_times, end_times, 
                                                                            boundary_list, tone_list, f0_list, energy_list,
                                                                            instruct_token, instruct_token_emb, instruct_token_len)
        # 4. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len)
        logits = self.llm_decoder(lm_output)
        speech_loss = self.criterion_ce(logits, lm_target)

        # 5. extract word-level embeddings for duration/pause
        lm_style_hiddens = []
        for b in range(word_lm_positions.size(0)):
            positions = word_lm_positions[b]
            positions = positions[positions != IGNORE_ID]
            lm_style_hiddens.append(lm_output[b, positions, :])

        lm_style_hiddens = pad_sequence(lm_style_hiddens, batch_first=True)

        # 7. style attributions prediction
        dur_pred = self.duration_predictor(lm_style_hiddens)
        # pau_pred = self.pause_predictor(lm_style_hiddens)
        bnd_pred = self.boundary_predictor(lm_style_hiddens)
        tone_pred = self.tone_predictor(lm_style_hiddens)
        f0_pred = self.f0_predictor(lm_style_hiddens)
        eng_pred = self.energy_predictor(lm_style_hiddens)

        # 8. flatten for cross entropy
        torch.set_printoptions(threshold=torch.inf)

        speech_loss = speech_loss / (2 * torch.exp(self.log_sigma_speech)**2) + self.log_sigma_speech
        # dur_loss = F.cross_entropy(dur_pred.view(-1, self.max_duration+1), dur_target.view(-1), ignore_index=IGNORE_ID)
        dur_loss = self.style_loss_module.duration_loss(dur_pred, dur_target)
        dur_loss = dur_loss / (2 * torch.exp(self.log_sigma_dur)**2) + self.log_sigma_dur
        # pau_loss = F.cross_entropy(pau_pred.view(-1, self.max_pause+1), pau_target.view(-1), ignore_index=IGNORE_ID)
        bnd_loss = self.style_loss_module.boundary_loss(bnd_pred, bnd_target)
        bnd_loss = bnd_loss / (2 * torch.exp(self.log_sigma_bnd)**2) + self.log_sigma_bnd
        tone_loss = self.style_loss_module.tone_loss(tone_pred, tone_target)
        tone_loss = tone_loss / (2 * torch.exp(self.log_sigma_tone)**2) + self.log_sigma_tone
        f0_loss = self.style_loss_module.f0_loss(f0_pred, f0_target) * 2
        f0_loss = f0_loss / (2 * torch.exp(self.log_sigma_f0)**2) + self.log_sigma_f0
        eng_loss = self.style_loss_module.energy_loss(eng_pred, eng_target)
        eng_loss = eng_loss / (2 * torch.exp(self.log_sigma_eng)**2) + self.log_sigma_eng

        # total_loss = speech_loss + dur_loss + pau_loss
        total_loss = speech_loss + dur_loss + bnd_loss + tone_loss + f0_loss + eng_loss
        acc = th_accuracy(logits.view(-1, self.llm_decoder.out_features), lm_target, ignore_label=IGNORE_ID)
        
        return {'loss': total_loss, 'speech_loss': speech_loss, 'acc': acc, 'dur_loss': dur_loss, 
                'bnd_loss': bnd_loss, 'tone_loss': tone_loss, 'f0_loss': f0_loss, 'eng_loss': eng_loss}

    @torch.inference_mode()
    def base_inference(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            word_list: List[torch.Tensor],       # 字 token 列表
            start_list: List[int],               # 字在音频中的起始索引
            dur_list: List[int],                 # 时长物理量
            bnd_list: List[int],
            tone_list: List[int],
            eng_list: List[int],
            f0_list: List[int],
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
            uuid: str = '',
            better_infer = True
    ) -> Generator[torch.Tensor, None, None]:
        device = text.device
        text = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        text_emb = self.llm.model.model.embed_tokens(text)
        
        if self.__class__.__name__ == 'CosyVoice3LM':
            assert 151646 in text, '<|endofprompt|> not detected in CosyVoice3 text or prompt_text, check your input!'

        # 常量预分配
        precomputed_bound_emb = self.speech_embedding.weight[self.bound_token].view(1, 1, -1)
        # 注意：这里改为 1D tensor，以便后续 logits[:, forbidden_ids] 索引不报错
        forbidden_stop_ids = torch.tensor([self.eos_token], device=device, dtype=torch.long)
        forbidden_bound_ids = torch.tensor([self.bound_token], device=device, dtype=torch.long)
        negative_inf = torch.tensor(-float('inf'), device=device)

        # 准备字特征 (Phase 1)
        style_embs = []
        word_embs_list = [] # 新增：保存纯净的字特征，用于后续动态生成 style_emb
        pau_list = [0] * len(word_list)
        
        for i in range(len(word_list)):
            w_emb = self.llm.model.model.embed_tokens(word_list[i].to(device))[0,0,:]
            word_embs_list.append(w_emb)
            
            # 此处 style_embs 仅用于组装 Prompt 部分（Prompt 部分通常不包含 MASK）
            # 对于后续自回归部分，若遇到 MASK，会在下方动态重新计算
            d = torch.tensor([dur_list[i]], device=device, dtype=torch.long)
            d = torch.clamp(d, 0, self.max_duration - 1)
            b = torch.tensor([bnd_list[i]], device=device, dtype=torch.long)
            t = torch.tensor([tone_list[i]], device=device, dtype=torch.long)
            e = torch.tensor([eng_list[i]], device=device, dtype=torch.long)
            f = torch.tensor([f0_list[i]], device=device, dtype=torch.long)
            
            d_emb = self.duration_embedding(d).view(-1)
            b_emb = self.boundary_embedding(b).view(-1)
            t_emb = self.tone_embedding(t).view(-1)
            e_emb = self.energy_embedding(e).view(-1)
            f_emb = self.f0_embedding(f).view(-1)
            
            style_emb = torch.stack([w_emb, d_emb, b_emb, t_emb, f_emb, e_emb], dim=0).mean(dim=0).view(1, 1, -1)
            style_embs.append(style_emb)

        # 准备 Prompt (Phase 2)
        word_idx = 0
        prompt_embs_list = []
        P = prompt_speech_token_len.item() if isinstance(prompt_speech_token_len, torch.Tensor) else prompt_speech_token_len

        if P > 0:
            p_speech_emb = self.speech_embedding(prompt_speech_token).squeeze(0)
            pre_dur=0
            for i in range(P): # 插入字级标签
                if word_idx < len(start_list) and start_list[word_idx] == i:
                    prompt_embs_list.append(self.speech_embedding.weight[self.bound_token])
                    prompt_embs_list.append(style_embs[word_idx].view(-1))
                    if word_idx > 0:
                        dur_list[word_idx-1] = pre_dur # 更新前一个字的时长为实际值
                        pre_dur = 0
                    word_idx += 1
                if word_idx > 0:
                    pre_dur += 1 
                prompt_embs_list.append(p_speech_emb[i])
            
            while word_idx < len(start_list) and start_list[word_idx] == P: # 补上没处理完的字级提示
                prompt_embs_list.append(self.speech_embedding.weight[self.bound_token])
                prompt_embs_list.append(style_embs[word_idx].view(-1))
                dur_list[word_idx-1] = pre_dur
                pre_dur = 0
                word_idx += 1
            prompt_speech_token_emb = torch.stack(prompt_embs_list, dim=0).unsqueeze(0)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text_emb.dtype).to(device)

        # 拼接初始 LLM Input
        sos_emb = self.speech_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.speech_embedding.weight[self.task_id].reshape(1, 1, -1)
        current_input = torch.concat([sos_emb, text_emb, task_id_emb, prompt_speech_token_emb], dim=1)

        # 开始推理：Prefill & Decode
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)
        
        cache = None        
        true_dur = pre_dur
        generate_speech_token = []
        final_dur = dur_list[word_idx-1]
        final_bnd = bnd_list[word_idx-1]
        for step in range(max_len):
            # 预填充是大三角，后续全都是 1x1
            lm_output, cache = self.llm.forward_one_step(
                current_input,
                masks=torch.tril(torch.ones((1, current_input.shape[1], current_input.shape[1]), device=current_input.device)).to(torch.bool),
                cache=cache
            )
            # logits = self.llm_decoder(lm_output[:, -1, :]) 
            logits = self.llm_decoder(lm_output[:, -1])
            
            # 控制停止符与边界符
            if word_idx < len(word_list):
                logits[:, forbidden_stop_ids] = negative_inf
            # else:
            #     logits[:, forbidden_bound_ids] = negative_inf
            
            if better_infer is True:
                if true_dur < final_dur:
                    # 比指定的时长短则强制模型输出有声音的token
                    logits[:, self.silent_tokens] = negative_inf
                    logits[:, forbidden_bound_ids] = negative_inf
                elif true_dur == final_dur:
                    # 留一帧缓冲
                    logits[:, forbidden_bound_ids] = negative_inf
                elif true_dur > final_dur:
                    # 最低静音与最大静音设置
                    silent_masks = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
                    if true_dur - final_dur < self.max_pause_lens[final_bnd]:
                        # 到了截止时间段则强制模型输出静音片段
                        silent_masks[self.silent_tokens] = False
                    if true_dur - final_dur > self.max_pause_lens[final_bnd-1]:
                        # 到了停顿时间自动截停
                        silent_masks[forbidden_bound_ids] = False
                        silent_masks[forbidden_stop_ids] = False
                    if silent_masks.all():
                        # 如果全被 mask 了，强制允许输出边界符，强行结束当前字的静音
                        silent_masks[forbidden_bound_ids] = False

                    logits[:, silent_masks] = negative_inf

            # 采样
            token_id = self.sampling_ids(logits.log_softmax(dim=-1).squeeze(dim=0), generate_speech_token, sampling, ignore_eos=True if i < min_len else False)
            if token_id in self.silent_tokens:
                pau_list[word_idx-1] += 1

            if better_infer is True:
                if true_dur == final_dur and final_bnd == 0:
                    token_id = self.bound_token

            if word_idx == len(word_list):
                if token_id == self.eos_token or token_id == self.bound_token:
                    break

            # 【核心修改区】：动态预测时长与停顿，构建 tot_emb
            if token_id == self.bound_token:
                # 1. 既然输出了 bound_token，我们需要先把它送进网络，拿到上下文 Hidden State
                inner_output, cache = self.llm.forward_one_step(
                    precomputed_bound_emb,
                    masks=torch.tril(torch.ones((1, precomputed_bound_emb.shape[1], precomputed_bound_emb.shape[1]), device=precomputed_bound_emb.device)).to(torch.bool),
                    cache=cache
                )
                # 2. 提取当前步骤的隐藏状态，过分类器
                hidden_state = inner_output[:, -1, :] # [1, D]
                dur_pred_logits = self.duration_predictor(hidden_state) # [1, max_duration+1]
                # pau_pred_logits = self.pause_predictor(hidden_state)    # [1, max_pause+1]
                bnd_pred_logits = self.boundary_predictor(hidden_state)
                tone_pred_logits = self.tone_predictor(hidden_state)
                f0_pred_logits = self.f0_predictor(hidden_state)
                eng_pred_logits = self.energy_predictor(hidden_state)
                
                # 取最大概率的值作为预测结果
                pred_dur = dur_pred_logits.argmax(dim=-1).item()
                # pred_pau = pau_pred_logits.argmax(dim=-1).item()
                pred_bnd = bnd_pred_logits.argmax(dim=-1).item()
                pred_tone = tone_pred_logits.argmax(dim=-1).item()
                pred_f0 = f0_pred_logits.argmax(dim=-1).item()
                pred_eng = eng_pred_logits.argmax(dim=-1).item()
                # print(pred_dur, pred_bnd, pred_tone, pred_f0, pred_eng)
                
                # 3. 判断是否使用预测值
                user_dur = dur_list[word_idx]
                user_bnd = bnd_list[word_idx]
                user_tone = tone_list[word_idx]
                user_f0 = f0_list[word_idx]
                user_eng = eng_list[word_idx]
                
                final_dur = pred_dur if user_dur == self.max_duration else user_dur # xxh
                # final_pau = pred_pau if user_pau == self.max_pause else user_pau # xxh
                final_bnd = pred_bnd if user_bnd == self.max_boundary else user_bnd
                final_tone = pred_tone if user_tone == self.max_tone else user_tone
                final_f0 = pred_f0 if user_f0 == self.max_f0 else user_f0
                final_eng = pred_eng if user_eng == self.max_energy else user_eng

                if better_infer == True:
                    if user_bnd == self.max_boundary:
                        final_bnd = min(final_bnd, 3) # 减少长停顿
                    if user_eng == self.max_energy:
                        final_eng = max(final_eng, min(eng_list)) # 防止极端小声

                # if word_idx == len(word_list)-1:
                #     final_bnd=self.max_boundary-1

                bnd_list[word_idx] = final_bnd
                tone_list[word_idx] = final_tone
                f0_list[word_idx] = final_f0
                eng_list[word_idx] = final_eng

                # 安全截断
                final_dur = min(max(final_dur, 0), self.max_duration - 1)
                # final_pau = min(max(final_pau, 0), self.max_pause - 1)
                
                # 4. 动态组装 Style Embedding
                w_emb = word_embs_list[word_idx]
                d_emb = self.duration_embedding(torch.tensor([final_dur], device=device, dtype=torch.long)).view(-1)
                # p_emb = self.pause_embedding(torch.tensor([final_pau], device=device, dtype=torch.long)).view(-1)
                b_emb = self.boundary_embedding(torch.tensor([final_bnd], device=device, dtype=torch.long)).view(-1)
                t_emb = self.tone_embedding(torch.tensor([final_tone], device=device, dtype=torch.long)).view(-1)
                f_emb = self.f0_embedding(torch.tensor([final_f0], device=device, dtype=torch.long)).view(-1)
                e_emb = self.energy_embedding(torch.tensor([final_eng], device=device, dtype=torch.long)).view(-1)
                
                dynamic_tot_emb = torch.stack([w_emb, d_emb, b_emb, t_emb, f_emb, e_emb], dim=0).mean(dim=0).view(1, 1, -1)
                
                # 5. 将组装好的字特征作为主循环的下一个输入
                current_input = dynamic_tot_emb[:]

                dur_list[word_idx-1] = true_dur
                true_dur = 0

                bnd_list[word_idx] = final_bnd
                tone_list[word_idx] = final_tone
                f0_list[word_idx] = final_f0
                eng_list[word_idx] = final_eng

                word_idx += 1
            else: # 输出语音token
                true_dur += 1
                generate_speech_token.append(token_id)
                current_input = self.speech_embedding.weight[token_id].reshape(1, 1, -1)
        dur_list[-1] = true_dur
        
        return generate_speech_token, dur_list, bnd_list, tone_list, f0_list, eng_list, pau_list

class CosyVoice3LM(Qwen2LM):
    def __init__(
            self,
            llm_input_size: int,
            llm_output_size: int,
            speech_token_size: int,
            llm: torch.nn.Module,
            sampling: Callable,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
            mix_ratio: List[int] = [5, 15],
    ):
        torch.nn.Module.__init__(self)
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size
        # 2. build speech token language model related modules
        self.sos = speech_token_size + 0
        self.eos_token = speech_token_size + 1
        self.task_id = speech_token_size + 2
        self.bound_token = speech_token_size + 3

        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 200, bias=False)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 200,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.max_duration = 35
        self.max_pause = 200
        self.max_boundary = 5
        self.max_tone = 7
        self.max_f0 = 20
        self.max_energy = 20
        self.silent_tokens = [1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323]
        self.max_pause_lens = [0, 1, 4, 10, 15]

        self.speech_embedding = torch.nn.Embedding(speech_token_size + 200, llm_input_size)
        self.duration_embedding = torch.nn.Embedding(self.max_duration + 1, llm_input_size)
        self.pause_embedding = torch.nn.Embedding(self.max_pause + 1, llm_input_size)
        self.boundary_embedding = torch.nn.Embedding(self.max_boundary + 1, llm_input_size)
        self.tone_embedding = torch.nn.Embedding(self.max_tone + 1, llm_input_size)
        self.f0_embedding = torch.nn.Embedding(self.max_f0 + 1, llm_input_size)
        self.energy_embedding = torch.nn.Embedding(self.max_energy + 1, llm_input_size)
        self.duration_predictor = nn.Linear(llm_output_size, self.max_duration + 1)
        self.pause_predictor = nn.Linear(llm_output_size, self.max_pause + 1)
        self.boundary_predictor = nn.Linear(llm_output_size, self.max_boundary + 1)
        self.tone_predictor = nn.Linear(llm_output_size, self.max_tone + 1)
        self.f0_predictor = nn.Linear(llm_output_size, self.max_f0 + 1)
        self.energy_predictor = nn.Linear(llm_output_size, self.max_energy + 1)

        # 初始化可学习的不确定性参数，初值可为 1.0
        self.log_sigma_speech = nn.Parameter(torch.zeros(()))  # log σ, 避免负值问题
        self.log_sigma_dur = nn.Parameter(torch.zeros(()))
        self.log_sigma_bnd = nn.Parameter(torch.zeros(()))
        self.log_sigma_tone = nn.Parameter(torch.zeros(()))
        self.log_sigma_f0 = nn.Parameter(torch.zeros(()))
        self.log_sigma_eng = nn.Parameter(torch.zeros(()))
        self.style_loss_module = DynamicStyleLoss(
            max_bnd_class = self.max_boundary + 1,
            max_tone_class = self.max_tone + 1,
            max_f0_class = self.max_f0 + 1,
            max_energy_class = self.max_energy + 1,
            max_dur_class = self.max_duration + 1,
            ignore_id = IGNORE_ID
        )

        # 4. sampling method
        self.sampling = sampling
        self.mix_ratio = mix_ratio

        # 5. vllm related
        self.stop_token_ids = [speech_token_size + i for i in range(200)]
        self.vllm_output_queue = {}
        if online_feature is True:
            self.speech_token_extractor = SpeechTokenExtractor(model_path=os.path.join(onnx_path, 'speech_tokenizer_v3.batch.onnx'))