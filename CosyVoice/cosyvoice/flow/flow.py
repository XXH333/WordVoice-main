# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
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
import os, logging
import random
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
from cosyvoice.utils.mask import make_pad_mask
from cosyvoice.utils.onnx import SpeechTokenExtractor, online_feature, onnx_path
from torch.nn.utils.rnn import pad_sequence
import math

class MaskedDiffWithXvec(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 output_type: str = "mel",
                 vocab_size: int = 4096,
                 input_frame_rate: int = 50,
                 only_mask_loss: bool = True,
                 encoder: torch.nn.Module = None,
                 length_regulator: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 decoder_conf: Dict = {'in_channels': 240, 'out_channel': 80, 'spk_emb_dim': 80, 'n_spks': 1,
                                       'cfm_params': DictConfig({'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine',
                                                                 'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7, 'reg_loss_type': 'l1'}),
                                       'decoder_params': {'channels': [256, 256], 'dropout': 0.0, 'attention_head_dim': 64,
                                                          'n_blocks': 4, 'num_mid_blocks': 12, 'num_heads': 8, 'act_fn': 'gelu'}}):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.length_regulator = length_regulator
        self.only_mask_loss = only_mask_loss

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        token = batch['speech_token'].to(device)
        token_len = batch['speech_token_len'].to(device)
        feat = batch['speech_feat'].to(device)
        feat_len = batch['speech_feat_len'].to(device)
        embedding = batch['embedding'].to(device)

        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h)
        h, h_lengths = self.length_regulator(h, feat_len)

        # get conditions
        conds = torch.zeros(feat.shape, device=token.device)
        for i, j in enumerate(feat_len):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(feat_len)).to(h)
        # NOTE this is unnecessary, feat/h already same shape
        loss, _ = self.decoder.compute_loss(
            feat.transpose(1, 2).contiguous(),
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            embedding,
            cond=conds
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  prompt_token,
                  prompt_token_len,
                  prompt_feat,
                  prompt_feat_len,
                  embedding,
                  flow_cache):
        assert token.shape[0] == 1
        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat speech token and prompt speech token
        token_len1, token_len2 = prompt_token.shape[1], token.shape[1]
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h)
        mel_len1, mel_len2 = prompt_feat.shape[1], int(token_len2 / self.input_frame_rate * 22050 / 256)
        h, h_lengths = self.length_regulator.inference(h[:, :token_len1], h[:, token_len1:], mel_len1, mel_len2, self.input_frame_rate)

        # get conditions
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
        feat, flow_cache = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10,
            prompt_len=mel_len1,
            cache=flow_cache
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.float(), flow_cache


class CausalMaskedDiffWithXvec(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 output_type: str = "mel",
                 vocab_size: int = 4096,
                 input_frame_rate: int = 50,
                 only_mask_loss: bool = True,
                 token_mel_ratio: int = 2,
                 pre_lookahead_len: int = 3,
                 encoder: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 decoder_conf: Dict = {'in_channels': 240, 'out_channel': 80, 'spk_emb_dim': 80, 'n_spks': 1,
                                       'cfm_params': DictConfig({'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine',
                                                                 'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7, 'reg_loss_type': 'l1'}),
                                       'decoder_params': {'channels': [256, 256], 'dropout': 0.0, 'attention_head_dim': 64,
                                                          'n_blocks': 4, 'num_mid_blocks': 12, 'num_heads': 8, 'act_fn': 'gelu'}}):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.only_mask_loss = only_mask_loss
        self.token_mel_ratio = token_mel_ratio
        self.pre_lookahead_len = pre_lookahead_len
        if online_feature is True:
            self.speech_token_extractor = SpeechTokenExtractor(model_path=os.path.join(onnx_path, 'speech_tokenizer_v2.batch.onnx'))

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if 'speech_token' not in batch:
            token, token_len = self.speech_token_extractor.inference(batch['whisper_feat'], batch['whisper_feat_len'], device)
        else:
            token = batch['speech_token'].to(device)
            token_len = batch['speech_token_len'].to(device)
        feat = batch['speech_feat'].to(device)
        print(feat[0])
        feat_len = batch['speech_feat_len'].to(device)
        embedding = batch['embedding'].to(device)

        # NOTE unified training, static_chunk_size > 0 or = 0
        streaming = True if random.random() < 0.5 else False

        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        h, h_lengths = self.encoder(token, token_len, streaming=streaming)
        h = self.encoder_proj(h)

        # get conditions
        conds = torch.zeros(feat.shape, device=token.device)
        for i, j in enumerate(feat_len):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(h_lengths.sum(dim=-1).squeeze(dim=1))).to(h)
        loss, _ = self.decoder.compute_loss(
            feat.transpose(1, 2).contiguous(),
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            embedding,
            cond=conds,
            streaming=streaming,
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  prompt_token,
                  prompt_token_len,
                  prompt_feat,
                  prompt_feat_len,
                  embedding,
                  streaming,
                  finalize):
        assert token.shape[0] == 1
        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        if finalize is True:
            h, h_lengths = self.encoder(token, token_len, streaming=streaming)
        else:
            token, context = token[:, :-self.pre_lookahead_len], token[:, -self.pre_lookahead_len:]
            h, h_lengths = self.encoder(token, token_len, context=context, streaming=streaming)
        mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]
        h = self.encoder_proj(h)

        # get conditions
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
        feat, _ = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10,
            streaming=streaming
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.float(), None


class CausalMaskedDiffWithDiT(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 output_type: str = "mel",
                 vocab_size: int = 4096,
                 input_frame_rate: int = 50,
                 only_mask_loss: bool = True,
                 token_mel_ratio: int = 2,
                 pre_lookahead_len: int = 3,
                 pre_lookahead_layer: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 decoder_conf: Dict = {'in_channels': 240, 'out_channel': 80, 'spk_emb_dim': 80, 'n_spks': 1,
                                       'cfm_params': DictConfig({'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine',
                                                                 'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7, 'reg_loss_type': 'l1'}),
                                       'decoder_params': {'channels': [256, 256], 'dropout': 0.0, 'attention_head_dim': 64,
                                                          'n_blocks': 4, 'num_mid_blocks': 12, 'num_heads': 8, 'act_fn': 'gelu'}}):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.pre_lookahead_len = pre_lookahead_len
        self.pre_lookahead_layer = pre_lookahead_layer
        self.decoder = decoder
        self.only_mask_loss = only_mask_loss
        self.token_mel_ratio = token_mel_ratio
        if online_feature is True:
            self.speech_token_extractor = SpeechTokenExtractor(model_path=os.path.join(onnx_path, 'speech_tokenizer_v3.batch.onnx'))

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if 'speech_token' not in batch:
            token, token_len = self.speech_token_extractor.inference(batch['whisper_feat'], batch['whisper_feat_len'], device)
        else:
            token = batch['speech_token'].to(device)
            token_len = batch['speech_token_len'].to(device)
        feat = batch['speech_feat'].to(device)
        feat_len = batch['speech_feat_len'].to(device)
        embedding = batch['embedding'].to(device)

        # NOTE unified training, static_chunk_size > 0 or = 0
        streaming = True if random.random() < 0.5 else False

        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        h = self.pre_lookahead_layer(token)
        h = h.repeat_interleave(self.token_mel_ratio, dim=1)
        mask = mask.repeat_interleave(self.token_mel_ratio, dim=1).squeeze(dim=-1)

        # xxh: revise feat
        target_feat_len = h.size(1)
        if feat.size(1) > target_feat_len:
            feat = feat[:, :target_feat_len]
        elif feat.size(1) < target_feat_len:
            pad_len = target_feat_len - feat.size(1)
            zero_pad = torch.zeros(
                feat.size(0), pad_len, *feat.shape[2:], 
                dtype=feat.dtype, device=feat.device
            )
            feat = torch.cat([feat, zero_pad], dim=1)
        feat_len = token_len * self.token_mel_ratio

        # get conditions
        conds = torch.zeros(feat.shape, device=token.device)
        for i, j in enumerate(feat_len):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)

        loss, _ = self.decoder.compute_loss(
            feat.transpose(1, 2).contiguous(),
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            embedding,
            cond=conds,
            streaming=streaming,
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  prompt_token,
                  prompt_token_len,
                  prompt_feat,
                  prompt_feat_len,
                  embedding,
                  streaming,
                  finalize):
        assert token.shape[0] == 1
        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        if finalize is True:
            h = self.pre_lookahead_layer(token)
        else:
            h = self.pre_lookahead_layer(token[:, :-self.pre_lookahead_len], context=token[:, -self.pre_lookahead_len:])
        h = h.repeat_interleave(self.token_mel_ratio, dim=1)
        mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]

        # get conditions
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
        feat, _ = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10,
            streaming=streaming
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.float(), None

class CausalMaskedDiffWithDiT_WV(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 output_type: str = "mel",
                 vocab_size: int = 4096,
                 input_frame_rate: int = 50,
                 only_mask_loss: bool = True,
                 token_mel_ratio: int = 2,
                 pre_lookahead_len: int = 3,
                 pre_lookahead_layer: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 decoder_conf: Dict = {'in_channels': 240, 'out_channel': 80, 'spk_emb_dim': 80, 'n_spks': 1,
                                       'cfm_params': DictConfig({'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine',
                                                                 'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7, 'reg_loss_type': 'l1'}),
                                       'decoder_params': {'channels': [256, 256], 'dropout': 0.0, 'attention_head_dim': 64,
                                                          'n_blocks': 4, 'num_mid_blocks': 12, 'num_heads': 8, 'act_fn': 'gelu'}}):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.pre_lookahead_len = pre_lookahead_len
        self.pre_lookahead_layer = pre_lookahead_layer
        self.decoder = decoder
        self.only_mask_loss = only_mask_loss
        self.token_mel_ratio = token_mel_ratio
        if online_feature is True:
            self.speech_token_extractor = SpeechTokenExtractor(model_path=os.path.join(onnx_path, 'speech_tokenizer_v3.batch.onnx'))

        self.max_bnd = 5
        self.max_tone = 7
        self.max_f0 = 20
        self.max_energy = 20

        feat_embed_dim = input_size // 4 
        # 词表大小为 max_val + 1，因为 max_val 本身被用作了 padding/起始符
        self.bnd_embed = nn.Embedding(self.max_bnd + 1, feat_embed_dim)
        self.tone_embed = nn.Embedding(self.max_tone + 1, feat_embed_dim)
        self.f0_embed = nn.Embedding(self.max_f0 + 1, feat_embed_dim)
        self.energy_embed = nn.Embedding(self.max_energy + 1, feat_embed_dim)
        
        # 定义字级的位置编码
        self.word_pos_embed = DynamicSinusoidalPositionalEncoding(input_size)
        self.control_modulator = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.SiLU(),
            nn.Linear(input_size, input_size * 2) # 输出 2*C，分别对应 Scale 和 Shift
        )  
        # 零初始化（Zero Initialization）, 保证在训练刚开始时，Scale接近0，Shift接近0，模型退化为标准的 h_modulated = h，确保训练极其平稳
        nn.init.zeros_(self.control_modulator[-1].weight)
        nn.init.zeros_(self.control_modulator[-1].bias)

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # NOTE unified training, static_chunk_size > 0 or = 0
        streaming = True if random.random() < 0.5 else False

        if 'speech_token' not in batch:
            token, token_len = self.speech_token_extractor.inference(batch['whisper_feat'], batch['whisper_feat_len'], device)
        else:
            token = batch['speech_token'].to(device)
            token_len = batch['speech_token_len'].to(device)
        feat = batch['speech_feat'].to(device)
        feat_len = batch['speech_feat_len'].to(device)
        embedding = batch['embedding'].to(device)

        # words_len = batch['word_len'].to(device)
        start_times = batch['start']
        # end_times = batch['end']
        boundary_list = batch['boundary']
        tone_list = batch['tone']
        f0_list = batch['f0']
        f0_list = [torch.clamp(torch.floor((f0 + 1) / 2 * self.max_f0), min=0, max=self.max_f0 - 1).to(torch.long) for f0 in f0_list]  # 量化为20个区间，并限制最大值为 self.max_f0
        energy_list = batch['energy']
        energy_list = [torch.clamp(torch.floor(energy * self.max_energy), min=0, max=self.max_energy - 1).to(torch.long) for energy in energy_list]  # 量化为20个区间，并限制最大值为 self.max_energy
        
        # 在字级标签最前面补上padding，值为最大值+1，表示静音部分
        boundary_list = [torch.cat([torch.tensor([self.max_bnd], dtype=torch.long, device=device),
                                    torch.tensor(bnd, dtype=torch.long, device=device)]) for bnd in boundary_list]
        tone_list = [torch.cat([torch.tensor([self.max_tone], dtype=torch.long, device=device), 
                                torch.tensor(tone, dtype=torch.long, device=device)]) for tone in tone_list]
        f0_list = [torch.cat([torch.tensor([self.max_f0], dtype=torch.long, device=device), 
                              f0.to(device)]) for f0 in f0_list]
        energy_list = [torch.cat([torch.tensor([self.max_energy], dtype=torch.long, device=device), 
                                  energy.to(device)]) for energy in energy_list]
        
        # 计算每个字的持续时长
        duration_list = []
        for b in range(len(start_times)):
            b_start = torch.tensor(start_times[b], dtype=torch.long, device=device)
            b_token_len = token_len[b]  # 保持为 Tensor
            # 拼接开始时间与总长度：[start_0, start_1, ..., start_N, total_len]
            b_start_extended = torch.cat([b_start, b_token_len.unsqueeze(0)])
            # 利用 diff 计算间隔，并 prepend 0，完美得到 [start_0, start_1-start_0, ..., total_len-start_N]
            b_dur = torch.diff(b_start_extended, prepend=torch.tensor([0], dtype=torch.long, device=device))
            duration_list.append(b_dur)

        # 将字级标签转为嵌入
        expanded_features = []
        batched_bnd = pad_sequence(boundary_list, batch_first=True, padding_value=0)
        batched_tone = pad_sequence(tone_list, batch_first=True, padding_value=0)
        batched_f0 = pad_sequence(f0_list, batch_first=True, padding_value=0)
        batched_energy = pad_sequence(energy_list, batch_first=True, padding_value=0)
        batched_dur = pad_sequence(duration_list, batch_first=True, padding_value=0)
        # 统一进行 Embedding 查表 (Shape: [B, max_word_len, feat_embed_dim])
        bnd_emb = self.bnd_embed(batched_bnd)
        tone_emb = self.tone_embed(batched_tone)
        f0_emb = self.f0_embed(batched_f0)
        energy_emb = self.energy_embed(batched_energy)

        # 结合字级时长，将字级特征展开到 token 级别
        word_feat = torch.cat([bnd_emb, tone_emb, f0_emb, energy_emb], dim=-1)
        # Positional Embedding
        word_feat = self.word_pos_embed(word_feat) # Shape: [1, max_word_len, input_size]
        # 展平
        flat_word_feat = word_feat.view(-1, word_feat.size(-1))  # Shape: [B * max_word_len, input_size]
        flat_dur = batched_dur.view(-1)                          # Shape: [B * max_word_len]
        # 根据duration展开到token级别
        flat_token_feat = torch.repeat_interleave(flat_word_feat, flat_dur, dim=0)  # Shape: [sum(token_len), input_size]
        # 计算每个 batch 样本展开后的真实长度 (即它们各自的 token_len)
        token_lengths = batched_dur.sum(dim=1).tolist()
        # 按照真实长度切分成 tuple，再用 pad_sequence 组装成最终的 Batch
        token_feat_list = torch.split(flat_token_feat, token_lengths)
        batched_token_feat = pad_sequence(token_feat_list, batch_first=True)

        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat speech_token and prompt_speech_token
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)
        token_emb = self.input_embedding(torch.clamp(token, min=0))
        if random.random() < 0.5:
            token_drop_mask = (torch.rand(token_emb.size(0), token_emb.size(1), 1, device=device) > 0.3).float()
            token_emb = token_emb * token_drop_mask
        token_emb = token_emb * mask

        # speech token encode
        h = self.pre_lookahead_layer(token_emb)
        h = h.repeat_interleave(self.token_mel_ratio, dim=1)

        # speech token modulation with word-level features
        batched_token_feat = batched_token_feat.repeat_interleave(self.token_mel_ratio, dim=1)
        scale, shift = self.control_modulator(batched_token_feat).chunk(2, dim=-1)
        h = h * (1.0 + torch.tanh(scale)) + shift  # 调制后的特征融合方式
        mask = mask.repeat_interleave(self.token_mel_ratio, dim=1).squeeze(dim=-1)

        # xxh: revise feat
        target_feat_len = h.size(1)
        if feat.size(1) > target_feat_len:
            feat = feat[:, :target_feat_len]
        elif feat.size(1) < target_feat_len:
            pad_len = target_feat_len - feat.size(1)
            zero_pad = torch.zeros(
                feat.size(0), pad_len, *feat.shape[2:], 
                dtype=feat.dtype, device=feat.device
            )
            feat = torch.cat([feat, zero_pad], dim=1)
        feat_len = token_len * self.token_mel_ratio
        feat_len = feat_len.cpu().tolist()

        # get conditions
        conds = torch.zeros(feat.shape, device=token.device)

        for i, j in enumerate(feat_len):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)

        loss, _ = self.decoder.compute_loss(
            feat.transpose(1, 2).contiguous(),
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            embedding,
            cond=conds,
            streaming=streaming,
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  prompt_token,
                  prompt_token_len,
                  prompt_feat,
                  prompt_feat_len,
                  embedding,
                  streaming,
                  finalize):
        assert token.shape[0] == 1
        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        if finalize is True:
            h = self.pre_lookahead_layer(token)
        else:
            h = self.pre_lookahead_layer(token[:, :-self.pre_lookahead_len], context=token[:, -self.pre_lookahead_len:])
        h = h.repeat_interleave(self.token_mel_ratio, dim=1)
        mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]

        # get conditions
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
        feat, _ = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10,
            streaming=streaming
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.float(), None

    @torch.inference_mode()
    def inference_wordvoice(self,
                  token,
                  token_len,
                  prompt_token,
                  prompt_token_len,
                  start_id,
                  dur_list,
                  bnd_list,
                  tone_list,
                  f0_list,
                  eng_list,
                  prompt_feat,
                  prompt_feat_len,
                  embedding,
                  streaming,
                  finalize):
        assert token.shape[0] == 1
        device = token.device
        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # 1. 处理字级特征 (Word-level Features) - 完全对齐训练代码
        # 获取原始长度标量
        orig_token_len = token_len.item() if isinstance(token_len, torch.Tensor) else token_len
        orig_prompt_len = prompt_token_len.item() if isinstance(prompt_token_len, torch.Tensor) else prompt_token_len
        total_expected_len = orig_prompt_len + orig_token_len # 拼接后的总 token 长度

        bnd_t = torch.as_tensor(bnd_list, dtype=torch.long, device=device)
        tone_t = torch.as_tensor(tone_list, dtype=torch.long, device=device)
        f0_t = torch.as_tensor(f0_list, dtype=torch.long, device=device)
        eng_t = torch.as_tensor(eng_list, dtype=torch.long, device=device)
        dur_t = torch.as_tensor(dur_list, dtype=torch.long, device=device)

        # 1.1 在最前面补上 Padding/静音 标志 (值为 max_xxx)
        bnd_t = torch.cat([torch.tensor([self.max_bnd], dtype=torch.long, device=device), bnd_t])
        tone_t = torch.cat([torch.tensor([self.max_tone], dtype=torch.long, device=device), tone_t])
        f0_t = torch.cat([torch.tensor([self.max_f0], dtype=torch.long, device=device), f0_t])
        eng_t = torch.cat([torch.tensor([self.max_energy], dtype=torch.long, device=device), eng_t])

        # 1.2 计算时长 (Duration)
        # 因为标签已包含 prompt，start_id 就是整个序列最开头的静音长度
        first_dur = start_id
        # 校验并补齐最后一个字的长度（对齐训练代码中 dur = b_token_len - b_start[i] 的逻辑）     
        full_dur = torch.cat([torch.tensor([first_dur], dtype=torch.long, device=device), dur_t])

        # 1.4 Embedding 查表 (添加 Batch 维度)
        bnd_emb = self.bnd_embed(bnd_t.unsqueeze(0))
        tone_emb = self.tone_embed(tone_t.unsqueeze(0))
        f0_emb = self.f0_embed(f0_t.unsqueeze(0))
        energy_emb = self.energy_embed(eng_t.unsqueeze(0))

        # 1.5 拼接并加上 Positional Embedding
        word_feat = torch.cat([bnd_emb, tone_emb, f0_emb, energy_emb], dim=-1)
        word_feat = self.word_pos_embed(word_feat)

        # 1.6 根据 duration 展开到 Token 级别
        flat_word_feat = word_feat.squeeze(0)
        batched_token_feat = torch.repeat_interleave(flat_word_feat, full_dur, dim=0).unsqueeze(0) # Shape: [1, total_expected_len, input_size]
        
        # 安全校验：防止长度溢出
        if batched_token_feat.size(1) > total_expected_len:
            print(f"Warning: Expanded token features length {batched_token_feat.size(1)} exceeds expected {total_expected_len}. Truncating.")
            batched_token_feat = batched_token_feat[:, :total_expected_len, :]

        # concat text and prompt_text
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        if finalize is True:
            h = self.pre_lookahead_layer(token)
            token_feat_sliced = batched_token_feat # 完整使用
        else:
            h = self.pre_lookahead_layer(token[:, :-self.pre_lookahead_len], context=token[:, -self.pre_lookahead_len:])
            # 如果是 streaming 切片，字级特征展开后也要做同样的切片对齐
            token_feat_sliced = batched_token_feat[:, :-self.pre_lookahead_len]

        # 2. 上采样并融合字级特征 (Upsample & Fusion)
        h = h.repeat_interleave(self.token_mel_ratio, dim=1)
        token_feat_sliced = token_feat_sliced.repeat_interleave(self.token_mel_ratio, dim=1)
        # 融合 Text Token 特征和 Word Level 特征
        scale, shift = self.control_modulator(token_feat_sliced).chunk(2, dim=-1)
        h = h * (1.0 + torch.tanh(scale)) + shift

        mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]

        # get conditions
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
        feat, _ = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10,
            streaming=streaming
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.float(), None


class DynamicSinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征，Shape 为 [B, T, C]，其中 C = d_model
        Returns:
            加上位置编码后的特征，Shape 为 [B, T, C]
        """
        device = x.device
        T = x.size(1)
        
        # 1. 动态生成位置索引 [T, 1]
        position = torch.arange(T, dtype=torch.float, device=device).unsqueeze(1)
        
        # 2. 计算衰减因子 div_term
        # 使用 10000.0 作为底数是标准 Transformer 的做法
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float, device=device) * 
            (-math.log(10000.0) / self.d_model)
        )
        
        # 3. 动态构建位置矩阵 [T, C]
        pe = torch.zeros(T, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 4. 广播相加 [B, T, C] + [1, T, C]
        return x + pe.unsqueeze(0)

if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    from hyperpyyaml import load_hyperpyyaml
    with open('./pretrained_models/Fun-CosyVoice3-0.5B/cosyvoice3.yaml', 'r') as f:
        configs = load_hyperpyyaml(f, overrides={'llm': None, 'hift': None})
    model = configs['flow']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    max_len = 10 * model.decoder.estimator.static_chunk_size
    chunk_size = model.decoder.estimator.static_chunk_size
    context_size = model.pre_lookahead_layer.pre_lookahead_len
    token = torch.randint(0, 6561, size=(1, max_len)).to(device)
    token_len = torch.tensor([max_len]).to(device)
    prompt_token = torch.randint(0, 6561, size=(1, chunk_size)).to(device)
    prompt_token_len = torch.tensor([chunk_size]).to(device)
    prompt_feat = torch.rand(1, chunk_size * 2, 80).to(device)
    prompt_feat_len = torch.tensor([chunk_size * 2]).to(device)
    prompt_embedding = torch.rand(1, 192).to(device)
    pred_gt, _ = model.inference(token, token_len, prompt_token, prompt_token_len, prompt_feat, prompt_feat_len, prompt_embedding, streaming=True, finalize=True)
    for i in range(0, max_len, chunk_size):
        finalize = True if i + chunk_size + context_size >= max_len else False
        pred_chunk, _ = model.inference(token[:, :i + chunk_size + context_size], torch.tensor([token[:, :i + chunk_size + context_size].shape[1]]).to(device),
                                        prompt_token, prompt_token_len, prompt_feat, prompt_feat_len, prompt_embedding, streaming=True, finalize=finalize)
        pred_chunk = pred_chunk[:, :, i * model.token_mel_ratio:]
        print((pred_gt[:, :, i * model.token_mel_ratio: i * model.token_mel_ratio + pred_chunk.shape[2]] - pred_chunk).abs().max().item())
