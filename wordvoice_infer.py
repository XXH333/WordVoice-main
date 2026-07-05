"""
WordVoice Inference Pipeline
This script provides the inference pipeline for WordVoice, supporting zero-shot TTS 
with explicit, decoupled multi-dimensional word-level control.
Attributes controlled: Duration, Energy, Pitch, Boundary, and Tone.
"""

import sys
import os
import re
import math
import time
import json
import logging
import tempfile
import torch
import torchaudio
import torchaudio.functional as F

# 动态添加依赖路径
sys.path.append('CosyVoice/third_party/Matcha-TTS')
sys.path.append('CosyVoice')
sys.path.append('xxh_tools')

from cosyvoice.cli.cosyvoice import WordVoice
from get_timestamp_mmsfa import MMSFA_Aligner
from get_dsp import extract_words_dsp
from en_punc import english_text_normalization
from zh_punc import chinese_text_normalization

logging.basicConfig(level=logging.INFO)

# 定义字级声学边界(Boundary)与音调(Tone)的分类体系
# Boundary: 5 levels (b0: continuous, b1: <=0.05s, b2: <=0.18s, b3: <=0.4s, b4: >0.4s)
boundary_class = ['b0', 'b1', 'b2', 'b3', 'b4']
# Tone: 7 morphologies
tone_class = ['flat', 'rise', 'rrise', 'fall', 'ffall', 'peak', 'valley']

CONTROL_TAG_PATTERN = re.compile(r"\[(eng|pit|dur|bnd|ton)\s*:\s*([-+]?(?:\d+(?:\.\d+)?)|[A-Za-z]\w*)\]")

def normalize_text(text, lan='zh'):
    """
    文本规范化与分词预处理。
    Args:
        text (str): 输入文本。
        lan (str): 语言类型 ('zh' 或 'en')。
    Returns:
        text (str): 规范化并按字/词加空格分隔的文本。
        special_dict (dict): 特殊字符（如数字、标点）的映射字典。
    """
    if lan == "en":
        # 英文：数字与符号规范化
        text, special_dict = english_text_normalization(text) 
    elif lan == "zh":
        # 中文：数字与符号规范化
        text, special_dict = chinese_text_normalization(text) 
        # 将标点符号替换为空格，以纯文本形式进行字级对齐
        text = re.sub(r"[^\w\s]|_", ' ', text) 
        # 在中文字符两边添加空格，强制实现“字级别(Character-level)”切分
        text = re.sub(r'([\u4e00-\u9fff])', r' \1 ', text) 
        # 清理多余空格
        text = re.sub(r'\s+', ' ', text).strip() 
    else:
        raise ValueError(f"Unsupported language: {lan}")
    
    return text, special_dict


def eval(prompt_text, prompt_speech, tts_text, control_dict, save_path, lan='zh'):
    """
    WordVoice 核心推理函数。
    支持零样本(Zero-shot)克隆，并允许用户通过 control_dict 显式干预生成文本的字级声学属性。
    
    Args:
        prompt_text (str): 提示音频对应的文本 (Prompt text)。
        prompt_speech (str): 提示音频的文件路径 (Prompt audio path)。
        tts_text (str): 需要合成的目标文本 (Target text)。
        control_dict (dict): 用户指定的字级控制字典。格式如 {'dur': {0: 120}, 'pit': {1: 0.5}}。
        save_path (str): 合成音频的保存路径。
        lan (str): 语言类型 ('zh' 或 'en')。
    """
    # 1. 文本预处理
    prompt_words, _ = normalize_text(prompt_text, lan)
    tts_words, control_dict2 = normalize_text(tts_text, lan)
    tts_text = CONTROL_TAG_PATTERN.sub("", tts_text)
    control_dict.update(control_dict2) # 合并标点/特殊字符带来的控制信息偏移

    print('--- Text Information ---')
    print('TTS Text:', tts_text)
    print('TTS Words:', [f'{i}: {w}' for i, w in enumerate(tts_words.split())])
    print('User Control Dict:', control_dict)
    print('------------------------')

    s = time.time()

    # ==========================================
    # 2. 提取 Prompt 音频的声学与对齐特征
    # ==========================================
    # 获取 Prompt 字级时间戳并进行响度优化 (Loudness optimization)
    prompt_align_words = Aligner_Model.infer(prompt_speech, prompt_words, lan)
    prompt_align_words = Aligner_Model.optimize(prompt_speech, prompt_align_words)

    word_list = [x["word"] for x in prompt_align_words]
    start_list = [x["start"] for x in prompt_align_words]
    end_list = [x["end"] for x in prompt_align_words]
    
    # 提取 Prompt 的五维声学属性，为 LLM 提供 Context
    f0_list, eng_list, tone_list, bnd_list = extract_words_dsp(prompt_speech, prompt_align_words)
    
    start_token_list = []
    end_token_list = []
    # 帧率转换：1秒 = 25帧 (即每帧 40ms)
    for i in range(len(word_list)):
        start = int(start_list[i] * 25 + 0.6)
        end = max(start + 1, int(end_list[i] * 25 + 0.6))
        start_token_list.append(start)
        end_token_list.append(end)
        
    # 计算 Prompt 的字级时长 (Token 数量)
    dur_list = []
    for i in range(len(word_list)):
        duration = end_token_list[i] - start_token_list[i]
        dur_list.append(duration)

    # 拼接 Prompt 与 Target 的词列表
    word_list = word_list + tts_words.split()
    
    # 将 Prompt 的离散属性(Boundary, Tone)转换为类别索引
    # 若不在预定义类别中，则赋予越界索引 (len(class))，在模型内部作为 Mask/Unconditional token 处理
    bnd_list = [boundary_class.index(i) if i in boundary_class else len(boundary_class) for i in bnd_list]
    tone_list = [tone_class.index(i) if i in tone_class else len(tone_class) for i in tone_list]

    # ==========================================
    # 3. 构建 Target 文本的控制序列
    # ==========================================
    target_len = len(word_list)
    
    # 初始化 Target 的属性列表。
    # 35: 默认最大时长(Token数); 1.1: 连续属性的默认/Mask值; 5/7: 离散属性的 Mask 索引 (触发 LLM 自适应预测)
    dur_list = dur_list + [35] * (target_len - len(dur_list))
    eng_list = eng_list + [1.1] * (target_len - len(eng_list))
    f0_list = f0_list + [1.1] * (target_len - len(f0_list))
    bnd_list = bnd_list + [len(boundary_class)] * (target_len - len(bnd_list)) # 5
    tone_list = tone_list + [len(tone_class)] * (target_len - len(tone_list))  # 7

    # 将用户定义的控制信息 (control_dict) 注入到对应的字索引中
    for key in control_dict:
        # orig_idx 是目标文本中的字索引(如 0)，val 是对应的控制值(如 200ms)
        for orig_idx, val in control_dict[key].items():
            # 加上 prompt 的长度，映射到全局 word_list 中的绝对索引
            idx = orig_idx + len(prompt_align_words)  
            idx = min(idx, target_len - 1)  # 边界保护
            
            if key == 'bnd':
                bnd_list[idx] = boundary_class.index(val) if val in boundary_class else len(boundary_class)
            elif key == 'ton':
                tone_list[idx] = tone_class.index(val) if val in tone_class else len(tone_class)
            else:
                assert key in ['eng', 'pit', 'dur'], f"Unsupported control key: {key}"
                assert isinstance(val, (int, float)), f"Control value for {key} at index {idx} must be a number, got {type(val)}"
                
                if key == 'eng':
                    eng_list[idx] = val
                elif key == 'pit':
                    f0_list[idx] = val
                elif key == 'dur':
                    # 将绝对时间(ms)转换为 Token 数量 (40ms/token)，并限制在 [1, 35] 范围内
                    dur_list[idx] = max(min(int(val / 40), 35), 1)

    # 连续属性量化 (Quantization)：与论文中 "uniformly quantized into 20 discrete bins" 对应
    # Energy 归一化范围 [0, 1] -> 映射到 [0, 20]
    eng_list = [max(min(int((x - 0.001) * 20), 20), 0) for x in eng_list]
    # Pitch 归一化范围 [-1, 1] -> 映射到 [0, 20]
    f0_list = [max(min(int((x + 0.999) * 10), 20), 0) for x in f0_list]

    # ==========================================
    # 4. 模型推理 (WordVoice LLM + Flow Matching)
    # ==========================================
    # 构造 LLM Prompt 格式
    llm_prompt = f'You are a helpful assistant.<|endofprompt|>{prompt_text}'
    
    # 调用 WordVoice 推理接口
    for i, j in enumerate(wordvoice.wordvoice_inference(
            tts_text, llm_prompt, prompt_speech, 
            word_list, start_token_list, dur_list,
            bnd_list, tone_list, eng_list, f0_list, stream=False)):
        
        # 获取生成的音频与 LLM 最终规划/使用的声学属性
        tts_speech = j['tts_speech']
        dur_list = j['dur_list']
        eng_list = j['eng_list']
        f0_list = j['f0_list']
        tone_list = j['tone_list']
        bnd_list = j['bnd_list']

    # 整理并打印模型最终生成的控制参数字典 (便于用户参考和二次微调)
    generated_control_dict = {
        'dur': {i: dur_list[i] for i in range(len(dur_list))},
        'eng': {i: eng_list[i] for i in range(len(eng_list))},
        'pit': {i: f0_list[i] for i in range(len(f0_list))},
        'bnd': {i: boundary_class[bnd_list[i]] for i in range(len(bnd_list))},
        'ton': {i: tone_class[tone_list[i]] for i in range(len(tone_list))},
    }
    
    print('tts words:', [f'{i}: {w}' for i, w in enumerate(tts_words.split())])
    print('generated_control_dict:')
    print('{')
    for key, value in generated_control_dict.items():
        print(f"\t\'{key}\': {value},")
    print('}')

    # 保存音频
    torchaudio.save(save_path, tts_speech, wordvoice.sample_rate)
    print(f"\n[Success] Audio saved to: {save_path}")
    print(f"Time elapsed: {time.time() - s:.2f}s")


if __name__ == '__main__':
    # ------------------------------------------
    # 1. 环境与路径配置
    # ------------------------------------------
    lan = 'en'

    # 推理音频
    prompt_texts = {
        'zh': "伊阳一约，但听了既惭愧，又害怕，回到书斋灭了灯。水下的。",
        'en': "The team that change what they're doing. If you don't change some of the coaches or perhaps change."
    }
    prompt_speechs = {
        'zh': 'demo/prompt_speech_zh.mp3',
        'en': 'demo/prompt_speech_en.mp3'
    }
    tts_texts = {
        'zh': "词声[bnd: b1]是一个强[eng:0.625][dur:200]大[pit: -0.35][ton: ffall]的零样本语音合成工具，支持显式的字级声学属性控制。",
        'en': "I will never[pit: 0][ton: ffall] agree to this[bnd:b4], are[pit:0.2] you crazy[eng:0.9][dur: 400]?"
    }

    # 基础模型路径
    cosyvoice_path = 'checkpoints/Fun-CosyVoice3-0.5B' 
    aligner_path = "checkpoints/mms_fa"
    
    # WordVoice 检查点路径
    llm_path = f'checkpoints/WordVoice-base-0.5B/wordvoice_llm_{lan}.pt'
    flow_path = 'checkpoints/WordVoice-base-0.5B/wordvoice_fm.pt'
    hyper_yaml_path = 'config/wordvoice.yaml'
    
    # ------------------------------------------
    # 2. 模型初始化
    # ------------------------------------------
    Aligner_Model = MMSFA_Aligner(model_path=aligner_path)
    wordvoice = WordVoice(
        model_dir=cosyvoice_path,
        llm_path=llm_path,
        flow_path=flow_path,
        hyper_yaml_path=hyper_yaml_path,
    )

    # ------------------------------------------
    # 3. 推理输入准备
    # ------------------------------------------
    prompt_text = prompt_texts[lan]
    prompt_speech = prompt_speechs[lan]
    tts_text = tts_texts[lan]
    save_path = f'demo_wordvoice_{lan}.wav'

    # 显式控制字典 (Explicit Control Dictionary)
    # Key 说明:
    # dur: 时长 (绝对时间 ms)
    # eng: 能量 (归一化 0~1)
    # pit: 音高 (归一化 -1~1)
    # bnd: 边界 (b0~b4)
    # ton: 音调 (flat, rise, rrise, fall, ffall, peak, valley)
    # 用户可自定义的字级控制信息 (示例): 属性名: {字的索引: 控制数值, ...}
    control_dict = { 
        # 'dur': {0: 240, 1: 240, 2: 280, 3: 200, 4: 160, 5: 200, 6: 160, 7: 120, 8: 200, 9: 200, 10: 160, 11: 160, 12: 200, 13: 200, 14: 240, 15: 200, 16: 200, 17: 200, 18: 200, 19: 240, 20: 200, 21: 160, 22: 160, 23: 200, 24: 240, 25: 200, 26: 240, 27: 240, 28: 240, 29: 200},
        # 'eng': {0: 0.375, 1: 0.525, 2: 0.575, 3: 0.825, 4: 0.775, 5: 0.625, 6: 0.675, 7: 0.625, 8: 0.725, 9: 0.725, 10: 0.725, 11: 0.525, 12: 0.675, 13: 0.625, 14: 0.525, 15: 0.725, 16: 0.475, 17: 0.775, 18: 0.675, 19: 0.625, 20: 0.575, 21: 0.575, 22: 0.625, 23: 0.575, 24: 0.725, 25: 0.625, 26: 0.475, 27: 0.575, 28: 0.475, 29: 0.325},
        # 'pit': {0: -0.55, 1: -0.65, 2: -0.65, 3: -0.55, 4: -0.55, 5: -0.65, 6: -0.35, 7: -0.75, 8: -0.65, 9: -0.55, 10: -0.65, 11: -0.85, 12: -0.65, 13: -0.65, 14: -0.75, 15: -0.55, 16: -0.55, 17: -0.35, 18: -0.45, 19: -0.55, 20: -0.65, 21: -0.75, 22: -0.75, 23: -0.75, 24: -0.45, 25: -0.75, 26: -0.75, 27: -0.75, 28: -0.75, 29: -0.85},
        # 'bnd': {0: 'b0', 1: 'b1', 2: 'b3', 3: 'b0', 4: 'b0', 5: 'b0', 6: 'b0', 7: 'b0', 8: 'b0', 9: 'b0', 10: 'b0', 11: 'b0', 12: 'b0', 13: 'b0', 14: 'b0', 15: 'b0', 16: 'b3', 17: 'b0', 18: 'b0', 19: 'b0', 20: 'b0', 21: 'b1', 22: 'b0', 23: 'b0', 24: 'b0', 25: 'b0', 26: 'b0', 27: 'b0', 28: 'b0', 29: 'b2'},
        # 'ton': {0: 'flat', 1: 'flat', 2: 'flat', 3: 'rise', 4: 'fall', 5: 'rrise', 6: 'ffall', 7: 'valley', 8: 'rrise', 9: 'ffall', 10: 'rise', 11: 'ffall', 12: 'peak', 13: 'valley', 14: 'valley', 15: 'rise', 16: 'flat', 17: 'flat', 18: 'valley', 19: 'flat', 20: 'fall', 21: 'flat', 22: 'flat', 23: 'rise', 24: 'peak', 25: 'valley', 26: 'flat', 27: 'flat', 28: 'peak', 29: 'flat'},
    }
    
    # ------------------------------------------
    # 4. 执行推理
    # ------------------------------------------
    eval(prompt_text, prompt_speech, tts_text, control_dict, save_path, lan)
