import os
# 自动配置国内 Hugging Face 镜像源，确保模型秒速下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torchaudio
import json
import numpy as np
from tqdm import tqdm
import copy
import scipy.signal
import librosa
import re
import string 
from pypinyin import lazy_pinyin

# ================= 新增/修改的静默配置 =================
import logging
import warnings

# 1. 屏蔽警告
warnings.filterwarnings("ignore")

# 2. 强制将 Numba (Librosa 的底层) 的日志级别设为 WARNING，屏蔽掉所有字节码编译的 Debug 信息
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('numba.core.byteflow').setLevel(logging.WARNING)
logging.getLogger('numba.core.ssa').setLevel(logging.WARNING)
logging.getLogger('numba.core.interpreter').setLevel(logging.WARNING)

# 3. 顺便把其他可能吵闹的库也强制静音
logging.getLogger('torchaudio').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
# =======================================================

# 导入原生 Torchaudio MMS-FA 强对齐 Bundle
from torchaudio.pipelines import MMS_FA as bundle


class MMSFA_Aligner:
    def __init__(self, model_path=None, device="cuda:0", dtype=torch.float32):
        """
        初始化原生 Torchaudio MMS-FA 强对齐器
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        
        # 优先使用传入的路径，否则使用您指定的本地 CPFS 路径
        if model_path is not None:
            # 【修改】将 print 改为 logging.info，这样在 WARNING 级别下就不会打印了
            logging.info(f"正在从本地目录加载 MMS-FA 强对齐模型: {model_path}")
            self.model = bundle.get_model(dl_kwargs={"model_dir": model_path})
        else:
            self.model = bundle.get_model()
        
        # 使用 Wav2Vec2FABundle (MMS_FA) 加载本地模型
        self.model.to(self.device).to(self.dtype)
        self.model.eval()
        
        # 初始化分词器和对齐器（纯本地算法，秒速加载）
        self.tokenizer = bundle.get_tokenizer()
        self.aligner = bundle.get_aligner()
        
        # 【修改】将 print 改为 logging.info
        logging.info("MMS-FA 模型及组件加载成功！")

        # 优化参数（完美保留）
        self.HOP_LENGTH_MS = 10  
        self.MIN_DUR_FRAMES = 4  
        self.SEARCH_RATIO = 0.10 
        self.MIN_SEARCH_MS = 0.03 
        self.THRESHOLD = 0.003  

    def _clean_word_english(self, word: str) -> str:
        """英文单字归一化：转小写，去除标点符号"""
        word = word.lower()
        word = word.translate(str.maketrans("", "", string.punctuation))
        return word

    def _clean_word_chinese(self, word: str) -> str:
        """中文单字/词归一化：转为不带声调的纯英文字母拼音"""
        py = "".join(lazy_pinyin(word))
        # 过滤掉非字母字符，确保只含有 a-z
        py = re.sub(r'[^a-z]', '', py.lower())
        return py

    def infer(self, audio_path, text, lan="zh"):
        """
        原生 MMS-FA 强对齐推理（实现 1对1 原始字词时间戳映射，彻底解决 KeyError）
        """
        # 1. 加载音频并重采样至 16000Hz (MMS-FA 强对齐模型标准采样率)
        waveform, sr = torchaudio.load(audio_path)
        if sr != bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True) # 转为单声道
            
        # 2. 智能切分原始文本为字词列表
        if lan == "zh":
            # 若无空格则按字切分，有空格则按空格切分
            if " " in text.strip():
                original_words = [w.strip() for w in text.strip().split() if w.strip()]
            else:
                original_words = [c for c in text.strip() if c.strip()]
        else:
            original_words = [w.strip() for w in text.strip().split() if w.strip()]

        # 3. 1对1 清洗并生成模型所需的拼音/英文 Token 列表
        valid_original_words = []
        valid_normalized_words = []
        
        for word in original_words:
            if lan == "zh":
                norm_w = self._clean_word_chinese(word)
            else:
                norm_w = self._clean_word_english(word)
            
            # 仅保留清洗后含有有效字符的词，防止特殊符号导致空 Token 报错
            if norm_w:
                valid_original_words.append(word)
                valid_normalized_words.append(norm_w)

        if not valid_normalized_words:
            raise ValueError("输入文本在清洗后无任何有效字符，无法进行对齐。")

        # 4. 转化为 Token ID (传入 List[str]，完美规避 KeyError: ' ')
        tokens = self.tokenizer(valid_normalized_words)

        # 5. 模型前向传播，计算发射概率 (Emission)
        with torch.inference_mode():
            emission, _ = self.model(waveform.to(self.device).to(self.dtype))

        # 6. 调用 Torchaudio 底层强对齐算子
        token_spans = self.aligner(emission[0], tokens)

        # 7. 计算帧率时间映射关系
        num_frames = emission.shape[1]
        audio_duration = waveform.shape[-1] / bundle.sample_rate
        ratio = audio_duration / num_frames  # 每帧对应的秒数

        # 8. 1对1 映射回原始输入的文本
        align_words = []
        for i, word_spans in enumerate(token_spans):
            start_time = word_spans[0].start * ratio
            end_time = word_spans[-1].end * ratio
            
            # 绑定回原始输入的汉字/单词
            align_words.append({
                "word": valid_original_words[i],
                "start": round(start_time, 2),
                "end": round(end_time, 2)
            })

        return align_words

    # ==================== 以下完美保留你原有的能量曲线优化算法 ====================
    def optimize(self, audio_path, words):
        audio_eng = self.process_audio_abs_mean(audio_path)
        optimized_mfa_words = self.process_single_data(audio_eng, words)
        return optimized_mfa_words

    def process_audio_abs_mean(self, audio_path, sr=16000):
        y, sr = librosa.load(audio_path, sr=sr)
        hop_length = int(sr * (self.HOP_LENGTH_MS / 1000.0))
        num_frames = len(y) // hop_length
        frames = np.abs(y[:num_frames * hop_length]).reshape(num_frames, hop_length)
        energy_curve = np.mean(frames, axis=1)  
        energy_curve = scipy.signal.savgol_filter(energy_curve, window_length=7, polyorder=2)
        return y, energy_curve, sr, hop_length

    def find_optimized_frame(self, f_start, f_end, f_left, f_right, energy_curve, low_pos=True, prefer_left=False):
        """
        基于帧(Frame)的优化寻找。传入的f_left和f_right必须已经是严格限制好的安全边界。
        """
        max_idx = len(energy_curve) - 1
        f_left = max(0, min(int(f_left), max_idx))
        f_right = max(0, min(int(f_right), max_idx))

        if f_left > f_right:
            f_right = min(f_left + 2, max_idx)

        if prefer_left:
            f_current = f_start
        else:
            f_current = f_end

        cur = f_current
        max_expand_steps = 1000 
        for i in range(max_expand_steps):
            segment = energy_curve[f_left:f_right+1]
            
            if len(segment) == 0:
                f_right = min(f_left + 2, max_idx)
                segment = energy_curve[f_left:f_right+1]
                if len(segment) == 0: 
                    return cur, 'none'

            below_thresh_indices = np.where(segment < self.THRESHOLD)[0]

            if len(below_thresh_indices) > 0:
                idx = below_thresh_indices[-1] if prefer_left else below_thresh_indices[0]
                new_pos = f_left + idx
                if new_pos == cur: return cur, 'thresh' 
                cur = new_pos
                if prefer_left and cur == f_right:
                    f_left = f_right
                    f_right = min(f_end, f_left + 2) 
                    if f_end == f_right:
                        return cur, 'thresh'
                elif (not prefer_left) and cur == f_left:
                    f_right = f_left
                    f_left = max(f_start, f_right - 2) 
                    if f_start == f_left:
                        return cur, 'thresh'
                else:
                    return cur, 'thresh'
            elif i == 0 and low_pos: 
                return f_left + np.argmin(segment), 'low'
            else: 
                return cur, 'thresh'

    def process_single_data(self, audio_eng, record_words):
        opt_record_words = copy.deepcopy(record_words)
        y, energy_curve, sr, hop_length = audio_eng
        words = opt_record_words
        max_frames = len(energy_curve) - 1
        min_search_frames = int((self.MIN_SEARCH_MS * sr) / hop_length)

        for w in words:
            w['f_start'] = min(max(0, int(w['start'] * sr / hop_length)), max_frames)
            w['f_end'] = min(max(0, int(w['end'] * sr / hop_length)), max_frames)
            if w['f_end'] > w['f_start']:
                w['peak'] = w['f_start'] + np.argmax(energy_curve[w['f_start']:w['f_end'] + 1])
            else:
                w['peak'] = w['f_start']

        for i in range(len(words)):
            w = words[i]
            orig_dur_f = w['f_end'] - w['f_start']
            search_right_f = max(int(orig_dur_f * self.SEARCH_RATIO), min_search_frames)
            if i == 0:
                search_left_f = max(int(orig_dur_f * self.SEARCH_RATIO), min_search_frames) 
                left_limit = max(0, w['f_start'] - search_left_f) 
                f_low_pos = True
                end_start_equal = False
            else:
                w_prev = words[i-1]
                dur_prev_f = w_prev['f_end'] - w_prev['f_start'] 
                search_left_f = max(int(dur_prev_f * self.SEARCH_RATIO), min_search_frames) 
                if w['f_start'] < w_prev['f_end']: 
                    w['f_start'] = w_prev['f_end']
                search_left_f = min(search_left_f, w['f_start'] - w_prev['f_end']) 
                left_limit = max(w['f_start'] - search_left_f, w_prev['peak']) 
                f_low_pos = True
                end_start_equal = (abs(w['f_start'] - w_prev['f_end']) < 0.001)
            right_limit = min(w['f_start'] + search_right_f, w['peak']) 
            w['f_start'], find_type = self.find_optimized_frame(w['f_start'], w['f_end'], left_limit, right_limit, energy_curve, low_pos=f_low_pos, prefer_left=True)
            
            if i > 0 and end_start_equal and find_type == 'low':
                if w['f_start'] > w_prev['f_end']:
                    w_prev['f_end'] = w['f_start']

            curr_dur_f = max(0, w['f_end'] - w['f_start']) 
            search_left_f = min(max(int(curr_dur_f * self.SEARCH_RATIO), min_search_frames), curr_dur_f) 
            if i < len(words) - 1:
                w_next = words[i+1]
                dur_next_f = w_next['f_end'] - w_next['f_start']  
                search_right_f = max(int(dur_next_f * self.SEARCH_RATIO), min_search_frames) 
                right_limit = min(w['f_end'] + search_right_f, w_next['peak']) 
            else:
                search_right_f = max(int(curr_dur_f * self.SEARCH_RATIO), min_search_frames) 
                right_limit = min(max_frames, w['f_end'] + search_right_f) 
            left_limit = max(w['f_end'] - search_left_f, w['peak']) 
            w['f_end'], _ = self.find_optimized_frame(w['f_start'], w['f_end'], left_limit, right_limit, energy_curve, prefer_left=False)

        for w in words:
            w['f_end'] = max(w['f_end'], w['f_start'] + self.MIN_DUR_FRAMES) 
            w['start'] = round(float(w['f_start'] * hop_length / sr), 2)
            w['end'] = round(float(w['f_end'] * hop_length / sr), 2)
            w.pop('f_start', None); w.pop('f_end', None); w.pop('peak', None); w.pop('f_dur', None)

        return opt_record_words