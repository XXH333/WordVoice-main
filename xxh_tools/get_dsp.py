import os
import torch
import torchcrepe
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator
import scipy.signal
import json
import torch.multiprocessing as mp
from tqdm import tqdm
from scipy.signal import resample

FIXED_F0_MIN = 4.0 # np.log(50.0) 约 3.91
FIXED_F0_MAX = 6.3 # np.log(800.0) 约 6.68
FIXED_ENERGY_MIN = -8.0
FIXED_ENERGY_MAX = -0.5
ENERGY_THRESHOLD = 0.005
low_thr = 2.0


def smooth_curve_savgol(data, window_len=7, poly_order=2):
    """
    使用 Savitzky-Golay 滤波器进行高级平滑
    window_len: 窗口长度（必须是奇数）。窗口越大，越平滑。
    poly_order: 多项式阶数。阶数越高，越贴合原曲线，但去噪效果减弱。
    """
    if len(data) <= window_len:
        return data
    
    # 1. 第一步：先用中值滤波去掉 1-2 帧的孤立毛刺（野点）
    # 这一步是为了防止 S-G 滤波器去“拟合”那些错误的野点
    data_med = scipy.signal.medfilt(data, kernel_size=3)
    # 2. 第二步：使用 S-G 滤波器进行形状保留的平滑
    # 对于 F0 和 Energy，poly_order 设为 2 或 3 通常效果最好
    data_smooth = scipy.signal.savgol_filter(data_med, window_length=window_len, polyorder=poly_order)
    
    return data_smooth


def normalize_minmax(x, min_val, max_val, target_range=(-1, 1)):
    """
    固定上下限的 Min-Max 归一化，支持映射到任意区间 (默认 -1 到 1)
    """
    # 1. 限制在物理上下限内，防止异常点导致归一化爆炸
    x_clipped = np.clip(x, min_val, max_val)
    # 2. 映射到 [0, 1]
    x_01 = (x_clipped - min_val) / (max_val - min_val)
    # 3. 映射到目标区间 [target_min, target_max]
    t_min, t_max = target_range
    x_norm = x_01 * (t_max - t_min) + t_min
    
    return x_norm


def get_tone_category(a, b, thresh_a=0.05, thresh_b=0.03):
    """
    根据二次多项式的曲率(a)和斜率(b)判断语调类别。
    返回: (语调名称, 符号)
    """
    is_curved = abs(a) > thresh_a
    # 顶点位置 x = -b / 2a
    vertex_x = -b / (2 * a) if a != 0 else 1 
    
    # 1. 如果斜率 b 极大，直接无视局部的微小弯曲 a
    if abs(b) > thresh_b * 3:
        return "rrise" if b > 0 else "ffall"
    # 2. 曲线且拐点在词内部/边缘附近
    if is_curved and abs(vertex_x) < 0.6:
        if a > 0:
            return "valley"
        else:
            return "peak"      
    # 3. 直线类 (平/升/降)
    else:
        if b > thresh_b:
            return "rise"
        elif b < -thresh_b:
            return "fall"
        else:
            return "flat"


def get_acoustic_boundaries(words, utt_duration):
    """
    计算当前字与下一个字之间的时间间隔，返回 b0~b3 边界标签列表。
    规则：保留两位小数后计算差值
    """
    boundaries = []
    num_words = len(words)
    
    for i, w in enumerate(words):
        end = w['end']
        if i < num_words - 1:
            next_start = words[i + 1]['start']
            gap = round(next_start - end, 2)
        else:
            gap = round(utt_duration - end, 2)

        if gap <= 0.00:
            label = "b0"
        elif gap <= 0.05:
            label = "b1"
        elif gap <= 0.18:
            label = "b2"
        elif gap <= 0.40:
            label = "b3"
        else:
            label = "b4"
        
        boundaries.append(label)
        
    return boundaries


def extract_f0_energy_crepe_words(
    path,
    words,
    sr=16000,
    hop_length=256,
    f0_min=50,
    f0_max=500,
    log_f0=True,
    log_energy=True,
    remove_outlier=False,
    periodicity_threshold=0.2,
    device='cuda'
):
    """
    TorchCREPE + FS2 风格 F0 & Energy 提取，映射到字级 (word-level)。
    包含 F0 插值平滑，解决清音/静音区域 Log F0 为 0 的异常问题。
    """
    results = []

    # 定义 IQR 裁剪函数（仅全局使用）
    def clip_iqr(x):
        p25 = np.percentile(x, 25)
        p75 = np.percentile(x, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        return np.clip(x, lower, upper)

    # 1. 加载音频
    y, _ = librosa.load(path, sr=sr)
    utt_duration = librosa.get_duration(path=path)
    wave_tensor = torch.tensor(y, dtype=torch.float32).to(device)

    # 2. 提取帧级波形平均、F0、能量和 periodicity
    # 波形绝对平均
    num_frames = len(y) // hop_length
    frames = np.abs(y[:num_frames * hop_length]).reshape(num_frames, hop_length)
    energy_curve = np.mean(frames, axis=1)
    # F0 和 periodicity
    with torch.no_grad():
        f0, periodicity = torchcrepe.predict(
            wave_tensor.unsqueeze(0), sr, hop_length, f0_min, f0_max,
            model='full', batch_size=1024, device=device, return_periodicity=True, pad=True
        )
    f0 = f0[0].cpu().numpy()
    periodicity = periodicity[0].cpu().numpy()
    # RMS 能量
    energy_rms = librosa.feature.rms(y=y, frame_length=hop_length, hop_length=hop_length, center=True)[0]
    # 对齐长度
    align_len = min(len(f0), len(periodicity), len(energy_curve), len(energy_rms))
    f0 = f0[:align_len]
    periodicity = periodicity[:align_len]
    energy_curve = energy_curve[:align_len]
    energy_rms = energy_rms[:align_len]

    # 4. log 变换
    if log_f0: f0 = np.log(f0 + 1e-8)
    if log_energy: energy_rms = np.log(energy_rms + 1e-8)

    # 保留一份原始 f0
    raw_f0_norm = normalize_minmax(f0.copy(), FIXED_F0_MIN, FIXED_F0_MAX, target_range=(-1, 1))

    # 5. 全局 IQR 裁剪去噪
    if remove_outlier:
        f0 = clip_iqr(f0)
        energy_rms = clip_iqr(energy_rms)

    # 对静音区或清音区的 F0 进行插值平滑
    voiced_mask = (periodicity > periodicity_threshold) & (energy_curve > ENERGY_THRESHOLD) # 获取浊音掩码
    if np.sum(voiced_mask) > 1:
        valid_idx = np.where(voiced_mask)[0]
        valid_f0 = f0[voiced_mask]
        # b. 采用连续曲线建模
        interp_func = PchipInterpolator(valid_idx, valid_f0, extrapolate=False)
        f0 = interp_func(np.arange(len(f0)))
        if np.isnan(f0).any(): # 将外推导致产生的 NaN，用最边缘的有效值进行平铺填充
            f0[:valid_idx[0]] = valid_f0[0]
            f0[valid_idx[-1]+1:] = valid_f0[-1]

    # 6. 平滑处理（S-G 滤波器）
    f0 = smooth_curve_savgol(f0, window_len=9, poly_order=2)
    energy_rms = smooth_curve_savgol(energy_rms, window_len=7, poly_order=2)
    
    # 归一化
    energy_rms = normalize_minmax(energy_rms, FIXED_ENERGY_MIN, FIXED_ENERGY_MAX, target_range=(0, 5))
    f0 = normalize_minmax(f0, FIXED_F0_MIN, FIXED_F0_MAX, target_range=(-1, 1))

    # 6. 字级 F0 & Energy 池化 (Pooling)
    n_frames = min(len(f0), len(energy_rms), len(periodicity))
    frame_times = np.arange(n_frames) * hop_length / sr

    f0, energy_rms, periodicity = f0[:n_frames], energy_rms[:n_frames], periodicity[:n_frames]
    energy_rms = np.where(energy_rms < low_thr, low_thr, energy_rms) # 能量下限裁剪，防止过多静音区导致的极端值

    word_boundaries = get_acoustic_boundaries(words, utt_duration)

    word_f0s = [] # a: F0均值
    word_f0_slopes = [] # b: F0斜率（线性回归斜率）
    word_f0_curves = [] # c: F0曲率（二阶导数均值）
    word_tones = [] # 语调类别（峰/谷/升/降/平）
    word_energys = [] # energy均值
    
    for j, w in enumerate(words):
        start, end = w['start'], w['end']
        duration = end - start
        core_start = start + duration * 0.10
        core_end = start + duration * 0.90
        assert duration > 0.03, f"Frame index length is less than 4 for word: {w['word']}"

        # 字级 F0
        frame_idx = np.where((frame_times >= core_start) & (frame_times < core_end))[0]
        y_curve = f0[frame_idx]
        if y_curve.size < 4: 
            coeffs = [0.0, 0.0, np.mean(y_curve) if len(y_curve) > 0 else 0.0]
            word_tone = 'flat'
        else:
            y_curve = resample(y_curve, 16)
            x_axis = np.linspace(-1, 1, len(y_curve))
            coeffs = np.polyfit(x_axis, y_curve, 2)
            word_tone = get_tone_category(coeffs[0], coeffs[1])
        word_tones.append(word_tone)
        word_f0_curves.append(coeffs[0]) 
        word_f0_slopes.append(coeffs[1]) 
        
        if len(frame_idx) > 0:
            word_f0 = np.mean(f0[frame_idx])
        else:
            word_f0 = f0[np.searchsorted(frame_times, core_start) - 1]
        word_f0s.append(word_f0)

        # 字级 Energy
        frame_idx = np.where((frame_times >= start) & (frame_times < end))[0]
        frame_e = energy_rms[frame_idx]
        
        if len(frame_e) == 0:
            word_energy = (energy_rms[np.searchsorted(frame_times, start) - 1] - low_thr) / (5 - low_thr)
        elif len(frame_e) < 4:
            topk_vals = (np.mean(frame_e) - low_thr) / (5 - low_thr)
        else:
            k = max(1, int(len(frame_e) * 0.5))  # top 50%
            topk_vals = np.partition(frame_e, -k)[-k:]
            topk_vals = (np.mean(topk_vals) - low_thr) / (5 - low_thr) 
        word_energy = np.clip(topk_vals, 0, 1)
        word_energys.append(word_energy)

    # 检查是否存在nan
    def check_validity(arr, name):
        if np.isnan(arr).any():
            raise ValueError(f"提取结果 {name} 中存在 NaN")
        if np.isinf(arr).any():
            raise ValueError(f"提取结果 {name} 中存在 Inf (无穷大)")
    word_f0s_np = np.array(word_f0s)
    word_energys_np = np.array(word_energys)
    check_validity(word_f0s_np, "f0_words")
    check_validity(word_energys_np, "energy_words")

    # 分布优化
    word_f0s_np = np.clip(word_f0s_np, -0.8, 0.75)
    word_f0s_np = np.round(-1 + (word_f0s_np - (-0.8)) * (2.0 / (0.75 - (-0.8))), 4)
    word_energys_np = np.clip(word_energys_np, 0.13, 0.95)
    word_energys_np = np.round((word_energys_np - 0.13) / (0.95 - 0.13), 4) # 批量归一化

    # 7. 保存结果
    result = {
        'wave': y,
        'utt_duration': utt_duration,
        'f0_frames': f0,
        'energy_frames': energy_rms / 5,
        'periodicity': periodicity,
        'f0_slopes': np.array(word_f0_slopes),   
        'f0_curves': np.array(word_f0_curves),   
        'f0_words': word_f0s_np, # f0
        'tone_words': np.array(word_tones), # int
        'energy_words': word_energys_np, # eng
        'boundary_words': np.array(word_boundaries), # bnd
        'n_frames': n_frames
    }

    return result

def extract_words_dsp(wav_path, words):
    result = extract_f0_energy_crepe_words(wav_path, words, log_f0=True, log_energy=True, device='cuda')
    f0 = result['f0_words'].tolist()
    eng = result['energy_words'].tolist()
    tone = result['tone_words'].tolist()
    bnd = result['boundary_words'].tolist()

    return f0, eng, tone, bnd