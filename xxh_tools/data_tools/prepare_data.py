import argparse
import logging
import os
from tqdm import tqdm
import json

# 配置日志记录器
logger = logging.getLogger()

def main():
    """
    主函数：解析原始的 jsonl 数据集文件，提取文本、音频路径以及字级声学属性，
    并将其分别保存为独立的 json 字典文件，供后续特征提取和模型训练使用。
    """
    
    # 初始化存储各项属性的字典，键均为 utterance id (utt)
    utt2wav = {}       # 存储音频的绝对路径
    utt2text = {}      # 存储清洗后的纯文本内容
    utt2words = {}     # 存储字级时间戳/时长信息 (对应论文中的 Duration)
    utt2f0 = {}        # 存储字级音高信息 (对应论文中的 Pitch)
    utt2energy = {}    # 存储字级能量信息 (对应论文中的 Energy)
    utt2tone = {}      # 存储字级音调分类信息 (对应论文中的 Tone)
    utt2boundary = {}  # 存储字级声学边界分类信息 (对应论文中的 Boundary)

    # 1. 读取原始数据文件
    # 注意：虽然参数名叫 src_dir，但在 bash 脚本中传入的是具体的 .jsonl 文件路径
    with open(args.src_dir, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f]
        
    # 2. 遍历并处理每一条语音数据
    for data in tqdm(datas, desc="Processing WordVoice Data"):
        # 从原始 json 行中提取对应字段
        utt = data["utt"]
        wav = data["audio_path"]
        content = data["text"]
        words = data["mfa_words"]  # 包含字级别时间戳/对齐信息的列表
        f0 = data["f0"]
        energy = data["eng"]
        tone = data["tone"]
        boundary = data["bnd"]
        
        # 文本清洗：移除可能存在的特殊标记，如填充词标记 [FIL] 和说话人标记 [SPK]
        content = content.replace('[FIL]', '')
        content = content.replace('[SPK]', '')
        
        # 拼接音频的绝对路径
        wav = os.path.join(args.wav_dir, wav)
        
        # 数据校验：如果音频文件在磁盘上不存在，则跳过该条数据，保证训练数据的有效性
        if not os.path.exists(wav):
            continue
            
        # 存入对应的字典中
        utt2wav[utt] = wav
        utt2text[utt] = content
        utt2words[utt] = words
        utt2f0[utt] = f0
        utt2energy[utt] = energy
        utt2tone[utt] = tone
        utt2boundary[utt] = boundary

    # 3. 将提取好的字典分别持久化保存到目标目录 (des_dir)
    # 使用 ensure_ascii=False 保证中文正常显示，indent=2 保证文件可读性
    with open(f'{args.des_dir}/wav.json', 'w', encoding="utf-8") as f:
        json.dump(utt2wav, f, ensure_ascii=False, indent=2)
        
    with open(f'{args.des_dir}/text.json', 'w', encoding="utf-8") as f:
        json.dump(utt2text, f, ensure_ascii=False, indent=2)
        
    with open(f'{args.des_dir}/utt2f0.json', 'w', encoding="utf-8") as f:
        json.dump(utt2f0, f, ensure_ascii=False, indent=2)
        
    with open(f'{args.des_dir}/utt2energy.json', 'w', encoding="utf-8") as f:
        json.dump(utt2energy, f, ensure_ascii=False, indent=2)
        
    with open(f'{args.des_dir}/utt2tone.json', 'w', encoding="utf-8") as f:
        json.dump(utt2tone, f, ensure_ascii=False, indent=2)
        
    with open(f'{args.des_dir}/utt2boundary.json', 'w', encoding="utf-8") as f:
        json.dump(utt2boundary, f, ensure_ascii=False, indent=2)
        
    # 针对 utt2words.json 的特殊写入处理：
    # 因为 words 列表通常包含复杂的嵌套字典（每个字的详细时间戳），直接 dump 整个大字典可能会导致内存占用过大或格式不易读。
    # 这里采用逐行手动构建 JSON 字符串的方式写入，既保证了格式，又提高了大文件写入的稳定性。
    with open(f'{args.des_dir}/utt2words.json', "w", encoding="utf-8") as f:
        f.write("{\n")
        items = list(utt2words.items())
        for i, (k, v) in enumerate(items):
            # 将单个句子的字级对齐信息转为 json 字符串
            line = f'\t"{k}": {json.dumps(v, ensure_ascii=False)}'
            # 如果不是最后一行，则添加逗号
            if i != len(items) - 1:
                line += ","
            line += "\n"
            f.write(line)
        f.write("}\n")
        
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 原始 jsonl 数据文件的路径 (例如: datasets/wordvoice-5a-zh/test.jsonl)
    parser.add_argument('--src_dir', type=str, help="Path to the source jsonl file")
    # 处理后输出 json 文件的目标目录 (例如: datasets_processed/wordvoice-5a-zh/test)
    parser.add_argument('--des_dir', type=str, help="Directory to save the output json files")
    # 音频文件的根目录，用于拼接绝对路径
    parser.add_argument('--wav_dir', type=str, help="Root directory of the audio files")
    
    args = parser.parse_args()
    main()