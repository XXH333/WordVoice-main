# pip install cn2an
import re
import cn2an
from typing import Match
from collections import defaultdict

# 匹配完整的数字串，包括整数、浮点数、带逗号的数字，以及以点结尾的数字 (如 12.)
NUMBER_REGEX = re.compile(r'\b[\d,]+(?:\.\d*)?|(?<!\w)\.\d+\b')

def _add_space_around_numbers(text: str) -> str:
    """
    预处理函数：智能地在**数字串两端**添加空格。
    这样做是为了确保主规范化函数中的 NUMBER_REGEX 能够可靠地通过空格边界匹配数字。
    """
    # 匹配完整的数字串 (与 NUMBER_REGEX 匹配的模式类似)
    number_pattern = re.compile(r'([\d,]+(?:\.\d*)?|\.\d+)')
    
    # 在每个捕获的数字串前后添加空格
    spaced_text = number_pattern.sub(r' \1 ', text)
    
    # 清理连续的多个空格，统一为一个空格
    spaced_text = re.sub(r'\s+', ' ', spaced_text).strip()
    
    return spaced_text

def _number_to_chinese_words(number_str: str) -> str:
    """
    核心转换函数：将阿拉伯数字字符串转换为中文单词 (使用 cn2an 的 low 模式)。
    """
    original_number = number_str
    
    try:
        cleaned_number_str = number_str.replace(',', '')
        
        # 标准化边缘情况：处理以点开头 (如 .015) 和以点结尾 (如 12.) 的数字
        if cleaned_number_str.startswith('.'):
            standardized_str = '0' + cleaned_number_str
        elif cleaned_number_str.endswith('.'):
            # 将 12. 转换为 12.0
            standardized_str = cleaned_number_str + '0'
        else:
            standardized_str = cleaned_number_str
        
        # 使用 cn2an.an2cn 进行转换，mode="low" 匹配口语习惯 ("十二" 而非 "一十二")
        return cn2an.an2cn(standardized_str, mode="low")

    except Exception as e:
        print(f"[错误] 转换 '{original_number}' 失败: {e}。将返回原文。")
        return original_number

def parse_control_text(input_text: str):
    """
    解析控制标签，并返回：
        cleaned_text
        control_dict
    """

    control_pattern = re.compile(
        r"\[(eng|pit|dur|bnd|ton)\s*:\s*([-+]?(?:\d+(?:\.\d+)?)|[A-Za-z]\w*)\]"
    )

    control_dict = defaultdict(dict)

    # 找出所有控制标签
    matches = list(control_pattern.finditer(input_text))

    # 删除控制标签后的文本
    cleaned_text = control_pattern.sub("", input_text)
    cleaned_words = cleaned_text.split()

    # 计算每个控制标签对应的是第几个词
    for m in matches:
        key = m.group(1)
        value_str = m.group(2)

        try:
            value = int(value_str)
        except ValueError:
            try:
                value = float(value_str)
            except ValueError:
                value = value_str

        prefix = control_pattern.sub("", input_text[:m.start()])
        word_idx = max(0, len(prefix.split()))
        if word_idx == 0:
            continue
        control_dict[key][word_idx-1] = value

    cleaned_text = " ".join(cleaned_words)

    return cleaned_text, dict(control_dict)

def chinese_text_normalization(text: str) -> str:
    """
    主函数：对中文文本进行数字规范化处理，并保留特殊插入符。
    """
    # ==========================================
    # 保护特殊插入符 (如 [*], [LAUGH] 等)
    # ==========================================
    tags = []
    def tag_replacer(match):
        tags.append(match.group(0))
        # 将索引数字转为纯字母 (0->a, 1->b)，防止被后续正则误伤
        alpha_idx = "".join(chr(97 + int(d)) for d in str(len(tags) - 1))
        return f" zzmask{alpha_idx}zz "

    # 匹配方括号及其中间的所有内容，并替换为纯字母占位符
    text = re.sub(r'\[.*?\]', tag_replacer, text)
    # ==========================================

    # 1. 预处理：确保数字串被空格隔离
    preprocessed_text = _add_space_around_numbers(text)
    
    # 定义 re.sub 调用的替换函数
    def replace_match(match: Match) -> str:
        """供 re.sub 调用的替换函数。"""
        number_token = match.group(0)
        # 调用核心转换函数
        return _number_to_chinese_words(number_token)

    # 2. 规范化：查找数字并替换为中文
    normalized_text = NUMBER_REGEX.sub(replace_match, preprocessed_text)

    # 3. 后处理 (将外部的逻辑移入，确保在恢复占位符前执行)
    normalized_text = re.sub(r"[^\w\s]|_", ' ', normalized_text) # 标点替换为空格
    normalized_text = re.sub(r'([\u4e00-\u9fff])', r' \1 ', normalized_text) # 中文字符两边加空格

    # ==========================================
    # 恢复特殊插入符
    # ==========================================
    for i, tag in enumerate(tags):
        alpha_idx = "".join(chr(97 + int(d)) for d in str(i))
        placeholder = f"zzmask{alpha_idx}zz"
        # 替换回原来的标签
        normalized_text = normalized_text.replace(placeholder, tag)
    # ==========================================

    # 4. 最终清理：多余空格归一化
    normalized_text = re.sub(r'\s+', ' ', normalized_text).strip()
    normalized_text, control_dict = parse_control_text(normalized_text)

    return normalized_text, control_dict

# --- 示例使用 ---
if __name__ == "__main__":
    test_sentences = [
        # 测试数字紧贴文字的情况
        "_“价格是1081.123美元',yes [*]。",
        "我们需要5000个，而不是123个[LAUGH]。",
        "费率只有.015，很低[PAUSE]。",
        "总数是5,000,000 [*]。",
        "ok小数是0.05，大数是1,234,567.89 [BGM_START]。",
        "这是一个整-数12. (十二点) [*]",
        # 测试数字周围已经有空格的情况
        "这里有 1000 个数字 [pit:1234]3.14 ."
    ]

    print("--- 原始文本 vs 中文规范化文本 ---")
    for sentence in test_sentences:
        # 直接调用函数即可，后处理逻辑已封装在函数内
        normalized, control_dict = chinese_text_normalization(sentence)
        print(f"原始: {sentence}")
        print(f"规范: {normalized}")
        print(f"控制信息: {control_dict}\n")