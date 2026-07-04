# pip install num2words
import re
from num2words import num2words
from typing import Match
from collections import defaultdict

ORDINAL_MAP = {
    "1st": "first",
    "2nd": "second",
    "3rd": "third",
    "5th": "fifth",
    "21st": "twenty first",
    "22nd": "twenty second",
    "31st": "thirty first"
}

# 匹配逻辑: (包含数字和逗号的串 + 可选的小数部分) OR (只包含小数部分)
NUMBER_REGEX = re.compile(r'\b[\d,]+(?:\.\d+)?\b|(?<!\w)\.\d+\b')

def _float_to_words(number_str: str) -> str:
    """
    内部函数：将数字字符串转换为英文单词，并确保小数部分逐位朗读。
    """
    original_number = number_str 
    
    try:
        # 清理逗号
        cleaned_number_str = number_str.replace(',', '')
        
        # 1. 如果是纯整数
        if '.' not in cleaned_number_str:
            integer_val = int(cleaned_number_str)
            return str(num2words(integer_val, lang='en'))

        # 2. 处理浮点数
        if cleaned_number_str.startswith('.'):
            standardized_str = '0' + cleaned_number_str
        elif cleaned_number_str.endswith('.'):
             standardized_str = cleaned_number_str + '0'
        else:
            standardized_str = cleaned_number_str
        
        integer_part, dot, decimal_part = standardized_str.partition('.')

        if not integer_part:
            integer_val = 0
        else:
            integer_val = int(integer_part)
            
        integer_words = str(num2words(integer_val, lang='en'))
            
        point_word = "point"

        # 2b. 转换小数部分 (严格逐位朗读)
        decimal_words = []
        if not decimal_part:
             decimal_words.append("zero")
        else:
            for digit in decimal_part:
                if digit.isdigit():
                    decimal_words.append(str(num2words(int(digit), lang='en')))
        
        # 3. 拼接结果
        return f"{integer_words} {point_word} {' '.join(decimal_words)}"

    except Exception as e:
        print(f"[ERROR] Failed to normalize '{original_number}': {e}. Returning original.")
        return original_number 


def split_mixed_token(match):
    token = match.group(0)

    # 只拆 token 内部的数字
    token = re.sub(r'([A-Za-z])(\d)', r'\1 \2', token)
    token = re.sub(r'(\d)([A-Za-z])', r'\1 \2', token)

    # 如果 token 里还有纯数字（比如 A19301s），再拆数字
    token = re.sub(r'\d{5,}', lambda m: " ".join(m.group(0)), token)

    return token

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

        # 标签之前有多少个词
        prefix = control_pattern.sub("", input_text[:m.start()])
        word_idx = max(0, len(prefix.split()))

        if word_idx == 0:
            continue
        control_dict[key][word_idx-1] = value

    cleaned_text = " ".join(cleaned_words)

    return cleaned_text, dict(control_dict)

def english_text_normalization(text: str) -> str:
    """
    主函数：对英文文本进行数字正则化处理。
    """
    # ==========================================
    # 保护特殊插入符 (如 [*], [LAUGH] 等)
    # ==========================================
    tags = []
    def tag_replacer(match):
        tags.append(match.group(0))
        # 将索引数字转为纯字母 (0->a, 1->b, 12->bc)，防止被 mix_pattern 拆分
        # 例如第一个标签变成 zzmaskazz，第二个变成 zzmaskbzz
        alpha_idx = "".join(chr(97 + int(d)) for d in str(len(tags) - 1))
        return f" zzmask{alpha_idx}zz "

    # 匹配方括号及其中间的所有内容，并替换为纯字母占位符
    text = re.sub(r'\[.*?\]', tag_replacer, text)

    # ==========================================

    # 文本规范化
    text = text.lower() # 转为小写
    # 特殊符号处理
    text = re.sub(r'\s+,', ',', text) # 删除,前空格
    text = re.sub(r',+', ',', text) # 合并,
    text = re.sub(r',(?!\d)', ', ', text) # ,后续不是数字，加空格
    text = re.sub(r'\s+&\s+', ' and ', text) # & 替换为 and
    text = re.sub(r'(\d+)%', r'\1 percent ', text) # 数字后跟%替换为 percent

    # 处理数字
    # 特殊数字处理（1st、2nd、3rd）
    for k, v in ORDINAL_MAP.items():
        text = text.replace(f" {k} ", f" {v} ")
    # 处理 四位年份+s
    text = re.sub(r'\b(\d{2})(\d{2})s\b', r'\1 \2 ', text)
    # 处理字母＋长数字组合
    mix_pattern = re.compile(r'\b(?=[A-Za-z]*\d|\d*[A-Za-z])[A-Za-z0-9]+\b')
    text = mix_pattern.sub(split_mixed_token, text)

    # 统一处理数字
    def replace_match(match: Match) -> str:
        """供 re.sub 调用的替换函数。"""
        number_token = match.group(0)
        return _float_to_words(number_token)
        
    # 使用正则表达式的 sub 方法，查找所有匹配项并用 replace_match 的结果替换
    normalized_text = NUMBER_REGEX.sub(replace_match, text)

    # 后处理
    normalized_text = re.sub(r"[^\w\s']|_", ' ', normalized_text) # 标点替换为空格
    normalized_text = re.sub(r"(?<!\w)'|'(?!\w)", "", normalized_text) # 处理单引号

    # ==========================================
    # 恢复特殊插入符
    # ==========================================
    for i, tag in enumerate(tags):
        alpha_idx = "".join(chr(97 + int(d)) for d in str(i))
        placeholder = f"zzmask{alpha_idx}zz"
        normalized_text = normalized_text.replace(placeholder, tag)
    # ==========================================

    normalized_text = re.sub(r'\s+', ' ', normalized_text).strip() # 多余空格归一化
    normalized_text, control_dict = parse_control_text(normalized_text)

    return normalized_text, control_dict

# --- 示例使用 ---
if __name__ == "__main__":
    test_sentences = [
        "1st."
        "The price is y1 1081.123 dollars.",
        "We need 5000 units, not 1234s.",
        "The rate is just .015, which is low.",
        "The count is 5,000,000.",
        "The small number is 0.05 and the big one is 1,234,567.89.",
        "Today is [bnd:b0]2025s. 3am, ab1345s,[eng: -0.12] 21312th"
    ]

    print("--- 原始文本 vs 规范化文本 ---")
    for sentence in test_sentences:
        normalized, control_dict = english_text_normalization(sentence) # mynorm
        # normalized = normalizer.normalize(sentence)  # wetext
        print(f"原始: {sentence}")
        print(f"规范: {normalized}")
        print(f"控制信息: {control_dict}\n")
        