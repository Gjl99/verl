import os
import json
import re
import asyncio
from typing import List, Union, Tuple, Optional, Dict, Any
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

SILICONFLOW_API_KEY = "sk-qqaypygblxgtabhaptwuyvoxhvcloxssoitnwsnrvbusmuod"
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
JUDGE_MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct" 

client = OpenAI(
    api_key=SILICONFLOW_API_KEY,
    base_url=SILICONFLOW_BASE_URL
)

LLM_JUDGE_SYSTEM_PROMPT_EN = """
You are an expert physics evaluator. Your task is to evaluate the quality of a student's answer to a physics question by comparing it with the reference answer.

Evaluation Criteria:
1. **Correctness (40%)**: Does the answer contain the correct physics principles, formulas, and conclusions?
2. **Completeness (30%)**: Does the answer cover all key points mentioned in the reference answer?
3. **Clarity (20%)**: Is the explanation clear, logical, and easy to understand?
4. **Reasoning (10%)**: Does the answer show proper reasoning and step-by-step thinking?

Scoring Guidelines:
- 0.9-1.0: Excellent - Correct, complete, clear, and well-reasoned
- 0.7-0.89: Good - Mostly correct with minor omissions or unclear explanations
- 0.5-0.69: Acceptable - Partially correct but missing important points
- 0.3-0.49: Poor - Contains some relevant information but significant errors
- 0.0-0.29: Very Poor - Incorrect or irrelevant answer

Output Constraints:
- Be extremely concise in your "reasoning", "strengths", and "weaknesses".
- The total length of your JSON response must strictly be under 768 tokens.

You must respond in JSON format with the following structure:
{
  "score": <float between 0 and 1>,
  "reasoning": "<concise explanation of why this score was given>",
  "strengths": "<briefly what the answer does well>",
  "weaknesses": "<briefly what the answer lacks or gets wrong>"
}
"""

LLM_JUDGE_SYSTEM_PROMPT_ZH = """
你是一位物理学专家评估员。你的任务是通过将学生的答案与参考答案进行比较，来评估学生答案的质量。

评估标准：
1. **正确性 (40%)**：答案是否包含正确的物理原理、公式和结论？
2. **完整性 (30%)**：答案是否涵盖了参考答案中提到的所有关键点？
3. **清晰度 (20%)**：解释是否清晰、有逻辑且易于理解？
4. **推理性 (10%)**：答案是否展示了适当的推理和逐步思考？

评分指南：
- 0.9-1.0：优秀 - 正确、完整、清晰且推理充分
- 0.7-0.89：良好 - 基本正确，有轻微遗漏或解释不够清晰
- 0.5-0.69：可接受 - 部分正确但缺少重要要点
- 0.3-0.49：较差 - 包含一些相关信息但有重大错误
- 0.0-0.29：很差 - 答案错误或不相关

输出限制：
- 请保持“reasoning”（理由）、“strengths”（优点）和“weaknesses”（缺点）非常简洁。
- 整个JSON回复的长度必须严格控制在 768 个Token以内。

你必须以JSON格式回复，结构如下：
{
  "score": <0到1之间的浮点数>,
  "reasoning": "<简洁解释为什么给出这个分数>",
  "strengths": "<简要说明答案做得好的地方>",
  "weaknesses": "<简要说明答案缺少或错误的地方>"
}
"""
def compute_format_reward(text: str) -> float:
    """
    计算格式奖励。
    目标格式严格为:
    <think>
    ...
    </think>
    <answer>
    \boxed{...}
    </answer>
    
    评分标准 (总分 1.0):
    1. 包含 <think>...</think>: +0.3
    2. 包含 <answer>...</answer>: +0.3
    3. <answer> 内部包含 \boxed{}: +0.4
    (如果顺序错误，如 answer 在 think 前面，可能会影响 XML 解析，视作格式错误)
    """
    score = 0.0
    if not text:
        return 0.0

    try:
        # 1.检查思考标签
        has_think = bool(re.search(r"<think>.*?</think>", text, re.DOTALL))
        if has_think:
            score += 0.3

        # 2.检查答案标签
        answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if answer_match:
            score += 0.3
            
            # 3.检查 Box 是否在 Answer 标签内部
            # 只有当 Answer 标签存在时，才检查里面的 Box
            content_inside_answer = answer_match.group(1)
            if "\\boxed{" in content_inside_answer:
                score += 0.4
    except Exception:
        # 正则解析如果因为极端情况崩溃，返回当前得分
        pass
        
    return score

def extract_answer(text: str, question_type: str = "single_select") -> str:
    """
    从文本中提取答案 - 根据题型优化提取策略
    
    Args:
        text: 模型生成的完整回答文本
        question_type: 题型，可选值：
            - "single_select": 单选题
            - "multi_select": 多选题  
            - "calculation": 填空题（数值计算）
            - "conversational_qa": 对话问答
    
    Returns:
        提取的答案字符串
    """
    if not text:
        return ""
    
    answer_tag_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if answer_tag_match:
        content_in_tag = answer_tag_match.group(1).strip()
        boxed_matches = re.findall(r'\\boxed\{([^}]+)\}', content_in_tag)
        if boxed_matches:
            answer = boxed_matches[-1].strip()
            
            # 根据题型进行后处理
            if question_type == "single_select":
                return extract_single_choice(answer)
            elif question_type == "multi_select":
                return extract_multi_choice(answer)
            elif question_type == "calculation":
                return extract_numerical_answer(answer)
            else:
                return answer
        
        # 如果没有boxed，直接返回answer标签内容
        cleaned_content = re.sub(r'\s+', ' ', content_in_tag).strip()
        if cleaned_content:
            return cleaned_content

    # 提取任意位置的 \boxed{}
    boxed_matches = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed_matches:
        answer = boxed_matches[-1].strip()
        
        if question_type == "single_select":
            return extract_single_choice(answer)
        elif question_type == "multi_select":
            return extract_multi_choice(answer)
        elif question_type == "calculation":
            return extract_numerical_answer(answer)
        else:
            return answer
    
    answer_pattern = r'####\s*(.+?)(?:\n|$)'
    answer_match = re.search(answer_pattern, text)
    if answer_match:
        return answer_match.group(1).strip()
    
    # 最后一行提取答案
    lines = text.strip().split('\n')
    if lines:
        last_line = lines[-1].strip()
        
        # 移除常见的答案前缀
        for prefix in ["答案是", "答案为", "最终答案是", "最终答案为", "答案：",
                      "Answer is", "Answer:", "The answer is", "Final answer:",
                      "因此", "所以", "Therefore", "Thus", "Hence"]:
            if last_line.lower().startswith(prefix.lower()):
                last_line = last_line[len(prefix):].strip()
                break
        
        if last_line:
            # 根据题型进行后处理
            if question_type == "single_select":
                return extract_single_choice(last_line)
            elif question_type == "multi_select":
                return extract_multi_choice(last_line)
            elif question_type == "calculation":
                return extract_numerical_answer(last_line)
            else:
                return last_line
    return ""


def extract_single_choice(text: str) -> str:
    """
    从文本中提取单选题答案（单个字母）
    
    Examples:
        "A" -> "A"
        "选项A" -> "A"
        "答案是B" -> "B"
        "A、这是解释" -> "A"
    """
    if not text:
        return ""
    
    # 查找第一个大写字母（A-Z）
    match = re.search(r'[A-Z]', text.upper())
    if match:
        return match.group(0)
    
    return ""


def extract_multi_choice(text: str) -> str:
    """
    从文本中提取多选题答案（多个字母，逗号分隔）
    
    Examples:
        "A, C" -> "A,C"
        "B,D,E" -> "B,D,E"
        "选项A和C" -> "A,C"
        "A、C、E" -> "A,C,E"
    """
    if not text:
        return ""
    
    # 查找所有大写字母
    letters = re.findall(r'[A-Z]', text.upper())
    
    # 去重并保持顺序
    seen = set()
    unique_letters = []
    for letter in letters:
        if letter not in seen:
            seen.add(letter)
            unique_letters.append(letter)
    return ','.join(unique_letters)


def extract_numerical_answer(text: str) -> str:
    """
    从文本中提取数值答案
    
    Examples:
        "10" -> "10"
        "3.14" -> "3.14"
        "3, 5, 1" -> "3,5,1"
        "答案是10.5" -> "10.5"
    """
    if not text:
        return ""
    
    # 移除单位
    cleaned = text
    for unit in ["V", "A", "Ω", "W", "Hz", "F", "H", "°C", "K", "m", "s"]:
        cleaned = cleaned.replace(unit, " ")
    # 查找所有数字
    numbers = re.findall(r'-?\d+\.?\d*', cleaned)
    if not numbers:
        return ""
    # 一个数字，直接返回
    if len(numbers) == 1:
        return numbers[0]
    # 多个数字，用逗号连接
    return ','.join(numbers)


def normalize_answer(answer: str, question_type: str = "single_select") -> str:
    """
    标准化答案格式 - 用于比较
    
    Args:
        answer: 原始答案
        question_type: 题型
        
    Returns:
        标准化后的答案
    """
    if not answer:
        return ""
    if isinstance(answer, list):
        answer = "".join(str(x) for x in answer)
    normalized = str(answer)
    # 移除所有空格
    normalized = answer.replace(" ", "").replace("_", "")
    
    if question_type in ["single_select", "multi_select"]:
        # 选择题：转大写，移除标点
        normalized = normalized.upper()
        normalized = re.sub(r'[^\w,]', '', normalized)
        
        # 针对多选题，进行排序以忽略顺序差异 (A,B == B,A)
        if question_type == "multi_select":
            parts = [p for p in normalized.split(',') if p]
            normalized = ",".join(sorted(parts))
    elif question_type == "calculation":
        # 数值题：只保留数字、小数点、负号和逗号
        normalized = re.sub(r'[^\d.,-]', '', normalized)
    
    return normalized


def compute_single_select_reward(prediction: str, ground_truth: str) -> float:
    """
    计算单选题准确性奖励
    
    Args:
        prediction: 模型预测的完整回答
        ground_truth: 标准答案（单个字母，如 "A" 或 "B"）
        
    Returns:
        奖励分数：1.0 (完全正确) 或 0.0 (错误)
    """
    try:
        # 提取预测答案
        pred_answer = extract_answer(prediction, question_type="single_select")
        pred_answer = normalize_answer(pred_answer, question_type="single_select")
        
        # 标准化
        gt_answer = normalize_answer(ground_truth, question_type="single_select")
        
        if not pred_answer:
            return 0.0
        if pred_answer == gt_answer:
            return 1.0
        
        return 0.0
        
    except Exception as e:
        print(f"Error in compute_single_select_reward: {e}")
        return 0.0


def compute_multi_select_reward(prediction: str, ground_truth: str) -> float:
    """
    计算多选题准确性奖励
    
    Args:
        prediction: 模型预测的完整回答
        ground_truth: 标准答案（多个字母，如 "A,C" 或 "B,D,E"）
        
    Returns:
        奖励分数：1.0 (完全正确) 或 0.0 (部分错误/完全错误)
    """
    try:
        # 提取预测答案
        pred_answer = extract_answer(prediction, question_type="multi_select")
        pred_answer = normalize_answer(pred_answer, question_type="multi_select")
        
        # 标准化正确答案
        gt_answer = normalize_answer(ground_truth, question_type="multi_select")
        
        if not pred_answer:
            return 0.0
        
        # 将答案转换为集合（不考虑顺序）
        pred_set = set(pred_answer.split(',')) if pred_answer else set()
        gt_set = set(gt_answer.split(',')) if gt_answer else set()
        
        # 移除空字符串
        pred_set.discard('')
        gt_set.discard('')
        
        # 完全匹配
        if pred_set == gt_set:
            return 1.0
        
       #部分正确给部分分
        correct_count = len(pred_set & gt_set) 
        wrong_count = len(pred_set - gt_set)   
        missed_count = len(gt_set - pred_set)   
        if wrong_count > 0:  
            return 0.0
        elif correct_count == len(gt_set):  
            return 1.0
        else:  
            return correct_count / len(gt_set) * 0.5  
        
        return 0.0
        
    except Exception as e:
        print(f"Error in compute_multi_select_reward: {e}")
        return 0.0


def compute_calculation_reward(prediction: str, ground_truth: str) -> float:
    """
    计算填空题/计算题准确性奖励
    
    Args:
        prediction: 模型预测的完整回答
        ground_truth: 标准答案（数值，如 "10" 或 "3.14,5.2,1.0"）
        
    Returns:
        奖励分数：1.0 (完全正确) 或 0.0 (错误)
    """
    try:
        # 提取
        pred_answer = extract_answer(prediction, question_type="calculation")
        pred_answer = normalize_answer(pred_answer, question_type="calculation")
        
        # 标准化
        gt_answer = normalize_answer(ground_truth, question_type="calculation")
        
        if not pred_answer:
            return 0.0
        
        # 数值比较
        try:
            pred_nums = [float(x.strip()) for x in pred_answer.split(',') if x.strip()]
            gt_nums = [float(x.strip()) for x in gt_answer.split(',') if x.strip()]

            if len(pred_nums) != len(gt_nums):
                return 0.0

            for p, g in zip(pred_nums, gt_nums):
                # 相对误差或绝对误差
                if abs(p - g) > max(1e-6, abs(g) * 1e-4):
                    return 0.0
            
            return 1.0
            
        except (ValueError, TypeError):
            # 字符串精确匹配
            if pred_answer == gt_answer:
                return 1.0
            return 0.0
        
    except Exception as e:
        print(f"Error in compute_calculation_reward: {e}")
        return 0.0


def _clean_json_response(text: str) -> str:
    """清理 LLM 返回的 JSON 字符串"""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text

def compute_conversational_reward_llm(
    question_text: str,
    reference_answer: str,
    explanation: str,
    student_raw_output: str,
    language: str = "en"
) -> float:
    """
    同步调用 SiliconFlow API 进行评分
    """
    student_answer = extract_answer(student_raw_output, question_type="conversational_qa")
    if not student_answer:
        return 0.0

    is_chinese = language.lower().startswith("zh") or "cn" in language.lower()
    system_prompt = LLM_JUDGE_SYSTEM_PROMPT_ZH if is_chinese else LLM_JUDGE_SYSTEM_PROMPT_EN
    
    user_template = """
[Question]
{question}

[Reference Answer]
{reference}

[Detailed Explanation]
{explanation}

[Student Answer]
{student_answer}

Please evaluate and respond in JSON as specified.
""" if not is_chinese else """
[问题]
{question}

[参考答案]
{reference}

[详细解释]
{explanation}

[学生答案]
{student_answer}

请按要求输出 JSON。
"""
    
    user_prompt = user_template.format(
        question=question_text,
        reference=reference_answer,
        explanation=explanation if explanation else "N/A",
        student_answer=student_answer
    )

    try:
        # 同步调用 (Blocking Call)
        response = client.chat.completions.create(
            model=JUDGE_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.01,
            max_tokens=512,
            response_format={"type": "json_object"}
        )
        
        raw_content = response.choices[0].message.content
        cleaned_content = _clean_json_response(raw_content)
        result = json.loads(cleaned_content)
        score = float(result.get("score", 0.0))
        return max(0.0, min(1.0, score))
        
    except Exception as e:
        print(f"Judge API Error: {e}")
        return 0.0

def compute_length_penalty(text: str, language: str = "en") -> float:
    """
    计算长度惩罚 (Length Penalty)。
    根据语言自动选择针对 768 Tokens 的字符软限制。
    
    Args:
        text: 模型生成的文本
        language: 语言代码 ('en', 'zh', 'zh-cn' 等)
    
    Returns:
        penalty: 负数或 0.0
    """
    if not text:
        return 0.0
        
    if language and ("zh" in language.lower() or "cn" in language.lower()):
        soft_limit_chars = 1200
    else:
        soft_limit_chars = 2200
        
    penalty = 0.0
    current_len = len(text)
    
    if "</answer>" not in text and "\\boxed" not in text and "####" not in text:
        penalty -= 1.0 

    if current_len > soft_limit_chars:
        excess = current_len - soft_limit_chars
        len_penalty_score = (excess / 100) * 0.1
        penalty -= min(1.0, len_penalty_score)
        
    return penalty


def compute_score(
    data: Dict[str, Any], 
    pred: str, 
    format_weight: float, 
    accuracy_weight: float,
    length_penalty_weight: float = 1.0
) -> float:
    """
    计算单样本的加权总分 = (FormatScore * fw) + (AccuracyScore * aw)
    """


    # 1. 计算格式分
    fmt_score = compute_format_reward(pred)
    
    # 2. 计算准确分
    q_type = data.get("question_type", "single_select")
    ground_truth = data.get("answer_text", "")
    
    acc_score = 0.0
    if q_type == "single_select":
        acc_score = compute_single_select_reward(pred, ground_truth)
    elif q_type == "multi_select":
        acc_score = compute_multi_select_reward(pred, ground_truth)
    elif q_type == "calculation":
        acc_score = compute_calculation_reward(pred, ground_truth)
    elif q_type == "conversational_qa":
        acc_score = compute_conversational_reward_llm(
            question_text=data.get("question", ""),
            reference_answer=ground_truth,
            explanation=data.get("explanation_text", ""),
            student_raw_output=pred,
            language=data.get("language", "en")
        )
    lang = data.get("language", "en")
    len_penalty = compute_length_penalty(pred, language=lang)

    # 3. 加权求和
    total_score = (fmt_score * format_weight) + (acc_score * accuracy_weight) + (len_penalty * length_penalty_weight)
    return total_score

