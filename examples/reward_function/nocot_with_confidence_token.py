import re
import numpy as np
from typing import Any, Dict, List, Optional
from mathruler.grader import grade_answer

# Metadata
REWARD_NAME = "math_with_confidence"
REWARD_TYPE = "batch"

# Confidence token IDs
HIGH_TOKEN_ID = 11892  # 设置为high对应的token_id
LOW_TOKEN_ID = 10303   # 设置为low对应的token_id


def format_reward(response: str) -> float:
    """检查格式：必须包含<confidence>标签，且high/low周围不能有空格"""
    pattern = re.compile(
        r"<confidence>(high|low)</confidence>",
        re.DOTALL | re.IGNORECASE
    )
    return 1.0 if pattern.search(response) else 0.0


def extract_answer(response: str) -> str:
    """从response中提取答案部分（移除confidence标签）"""
    answer = re.sub(
        r"<confidence>(high|low)</confidence>",
        "",
        response,
        flags=re.IGNORECASE | re.DOTALL
    )
    return answer.strip()


def accuracy_reward(response: str, ground_truth: str) -> float:
    """检查答案正确性"""
    answer = extract_answer(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def extract_confidence_label(response: str) -> Optional[str]:
    """从response中提取声明的confidence标签"""
    match = re.search(r"<confidence>(high|low)</confidence>", response, re.IGNORECASE)
    return match.group(1).lower() if match else None


def find_confidence_token_position(
    response_logprobs: List[Dict],
) -> Optional[Dict]:
    """
    找到confidence token的位置及其logprobs分布
    
    Returns:
        包含该位置所有候选token的logprobs字典，如果未找到返回None
    """
    if not response_logprobs or HIGH_TOKEN_ID is None or LOW_TOKEN_ID is None:
        return None
    
    # 从后往前搜索包含HIGH_TOKEN_ID或LOW_TOKEN_ID的位置
    for i in range(len(response_logprobs) - 1, max(0, len(response_logprobs) - 20), -1):
        token_logprobs_dict = response_logprobs[i]
        if not token_logprobs_dict:
            continue
        
        # 检查是否包含confidence token
        for token_id in token_logprobs_dict.keys():
            if token_id == HIGH_TOKEN_ID or token_id == LOW_TOKEN_ID:
                return token_logprobs_dict
    
    return None


def calculate_entropy(logprobs_dict: Dict) -> float:
    """计算该位置所有候选token的熵"""
    logprobs = np.array([
        lp.logprob if hasattr(lp, 'logprob') else float(lp)
        for lp in logprobs_dict.values()
    ])
    probs = np.exp(logprobs)
    probs = probs / probs.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return entropy


def confidence_reward_prob_based(
    is_correct: bool,
    declared_confidence: str,
    confidence_token_id: int,
    token_logprobs_dict: Dict,
) -> float:
    """
    基于confidence token概率和熵的奖励
    
    核心思想：
    - 期望confidence: 答对->high, 答错->low
    - 实际confidence正确：奖励 = prob × (base + certainty_bonus × certainty)
    - 实际confidence错误：惩罚 = -prob × (base + certainty_bonus × certainty)
    - certainty从熵计算：熵越低越确定，奖励/惩罚越强
    
    Returns:
        reward in [-1, 1]
    """
    expected_confidence = "high" if is_correct else "low"
    is_correct_confidence = (declared_confidence == expected_confidence)
    
    # 获取confidence token的logprob
    confidence_logprob = None
    for token_id, logprob_obj in token_logprobs_dict.items():
        if token_id == confidence_token_id:
            confidence_logprob = logprob_obj.logprob if hasattr(logprob_obj, 'logprob') else float(logprob_obj)
            break
    
    if confidence_logprob is None:
        return 0.0
    
    confidence_prob = np.exp(confidence_logprob)
    
    # 计算熵并转换为确定性分数
    entropy = calculate_entropy(token_logprobs_dict)
    # 动态计算最大熵：均匀分布的熵 = log(候选数量)
    num_candidates = len(token_logprobs_dict)
    max_entropy = np.log(num_candidates) if num_candidates > 1 else 1.0
    certainty = 1.0 - min(entropy / max_entropy, 1.0)  # [0, 1]，越大越确定
    
    if is_correct_confidence:
        # 正确且确定 -> 最大奖励
        return confidence_prob * (0.5 + 0.5 * certainty)
    else:
        # 错误且确定 -> 最大惩罚
        return -confidence_prob * (0.5 + 0.5 * certainty)


def compute_score(
    reward_inputs: List[Dict[str, Any]],
    format_weight: float = 0.1,
    confidence_weight: float = 0.2,
) -> List[Dict[str, float]]:
    """
    计算奖励分数
    
    Args:
        reward_inputs: 每个元素包含:
            - "response": 生成的文本
            - "ground_truth": 正确答案
            - "response_logprobs": vLLM返回的logprobs
        format_weight: 格式权重
        confidence_weight: 置信度权重
    
    规则：如果format未通过，overall_score = 0（不给任何奖励）
    """
    accuracy_weight = 1.0 - format_weight - confidence_weight
    
    scores = []
    for idx, reward_input in enumerate(reward_inputs):
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        
        declared_conf = extract_confidence_label(response)
        response_logprobs = reward_input.get("response_logprobs")
        
        conf_reward = 0.0
        confidence_prob = 0.0
        position_entropy = 0.0
        declared_conf_numeric = 0.0  # high=1.0, low=0.0, none=0.0
        
        if declared_conf:
            declared_conf_numeric = 1.0 if declared_conf == "high" else 0.0
        
        if declared_conf and response_logprobs:
            token_logprobs_dict = find_confidence_token_position(response_logprobs)
            
            if token_logprobs_dict:
                # 确定实际生成的token_id
                confidence_token_id = HIGH_TOKEN_ID if declared_conf == "high" else LOW_TOKEN_ID
                
                conf_reward = confidence_reward_prob_based(
                    is_correct=(accuracy_score == 1.0),
                    declared_confidence=declared_conf,
                    confidence_token_id=confidence_token_id,
                    token_logprobs_dict=token_logprobs_dict,
                )
                
                # 记录统计信息
                for token_id, logprob_obj in token_logprobs_dict.items():
                    if token_id == confidence_token_id:
                        logprob = logprob_obj.logprob if hasattr(logprob_obj, 'logprob') else float(logprob_obj)
                        confidence_prob = float(np.exp(logprob))
                        break
                
                position_entropy = calculate_entropy(token_logprobs_dict)
        
        # 计算总分：如果format未通过，overall_score = 0
        if format_score > 0:
            overall_score = (
                accuracy_weight * accuracy_score +
                format_weight * format_score +
                confidence_weight * conf_reward
            )
            overall_score = max(0.0, min(1.0, overall_score))
        else:
            overall_score = 0.0
        
        # 分类统计
        is_correct = (accuracy_score == 1.0)
        is_high_declared = (declared_conf == "high")
        expected_conf = "high" if is_correct else "low"
        is_confidence_correct = (declared_conf == expected_conf)
        
        scores.append({
            "overall": overall_score,
            "format": format_score,
            "accuracy": accuracy_score,
            "confidence_reward": conf_reward,
            "declared_confidence": declared_conf_numeric,  # high=1.0, low=0.0
            "confidence_prob": confidence_prob,
            "position_entropy": position_entropy,
            # 分类统计
            "correct_high": 1.0 if (is_correct and is_high_declared) else 0.0,
            "correct_low": 1.0 if (is_correct and not is_high_declared) else 0.0,
            "wrong_high": 1.0 if (not is_correct and is_high_declared) else 0.0,
            "wrong_low": 1.0 if (not is_correct and not is_high_declared) else 0.0,
            "confidence_correct": 1.0 if is_confidence_correct else 0.0,
        })
    
    return scores