# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""
Confidence Calibration Reward for RL Training (FINAL VERSION)

使用绝对Top-1概率（修正版）：
1. ✅ 使用top-1的绝对概率（不做归一化）
2. ✅ 对称的四象限reward设计（V4）
3. ✅ 完整的校准指标监控

关键修正：
之前用归一化会高估confidence（因为忽略top-5外的概率）
现在用绝对概率，真实反映模型不确定性
"""

import re
import numpy as np
from typing import Any, Dict, List, Optional
from mathruler.grader import extract_boxed_content, grade_answer

# Metadata
REWARD_NAME = "math_with_entropy"
REWARD_TYPE = "batch"

# Debug flag
DEBUG = True
DEBUG_SAMPLE_COUNT = 3


def format_reward(response: str) -> float:
    """Check format"""
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    """Check accuracy"""
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def extract_logprob_values(vllm_logprobs_dict: Dict) -> Dict[int, float]:
    """
    从 vLLM 的 Logprob 对象中提取真正的 logprob 值
    
    Args:
        vllm_logprobs_dict: vLLM 返回的每个token位置的 logprobs (top-5)
    
    Returns:
        {token_id: logprob_value}
    """
    if not vllm_logprobs_dict:
        return {}
    
    logprobs_dict = {}
    for token_id, logprob_obj in vllm_logprobs_dict.items():
        if hasattr(logprob_obj, 'logprob'):
            logprobs_dict[token_id] = logprob_obj.logprob
        else:
            logprobs_dict[token_id] = float(logprob_obj)
    
    return logprobs_dict


def get_token_confidence_absolute(logprobs_dict: Dict[int, float]) -> float:
    """
    ⭐⭐⭐ 最终推荐方法: 绝对Top-1概率
    
    直接使用top-1的绝对概率，不做归一化
    
    为什么这样更好：
    1. 真实反映模型不确定性（不会因为只看top-5而高估）
    2. 数学上正确（模型输出的概率已经对全词表归一化）
    3. 避免归一化带来的系统性高估（当top-5外有较多概率时）
    
    例子：
    - 如果top-1占70%的概率 → confidence = 0.70
    - 如果top-1只占20%（因为有很多其他选项）→ confidence = 0.20
    
    对比归一化方法：
    - 场景：top-1=0.37, top-5总和=0.58（top-5外有42%）
    - 归一化：0.37/0.58 = 0.64 ❌（高估73%）
    - 绝对值：0.37 ✅（真实值）
    
    Args:
        logprobs_dict: {token_id: logprob} for top-5 tokens
    
    Returns:
        confidence: float in [0, 1], top-1的绝对概率
    """
    if not logprobs_dict:
        return 0.0
    
    # 直接用top-1的logprob
    top1_logprob = list(logprobs_dict.values())[0]
    
    # 转换为概率（绝对值，不归一化）
    confidence = float(np.exp(top1_logprob))
    
    return confidence


def calculate_response_confidence(
    response_logprobs: List[Dict], 
    debug_info: dict = None,
    aggregation_method: str = "mean"
) -> Optional[float]:
    """
    计算整个response的confidence（使用绝对概率）
    
    步骤：
    1. 对每个token，使用top-1的绝对概率作为confidence
    2. 聚合所有token的confidence（默认：算术平均）
    
    Args:
        response_logprobs: vLLM 返回的 response 中所有 logprobs 列表
        debug_info: 用于收集调试信息的字典（可选）
        aggregation_method: 聚合方法，可选 "mean"（推荐）, "geometric", "min"
    
    Returns:
        整个 response 的 confidence [0, 1]
    """
    if not response_logprobs:
        if debug_info is not None:
            debug_info["error"] = "response_logprobs is empty or None"
        return None
    
    if debug_info is not None:
        debug_info["num_tokens"] = len(response_logprobs)
        debug_info["first_token_logprobs_type"] = str(type(response_logprobs[0]))
        debug_info["first_token_logprobs_keys"] = list(response_logprobs[0].keys())[:5] if response_logprobs[0] else "empty"
    
    token_confidences = []
    
    for i, vllm_logprobs in enumerate(response_logprobs):
        # 提取logprob值
        logprobs_dict = extract_logprob_values(vllm_logprobs)
        
        if debug_info is not None and i == 0:
            debug_info["first_token_extracted_logprobs"] = {k: float(v) for k, v in list(logprobs_dict.items())[:5]}
        
        if logprobs_dict:
            # 使用绝对概率方法
            token_conf = get_token_confidence_absolute(logprobs_dict)
            token_confidences.append(token_conf)
            
            if debug_info is not None and i < 3:
                if "sample_token_confidences" not in debug_info:
                    debug_info["sample_token_confidences"] = []
                debug_info["sample_token_confidences"].append(float(token_conf))
    
    if not token_confidences:
        if debug_info is not None:
            debug_info["error"] = "No valid token confidences calculated"
        return None
    
    # 聚合token级别的confidence
    if aggregation_method == "mean":
        response_confidence = np.mean(token_confidences)
    elif aggregation_method == "geometric":
        # 几何平均，对极端低值更敏感
        response_confidence = np.exp(np.mean(np.log(np.array(token_confidences) + 1e-10)))
    elif aggregation_method == "min":
        # 最保守：取最小值
        response_confidence = np.min(token_confidences)
    else:
        response_confidence = np.mean(token_confidences)
    
    if debug_info is not None:
        debug_info["num_valid_tokens"] = len(token_confidences)
        debug_info["response_confidence"] = float(response_confidence)
        debug_info["confidence_min"] = float(np.min(token_confidences))
        debug_info["confidence_max"] = float(np.max(token_confidences))
        debug_info["confidence_std"] = float(np.std(token_confidences))
        debug_info["aggregation_method"] = aggregation_method
    
    return response_confidence


def confidence_reward_v4(is_correct: bool, confidence: float) -> float:
    """
    ⭐ 对称的四象限校准reward
    
    设计理念：
    - 在confidence=0.5时为中性点（reward=0）
    - 正确答案：鼓励confidence>0.5，惩罚confidence<0.5
    - 错误答案：鼓励confidence<0.5，惩罚confidence>0.5
    - 完全对称，避免模型偏向某一策略
    
    数学形式：
    - 正确时：reward = 2 * (confidence - 0.5) ∈ [-1.0, +1.0]
    - 错误时：reward = 2 * (0.5 - confidence) ∈ [-1.0, +1.0]
    """
    if is_correct:
        return 2.0 * (confidence - 0.5)
    else:
        return 2.0 * (0.5 - confidence)


def calculate_calibration_metrics(scores: List[Dict]) -> Dict[str, float]:
    """计算校准指标"""
    correct_confidences = [s["confidence_value"] for s in scores 
                          if s["accuracy"] == 1.0 and s["confidence_value"] > 0]
    wrong_confidences = [s["confidence_value"] for s in scores 
                        if s["accuracy"] == 0.0 and s["confidence_value"] > 0]
    
    all_confidences = [s["confidence_value"] for s in scores if s["confidence_value"] > 0]
    all_accuracies = [s["accuracy"] for s in scores if s["confidence_value"] > 0]
    
    metrics = {}
    
    # 1. Separation（分离度）
    if correct_confidences and wrong_confidences:
        metrics["separation"] = np.mean(correct_confidences) - np.mean(wrong_confidences)
    else:
        metrics["separation"] = 0.0
    
    # 2. ECE (Expected Calibration Error)
    if all_confidences and all_accuracies:
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        confidences_arr = np.array(all_confidences)
        accuracies_arr = np.array(all_accuracies)
        
        for i in range(n_bins):
            bin_mask = (confidences_arr >= bins[i]) & (confidences_arr < bins[i+1])
            if bin_mask.sum() > 0:
                bin_acc = accuracies_arr[bin_mask].mean()
                bin_conf = confidences_arr[bin_mask].mean()
                ece += bin_mask.sum() / len(all_confidences) * abs(bin_acc - bin_conf)
        
        metrics["ece"] = ece
    else:
        metrics["ece"] = 0.0
    
    # 3. Brier Score
    if all_confidences and all_accuracies:
        brier = np.mean([(c - a)**2 for c, a in zip(all_confidences, all_accuracies)])
        metrics["brier_score"] = brier
    else:
        metrics["brier_score"] = 0.0
    
    # 4. Sharpness (区分度)
    if all_confidences:
        metrics["sharpness"] = np.std(all_confidences)
    else:
        metrics["sharpness"] = 0.0
    
    # 5. 各组统计
    if correct_confidences:
        metrics["correct_conf_mean"] = np.mean(correct_confidences)
        metrics["correct_conf_std"] = np.std(correct_confidences)
    else:
        metrics["correct_conf_mean"] = 0.0
        metrics["correct_conf_std"] = 0.0
    
    if wrong_confidences:
        metrics["wrong_conf_mean"] = np.mean(wrong_confidences)
        metrics["wrong_conf_std"] = np.std(wrong_confidences)
    else:
        metrics["wrong_conf_mean"] = 0.0
        metrics["wrong_conf_std"] = 0.0
    
    return metrics


def build_confidence_confusion_matrix(
    scores: List[Dict],
    high_confidence_threshold: float = 0.5
) -> Dict[str, Any]:
    """构建置信度混淆矩阵（仅用于统计）"""
    correct_high_conf = 0
    correct_low_conf = 0
    wrong_high_conf = 0
    wrong_low_conf = 0
    
    correct_high_conf_list = []
    correct_low_conf_list = []
    wrong_high_conf_list = []
    wrong_low_conf_list = []
    
    for s in scores:
        is_correct = (s["accuracy"] == 1.0)
        confidence = s["confidence_value"]
        is_high_conf = confidence >= high_confidence_threshold
        
        if confidence == 0:
            continue
        
        if is_correct and is_high_conf:
            correct_high_conf += 1
            correct_high_conf_list.append(confidence)
        elif is_correct and not is_high_conf:
            correct_low_conf += 1
            correct_low_conf_list.append(confidence)
        elif not is_correct and is_high_conf:
            wrong_high_conf += 1
            wrong_high_conf_list.append(confidence)
        else:
            wrong_low_conf += 1
            wrong_low_conf_list.append(confidence)
    
    total = correct_high_conf + correct_low_conf + wrong_high_conf + wrong_low_conf
    
    matrix = {
        "total_samples": total,
        "correct_high_conf": {
            "count": correct_high_conf,
            "percentage": correct_high_conf / total * 100 if total > 0 else 0,
            "avg_confidence": np.mean(correct_high_conf_list) if correct_high_conf_list else 0,
        },
        "correct_low_conf": {
            "count": correct_low_conf,
            "percentage": correct_low_conf / total * 100 if total > 0 else 0,
            "avg_confidence": np.mean(correct_low_conf_list) if correct_low_conf_list else 0,
        },
        "wrong_high_conf": {
            "count": wrong_high_conf,
            "percentage": wrong_high_conf / total * 100 if total > 0 else 0,
            "avg_confidence": np.mean(wrong_high_conf_list) if wrong_high_conf_list else 0,
        },
        "wrong_low_conf": {
            "count": wrong_low_conf,
            "percentage": wrong_low_conf / total * 100 if total > 0 else 0,
            "avg_confidence": np.mean(wrong_low_conf_list) if wrong_low_conf_list else 0,
        },
    }
    
    return matrix


def compute_score(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = 0.1,
    confidence_weight: float = 0.15,
    confusion_matrix_threshold: float = 0.5,
    aggregation_method: str = "mean"
) -> list[dict[str, float]]:
    """
    计算奖励分数（最终版 - 使用绝对概率）
    
    Args:
        reward_inputs: 输入列表，每个元素包含:
            - "response": 模型生成的 response 文本
            - "ground_truth": 正确答案
            - "response_logprobs": vLLM 返回的 response logprobs（top-5）
        format_weight: 格式奖励权重
        confidence_weight: 置信度奖励权重
        confusion_matrix_threshold: 混淆矩阵的阈值（仅用于统计，不影响训练）
        aggregation_method: token confidence聚合方法 ("mean", "geometric", "min")
    
    核心特性：
    1. ⭐ 使用绝对top-1概率（不归一化，避免高估）
    2. ⭐ 对称的四象限reward设计（confidence_reward_v4）
    3. ⭐ 完整的校准指标监控
    """
    accuracy_weight = 1.0 - format_weight - confidence_weight
    
    if DEBUG:
        print("\n" + "="*80)
        print("DEBUG: compute_score called (FINAL VERSION - ABSOLUTE PROBABILITY)")
        print(f"Batch size: {len(reward_inputs)}")
        print(f"Weights: accuracy={accuracy_weight:.3f}, format={format_weight:.3f}, confidence={confidence_weight:.3f}")
        print(f"Aggregation method: {aggregation_method}")
        print(f"Confusion matrix threshold (for statistics only): {confusion_matrix_threshold}")
        print("\n🎯 Confidence method: Absolute Top-1 Probability (绝对top-1概率)")
        print("   → 直接用top-1的概率，不做归一化")
        print("   → 避免因只看top-5而高估confidence")
        print("\n🎯 Reward design: Symmetric Four-Quadrant (V4)")
        print("="*80 + "\n")
    
    scores = []
    for idx, reward_input in enumerate(reward_inputs):
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        
        # 计算整个 response 的 confidence
        response_logprobs = reward_input.get("response_logprobs", None)
        
        debug_info = {} if (DEBUG and idx < DEBUG_SAMPLE_COUNT) else None
        
        if response_logprobs is not None:
            response_confidence = calculate_response_confidence(
                response_logprobs, 
                debug_info,
                aggregation_method=aggregation_method
            )
            
            if response_confidence is not None:
                # 使用V4对称reward
                conf_reward = confidence_reward_v4(
                    is_correct=(accuracy_score == 1.0),
                    confidence=response_confidence
                )
                
                confidence_value = response_confidence
                
                # 是否校准（仅用于统计）
                is_calibrated = 1.0 if (
                    (accuracy_score == 1.0 and response_confidence >= confusion_matrix_threshold) or
                    (accuracy_score == 0.0 and response_confidence < confusion_matrix_threshold)
                ) else 0.0
                
                if DEBUG and idx < DEBUG_SAMPLE_COUNT:
                    print(f"\n--- Sample {idx} ---")
                    print(f"Response (first 100 chars): {response[:100]}...")
                    print(f"Accuracy: {accuracy_score}, Format: {format_score}")
                    print(f"Debug info: {debug_info}")
                    print(f"Response confidence (absolute): {response_confidence:.4f}")
                    print(f"Confidence reward (v4): {conf_reward:.4f}")
                    
                    # 展示reward设计的效果
                    if accuracy_score == 1.0:
                        print(f"  → ✅ Correct answer, conf={response_confidence:.2f}")
                        print(f"     Reward range: [-1.00 (conf=0) to +1.00 (conf=1)]")
                        print(f"     Current: {conf_reward:.2f}")
                    else:
                        print(f"  → ❌ Wrong answer, conf={response_confidence:.2f}")
                        print(f"     Reward range: [+1.00 (conf=0) to -1.00 (conf=1)]")
                        print(f"     Current: {conf_reward:.2f}")
            else:
                conf_reward = 0.0
                confidence_value = 0.0
                is_calibrated = 0.0
                
                if DEBUG and idx < DEBUG_SAMPLE_COUNT:
                    print(f"\n--- Sample {idx} (confidence calculation failed) ---")
                    print(f"Debug info: {debug_info}")
        else:
            conf_reward = 0.0
            confidence_value = 0.0
            is_calibrated = 0.0
            
            if DEBUG and idx < DEBUG_SAMPLE_COUNT:
                print(f"\n--- Sample {idx} (no logprobs) ---")
                print("response_logprobs is None!")
        
        # 计算混淆矩阵分类
        is_correct = (accuracy_score == 1.0)
        is_high_conf = (confidence_value >= confusion_matrix_threshold)
        
        confusion_correct_high = 1.0 if (is_correct and is_high_conf) else 0.0
        confusion_correct_low = 1.0 if (is_correct and not is_high_conf) else 0.0
        confusion_wrong_high = 1.0 if (not is_correct and is_high_conf) else 0.0
        confusion_wrong_low = 1.0 if (not is_correct and not is_high_conf) else 0.0
        confusion_valid_sample = 1.0 if confidence_value > 0 else 0.0
        
        # 计算总分
        base_score = accuracy_weight * accuracy_score + format_weight * format_score
        overall_score = base_score + confidence_weight * conf_reward
        overall_score = max(0.0, min(1.0, overall_score))
        
        scores.append({
            "overall": overall_score,
            "format": format_score,
            "accuracy": accuracy_score,
            "confidence_calibration": conf_reward,
            "confidence_value": confidence_value,
            "is_calibrated": is_calibrated,
            "confusion_correct_high_conf": confusion_correct_high,
            "confusion_correct_low_conf": confusion_correct_low,
            "confusion_wrong_high_conf": confusion_wrong_high,
            "confusion_wrong_low_conf": confusion_wrong_low,
            "confusion_valid_sample": confusion_valid_sample,
        })
    
    if DEBUG:
        print("\n" + "="*80)
        print("BATCH STATISTICS:")
        
        all_confidences = [s["confidence_value"] for s in scores if s["confidence_value"] > 0]
        all_accuracies = [s["accuracy"] for s in scores]
        
        print(f"Valid samples: {len(all_confidences)} / {len(scores)}")
        if all_confidences:
            print(f"Confidence - mean: {np.mean(all_confidences):.4f}, std: {np.std(all_confidences):.4f}, "
                  f"min: {np.min(all_confidences):.4f}, max: {np.max(all_confidences):.4f}")
        print(f"Accuracy - mean: {np.mean(all_accuracies):.4f}")
        
        # 校准指标
        print("\n" + "-"*80)
        print("CALIBRATION METRICS:")
        cal_metrics = calculate_calibration_metrics(scores)
        print(f"✓ Separation: {cal_metrics['separation']:.4f} (target: >0.30)")
        print(f"✓ ECE: {cal_metrics['ece']:.4f} (target: <0.10)")
        print(f"✓ Brier Score: {cal_metrics.get('brier_score', 0):.4f} (target: <0.15)")
        print(f"✓ Sharpness: {cal_metrics.get('sharpness', 0):.4f} (target: >0.20)")
        
        # 混淆矩阵
        print("\n" + "-"*80)
        conf_matrix = build_confidence_confusion_matrix(scores, confusion_matrix_threshold)
        print(f"CONFIDENCE CONFUSION MATRIX (threshold={confusion_matrix_threshold}):")
        
        total = conf_matrix['total_samples']
        if total > 0:
            good_calib = conf_matrix['correct_high_conf']['percentage'] + conf_matrix['wrong_low_conf']['percentage']
            print(f"✅ Correct+High: {conf_matrix['correct_high_conf']['percentage']:.1f}%")
            print(f"❌ Correct+Low:  {conf_matrix['correct_low_conf']['percentage']:.1f}%")
            print(f"❌ Wrong+High:   {conf_matrix['wrong_high_conf']['percentage']:.1f}%")
            print(f"✅ Wrong+Low:    {conf_matrix['wrong_low_conf']['percentage']:.1f}%")
            print(f"\n📊 Good Calibration: {good_calib:.1f}% (target: >90%)")
            
            if good_calib < 70:
                print("   ⚠️  Needs more training")
            elif good_calib < 85:
                print("   🔄 Improving")
            else:
                print("   ✅ Well-calibrated!")
        
        print("="*80 + "\n")
    
    return scores
