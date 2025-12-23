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
        vllm_logprobs_dict: vLLM 返回的每个token位置的 logprobs
    
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


def calculate_entropy(logprobs_dict: Dict) -> float:
    """
    从单个token的 top-k logprobs 计算熵
    
    Args:
        logprobs_dict: {token_id: logprob}
    
    Returns:
        熵值
    """
    if not logprobs_dict:
        return 0.0
    
    logprobs = np.array(list(logprobs_dict.values()))
    probs = np.exp(logprobs)
    probs = probs / probs.sum()
    
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return entropy


def calculate_response_entropy(response_logprobs: List[Dict], debug_info: dict = None) -> Optional[float]:
    """
    计算整个 response 的平均熵（不确定性）
    
    Args:
        response_logprobs: vLLM 返回的 response 中所有 logprobs 列表
        debug_info: 用于收集调试信息的字典（可选）
    
    Returns:
        整个 response 的平均熵
    """
    if not response_logprobs:
        if debug_info is not None:
            debug_info["error"] = "response_logprobs is empty or None"
        return None
    
    if debug_info is not None:
        debug_info["num_tokens"] = len(response_logprobs)
        debug_info["first_token_logprobs_type"] = str(type(response_logprobs[0]))
        debug_info["first_token_logprobs_keys"] = list(response_logprobs[0].keys())[:3] if response_logprobs[0] else "empty"
    
    entropies = []
    for i, vllm_logprobs in enumerate(response_logprobs):
        logprobs_dict = extract_logprob_values(vllm_logprobs)
        
        if debug_info is not None and i == 0:
            debug_info["first_token_extracted_logprobs"] = {k: float(v) for k, v in list(logprobs_dict.items())[:3]}
        
        if logprobs_dict:
            entropy = calculate_entropy(logprobs_dict)
            entropies.append(entropy)
            
            if debug_info is not None and i < 3:
                if "sample_entropies" not in debug_info:
                    debug_info["sample_entropies"] = []
                debug_info["sample_entropies"].append(float(entropy))
    
    if not entropies:
        if debug_info is not None:
            debug_info["error"] = "No valid entropies calculated"
        return None
    
    avg_entropy = np.mean(entropies)
    
    if debug_info is not None:
        debug_info["num_valid_entropies"] = len(entropies)
        debug_info["avg_entropy"] = float(avg_entropy)
        debug_info["min_entropy"] = float(np.min(entropies))
        debug_info["max_entropy"] = float(np.max(entropies))
    
    return avg_entropy


def entropy_to_confidence(
    entropy: float,
    low_entropy_threshold: float = 0.5,
    high_entropy_threshold: float = 1.2
) -> float:
    """
    将熵转换为置信度，范围 [0, 1]
    
    修改点：根据实际数据分布调整阈值
    - 从 (0.3, 2.0) 改为 (0.5, 1.2)
    """
    if entropy <= low_entropy_threshold:
        return 1.0
    elif entropy >= high_entropy_threshold:
        return 0.0
    else:
        return 1.0 - (entropy - low_entropy_threshold) / (high_entropy_threshold - low_entropy_threshold)


def confidence_reward_v2(
    is_correct: bool,
    confidence: float,
    high_confidence_threshold: float = 0.8
) -> float:
    """
    改进的置信度奖励函数
    
    核心思想：
    1. 只奖励"答对且自信"
    2. 中性对待"答对但不自信"和"答错且不自信"
    3. 严惩"答错但自信"
    
    修改点：
    - 提高high_confidence_threshold从0.7到0.8，鼓励更高的置信度分离
    - 移除wrong_low_bonus（不再奖励答错）
    - 移除correct_low_penalty（不惩罚答对但谨慎的情况）
    """
    is_high_confidence = confidence >= high_confidence_threshold
    
    if is_correct and is_high_confidence:
        return 0.4      # 答对且自信：大奖励
    elif is_correct and not is_high_confidence:
        return 0.0      # 答对但不自信：中性（不惩罚谨慎）
    elif not is_correct and not is_high_confidence:
        return 0.0      # 答错且不自信：中性（诚实不确定是可接受的）
    else:  # 答错且自信
        return -0.6     # 答错还自信：严厉惩罚


def calculate_calibration_metrics(scores: List[Dict]) -> Dict[str, float]:
    """
    计算更好的校准指标
    
    Returns:
        包含多个校准指标的字典
    """
    # 修正：直接用 == 1.0 判断，因为accuracy是严格二元的
    correct_confidences = [s["confidence_value"] for s in scores 
                          if s["accuracy"] == 1.0 and s["confidence_value"] > 0]
    wrong_confidences = [s["confidence_value"] for s in scores 
                        if s["accuracy"] == 0.0 and s["confidence_value"] > 0]
    
    all_confidences = [s["confidence_value"] for s in scores if s["confidence_value"] > 0]
    all_accuracies = [s["accuracy"] for s in scores if s["confidence_value"] > 0]
    
    metrics = {}
    
    # 1. Separation（分离度）：正确和错误样本的置信度差距
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
    
    # 3. 各组的平均值和标准差
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
    high_confidence_threshold: float = 0.8
) -> Dict[str, Any]:
    """
    构建置信度混淆矩阵
    
    Returns:
        包含混淆矩阵统计的字典
    """
    # 分类统计
    correct_high_conf = 0  # 答对且高置信度
    correct_low_conf = 0   # 答对但低置信度
    wrong_high_conf = 0    # 答错但高置信度
    wrong_low_conf = 0     # 答错且低置信度
    
    correct_high_conf_list = []
    correct_low_conf_list = []
    wrong_high_conf_list = []
    wrong_low_conf_list = []
    
    for s in scores:
        # 修正：直接判断是否等于1.0，因为accuracy是严格二元的
        is_correct = (s["accuracy"] == 1.0)
        confidence = s["confidence_value"]
        is_high_conf = confidence >= high_confidence_threshold
        
        if confidence == 0:  # 跳过无效样本
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
        else:  # not is_correct and not is_high_conf
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
    low_entropy_threshold: float = 0.5,
    high_entropy_threshold: float = 1.2,
    high_confidence_threshold: float = 0.8
) -> list[dict[str, float]]:
    """
    计算奖励分数
    
    Args:
        reward_inputs: 输入列表，每个元素包含:
            - "response": 模型生成的 response 文本
            - "ground_truth": 正确答案
            - "response_logprobs": vLLM 返回的 response logprobs（可选）
    
    修改点：
    - 使用改进的confidence_reward_v2
    - 调整熵阈值：(0.3, 2.0) -> (0.5, 1.2)
    - 提高置信度阈值：0.7 -> 0.8
    - 添加混淆矩阵和更多校准指标
    - 修正所有 > 0.5 为 == 1.0 的二元判断
    - 在每个样本输出中添加混淆矩阵分类指标（4个二值变量）
    """
    accuracy_weight = 1.0 - format_weight - confidence_weight
    
    if DEBUG:
        print("\n" + "="*80)
        print("DEBUG: compute_score called")
        print(f"Batch size: {len(reward_inputs)}")
        print(f"Weights: accuracy={accuracy_weight:.3f}, format={format_weight:.3f}, confidence={confidence_weight:.3f}")
        print(f"Thresholds: low_entropy={low_entropy_threshold}, high_entropy={high_entropy_threshold}, high_conf={high_confidence_threshold}")
        
        if reward_inputs:
            first_input = reward_inputs[0]
            print(f"\nFirst sample keys: {list(first_input.keys())}")
            print(f"Has 'response_logprobs': {'response_logprobs' in first_input}")
            if 'response_logprobs' in first_input:
                logprobs = first_input['response_logprobs']
                print(f"response_logprobs type: {type(logprobs)}")
                if logprobs is not None:
                    print(f"response_logprobs length: {len(logprobs) if hasattr(logprobs, '__len__') else 'N/A'}")
                else:
                    print("response_logprobs is None!")
        print("="*80 + "\n")
    
    scores = []
    for idx, reward_input in enumerate(reward_inputs):
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        
        # 计算整个 response 的平均熵
        response_logprobs = reward_input.get("response_logprobs", None)
        
        debug_info = {} if (DEBUG and idx < DEBUG_SAMPLE_COUNT) else None
        
        if response_logprobs is not None:
            response_entropy = calculate_response_entropy(response_logprobs, debug_info)
            
            if response_entropy is not None:
                confidence = entropy_to_confidence(
                    response_entropy,
                    low_entropy_threshold=low_entropy_threshold,
                    high_entropy_threshold=high_entropy_threshold
                )
                
                # 使用改进的奖励函数，修正：直接判断 == 1.0
                conf_reward = confidence_reward_v2(
                    is_correct=(accuracy_score == 1.0),
                    confidence=confidence,
                    high_confidence_threshold=high_confidence_threshold
                )
                
                confidence_value = confidence
                # 修正：使用 == 1.0 和 == 0.0 进行二元判断
                is_calibrated = 1.0 if (
                    (accuracy_score == 1.0 and confidence >= high_confidence_threshold) or
                    (accuracy_score == 0.0 and confidence < high_confidence_threshold)
                ) else 0.0
                
                if DEBUG and idx < DEBUG_SAMPLE_COUNT:
                    print(f"\n--- Sample {idx} ---")
                    print(f"Response (first 100 chars): {response[:100]}...")
                    print(f"Ground truth: {reward_input['ground_truth']}")
                    print(f"Accuracy: {accuracy_score}")
                    print(f"Format: {format_score}")
                    print(f"Debug info: {debug_info}")
                    print(f"Entropy: {response_entropy:.4f}")
                    print(f"Confidence: {confidence:.4f}")
                    print(f"Confidence reward: {conf_reward:.4f}")
                    print(f"Is calibrated: {is_calibrated}")
            else:
                conf_reward = 0.0
                confidence_value = 0.0
                is_calibrated = 0.0
                response_entropy = 0.0
                
                if DEBUG and idx < DEBUG_SAMPLE_COUNT:
                    print(f"\n--- Sample {idx} (entropy calculation failed) ---")
                    print(f"Debug info: {debug_info}")
        else:
            conf_reward = 0.0
            confidence_value = 0.0
            is_calibrated = 0.0
            response_entropy = 0.0
            
            if DEBUG and idx < DEBUG_SAMPLE_COUNT:
                print(f"\n--- Sample {idx} (no logprobs) ---")
                print(f"Response (first 100 chars): {response[:100]}...")
                print("response_logprobs is None!")
        
        # 计算混淆矩阵分类（4个二值指标）
        # 这4个指标可以被后续流程平均，得到各类别的占比
        is_correct = (accuracy_score == 1.0)
        is_high_conf = (confidence_value >= high_confidence_threshold)
        
        confusion_correct_high = 1.0 if (is_correct and is_high_conf) else 0.0
        confusion_correct_low = 1.0 if (is_correct and not is_high_conf) else 0.0
        confusion_wrong_high = 1.0 if (not is_correct and is_high_conf) else 0.0
        confusion_wrong_low = 1.0 if (not is_correct and not is_high_conf) else 0.0
        
        # 计算总分
        base_score = accuracy_weight * accuracy_score + format_weight * format_score
        overall_score = base_score + confidence_weight * conf_reward
        overall_score = max(0.0, min(1.0, overall_score))
        
        if DEBUG and idx < DEBUG_SAMPLE_COUNT:
            print(f"Overall score: {overall_score:.4f} = {accuracy_weight:.2f}*{accuracy_score:.2f} + {format_weight:.2f}*{format_score:.2f} + {confidence_weight:.2f}*{conf_reward:.2f}")
            print(f"Confusion category: correct_high={confusion_correct_high}, correct_low={confusion_correct_low}, wrong_high={confusion_wrong_high}, wrong_low={confusion_wrong_low}")
        
        scores.append({
            "overall": overall_score,
            "format": format_score,
            "accuracy": accuracy_score,
            "confidence_calibration": conf_reward,
            "confidence_value": confidence_value,
            "response_entropy": response_entropy,
            "is_calibrated": is_calibrated,
            # 混淆矩阵分类指标（可平均得到各类别占比）
            "confusion_correct_high_conf": confusion_correct_high,
            "confusion_correct_low_conf": confusion_correct_low,
            "confusion_wrong_high_conf": confusion_wrong_high,
            "confusion_wrong_low_conf": confusion_wrong_low,
        })
    
    if DEBUG:
        print("\n" + "="*80)
        print("BATCH STATISTICS:")
        
        # 基础统计
        all_entropies = [s["response_entropy"] for s in scores if s["response_entropy"] > 0]
        all_confidences = [s["confidence_value"] for s in scores if s["confidence_value"] > 0]
        all_accuracies = [s["accuracy"] for s in scores]
        
        print(f"Samples with valid entropy: {len(all_entropies)} / {len(scores)}")
        if all_entropies:
            print(f"Entropy - mean: {np.mean(all_entropies):.4f}, std: {np.std(all_entropies):.4f}, min: {np.min(all_entropies):.4f}, max: {np.max(all_entropies):.4f}")
        if all_confidences:
            print(f"Confidence - mean: {np.mean(all_confidences):.4f}, std: {np.std(all_confidences):.4f}, min: {np.min(all_confidences):.4f}, max: {np.max(all_confidences):.4f}")
        
        print(f"Accuracy - mean: {np.mean(all_accuracies):.4f}")
        
        calibration_rewards = [s["confidence_calibration"] for s in scores]
        print(f"Confidence calibration reward - mean: {np.mean(calibration_rewards):.4f}, min: {np.min(calibration_rewards):.4f}, max: {np.max(calibration_rewards):.4f}")
        
        # 计算校准指标
        print("\n" + "-"*80)
        print("CALIBRATION METRICS:")
        cal_metrics = calculate_calibration_metrics(scores)
        print(f"Separation (correct - wrong): {cal_metrics['separation']:.4f}")
        print(f"ECE (Expected Calibration Error): {cal_metrics['ece']:.4f}")
        print(f"Correct samples confidence: {cal_metrics['correct_conf_mean']:.4f} ± {cal_metrics['correct_conf_std']:.4f}")
        print(f"Wrong samples confidence: {cal_metrics['wrong_conf_mean']:.4f} ± {cal_metrics['wrong_conf_std']:.4f}")
        
        # 打印混淆矩阵
        print("\n" + "-"*80)
        print("CONFIDENCE CONFUSION MATRIX:")
        print(f"(Threshold: {high_confidence_threshold})")
        print("")
        conf_matrix = build_confidence_confusion_matrix(scores, high_confidence_threshold)
        
        print(f"{'Category':<20} {'Count':<8} {'Percentage':<12} {'Avg Confidence':<15}")
        print("-" * 60)
        print(f"{'Correct + High Conf':<20} {conf_matrix['correct_high_conf']['count']:<8} "
              f"{conf_matrix['correct_high_conf']['percentage']:<11.2f}% "
              f"{conf_matrix['correct_high_conf']['avg_confidence']:<15.4f}")
        print(f"{'Correct + Low Conf':<20} {conf_matrix['correct_low_conf']['count']:<8} "
              f"{conf_matrix['correct_low_conf']['percentage']:<11.2f}% "
              f"{conf_matrix['correct_low_conf']['avg_confidence']:<15.4f}")
        print(f"{'Wrong + High Conf':<20} {conf_matrix['wrong_high_conf']['count']:<8} "
              f"{conf_matrix['wrong_high_conf']['percentage']:<11.2f}% "
              f"{conf_matrix['wrong_high_conf']['avg_confidence']:<15.4f}")
        print(f"{'Wrong + Low Conf':<20} {conf_matrix['wrong_low_conf']['count']:<8} "
              f"{conf_matrix['wrong_low_conf']['percentage']:<11.2f}% "
              f"{conf_matrix['wrong_low_conf']['avg_confidence']:<15.4f}")
        print("-" * 60)
        print(f"{'Total':<20} {conf_matrix['total_samples']:<8}")
        
        # 验证混淆矩阵指标的平均值
        print("\n" + "-"*80)
        print("CONFUSION MATRIX METRICS (from per-sample indicators):")
        print(f"Correct + High Conf: {np.mean([s['confusion_correct_high_conf'] for s in scores]):.4f} ({np.mean([s['confusion_correct_high_conf'] for s in scores])*100:.2f}%)")
        print(f"Correct + Low Conf:  {np.mean([s['confusion_correct_low_conf'] for s in scores]):.4f} ({np.mean([s['confusion_correct_low_conf'] for s in scores])*100:.2f}%)")
        print(f"Wrong + High Conf:   {np.mean([s['confusion_wrong_high_conf'] for s in scores]):.4f} ({np.mean([s['confusion_wrong_high_conf'] for s in scores])*100:.2f}%)")
        print(f"Wrong + Low Conf:    {np.mean([s['confusion_wrong_low_conf'] for s in scores]):.4f} ({np.mean([s['confusion_wrong_low_conf'] for s in scores])*100:.2f}%)")
        
        # 理想分布提示
        total = conf_matrix['total_samples']
        if total > 0:
            correct_high_pct = conf_matrix['correct_high_conf']['percentage']
            wrong_low_pct = conf_matrix['wrong_low_conf']['percentage']
            good_calibration_pct = correct_high_pct + wrong_low_pct
            
            print(f"\nGood calibration ratio: {good_calibration_pct:.2f}% (Correct+High + Wrong+Low)")
            print(f"Target: >90% for well-calibrated model")
            
            if conf_matrix['correct_high_conf']['count'] > 0 and conf_matrix['wrong_low_conf']['count'] > 0:
                ideal_separation = conf_matrix['correct_high_conf']['avg_confidence'] - conf_matrix['wrong_low_conf']['avg_confidence']
                print(f"Ideal separation: {ideal_separation:.4f} (target: >0.30)")
        
        print("="*80 + "\n")
    
    return scores
