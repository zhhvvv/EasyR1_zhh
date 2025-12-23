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
from typing import Any, Optional
from mathruler.grader import extract_boxed_content, grade_answer

# Metadata
REWARD_NAME = "math_with_confidence"
REWARD_TYPE = "batch"


def extract_confidence(response: str) -> Optional[str]:
    """Extract confidence level from response."""
    pattern = re.compile(r"<confidence>\s*(high|low)\s*</confidence>", re.IGNORECASE | re.DOTALL)
    match = pattern.search(response)
    if match:
        return match.group(1).lower()
    return None


def format_reward(response: str) -> float:
    """Check if response follows the required format with confidence."""
    pattern = re.compile(
        r"<think>.*?</think>.*?\\boxed\{.*?\}.*?<confidence>\s*(high|low)\s*</confidence>",
        re.DOTALL | re.IGNORECASE
    )
    format_match = pattern.search(response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    """Check if the answer is correct."""
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def confidence_reward(
    response: str, 
    ground_truth: str,
    correct_high_bonus: float = 0.2,      # ✅正确+high：给奖励
    correct_low_bonus: float = 0.0,       # ✅正确+low：不奖励不惩罚
    wrong_low_bonus: float = 0.1,         # ❌错误+low：给奖励（拒识能力）
    wrong_high_bonus: float = 0.0         # ❌错误+high：不奖励不惩罚
) -> dict[str, float]:
    """
    纯奖励机制 - 只奖励不惩罚
    
    Reward logic (NO PENALTIES):
    - ✅ Correct + High: +0.2 (完美校准，给奖励)
    - ✅ Correct + Low:   0.0 (做对了但不够自信，不给额外奖励)
    - ❌ Wrong + Low:   +0.1 (知道自己不确定，拒识能力好，给奖励)
    - ❌ Wrong + High:   0.0 (过度自信，但不额外惩罚，accuracy已经是0)
    
    核心思想：
    - 用正向激励代替负向惩罚
    - 让模型追求奖励而不是规避惩罚
    - 避免reward hacking
    """
    confidence = extract_confidence(response)
    is_correct = accuracy_reward(response, ground_truth) > 0.5
    
    if confidence is None:
        return {
            "confidence_score": 0.0,
            "has_confidence": 0.0,
            "confidence_numeric": 0.0
        }
    
    # Calculate confidence reward - 只有正数或0，没有负数
    if is_correct and confidence == "high":
        conf_reward = correct_high_bonus
    elif is_correct and confidence == "low":
        conf_reward = correct_low_bonus
    elif not is_correct and confidence == "low":
        conf_reward = wrong_low_bonus
    elif not is_correct and confidence == "high":
        conf_reward = wrong_high_bonus
    else:
        conf_reward = 0.0
    
    confidence_numeric = 1.0 if confidence == "high" else 0.0
    
    return {
        "confidence_score": conf_reward,
        "has_confidence": 1.0,
        "confidence_numeric": confidence_numeric
    }


def compute_score(
    reward_inputs: list[dict[str, Any]], 
    format_weight: float = 0.1,
    confidence_weight: float = 0.15,
    accuracy_weight: float = 0.75
) -> list[dict[str, float]]:
    """
    Compute overall score with format, accuracy, and confidence components.
    """
    # Normalize weights
    total_weight = format_weight + confidence_weight + accuracy_weight
    format_weight /= total_weight
    confidence_weight /= total_weight
    accuracy_weight /= total_weight
    
    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        conf_rewards = confidence_reward(response, reward_input["ground_truth"])
        
        base_score = accuracy_weight * accuracy_score + format_weight * format_score
        overall_score = base_score + confidence_weight * conf_rewards["confidence_score"]
        overall_score = max(0.0, min(1.0, overall_score))
        
        scores.append({
            "overall": overall_score,
            "format": format_score,
            "accuracy": accuracy_score,
            "confidence_calibration": conf_rewards["confidence_score"],
            "confidence_numeric": conf_rewards["confidence_numeric"],
            "has_confidence_tag": conf_rewards["has_confidence"],
        })
    
    return scores


def compute_score_custom(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = 0.1,
    accuracy_weight: float = 0.75,
    confidence_weight: float = 0.15,
    correct_high_bonus: float = 0.2,
    correct_low_bonus: float = 0.0,
    wrong_low_bonus: float = 0.1,
    wrong_high_bonus: float = 0.0
) -> list[dict[str, float]]:
    """
    纯奖励机制的自定义版本
    """
    total_weight = format_weight + confidence_weight + accuracy_weight
    format_weight /= total_weight
    confidence_weight /= total_weight
    accuracy_weight /= total_weight
    
    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        conf_rewards = confidence_reward(
            response, 
            reward_input["ground_truth"],
            correct_high_bonus=correct_high_bonus,
            correct_low_bonus=correct_low_bonus,
            wrong_low_bonus=wrong_low_bonus,
            wrong_high_bonus=wrong_high_bonus
        )
        
        base_score = accuracy_weight * accuracy_score + format_weight * format_score
        overall_score = base_score + confidence_weight * conf_rewards["confidence_score"]
        overall_score = max(0.0, min(1.0, overall_score))
        
        scores.append({
            "overall": overall_score,
            "format": format_score,
            "accuracy": accuracy_score,
            "confidence_calibration": conf_rewards["confidence_score"],
            "confidence_numeric": conf_rewards["confidence_numeric"],
            "has_confidence_tag": conf_rewards["has_confidence"],
        })
    
    return scores
