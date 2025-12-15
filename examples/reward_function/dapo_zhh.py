import re
from typing import Any
from mathruler.grader import extract_boxed_content, grade_answer

# Metadata
REWARD_NAME = "math"
REWARD_TYPE = "batch"


def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else -1.0


def soft_overlong_punishment(
    response_length: int,
    max_response_length: int,
    overlong_buffer_length: int
) -> float:
    expected_len = max_response_length - overlong_buffer_length
    if response_length <= expected_len:
        return 0.0
    elif response_length <= max_response_length:
        return (expected_len - response_length) / overlong_buffer_length
    else:
        return -1.0


def compute_score(
    reward_inputs: list[dict[str, Any]],
    max_response_length: int,
    overlong_buffer_length: int,
    overlong_penalty_factor: float,
    format_weight: float = 0.1,
) -> list[dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        overlong_score = soft_overlong_punishment(
            reward_input["response_length"],
            max_response_length,
            overlong_buffer_length
        )
        
        overall = (
            (1 - format_weight) * accuracy_score
            + format_weight * format_score
            + overlong_score * overlong_penalty_factor
        )
        
        scores.append({
            "overall": overall,
            "accuracy": accuracy_score,
            "format": format_score,
            "overlong": overlong_score,
            "accuracy_normalized": 0.5 * (accuracy_score + 1.0),
        })
    
    return scores
