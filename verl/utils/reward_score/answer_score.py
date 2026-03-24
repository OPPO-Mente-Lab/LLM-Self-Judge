import re

from mathruler.grader import extract_boxed_content, grade_answer


def format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 0.0 if match_result else -0.5


def acc_reward(predict_str: str, ground_truth: str, use_boxed: bool = True) -> float:
    if use_boxed:
        answer = extract_boxed_content(predict_str)
    else:
        answer = predict_str
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(
    predict_str: str,
    ground_truth: str,
    accuracy_score,
    use_boxed: bool = True,
) -> float:

    # if accuracy_score is None:
    #     acc = acc_reward(predict_str, ground_truth, use_boxed)
    # else:
    #     acc = float(accuracy_score)
    acc = float(accuracy_score)

    # 2) format penalty
    fmt_penalty = format_reward(predict_str)  # 0.0 (ok) or -0.5 (bad)

    return acc + fmt_penalty