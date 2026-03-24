import re
from collections import Counter
from typing import List, Optional

from mathruler.grader import extract_boxed_content
import math


def format_reward(predict_str: str) -> float:
    """
    Format penalty aligned with current training-time rule:
      - Return 0.0 if output matches the strict pattern:
        <think>...</think> ... \\boxed{...}
      - Otherwise return -0.5
    """
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 0.0 if match_result else -0.5


def _normalize_answer_for_vote(predict_str: str, use_boxed: bool = True) -> str:
    """
    Normalize a prediction into an answer string for voting comparison.
    When use_boxed=True, extract the content inside \\boxed{...}.
    """
    if use_boxed:
        try:
            answer = extract_boxed_content(predict_str)
        except Exception:
            answer = predict_str
    else:
        answer = predict_str
    return (answer or "").strip()


def _unique_majority_label(answers: List[str]) -> Optional[str]:
    """
    Determine a unique majority label (mode) among answers.
    - Returns the label if there is a unique mode with count >= 2.
    - Returns None if all answers are different (max count == 1) or there is a tie.
    """
    if not answers:
        return None
    counts = Counter(answers)
    most_common = counts.most_common()
    if not most_common:
        return None
    # If the top count is 1, all are different -> no pseudo label
    if most_common[0][1] <= 1:
        return None
    # If there is a tie on the top frequency, we treat it as no unique majority
    if len(most_common) >= 2 and most_common[0][1] == most_common[1][1]:
        return None
    return most_common[0][0]


def compute_majority_vote_rewards(predict_strs: List[str], use_boxed: bool = True) -> List[float]:
    """
    Compute base rewards (without format penalty) for a group of rollouts using majority vote.
    - Extract normalized answers (optionally via \\boxed{...}).
    - Find a unique majority label (pseudo label).
      * If unique majority exists: reward = 1.0 for answers matching the label, else 0.0.
      * If all different or tie: all rewards = 0.0.
    """
    normalized = [_normalize_answer_for_vote(s, use_boxed=use_boxed) for s in predict_strs]
    label = _unique_majority_label(normalized)
    if label is None:
        return [0.0 for _ in predict_strs]
    return [1.0 if ans == label else 0.0 for ans in normalized]


def compute_majority_vote_rewards_with_format(predict_strs: List[str], use_boxed: bool = True) -> List[float]:
    """
    Compute final rewards for a group of rollouts:
      final_reward[i] = majority_vote_reward[i] + format_penalty[i]
    where:
      - majority_vote_reward[i] in {0.0, 1.0} from compute_majority_vote_rewards
      - format_penalty[i] is 0.0 if pattern matches, else -0.5
    """
    base = compute_majority_vote_rewards(predict_strs, use_boxed=use_boxed)
    penalties = [format_reward(s) for s in predict_strs]
    return [b + p for b, p in zip(base, penalties)]


def compute_group_score(predict_strs: List[str], use_boxed: bool = True) -> List[float]:
    """
    Alias of compute_majority_vote_rewards_with_format for convenience.
    """
    return compute_majority_vote_rewards_with_format(predict_strs, use_boxed=use_boxed)


# ---------------- Advanced majority score with Dirichlet-smoothed frequencies ----------------
def _sigmoid(x: float) -> float:
    # numerically stable sigmoid for scalar x
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def compute_dirichlet_majority_rewards(
    predict_strs: List[str],
    *,
    alpha_dirichlet: float = 1.0,
    beta: float = 22.0,
    margin: float = 0.06,
    use_boxed: bool = True,
    eps: float = 1e-12,
) -> List[float]:
    """
    Compute a soft "majority" score per trajectory using a Dirichlet-smoothed frequency model.

    For a group of n answers (strings) to the same question:
      - Let counts c(a) be the frequency of each unique answer a (after optional \\boxed{} extraction).
      - Let A be the number of unique answers.
      - Define posterior-mean category probabilities:
            π̄_a = (c(a) + α) / (n + α * A)
      - Define Δ_k = log π̇(a_k) - log max_a π̄_a  (≤ 0).
      - Define certainty factor C = 1 - H(π̄)/log A, where H is Shannon entropy of π̄.
      - Initial score for trajectory k:
            r0_k = C * σ( β * (Δ_k + margin) ),  0 ≤ r0_k ≤ 1

    Returns:
      List[float] of length n with values in [0, 1].
    """
    n = len(predict_strs)
    if n == 0:
        return []

    # 1) Normalize answers (optionally extract \boxed{...})
    normalized = [_normalize_answer_for_vote(s, use_boxed=use_boxed) for s in predict_strs]
    counts = Counter(normalized)
    A = len(counts)
    if A == 0:
        return [0.0 for _ in range(n)]

    # Edge case: if all answers identical => C = 1, Δ=0 ⇒ r0 = σ(β * margin)
    if A == 1:
        val = _sigmoid(beta * (0.0 + margin))
        return [float(val) for _ in range(n)]

    denom = float(n + alpha_dirichlet * A)
    # 2) Posterior mean probabilities π̄_a
    pi_bar = {a: (cnt + alpha_dirichlet) / denom for a, cnt in counts.items()}

    # 3) Certainty factor C = 1 - H(π̄)/log A
    H = 0.0
    for a, p in pi_bar.items():
        if p > 0.0:
            H -= p * math.log(p + eps)
    logA = math.log(float(A))
    C = 0.0
    if logA > 0:
        C = 1.0 - (H / logA)
        # Clamp to [0,1] for numeric safety
        C = max(0.0, min(1.0, C))

    # 4) Δ_k per trajectory: log π̄(a_k) - log π̄_max  (≤ 0)
    pi_max = max(pi_bar.values())
    log_pi_max = math.log(pi_max + 0.0 + eps)

    out: List[float] = []
    for a_k in normalized:
        p_k = pi_bar.get(a_k, 0.0)
        log_pk = math.log(p_k + eps)
        delta = log_pk - log_pi_max  # ≤ 0
        z = beta * (delta + margin)
        r0 = C * _sigmoid(z)
        out.append(float(r0))
    return out

# ---------------- New rule: thresholded hard-majority else raw-frequency probability ----------------
def compute_threshold_majority_rewards(
    predict_strs: List[str],
    *,
    threshold: float = 0.85,
    use_boxed: bool = True,
) -> List[float]:
    """
    Compute per-trajectory rewards for a group of n rollouts under a simple rule:
      - Normalize answers (optionally extract \\boxed{...}) and count frequencies.
      - Let c_max be the highest count of any answer and p_max = c_max / n.
      - If p_max > threshold:
          * Assign reward 1.0 to all trajectories whose answer equals the majority one;
            others get 0.0.
        Else:
          * Assign raw frequency probability per answer: reward = count(answer) / n
            (e.g., n=8, an answer appearing twice gets 0.25 for each matching trajectory).

    Args:
        predict_strs: The list of answers for the same question (length n).
        threshold: The strict "exceeds" ratio to activate hard majority (default 0.85).
        use_boxed: Whether to normalize by extracting \\boxed{...}.

    Returns:
        List[float]: Rewards of length n in [0, 1].
    """
    n = len(predict_strs)
    if n == 0:
        return []

    normalized = [_normalize_answer_for_vote(s, use_boxed=use_boxed) for s in predict_strs]
    counts = Counter(normalized)

    if not counts:
        return [0.0 for _ in range(n)]

    # Identify top-frequency answer and its proportion
    most_common_answer, c_max = counts.most_common(1)[0]
    p_max = c_max / float(n)

    if p_max > threshold:
        # Hard majority: winners get 1.0, others 0.0
        return [1.0 if ans == most_common_answer else 0.0 for ans in normalized]

    # Otherwise: assign raw frequency probability per answer
    return [counts.get(ans, 0) / float(n) for ans in normalized]