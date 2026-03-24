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

from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
# from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from verl.utils.reward_score import answer_score
from verl.utils.reward_score import major_vote_score

def _judge_score_to_factor(
    s: torch.Tensor,
    *,
    t_h: float = 0.95,
    t_l: float = 0.40,
    lam_pos: float = 0.2,
    lam_neg: float = 0.2,
    tau_h: float = 0.02,
    tau_l: float = 0.05,
    clamp_min: float = 0.8,
    clamp_max: float = 1.2,
) -> torch.Tensor:
    """
    Smoothly map judge score s in [0,1] to a multiplicative factor g(s) in [clamp_min, clamp_max]:

      g(s) = 1 + lam_pos * sigmoid((s - t_h)/tau_h) - lam_neg * sigmoid((t_l - s)/tau_l)
    """
    s = torch.clamp(s, 0.0, 1.0)
    tau_h = float(tau_h)
    tau_l = float(tau_l)
    assert tau_h > 0 and tau_l > 0, "tau_h and tau_l must be > 0"
    sig_hi = torch.sigmoid((s - t_h) / tau_h)
    sig_lo = torch.sigmoid((t_l - s) / tau_l)
    g = 1.0 + lam_pos * sig_hi - lam_neg * sig_lo
    return torch.clamp(g, clamp_min, clamp_max)

@register("judge")
# @register("verifier")  # backward-compatible name
class VerifierRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", **kwargs): 
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.mode = kwargs.get("mode", None)
        # Hyperparameters for initial (majority-based soft scoring) and judge scaling.
        # Provided via config.reward_model.reward_kwargs.*
        self.mv_attachment = None  # placeholder to maintain indentation style
        self.mv_alpha = float(kwargs.get("alpha_dirichlet", 1.0))
        # NOTE: `beta` is used as the distribution sharpness for group-wise log-softmax normalization below.
        # Default requested: 1.0
        self.mv_beta = float(kwargs.get("beta", 1.0))
        self.mv_margin = float(kwargs.get("margin", 0.06))
        # Smooth judge-score-to-factor mapping params (defaults match Judge.py)
        self.judge_t_h = float(kwargs.get("judge_t_h", 0.95))
        self.judge_t_l = float(kwargs.get("judge_t_l", 0.40))
        self.judge_lam_pos = float(kwargs.get("judge_lam_pos", 0.2))
        self.judge_lam_neg = float(kwargs.get("judge_lam_neg", 0.2))
        # Keep tau fixed (do not expose as hyperparameters)
        self.judge_tau_h = 0.02
        self.judge_tau_l = 0.05
        self.judge_clamp_min = float(kwargs.get("judge_clamp_min", 0.8))
        self.judge_clamp_max = float(kwargs.get("judge_clamp_max", 1.2))
        clip_adv_val = kwargs.get("clip_adv", None)
        try:
            self.clip_adv = None if clip_adv_val in [None, "None", "none", "null", "Null"] else float(clip_adv_val)
        except Exception:
            self.clip_adv = None

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        # if "rm_scores" in data.batch.keys():
        #     if return_dict:
        #         reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
        #         reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
        #         return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
        #     else:
        #         return data.batch["rm_scores"]

        # assert 'rm_scores' in data.batch.keys()

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        # Collect decoded strings and metadata for grouping (uid) and scoring
        uid_to_indices = {}
        uid_to_responses = {}
        index_to_valid_resp_len = {}
        index_to_prompt = {}
        index_to_response = {}
        index_to_ground_truth = {}
        index_to_data_source = {}
        index_to_accuracy_score = {}
        index_to_extra_info = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores
            accuracy_score = (
                float(data_item.batch["rm_scores"].max().item())
                if "rm_scores" in data_item.batch.keys()
                else 0.0
            )
            uid = data_item.non_tensor_batch.get("uid", None)
            group_key = uid if uid is not None else f"__idx_{i}"

            index_to_valid_resp_len[i] = int(valid_response_length)
            index_to_prompt[i] = prompt_str
            index_to_response[i] = response_str
            index_to_ground_truth[i] = ground_truth
            index_to_data_source[i] = data_source
            index_to_accuracy_score[i] = accuracy_score
            index_to_extra_info[i] = extra_info

            uid_to_indices.setdefault(group_key, []).append(i)
            uid_to_responses.setdefault(group_key, []).append(response_str)

        if self.mode == "train":
            # Majority vote per uid-group, add format penalty; for winners add Judge penalty (1 - judge_score)
            for group_key, indices in uid_to_indices.items():
                responses = uid_to_responses[group_key]
                # Use Dirichlet-smoothed frequency-based initial scores per trajectory
                base_list = major_vote_score.compute_threshold_majority_rewards(
                    responses,
                    threshold=0.85,
                    use_boxed=True,
                )
                # format penalty: 0.0 if ok, -0.5 if not ok
                fmt_list = [answer_score.format_reward(resp) for resp in responses]
                # 1) compute raw rewards per group (after judge & format)
                group_rewards = []
                for local_idx, i in enumerate(indices):
                    base = float(base_list[local_idx])
                    fmt_pen = float(fmt_list[local_idx])  # 0.0 or -0.5
                    # Scale per-trajectory base by smooth multiplier based on judge score, then add format penalty
                    judge_score = float(index_to_accuracy_score[i])  # in [0,1]
                    # clamp to [0,1]
                    js = judge_score
                    if js < 0.0:
                        js = 0.0
                    elif js > 1.0:
                        js = 1.0
                    # smooth multiplier policy (sigmoid gates + clamp)
                    judge_factor = float(
                        _judge_score_to_factor(
                            torch.tensor(js, dtype=torch.float32),
                            t_h=self.judge_t_h,
                            t_l=self.judge_t_l,
                            lam_pos=self.judge_lam_pos,
                            lam_neg=self.judge_lam_neg,
                            tau_h=self.judge_tau_h,
                            tau_l=self.judge_tau_l,
                            clamp_min=self.judge_clamp_min,
                            clamp_max=self.judge_clamp_max,
                        ).item()
                    )
                    # final reward: base * judge_factor + format penalty; clamp to [0, 1]
                    combined = (base * judge_factor) + fmt_pen
                    if combined < 0.0:
                        combined = 0.0
                    elif combined > 1.0:
                        combined = 1.0
                    reward_val = combined
                    group_rewards.append(reward_val)

                # 2) After computing all per-trajectory rewards in this uid-group,
                #    model them as a distribution by replacing reward with:
                #      a_i = beta * r_i - logsumexp(beta * r)
                #    i.e. log-softmax(beta * r)
                #    Optional clip to [-clip_adv, clip_adv].
                try:
                    r = torch.tensor(group_rewards, dtype=torch.float32, device=reward_tensor.device)
                    r_scaled = self.mv_beta * r
                    b = torch.logsumexp(r_scaled, dim=0, keepdim=False)
                    a = r_scaled - b
                    if self.clip_adv is not None:
                        a = a.clamp(-self.clip_adv, self.clip_adv)
                    # overwrite group_rewards with transformed values (now log-prob like, <= 0)
                    group_rewards = [float(x.item()) for x in a]
                except Exception as _e:
                    # best-effort; fall back to raw rewards if distribution modeling fails
                    pass

                # write back raw (non-normalized) rewards and log
                for local_idx, i in enumerate(indices):
                    reward_val = float(group_rewards[local_idx])
                    reward_tensor[i, index_to_valid_resp_len[i] - 1] = reward_val

                    data_answer_source = index_to_data_source[i]
                    if data_answer_source not in already_print_data_sources:
                        already_print_data_sources[data_answer_source] = 0
                    if already_print_data_sources[data_answer_source] < self.num_examine:
                        already_print_data_sources[data_answer_source] += 1
                        print("[prompt]", index_to_prompt[i])
                        print("[response]", index_to_response[i])
                        print("[ground_truth]", index_to_ground_truth[i])
                        print("[score]", reward_val)
        else:
            # Original per-sample scoring for non-train modes
            for i in range(len(data)):
                score = self.compute_score(
                    data_source=index_to_data_source[i],
                    solution_str=index_to_response[i],
                    ground_truth=index_to_ground_truth[i],
                    extra_info=index_to_extra_info[i],
                    accuracy_score=index_to_accuracy_score[i],
                    mode=self.mode,
                )

                if isinstance(score, dict):
                    reward = score["score"]
                    for key, value in score.items():
                        reward_extra_info[key].append(value)
                else:
                    reward = score

                reward_tensor[i, index_to_valid_resp_len[i] - 1] = reward

                data_source = index_to_data_source[i]
                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0
                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print("[prompt]", index_to_prompt[i])
                    print("[response]", index_to_response[i])
                    print("[ground_truth]", index_to_ground_truth[i])
                    if isinstance(score, dict):
                        for key, value in score.items():
                            print(f"[{key}]", value)
                    else:
                        print("[score]", score)

        if return_dict:
            # build extra info for downstream logging
            # 1) judge_score comes from rm_scores (accuracy_score)
            # 2) format_penalty from format checker
            # 3) final_reward equals 'score' above
            # 4) judge_response propagated from RM worker if available
            # Re-walk the batch to assemble lists (guaranteed same order)
            judge_scores = []
            format_penalties = []
            final_rewards = []
            judge_responses = []
            for i in range(len(data)):
                item = data[i]
                # judge score (aka accuracy_score)
                acc_sc = float(item.batch["rm_scores"].max().item()) if "rm_scores" in item.batch.keys() else 0.0
                judge_scores.append(acc_sc)
                # format penalty
                fmt_pen = answer_score.format_reward(
                    self.tokenizer.decode(
                        item.batch["responses"][: int(item.batch["attention_mask"][len(item.batch["prompts"]) :].sum())],
                        skip_special_tokens=True,
                    )
                )
                format_penalties.append(fmt_pen)
                # final reward equals tensor last position value
                valid_resp_len = int(item.batch["attention_mask"][len(item.batch["prompts"]) :].sum())
                final_rewards.append(float(reward_tensor[i, valid_resp_len - 1].item()))
                # judge raw response (if RM provided)
                jr = item.non_tensor_batch.get("judge_response", None)
                if jr is None:
                    jr = ""
                judge_responses.append(jr)

            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {
                    **reward_extra_info,
                    "judge_score": judge_scores,
                    "format_penalty": format_penalties,
                    "final_reward": final_rewards,
                    "judge_response": judge_responses,
                },
            }
        else:
            return reward_tensor