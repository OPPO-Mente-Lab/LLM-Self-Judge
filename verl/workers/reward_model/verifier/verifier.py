import logging
import os
import re
import json

import torch
from vllm import LLM, SamplingParams
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils import hf_tokenizer
from verl import DataProto
from tensordict import TensorDict
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from .prompts import (
    QWEN_2_5_JUDGE_PROMPT,
    QWEN_2_5_JUDGE_SYSTEM_PROMPT,
    QWEN3_JUDGE_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))

# QUERY_PROMPT = """
# Problem: {question}
# Reasoning and Answer (**Model Predicted Answer**): {prediction}
# """

QUERY_PROMPT = """
[PROBLEM]
{question}

[CANDIDATE_SOLUTION]
{prediction}

The image is provided above. Evaluate the candidate solution according to the system instructions.
"""

def extract_solution(solution_str: str) -> str:
    # Robustly extract content of \boxed{...} with balanced brace parsing (handles nesting).
    m = re.search(r'\\boxed\s*\{', solution_str)
    if not m:
        return "Invalid Answer. Please output [[NO]]."
    i = m.end()  # position right after the opening '{'
    depth = 1
    buf = []
    while i < len(solution_str) and depth > 0:
        ch = solution_str[i]
        # keep escaped chars as literals (avoid counting escaped braces)
        if ch == "\\" and i + 1 < len(solution_str):
            buf.append(ch)
            buf.append(solution_str[i + 1])
            i += 2
            continue
        if ch == "{":
            depth += 1
            buf.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                i += 1
                break
            buf.append(ch)
        else:
            buf.append(ch)
        i += 1
    ans = "".join(buf).strip().strip("$")
    return ans if ans else "Invalid Answer. Please output [[NO]]."

def qwen2_5_extract_judge(cur_judge: str) -> str:
    return cur_judge

def qwen3_extract_judge(cur_judge: str) -> str:
    match = re.search(r"</think>\s*(.*)", cur_judge, re.DOTALL)
    result = match.group(1).strip() if match else ""
    return result

def extract_overall_score(text: str) -> float:
    """
    Extract overall_score from the STRICT output format:
    [BEGIN THOUGHT]
    ...
    [END THOUGHT]
    [BEGIN SCORES]
    { "overall_score": <float>, ... }
    [END SCORES]
    Returns 0.0 on failure and clamps to [0, 1].
    """
    if not isinstance(text, str):
        return 0.0
    # Helper: clamp to [0,1]
    def _clamp_01(x: float) -> float:
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return x

    # Helper: extract last \boxed{...} content as float
    def _extract_last_boxed_value(s: str):
        key = r"\boxed{"
        start = s.rfind(key)
        if start == -1:
            return None
        i = start + len(key)
        depth = 1
        buf = []
        while i < len(s) and depth > 0:
            ch = s[i]
            # keep escaped sequence as literal
            if ch == "\\" and i + 1 < len(s):
                buf.append(ch)
                buf.append(s[i + 1])
                i += 2
                continue
            if ch == "{":
                depth += 1
                buf.append(ch)
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
                buf.append(ch)
            else:
                buf.append(ch)
            i += 1
        if depth != 0:
            return None
        ans = "".join(buf).strip().strip("$")
        try:
            val = float(ans)
            return _clamp_01(val)
        except Exception:
            return None

    # Helper: parse first balanced JSON in a snippet and fetch overall_score
    def _parse_overall_from_json_snippet(snippet: str):
        start_brace = snippet.find("{")
        if start_brace == -1:
            return None
        depth = 0
        end_brace = -1
        for idx in range(start_brace, len(snippet)):
            ch = snippet[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end_brace = idx
                    break
        if end_brace == -1:
            return None
        json_str = snippet[start_brace : end_brace + 1]
        try:
            obj = json.loads(json_str)
            score = obj.get("overall_score", 0.0)
            if isinstance(score, str):
                score = float(score.strip())
            return _clamp_01(float(score))
        except Exception:
            return None

    # 1) Try boxed fallback first (new strongest rule)
    boxed_val = _extract_last_boxed_value(text)
    if boxed_val is not None:
        return boxed_val

    # Define tags
    begin_tag = "[BEGIN SCORES]"
    end_tag = "[END SCORES]"
    alt_begin_tag = "[SCORES]"

    # Locate positions
    begin_pos = text.find(begin_tag)
    end_pos = text.find(end_tag, begin_pos + len(begin_tag)) if begin_pos != -1 else -1
    alt_begin_pos = text.find(alt_begin_tag)
    alt_end_pos = text.find(end_tag, alt_begin_pos + len(alt_begin_tag)) if alt_begin_pos != -1 else -1

    if begin_pos != -1 and end_pos != -1:
        candidate = text[begin_pos + len(begin_tag) : end_pos]
        val = _parse_overall_from_json_snippet(candidate)
        if val is not None:
            return val

    if alt_begin_pos != -1 and alt_end_pos != -1:
        candidate = text[alt_begin_pos + len(alt_begin_tag) : alt_end_pos]
        val = _parse_overall_from_json_snippet(candidate)
        if val is not None:
            return val

    if begin_pos != -1:
        candidate = text[begin_pos + len(begin_tag) :]
        val = _parse_overall_from_json_snippet(candidate)
        if val is not None:
            return val

    if alt_begin_pos != -1:
        candidate = text[alt_begin_pos + len(alt_begin_tag) :]
        val = _parse_overall_from_json_snippet(candidate)
        if val is not None:
            return val

    # 6) Last resort: not found
    return 0.0

class RewardModelWorker(Worker):
    def __init__(self, config):
        """
        Initializes the reward model worker with its configuration and sampling parameters.
        """
        super().__init__()
        self.config = config
        # Judge sampling parameters (configurable)
        judge_temperature = float(self.config.model.get("judge_temperature", 0.7))
        judge_top_p = float(self.config.model.get("judge_top_p", 0.9))
        params = {
            "temperature": judge_temperature,  # >0 enables stochastic sampling
            "top_p": judge_top_p,
            "max_tokens": 1024,
        }
        self.sampling_params = SamplingParams(**params)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """
        Initialize the language model and tokenizer.
        """
        self.verifier = LLM(
            # enable_sleep_mode=True,
            model=self.config.model.path,
            gpu_memory_utilization=self.config.model.gpu_memory_utilization,
            enable_prefix_caching=True,
        )
        self.tokenizer = hf_tokenizer(
            self.config.model.path,
            trust_remote_code=self.config.model.get("trust_remote_code", False)
        )
        self.processor = AutoProcessor.from_pretrained(
            self.config.model.path,
            trust_remote_code=self.config.model.get("trust_remote_code", False)
        )
        self.verifier.sleep(2)
        torch.cuda.empty_cache()

    def extract_responses_list(self, tokenizer, input_ids: torch.Tensor, multi_turn_response_mask: torch.Tensor):
        diff = torch.diff(multi_turn_response_mask, prepend=torch.tensor([0], device=multi_turn_response_mask.device))
        starts = torch.where(diff == 1)[0]
        mask_appended = torch.cat([multi_turn_response_mask, torch.tensor([0], device=multi_turn_response_mask.device)], dim=0)
        diff_end = torch.diff(mask_appended)
        ends = torch.where(diff_end == -1)[0] - 1
        segments = []
        for s, e in zip(starts, ends):
            segments.append(input_ids[s:e+1].tolist())

        # Decode each segment
        # decoded_responses = [tokenizer.decode(seg, skip_special_tokens=True) for seg in segments]
        decoded_responses = tokenizer.batch_decode(segments, skip_special_tokens=True)
        return decoded_responses

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @torch.no_grad()
    def compute_rm_score(self, data: DataProto) -> DataProto:
        """
        Compute the reward model score for each data item.
        
        For every data instance, the function decodes the sequence of prompt and response
        tokens, extracts the solution, and then uses a language model to verify the answer.
        A reward score is then computed based on whether the verified answer is correct and the
        token length difference from the ground truth.
        
        Returns:
            A DataProto object containing the computed reward scores.
        """
        torch.cuda.empty_cache()
        self.verifier.wake_up()
        response_strs = []
        ground_truths = []
        question_texts = []
        valid_response_lengths = []

        # Process each data item to create a sequence string and extract necessary fields.
        for i in range(len(data)):
            data_item = data[i]
                
            prompt_ids = data_item.batch["prompts"]  # multimodal_dataset line269
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = int(data_item.batch["attention_mask"][:prompt_length].sum())
            response_ids = data_item.batch["responses"]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum())
            valid_response_lengths.append(valid_response_length)

            if 'multi_turn_response_mask' in data_item.batch:
                response_str_list = self.extract_responses_list(
                    self.tokenizer,
                    data_item.batch['input_ids'],
                    data_item.batch['multi_turn_response_mask']
                )
                response_str = ' '.join(response_str_list)
            else:
                response_str = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            response_strs.append(response_str)
            # Extract question and ground truth from non-tensor batch.
            # question = data_item.non_tensor_batch["problem"]
            rm = data_item.non_tensor_batch.get("reward_model", {})
            ground_truth = rm.get("ground_truth", None)

            raw_prompt_messages = data_item.non_tensor_batch.get("raw_prompt", [])
            base_content = []

            if isinstance(raw_prompt_messages, list) and len(raw_prompt_messages) > 0:
                base_content = raw_prompt_messages[0].get("content", [])

            q_text = " ".join([seg.get("text", "") for seg in base_content if isinstance(seg, dict) and seg.get("type") == "text"])
            # 'Find x. You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}.'
            # breakpoint()
            question_texts.append(q_text)
            ground_truths.append(ground_truth)

        if "Qwen2.5" in self.config.model.path:
            # SYSTEM_PROMPT = QWEN_2_5_JUDGE_SYSTEM_PROMPT
            SYSTEM_PROMPT = QWEN_2_5_JUDGE_PROMPT
            extract_judge = qwen2_5_extract_judge
        elif "Qwen3" in self.config.model.path:
            SYSTEM_PROMPT = QWEN3_JUDGE_SYSTEM_PROMPT
            extract_judge = qwen3_extract_judge
        else:
            # raise NotImplementedError(f"{self.config.model.path} is NOT Supported for LLM-as-Judge Reward Model.")
            SYSTEM_PROMPT = QWEN_2_5_JUDGE_PROMPT
            extract_judge = qwen2_5_extract_judge
            
        # Extract solutions from the decoded sequences.
        
        # solutions = [extract_solution(response_str) for response_str in response_strs]
        solutions = response_strs
        # breakpoint()
        # Prepare messages for the verification prompt.
        llm_inputs = []
        for i in range(len(solutions)):
            # reuse original raw_prompt for image parts
            raw_prompt_messages = data[i].non_tensor_batch.get("raw_prompt", [])
            base_content = []
            if isinstance(raw_prompt_messages, list) and len(raw_prompt_messages) > 0:
                base_content = raw_prompt_messages[0].get("content", [])
            image_parts = [seg for seg in base_content if isinstance(seg, dict) and seg.get("type") == "image"]

            input_query = QUERY_PROMPT.format(question=question_texts[i], prediction=solutions[i])
            # breakpoint()
            judge_messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": SYSTEM_PROMPT},
                    ],
                },
                {
                    "role": "user",
                    "content": image_parts + [{"type": "text", "text": input_query}],
                },
            ]
            prompt = self.processor.apply_chat_template(
                judge_messages, tokenize=False, add_generation_prompt=True
            )
            mm_data = {}
            # Use images already attached in non_tensor_batch (produced by rollout), not from judge_messages
            sample_mmd = data[i].non_tensor_batch.get("multi_modal_data", None)
            if isinstance(sample_mmd, dict) and "image" in sample_mmd:
                mm_data["image"] = sample_mmd["image"]
            llm_inputs.append({"prompt": prompt, "multi_modal_data": mm_data})
        # Generate verification responses using the language model.
        # breakpoint()
        print(">>> LLM-as-Judge Inference Start.")
        # Repeat judge multiple times and average scores to reduce variance
        num_repeats = int(self.config.model.get("num_judge_repeats", 1))
        all_scores = []
        responses = None
        # for _ in range(num_repeats):
        #     outputs = self.verifier.generate(llm_inputs, sampling_params=self.sampling_params)
        #     cur_responses = [extract_judge(output.outputs[0].text.strip()) for output in outputs]
        #     # Keep the most recent responses for logging purpose (minimal change to return payload)
        #     responses = cur_responses
        #     cur_scores = [extract_overall_score(r) for r in cur_responses]
        #     all_scores.append(cur_scores)
        # Single-call multi-sample generate to reduce per-call overhead while keeping statistics identical
        repeat_params = SamplingParams(
            temperature=self.sampling_params.temperature,
            top_p=self.sampling_params.top_p,
            max_tokens=self.sampling_params.max_tokens,
            n=num_repeats,
        )
        outputs = self.verifier.generate(llm_inputs, sampling_params=repeat_params)
        # Build as [num_repeats][batch] to match downstream LCB aggregation
        for k in range(num_repeats):
            cur_scores = []
            for out in outputs:
                cand_text = extract_judge(out.outputs[k].text.strip())
                cur_scores.append(extract_overall_score(cand_text))
            all_scores.append(cur_scores)
        # Keep "responses" close to previous behavior (use last repeat result)
        responses = [extract_judge(out.outputs[-1].text.strip()) for out in outputs]
        print(">>> LLM-as-Judge Inference End.")
        # Combine mean and uncertainty (LCB-style): s_eff = clamp(mu - alpha * sigma, 0, 1)
        _scores = torch.tensor(all_scores, dtype=torch.float32)  # shape: [num_repeats, batch]
        mu = _scores.mean(dim=0)
        overall_scores = torch.clamp(mu, min=0.0, max=1.0).tolist()
        # breakpoint()
        # Initialize reward tensor with the same shape as responses.
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        # breakpoint()
        # Compute a reward score for each data item.
        already_print_data = 0
        for i, (question, solution, ground_truth, verification, valid_response_length, score) in enumerate(
            zip(question_texts, solutions, ground_truths, responses, valid_response_lengths, overall_scores)
        ):
            # Record the score at the final valid response token index.
            reward_tensor[i, valid_response_length - 1] = score
            if already_print_data < 5:
                already_print_data += 1
                print("### Verification Result: ###")
                print("[QUESTION]", question)
                print("[CANDIDATE_SOLUTION]", solution)
                print("[GROUND_TRUTH]", ground_truth)
                print("[VERIFICATION]", verification)
                print("[OVERALL_SCORE]", score)
        batch = TensorDict({"rm_scores": reward_tensor}, batch_size=reward_tensor.shape[0])
        # Also return judge raw responses via non-tensor batch for downstream optional logging
        import numpy as np
        non_tensor_batch = {
            "judge_response": np.array(responses, dtype=object),
        }
        # Reduce idle wait to minimize per-batch overhead
        self.verifier.sleep(0)
        torch.cuda.empty_cache()
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)