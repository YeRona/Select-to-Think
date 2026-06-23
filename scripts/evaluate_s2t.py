"""Evaluate Select-to-Think decoding policies and collect S2T-local data."""

from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from parser import extract_answer, math_equal, parse_ground_truth, parse_question
from utils import (
    ModelBundle,
    append_jsonl,
    ensure_dir,
    load_data_splits,
    load_model_bundle,
    masked_group_norm_2d,
    resolve_stop_token_ids,
    score_candidates_from_reserved_logits,
    set_seed,
    write_json,
)


@dataclass(frozen=True)
class DecodingPolicy:
    """A single evaluation policy."""

    name: str
    model_role: str = "hybrid"  # "slm", "llm", or "hybrid"
    trigger_type: str = "none"  # "none", "kl", "entropy", or "random"
    guidance_type: str = "student_greedy"  # teacher_takeover, teacher_rerank, or s2t_local
    do_sample: bool = False
    self_consistency_k: int = 1


@dataclass
class EvaluationConfig:
    """Runtime configuration for S2T evaluation."""

    seed: int = 2025
    dataset: str = "math"
    train_num: int = 1000
    test_num: int = 300
    start_idx: int = 0
    output_dir: str = "./runs/s2t_eval"
    run_dir: str = ""

    policy_spec: str = "s2t_local"
    slm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    llm_model: str = "Qwen/Qwen2.5-32B-Instruct"
    slm_device: str = "cuda:0"
    llm_device: str = "cuda:1"

    student_top_k: int = 64
    candidate_k: int = 16
    candidate_prob_ratio: float = 0.1
    self_consistency_k: int = 5
    max_new_tokens: int = 0
    warmup_steps: int = 3

    calibration_num: int = 20
    trigger_quantile: float = 0.95
    kl_threshold: Optional[float] = None
    entropy_threshold: Optional[float] = None

    collect_s2t_data: bool = False
    s2t_data_dir: str = ""
    s2t_lora_ckpt: str = ""
    use_s2t_lora: bool = False
    num_bins: int = 16
    max_context_len: int = 512
    label_mode: str = "prob_bin"
    score_mode: str = "expected_bin"
    bin_softmax_temperature: float = 1.0
    score_group_norm: bool = False


@dataclass
class StepState:
    """Cached model state for one decoding problem."""

    generated_ids: List[int]
    prompt_len: int
    slm_out: Any
    llm_out: Any


def safe_policy_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_")


def default_max_new_tokens(dataset_name: str) -> int:
    if any(name in dataset_name for name in ["aime", "olympiadbench", "mmlu"]):
        return 4096
    return 1024


def build_decoding_policies(policy_spec: str, candidate_k: int, self_consistency_k: int) -> List[DecodingPolicy]:
    """Translate a CLI policy spec into explicit decoding policies."""

    tokens = [token.strip().lower() for token in re.split(r"[, ]+", policy_spec) if token.strip()]
    policies: List[DecodingPolicy] = []

    for token in tokens:
        if token in {"slm", "slm_only"}:
            policies.append(DecodingPolicy(name="SLM-only", model_role="slm"))
        elif token in {"slm_multi", "slm_sc"}:
            policies.append(
                DecodingPolicy(
                    name="SLM-only (self-consistency)",
                    model_role="slm",
                    do_sample=True,
                    self_consistency_k=self_consistency_k,
                )
            )
        elif token in {"llm", "llm_only"}:
            policies.append(DecodingPolicy(name="LLM-only", model_role="llm"))
        elif token in {"llm_multi", "llm_sc"}:
            policies.append(
                DecodingPolicy(
                    name="LLM-only (self-consistency)",
                    model_role="llm",
                    do_sample=True,
                    self_consistency_k=self_consistency_k,
                )
            )
        elif token in {"kl", "kl_takeover"}:
            policies.append(DecodingPolicy(name="KL teacher-takeover", trigger_type="kl", guidance_type="teacher_takeover"))
        elif token in {"entropy", "entropy_takeover"}:
            policies.append(
                DecodingPolicy(name="Entropy teacher-takeover", trigger_type="entropy", guidance_type="teacher_takeover")
            )
        elif token in {"random", "random_takeover"}:
            policies.append(DecodingPolicy(name="Random teacher-takeover", trigger_type="random", guidance_type="teacher_takeover"))
        elif token in {"teacher", "teacher_rerank", "help", "nope", "collect"}:
            policies.append(DecodingPolicy(name="KL teacher-rerank", trigger_type="kl", guidance_type="teacher_rerank"))
        elif token in {"entropy_teacher", "entropy_rerank"}:
            policies.append(
                DecodingPolicy(name="Entropy teacher-rerank", trigger_type="entropy", guidance_type="teacher_rerank")
            )
        elif token in {"random_teacher", "random_rerank"}:
            policies.append(DecodingPolicy(name="Random teacher-rerank", trigger_type="random", guidance_type="teacher_rerank"))
        elif token in {"s2t", "s2t_local", "ziprank"}:
            policies.append(DecodingPolicy(name="KL S2T-local", trigger_type="kl", guidance_type="s2t_local"))
        elif token in {"entropy_s2t", "entropy_s2t_local"}:
            policies.append(DecodingPolicy(name="Entropy S2T-local", trigger_type="entropy", guidance_type="s2t_local"))
        elif token in {"random_s2t", "random_s2t_local", "ranno"}:
            policies.append(DecodingPolicy(name="Random S2T-local", trigger_type="random", guidance_type="s2t_local"))
        else:
            raise ValueError(f"Unknown policy token: {token}")

    if not policies:
        policies.append(DecodingPolicy(name="KL S2T-local", trigger_type="kl", guidance_type="s2t_local"))
    return policies


def policy_needs_llm(policy: DecodingPolicy) -> bool:
    """Return whether a policy needs the teacher model during evaluation."""

    if policy.model_role == "llm":
        return True
    if policy.trigger_type == "kl":
        return True
    return policy.guidance_type in {"teacher_takeover", "teacher_rerank"}


class S2TDecoder:
    """Step-by-step decoder for hybrid S2T policies."""

    def __init__(self, config: EvaluationConfig, bundle: ModelBundle):
        self.config = config
        self.bundle = bundle
        if bundle.slm is not None:
            self.slm = bundle.slm
            self.slm_tokenizer = bundle.slm_tokenizer
            self.slm_formatter = bundle.slm_formatter
            self.stop_ids = resolve_stop_token_ids(self.slm_tokenizer)
            self.s2t_ids_tensor = torch.tensor(bundle.s2t_token_ids, device=self.slm.device, dtype=torch.long)
        else:
            self.slm = self.slm_tokenizer = self.slm_formatter = None
            self.stop_ids = []
            self.s2t_ids_tensor = torch.empty(0, dtype=torch.long)

    @torch.no_grad()
    def generate_baseline(self, question: str, data_name: str, policy: DecodingPolicy) -> Dict[str, Any]:
        """Run ordinary model.generate for SLM-only or LLM-only baselines."""

        if policy.model_role == "llm":
            model = self.bundle.llm
            tokenizer = self.bundle.llm_tokenizer
            formatter = self.bundle.llm_formatter
            temperature, top_p, top_k = 0.6, 0.95, None
        else:
            model = self.bundle.slm
            tokenizer = self.bundle.slm_tokenizer
            formatter = self.bundle.slm_formatter
            temperature, top_p, top_k = 0.7, 0.8, 20

        if model is None or tokenizer is None or formatter is None:
            raise RuntimeError(f"Policy {policy.name} requires a model that was not loaded.")

        inputs = formatter.build_inputs(question, data_name)
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device) if "attention_mask" in inputs else None
        prompt_len = input_ids.shape[1]
        stop_ids = resolve_stop_token_ids(tokenizer)

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id or stop_ids[0]

        start = time.time()
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=policy.do_sample,
            temperature=temperature if policy.do_sample else None,
            top_p=top_p if policy.do_sample else None,
            top_k=top_k,
            num_return_sequences=max(1, int(policy.self_consistency_k)),
            eos_token_id=stop_ids,
            pad_token_id=tokenizer.pad_token_id,
        )
        sequences = outputs[:, prompt_len:]
        texts = [text.strip() for text in tokenizer.batch_decode(sequences, skip_special_tokens=True)]
        return {
            "policy_name": policy.name,
            "generated_text": texts if len(texts) > 1 else texts[0],
            "total_steps": int(sequences.shape[1]) if sequences.ndim == 2 else 0,
            "generation_time_sec": time.time() - start,
        }

    def initialize_state(self, question: str, data_name: str, require_llm: bool, need_hidden: bool) -> StepState:
        """Run the prompt once and keep the initial KV cache."""

        if self.bundle.slm is None or self.bundle.slm_formatter is None:
            raise RuntimeError("Hybrid decoding requires the SLM.")

        prompt_slm = self.bundle.slm_formatter.build_prompt(question, data_name)
        slm_inputs = self.bundle.slm_tokenizer(prompt_slm, return_tensors="pt").to(self.bundle.slm.device)
        slm_out = self.bundle.slm(
            input_ids=slm_inputs.input_ids,
            attention_mask=slm_inputs.get("attention_mask"),
            use_cache=True,
            output_hidden_states=need_hidden,
        )

        llm_out = None
        if require_llm:
            if self.bundle.llm is None or self.bundle.llm_formatter is None:
                raise RuntimeError("This policy requires the teacher LLM.")
            if not self.bundle.same_vocab:
                raise NotImplementedError("This refactor supports teacher guidance only when SLM and LLM share a tokenizer.")
            prompt_llm = self.bundle.llm_formatter.build_prompt(question, data_name)
            llm_inputs = self.bundle.llm_tokenizer(prompt_llm, return_tensors="pt").to(self.bundle.llm.device)
            llm_out = self.bundle.llm(
                input_ids=llm_inputs.input_ids,
                attention_mask=llm_inputs.get("attention_mask"),
                use_cache=True,
            )

        return StepState(
            generated_ids=slm_inputs.input_ids[0].tolist(),
            prompt_len=int(slm_inputs.input_ids.shape[1]),
            slm_out=slm_out,
            llm_out=llm_out,
        )

    def compute_trigger(
        self,
        policy: DecodingPolicy,
        step: int,
        student_probs_on_support: torch.Tensor,
        teacher_probs_on_student_support: Optional[torch.Tensor],
        metrics: Dict[str, List[float]],
    ) -> bool:
        """Decide whether this step should invoke guidance.

        Trigger statistics are computed on the SLM top-k support only. The
        teacher distribution is used only for comparison on those same token ids;
        it does not expand the candidate set used by teacher-rerank or S2T-local.
        """

        if step < int(self.config.warmup_steps):
            return False
        if policy.trigger_type == "none":
            return False

        entropy = -torch.sum(student_probs_on_support * torch.log(student_probs_on_support.clamp_min(1e-12)))
        metrics["entropy"].append(float(entropy.item()))

        if policy.trigger_type == "entropy":
            threshold = self.config.entropy_threshold
            return False if threshold is None else bool(entropy > float(threshold))

        if policy.trigger_type == "random":
            return bool(torch.rand(1).item() > float(self.config.trigger_quantile))

        if policy.trigger_type == "kl":
            if teacher_probs_on_student_support is None:
                return False
            mask = teacher_probs_on_student_support > 1e-12
            if mask.sum().item() == 0:
                kl_value = torch.zeros((), device=student_probs_on_support.device)
            else:
                student_support = student_probs_on_support[mask]
                teacher_support = teacher_probs_on_student_support[mask]
                student_support = student_support / student_support.sum().clamp_min(1e-12)
                teacher_support = teacher_support / teacher_support.sum().clamp_min(1e-12)
                kl_value = torch.sum(
                    student_support
                    * torch.log(student_support.clamp_min(1e-12) / teacher_support.clamp_min(1e-12))
                )
            metrics["kl"].append(float(kl_value.item()))
            threshold = self.config.kl_threshold
            return False if threshold is None else bool(kl_value > float(threshold))

        raise ValueError(f"Unknown trigger_type: {policy.trigger_type}")

    def sample_student_candidates(
        self,
        slm_topk_ids: torch.Tensor,
        slm_probs_topk: torch.Tensor,
    ) -> torch.Tensor:
        """Build the candidate set using only SLM top-k information."""

        candidate_k = int(self.config.candidate_k)
        if candidate_k == 0:
            max_prob = float(slm_probs_topk.max().item())
            keep = slm_probs_topk > max_prob * float(self.config.candidate_prob_ratio)
            candidate_ids = slm_topk_ids[torch.nonzero(keep, as_tuple=True)[0]]
            if candidate_ids.numel() == 0:
                candidate_ids = slm_topk_ids[torch.argmax(slm_probs_topk)].view(1)
            return candidate_ids

        candidate_count = min(max(1, candidate_k), int(slm_topk_ids.numel()))
        return slm_topk_ids[:candidate_count]

    @torch.no_grad()
    def score_s2t_local_candidates(self, prefix_ids: List[int], candidate_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score candidates with the S2T-local reserved-bin head."""

        max_prefix = max(1, int(self.config.max_context_len) - 1)
        prefix_ids = prefix_ids[-max_prefix:]
        prefix = torch.tensor(prefix_ids, device=self.slm.device, dtype=torch.long).unsqueeze(0)
        candidate_count = int(candidate_ids.numel())
        prefix_batch = prefix.expand(candidate_count, -1)
        candidate_col = candidate_ids.view(candidate_count, 1).to(self.slm.device)
        input_ids = torch.cat([prefix_batch, candidate_col], dim=1)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        out = self.slm(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits_bins = out.logits[:, -1, :][:, self.s2t_ids_tensor]
        scores, probs = score_candidates_from_reserved_logits(
            logits_bins,
            label_mode=self.config.label_mode,
            score_mode=self.config.score_mode,
            bin_softmax_temperature=self.config.bin_softmax_temperature,
        )
        if self.config.score_group_norm:
            mask = torch.ones((1, scores.numel()), device=scores.device, dtype=torch.float32)
            scores = masked_group_norm_2d(scores.view(1, -1), mask).view(-1)
        return scores, probs

    def collect_training_group(
        self,
        policy: DecodingPolicy,
        problem_id: int,
        step: int,
        prefix_ids: List[int],
        candidate_ids: torch.Tensor,
        teacher_scores: torch.Tensor,
    ) -> None:
        """Write a teacher-labeled candidate group for S2T-local training."""

        if not self.config.collect_s2t_data:
            return
        if policy.guidance_type != "teacher_rerank":
            return
        if teacher_scores.sum().item() <= 0.0:
            return

        max_prefix = max(1, int(self.config.max_context_len) - 1)
        record = {
            "problem_id": int(problem_id),
            "step": int(step),
            "group_id": f"{int(problem_id)}_{int(step)}",
            "context_ids": list(prefix_ids[-max_prefix:]),
            "cand_ids": [int(x) for x in candidate_ids.detach().cpu().tolist()],
            "teacher_prob_abs": [float(x) for x in teacher_scores.detach().float().cpu().tolist()],
            "teacher_sum_on_K": float(teacher_scores.sum().item()),
            "k_eff": int(candidate_ids.numel()),
            "candidate_source": "slm_topk",
            "teacher_label_source": "teacher_full_distribution_on_candidate_ids",
        }
        file_name = f"s2t_samples_{safe_policy_name(policy.name)}.jsonl"
        append_jsonl(os.path.join(self.config.s2t_data_dir, file_name), record)

    @torch.no_grad()
    def generate_hybrid(
        self,
        question: str,
        data_name: str,
        policy: DecodingPolicy,
        problem_id: int,
        log_dir: str,
    ) -> Dict[str, Any]:
        """Run step-wise S2T decoding for one problem."""

        require_llm = policy_needs_llm(policy)
        state = self.initialize_state(question, data_name, require_llm=require_llm, need_hidden=False)

        start = time.time()
        metrics: Dict[str, List[float]] = {
            "kl": [],
            "entropy": [],
            "slm_selected_probs": [],
            "teacher_selected_probs": [],
            "candidate_count": [],
        }
        token_history: List[str] = []
        trigger_count = 0
        flipped_count = 0
        teacher_top1_hits = 0

        for step in range(int(self.config.max_new_tokens)):
            slm_logits = state.slm_out.logits[:, -1, :].squeeze(0).float()
            slm_logits[self.s2t_ids_tensor] = -float("inf")

            teacher_logits = None
            teacher_probs_full = None
            teacher_top_id = None
            if require_llm and state.llm_out is not None:
                teacher_logits = state.llm_out.logits[:, -1, :].squeeze(0).to(self.slm.device).float()
                if self.bundle.same_vocab:
                    teacher_logits[self.s2t_ids_tensor] = -float("inf")
                teacher_probs_full = F.softmax(teacher_logits, dim=0)
                teacher_top_id = int(torch.argmax(teacher_probs_full).item())

            top_k = min(int(self.config.student_top_k), slm_logits.numel())
            slm_topk_logits, slm_topk_ids = torch.topk(slm_logits, k=top_k)
            slm_probs_full = F.softmax(slm_logits, dim=0)

            # Trigger comparison is intentionally restricted to the SLM top-k support.
            # Teacher probabilities are only read on those same token ids.
            slm_probs_topk_abs = slm_probs_full[slm_topk_ids]
            slm_probs_on_trigger_support = slm_probs_topk_abs / slm_probs_topk_abs.sum().clamp_min(1e-12)
            slm_probs_topk_cond = F.softmax(slm_topk_logits, dim=0)
            teacher_probs_on_trigger_support = (
                teacher_probs_full[slm_topk_ids] if teacher_probs_full is not None else None
            )

            triggered = self.compute_trigger(
                policy,
                step,
                slm_probs_on_trigger_support,
                teacher_probs_on_trigger_support,
                metrics,
            )

            student_top_id = int(slm_topk_ids[0].item())
            next_token_id = student_top_id
            selection_scope = "student_greedy"
            candidate_ids_for_log: Optional[torch.Tensor] = None

            if triggered:
                trigger_count += 1
                if policy.guidance_type == "teacher_takeover":
                    # Takeover is the only mode allowed to select from the teacher full vocabulary.
                    if teacher_top_id is not None:
                        next_token_id = teacher_top_id
                        selection_scope = "teacher_full_vocab"
                elif policy.guidance_type in {"teacher_rerank", "s2t_local"}:
                    # Rerank/local modes may only choose among candidates proposed by the SLM.
                    candidate_ids = self.sample_student_candidates(slm_topk_ids, slm_probs_topk_cond)
                    candidate_ids_for_log = candidate_ids
                    metrics["candidate_count"].append(int(candidate_ids.numel()))

                    if policy.guidance_type == "teacher_rerank":
                        selection_scope = "slm_topk_teacher_rerank"
                        if teacher_probs_full is not None:
                            teacher_scores = teacher_probs_full[candidate_ids]
                            self.collect_training_group(
                                policy,
                                problem_id,
                                step,
                                state.generated_ids,
                                candidate_ids,
                                teacher_scores,
                            )
                            if teacher_scores.sum().item() > 0:
                                best_local = int(torch.argmax(teacher_scores).item())
                                next_token_id = int(candidate_ids[best_local].item())
                    elif policy.guidance_type == "s2t_local":
                        selection_scope = "slm_topk_s2t_local"
                        scores, _ = self.score_s2t_local_candidates(state.generated_ids, candidate_ids)
                        best_local = int(torch.argmax(scores).item())
                        next_token_id = int(candidate_ids[best_local].item())
                elif policy.guidance_type == "student_greedy":
                    next_token_id = student_top_id
                else:
                    raise ValueError(f"Unknown guidance_type: {policy.guidance_type}")

            flipped_count += int(triggered and next_token_id != student_top_id)
            
            # Log the selected token and its probability on trigger steps (also teacher info for deep analysis). 
            if triggered:
                metrics["slm_selected_probs"].append(float(slm_probs_full[next_token_id].item()))
                if teacher_probs_full is not None:
                    metrics["teacher_selected_probs"].append(float(teacher_probs_full[next_token_id].item()))
                    teacher_top1_hits += int(next_token_id == teacher_top_id)

                append_jsonl(
                    os.path.join(log_dir, f"trigger_log_{safe_policy_name(policy.name)}.jsonl"),
                    {
                        "problem_id": int(problem_id),
                        "step": int(step),
                        "policy": policy.name,
                        "trigger_type": policy.trigger_type,
                        "guidance_type": policy.guidance_type,
                        "trigger_support": "slm_topk",
                        "teacher_comparison_source": (
                            "teacher_probs_on_slm_topk" if teacher_probs_full is not None else None
                        ),
                        "selection_scope": selection_scope,
                        "candidate_source": "slm_topk" if candidate_ids_for_log is not None else None,
                        "candidate_ids": (
                            [int(x) for x in candidate_ids_for_log.detach().cpu().tolist()]
                            if candidate_ids_for_log is not None
                            else None
                        ),
                        "selected_id": next_token_id,
                        "student_top_id": student_top_id,
                        "teacher_top_id": teacher_top_id,
                        "flipped_from_student": bool(next_token_id != student_top_id),
                        "student_prob_selected": float(slm_probs_full[next_token_id].item()),
                        "teacher_prob_selected": (
                            float(teacher_probs_full[next_token_id].item()) if teacher_probs_full is not None else None
                        ),
                    },
                )

            state.generated_ids.append(next_token_id)
            token_history.append(self.slm_tokenizer.decode([next_token_id], clean_up_tokenization_spaces=False))

            if next_token_id in self.stop_ids:
                break

            next_input_slm = torch.tensor([[next_token_id]], device=self.slm.device, dtype=torch.long)
            state.slm_out = self.slm(input_ids=next_input_slm, use_cache=True, past_key_values=state.slm_out.past_key_values)
            if require_llm and state.llm_out is not None:
                next_input_llm = torch.tensor([[next_token_id]], device=self.bundle.llm.device, dtype=torch.long)
                state.llm_out = self.bundle.llm(
                    input_ids=next_input_llm,
                    use_cache=True,
                    past_key_values=state.llm_out.past_key_values,
                )

        generated_text = self.slm_tokenizer.decode(state.generated_ids[state.prompt_len :], skip_special_tokens=True).strip()
        total_steps = len(token_history)
        trigger_rate = trigger_count / total_steps if total_steps else 0.0
        flip_rate = flipped_count / total_steps if total_steps else 0.0
        return {
            "policy_name": policy.name,
            "generated_text": generated_text,
            "total_steps": total_steps,
            "trigger_count": trigger_count,
            "trigger_rate": trigger_rate,
            "flipped_num": flipped_count,
            "flip_rate": flip_rate,
            "hit_at_teacher_top1": teacher_top1_hits,
            "generation_time_sec": time.time() - start,
            "token_history": token_history,
            "d_kl_history": metrics["kl"],
            "d_entropy_history": metrics["entropy"],
            "slm_selected_probs_on_trigger": metrics["slm_selected_probs"],
            "llm_selected_probs_on_trigger": metrics["teacher_selected_probs"],
            "candidate_count_on_trigger": metrics["candidate_count"],
        }

    def generate(self, question: str, data_name: str, policy: DecodingPolicy, problem_id: int, log_dir: str) -> Dict[str, Any]:
        if policy.model_role in {"slm", "llm"}:
            return self.generate_baseline(question, data_name, policy)
        return self.generate_hybrid(question, data_name, policy, problem_id, log_dir)


def calibrate_thresholds(
    config: EvaluationConfig,
    decoder: S2TDecoder,
    calibration_data,
    need_kl: bool,
    need_entropy: bool,
) -> None:
    """Estimate trigger thresholds from a held-out calibration prefix."""

    if not need_kl and not need_entropy:
        return
    kl_values: List[float] = []
    entropy_values: List[float] = []
    calibration_policy = DecodingPolicy(name="Calibration", trigger_type="kl" if need_kl else "entropy", guidance_type="student_greedy")

    saved_kl = config.kl_threshold
    saved_entropy = config.entropy_threshold
    config.kl_threshold = None
    config.entropy_threshold = None

    for idx, example in enumerate(tqdm(calibration_data, desc="Calibrating triggers")):
        question = parse_question(example, config.dataset)
        result = decoder.generate_hybrid(question, config.dataset, calibration_policy, idx, config.output_dir)
        kl_values.extend(result.get("d_kl_history", []))
        entropy_values.extend(result.get("d_entropy_history", []))

    if need_kl:
        config.kl_threshold = float(np.quantile(np.array(kl_values, dtype=float), config.trigger_quantile)) if kl_values else saved_kl
        print(f"[Calibration] KL threshold={config.kl_threshold}")
    else:
        config.kl_threshold = saved_kl

    if need_entropy:
        config.entropy_threshold = (
            float(np.quantile(np.array(entropy_values, dtype=float), config.trigger_quantile)) if entropy_values else saved_entropy
        )
        print(f"[Calibration] entropy threshold={config.entropy_threshold}")
    else:
        config.entropy_threshold = saved_entropy


def evaluate_policy(config: EvaluationConfig, decoder: S2TDecoder, policy: DecodingPolicy, dataset) -> Dict[str, Any]:
    """Evaluate a policy and save per-example JSONL results."""

    policy_file = safe_policy_name(policy.name)
    results_path = os.path.join(config.output_dir, f"results_{policy_file}.jsonl")

    total = 0
    correct = 0
    total_steps: List[int] = []
    total_times: List[float] = []
    trigger_rates: List[float] = []
    flip_rates: List[float] = []
    candidate_counts: List[int] = []
    hit_top1 = 0
    trigger_count = 0

    with open(results_path, "w", encoding="utf-8") as handle:
        for idx, example in enumerate(tqdm(dataset, desc=policy.name)):
            if idx < int(config.start_idx):
                continue
            question = parse_question(example, config.dataset)
            _, answer = parse_ground_truth(example, config.dataset)
            result = decoder.generate(question, config.dataset, policy, idx, config.output_dir)

            generated = result["generated_text"]
            generated_texts = generated if isinstance(generated, list) else [generated]
            is_correct = any(math_equal(extract_answer(text, config.dataset), answer) for text in generated_texts)
            correct += int(is_correct)
            total += 1

            result.update(
                {
                    "problem_id": f"{config.dataset}_{idx}",
                    "question": question,
                    "answer": answer,
                    "is_correct": bool(is_correct),
                }
            )
            handle.write(__import__("json").dumps(result, ensure_ascii=False) + "\n")

            total_steps.append(int(result.get("total_steps", 0)))
            total_times.append(float(result.get("generation_time_sec", 0.0)))
            trigger_rates.append(float(result.get("trigger_rate", 0.0)))
            flip_rates.append(float(result.get("flip_rate", 0.0)))
            candidate_counts.extend(int(x) for x in result.get("candidate_count_on_trigger", []))
            hit_top1 += int(result.get("hit_at_teacher_top1", 0))
            trigger_count += int(result.get("trigger_count", 0))

            flat_generated = str(generated_texts[0] if generated_texts else "").replace("\n", " ")
            print(f"\nPROBLEM: {question.replace(chr(10), ' ')}")
            print(f"GROUND TRUTH: {answer}")
            print(f"GENERATED ANSWER: {flat_generated[-120:]} ======> {is_correct}")

    accuracy = 100.0 * correct / max(total, 1)
    summary = {
        "policy": policy.name,
        "dataset": config.dataset,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "avg_tokens": float(np.mean(total_steps)) if total_steps else 0.0,
        "avg_time_sec": float(np.mean(total_times)) if total_times else 0.0,
        "avg_trigger_rate": float(np.mean(trigger_rates)) if trigger_rates else 0.0,
        "avg_flip_rate": float(np.mean(flip_rates)) if flip_rates else 0.0,
        "teacher_top1_hit_rate_on_triggers": float(hit_top1 / trigger_count) if trigger_count else 0.0,
        "avg_candidate_count": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        "results_path": results_path,
    }
    write_json(os.path.join(config.output_dir, f"summary_{policy_file}.json"), summary)

    print(f"\n--- Summary: {policy.name} on {config.dataset} ---")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"Average tokens: {summary['avg_tokens']:.2f}")
    print(f"Average trigger rate: {100.0 * summary['avg_trigger_rate']:.2f}%")
    print(f"Average flip rate: {100.0 * summary['avg_flip_rate']:.2f}%")
    print(f"Results saved to: {results_path}")
    return summary


def prepare_config(config: EvaluationConfig) -> EvaluationConfig:
    """Fill derived paths and defaults."""

    timestamp = datetime.now().strftime("%m-%d-%H_%M_%S")
    if config.run_dir:
        run_dir = os.path.abspath(config.run_dir)
    else:
        run_dir = os.path.abspath(os.path.join(config.output_dir, timestamp))
    config.output_dir = ensure_dir(os.path.join(run_dir, "eval"))
    if not config.s2t_data_dir:
        config.s2t_data_dir = ensure_dir(os.path.join(run_dir, "data"))
    else:
        config.s2t_data_dir = ensure_dir(config.s2t_data_dir)
    if config.max_new_tokens <= 0:
        config.max_new_tokens = default_max_new_tokens(config.dataset)
    if config.label_mode == "topk_binary":
        config.num_bins = 2
    write_json(os.path.join(run_dir, "run_config.json"), asdict(config))
    return config


def run_evaluation(config: EvaluationConfig) -> List[Dict[str, Any]]:
    """Load data/models, calibrate triggers, and evaluate requested policies."""

    config = prepare_config(config)
    set_seed(config.seed)

    policies = build_decoding_policies(config.policy_spec, config.candidate_k, config.self_consistency_k)
    needs_slm = any(policy.model_role != "llm" for policy in policies)
    needs_llm = any(policy_needs_llm(policy) for policy in policies)

    train_data, test_data = load_data_splits(
        dataset_name=config.dataset,
        train_num=config.train_num,
        test_num=config.test_num,
        seed=config.seed,
    )

    bundle = load_model_bundle(
        slm_model=config.slm_model,
        llm_model=config.llm_model,
        slm_device=config.slm_device,
        llm_device=config.llm_device,
        load_slm=needs_slm,
        load_llm=needs_llm,
        s2t_lora_ckpt=config.s2t_lora_ckpt,
        use_s2t_lora=config.use_s2t_lora,
        num_bins=config.num_bins,
    )
    decoder = S2TDecoder(config, bundle)

    need_kl = any(policy.trigger_type == "kl" for policy in policies) and config.kl_threshold is None
    need_entropy = any(policy.trigger_type == "entropy" for policy in policies) and config.entropy_threshold is None
    if need_kl or need_entropy:
        calibration_count = min(int(config.calibration_num), len(test_data))
        calibration_data = test_data.select(range(calibration_count))
        eval_data = test_data.select(range(calibration_count, len(test_data))) if "aime" not in config.dataset else test_data
        calibrate_thresholds(config, decoder, calibration_data, need_kl, need_entropy)
    else:
        eval_data = test_data

    summaries = []
    for policy in policies:
        summaries.append(evaluate_policy(config, decoder, policy, eval_data))
    write_json(os.path.join(config.output_dir, "summary_all.json"), {"summaries": summaries})
    return summaries


def parse_args() -> EvaluationConfig:
    parser = argparse.ArgumentParser(description="Evaluate Select-to-Think decoding policies.")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--dataset", type=str, default="math")
    parser.add_argument("--train_num", type=int, default=1000)
    parser.add_argument("--test_num", type=int, default=300)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./runs/s2t_eval")
    parser.add_argument("--run_dir", type=str, default="")

    parser.add_argument("--policy_spec", type=str, default="s2t_local")
    parser.add_argument("--slm_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--llm_model", type=str, default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--slm_device", type=str, default="cuda:0")
    parser.add_argument("--llm_device", type=str, default="cuda:1")

    parser.add_argument("--student_top_k", type=int, default=64)
    parser.add_argument("--candidate_k", type=int, default=16)
    parser.add_argument("--candidate_prob_ratio", type=float, default=0.1)
    parser.add_argument("--self_consistency_k", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=0)
    parser.add_argument("--warmup_steps", type=int, default=3)

    parser.add_argument("--calibration_num", type=int, default=20)
    parser.add_argument("--trigger_quantile", type=float, default=0.95)
    parser.add_argument("--kl_threshold", type=float, default=None)
    parser.add_argument("--entropy_threshold", type=float, default=None)

    parser.add_argument("--collect_s2t_data", action="store_true")
    parser.add_argument("--s2t_data_dir", type=str, default="")
    parser.add_argument("--s2t_lora_ckpt", type=str, default="")
    parser.add_argument("--use_s2t_lora", action="store_true")
    parser.add_argument("--num_bins", type=int, default=16)
    parser.add_argument("--max_context_len", type=int, default=512)
    parser.add_argument("--label_mode", type=str, default="prob_bin", choices=["prob_bin", "topk_binary"])
    parser.add_argument("--score_mode", type=str, default="expected_bin", choices=["expected_bin"])
    parser.add_argument("--bin_softmax_temperature", type=float, default=1.0)
    parser.add_argument("--score_group_norm", action="store_true")

    return EvaluationConfig(**vars(parser.parse_args()))


def main() -> None:
    run_evaluation(parse_args())


if __name__ == "__main__":
    main()
