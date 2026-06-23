"""Train the S2T-local candidate selector.

The selector is implemented by fine-tuning the student model so that reserved
output-token logits act as scoring bins for each candidate token. Training uses
listwise groups collected during teacher-rerank evaluation.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except Exception:  # pragma: no cover
    PeftModel = None

from utils import (
    S2TLocalDataset,
    add_lora_to_student,
    collate_s2t_local_groups,
    count_trainable_parameters,
    ensure_dir,
    get_s2t_reserved_token_ids,
    masked_group_norm_2d,
    score_candidates_from_reserved_logits,
    set_seed,
    write_json,
)


@dataclass
class S2TLocalTrainingConfig:
    """Configuration for S2T-local LoRA training."""

    seed: int = 2025
    slm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    slm_device: str = "cuda:0"
    reference_device: str = "cuda:1"
    output_dir: str = "./runs/s2t_local"
    data_glob: str = "./runs/*/data/s2t_samples*.jsonl"

    num_bins: int = 16
    max_length: int = 512
    expected_candidates: int = 16
    label_mode: str = "prob_bin"
    score_mode: str = "expected_bin"
    bin_softmax_temperature: float = 1.0
    candidate_softmax_temperature: float = 1.0
    score_group_norm: bool = False

    loss_mode: str = "top1_ce"
    teacher_kl_weight: float = 0.1
    lm_kl_weight: float = 0.1
    pairwise_margin: float = 1.0
    teacher_filter_mode: str = "none"
    teacher_min_margin: float = 0.0
    teacher_margin_power: float = 1.0
    teacher_weight_eps: float = 1e-3

    lr: float = 5e-5
    batch_size: int = 4
    num_epochs: int = 3
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    save_steps: int = 500
    num_workers: int = 2

    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_ckpt: str = ""
    reference_placement: str = "same"


@dataclass
class DistributedState:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    is_main: bool
    train_device: str
    reference_device: str
    ddp_device_id: int | None


def resolve_distributed_state(config: S2TLocalTrainingConfig) -> DistributedState:
    """Infer DDP settings from torchrun environment variables."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    enabled = world_size > 1

    if not enabled:
        train_device = config.slm_device if torch.cuda.is_available() else "cpu"
        ref_device = config.reference_device if torch.cuda.is_available() else "cpu"
        return DistributedState(False, rank, local_rank, world_size, True, train_device, ref_device, None)

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed S2T-local training requires CUDA.")
    if local_rank < 0:
        raise RuntimeError("WORLD_SIZE>1 was detected but LOCAL_RANK is missing. Use torchrun.")

    dist.init_process_group(backend="nccl")
    placement = config.reference_placement
    if placement == "paired":
        train_id = local_rank * 2
        ref_id = train_id + 1
        if ref_id >= torch.cuda.device_count():
            raise RuntimeError("reference_placement='paired' requires two visible GPUs per rank.")
        torch.cuda.set_device(train_id)
        train_device = f"cuda:{train_id}"
        ref_device = f"cuda:{ref_id}"
        ddp_device_id = train_id
    elif placement == "cpu":
        torch.cuda.set_device(local_rank)
        train_device = f"cuda:{local_rank}"
        ref_device = "cpu"
        ddp_device_id = local_rank
    else:
        torch.cuda.set_device(local_rank)
        train_device = f"cuda:{local_rank}"
        ref_device = train_device
        ddp_device_id = local_rank

    return DistributedState(enabled, rank, local_rank, world_size, rank == 0, train_device, ref_device, ddp_device_id)


def log_main(state: DistributedState, *parts) -> None:
    if state.is_main:
        print(*parts)


def register_reserved_row_gradient_mask(model: torch.nn.Module, reserved_ids: torch.Tensor):
    """Keep lm_head updates limited to the reserved S2T scoring rows."""

    model_ref = model.module if hasattr(model, "module") else model
    lm_head = model_ref.get_output_embeddings()
    lm_head.weight.requires_grad = True

    def mask_gradient(grad: torch.Tensor) -> torch.Tensor:
        ids = reserved_ids.to(grad.device)
        kept = grad.index_select(0, ids).clone()
        grad.zero_()
        grad.index_copy_(0, ids, kept)
        return grad

    return lm_head.weight.register_hook(mask_gradient)


def build_group_weights(
    p_teacher: torch.Tensor,
    config: S2TLocalTrainingConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """Compute teacher top-1 targets and confidence weights."""

    if p_teacher.size(1) >= 2:
        top_values, top_indices = torch.topk(p_teacher, k=2, dim=1)
        teacher_best = top_indices[:, 0]
        teacher_second = top_indices[:, 1]
        margin = (top_values[:, 0] - top_values[:, 1]).clamp_min(0.0)
    else:
        _, teacher_best = torch.max(p_teacher, dim=1)
        teacher_second = teacher_best
        margin = torch.ones_like(teacher_best, dtype=torch.float32)

    keep_mask = torch.ones_like(margin, dtype=torch.bool)
    if config.teacher_min_margin > 0.0:
        keep_mask = margin >= float(config.teacher_min_margin)

    if config.teacher_filter_mode == "none":
        weights = torch.ones_like(margin)
    else:
        base = (margin + float(config.teacher_weight_eps)).pow(max(float(config.teacher_margin_power), 0.0))
        weights = base * keep_mask.float() if config.teacher_filter_mode == "drop" else base

    return (
        teacher_best,
        teacher_second,
        weights,
        float(margin.mean().item()),
        float(keep_mask.float().mean().item()),
    )


def listwise_loss(
    scores: torch.Tensor,
    teacher_abs: torch.Tensor,
    candidate_mask: torch.Tensor,
    config: S2TLocalTrainingConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the candidate-level S2T-local supervision loss."""

    if config.score_group_norm:
        scores = masked_group_norm_2d(scores, candidate_mask)

    candidate_temperature = max(float(config.candidate_softmax_temperature), 1e-6)
    scores_eff = scores / candidate_temperature
    scores_masked = scores_eff.masked_fill(candidate_mask <= 0, -1e9)

    teacher_mass = teacher_abs.float().clamp_min(0.0) * candidate_mask
    teacher_sum = teacher_mass.sum(dim=1, keepdim=True)
    valid_count = candidate_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    uniform_valid = candidate_mask / valid_count
    p_teacher = torch.where(teacher_sum > 0, teacher_mass / teacher_sum.clamp_min(1e-12), uniform_valid)
    p_student = torch.softmax(scores_masked, dim=-1)

    kl_per_group = (
        p_teacher
        * (torch.log(p_teacher.clamp_min(1e-12)) - torch.log(p_student.clamp_min(1e-12)))
    ).sum(dim=1)

    teacher_best, teacher_second, group_weights, margin_mean, keep_ratio = build_group_weights(p_teacher, config)
    weight_denom = group_weights.sum().clamp_min(1e-12)
    kl_mean = (group_weights * kl_per_group).sum() / weight_denom

    if config.loss_mode == "kl":
        objective = float(config.teacher_kl_weight) * kl_mean
    elif config.loss_mode == "top1_ce":
        ce = F.cross_entropy(scores_masked, teacher_best, reduction="none")
        pairwise = torch.zeros_like(ce)
        if config.pairwise_margin > 1e-9 and scores_masked.size(1) >= 2:
            target_score = scores_eff.gather(1, teacher_best.unsqueeze(1)).squeeze(1)
            negatives = scores_masked.clone()
            negatives.scatter_(1, teacher_best.unsqueeze(1), -1e9)
            hard_negative = negatives.max(dim=1).values
            pairwise = F.relu(float(config.pairwise_margin) - (target_score - hard_negative))
        objective = (group_weights * (ce + pairwise)).sum() / weight_denom
        objective = objective + float(config.teacher_kl_weight) * kl_mean
    elif config.loss_mode == "pairwise_margin":
        if scores_masked.size(1) < 2:
            objective = scores.sum() * 0.0
        else:
            best_score = scores_eff.gather(1, teacher_best.unsqueeze(1)).squeeze(1)
            second_score = scores_eff.gather(1, teacher_second.unsqueeze(1)).squeeze(1)
            pairwise = F.relu(float(config.pairwise_margin) - (best_score - second_score))
            objective = (group_weights * pairwise).sum() / weight_denom
        objective = objective + float(config.teacher_kl_weight) * kl_mean
    else:
        raise ValueError(f"Unknown loss_mode: {config.loss_mode}")

    with torch.no_grad():
        pred_best = torch.argmax(scores_masked, dim=1)
        hit1 = (pred_best == teacher_best).float().mean().item()

    metrics = {
        "candidate_hit1": float(hit1),
        "teacher_margin": float(margin_mean),
        "teacher_keep_ratio": float(keep_ratio),
        "listwise_kl": float(kl_mean.detach().item()),
    }
    return objective, metrics


def language_model_kl(
    student_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    action_positions: torch.Tensor,
) -> torch.Tensor:
    """KL regularizer on prefix tokens, excluding the candidate action token."""

    reference_logits = reference_logits.to(student_logits.device)
    log_p_ref = F.log_softmax(reference_logits, dim=-1)
    log_p_student = F.log_softmax(student_logits, dim=-1)
    p_ref = log_p_ref.exp()
    token_kl = (p_ref * (log_p_ref - log_p_student)).sum(dim=-1)

    mask = attention_mask.float().clone()
    row_idx = torch.arange(mask.size(0), device=mask.device)
    mask[row_idx, action_positions] = 0.0
    return (token_kl * mask).sum() / mask.sum().clamp_min(1.0)


def train_s2t_local(config: S2TLocalTrainingConfig) -> None:
    """Train and save an S2T-local LoRA selector."""

    state = resolve_distributed_state(config)
    set_seed(config.seed + state.rank)
    ensure_dir(config.output_dir)
    if state.is_main:
        write_json(os.path.join(config.output_dir, "training_config.json"), asdict(config))

    log_main(
        state,
        f"[S2T-Local] world_size={state.world_size} rank={state.rank} "
        f"train_device={state.train_device} reference_device={state.reference_device}",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.slm_model,
        trust_remote_code=True,
        use_fast=False,
    )
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define either pad_token_id or eos_token_id.")

    student = AutoModelForCausalLM.from_pretrained(
        config.slm_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": state.train_device} if torch.cuda.is_available() else "cpu",
    )
    reference = AutoModelForCausalLM.from_pretrained(
        config.slm_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": state.reference_device} if torch.cuda.is_available() else "cpu",
    )
    reference.eval()
    for param in reference.parameters():
        param.requires_grad = False

    if config.lora_ckpt:
        if PeftModel is None:
            raise ImportError("peft is required to resume S2T-local LoRA training.")
        student = PeftModel.from_pretrained(student, config.lora_ckpt, is_trainable=True)
    else:
        student = add_lora_to_student(
            student,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        )

    embedding_size = int(student.get_input_embeddings().weight.shape[0])
    reserved_ids = get_s2t_reserved_token_ids(tokenizer, embedding_size, num_bins=config.num_bins)
    reserved_ids_tensor = torch.tensor(reserved_ids, device=state.train_device, dtype=torch.long)
    grad_hook = register_reserved_row_gradient_mask(student, reserved_ids_tensor)

    log_main(state, f"[S2T-Local] reserved_token_ids={reserved_ids}")
    log_main(state, f"[S2T-Local] trainable_parameters={count_trainable_parameters(student):,}")

    dataset = S2TLocalDataset(
        data_glob=config.data_glob,
        num_bins=config.num_bins,
        max_len=config.max_length,
        expected_candidates=config.expected_candidates,
    )
    if len(dataset) == 0:
        raise ValueError(f"No S2T-local training groups matched {config.data_glob!r}.")
    log_main(state, f"[S2T-Local] training_groups={len(dataset)} files={len(dataset.files)}")

    sampler = DistributedSampler(dataset, shuffle=True, drop_last=False) if state.enabled else None
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda batch: collate_s2t_local_groups(batch, pad_token_id=pad_token_id),
    )

    optimizer = torch.optim.AdamW(
        [param for param in student.parameters() if param.requires_grad],
        lr=float(config.lr),
        weight_decay=0.01,
    )

    if state.enabled:
        student = DDP(student, device_ids=[state.ddp_device_id], output_device=state.ddp_device_id)
    model_to_save = student.module if hasattr(student, "module") else student
    try:
        student.config.use_cache = False
    except Exception:
        pass

    grad_accum = max(1, int(config.grad_accum_steps))
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    try:
        for epoch in range(int(config.num_epochs)):
            if sampler is not None:
                sampler.set_epoch(epoch)
            student.train()
            running = {
                "loss": 0.0,
                "listwise": 0.0,
                "lm_kl": 0.0,
                "hit1": 0.0,
                "teacher_margin": 0.0,
                "teacher_keep_ratio": 0.0,
            }
            running_items = 0

            progress = tqdm(loader, disable=not state.is_main, desc=f"S2T-local epoch {epoch}")
            for step_idx, batch in enumerate(progress):
                batch = {key: value.to(state.train_device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                batch_size = int(batch["batch_size"].item())
                num_candidates = int(batch["num_candidates"].item())

                output = student(input_ids=input_ids, attention_mask=attention_mask, labels=None, use_cache=False)
                logits = output.logits
                action_positions = attention_mask.sum(dim=1) - 1
                row_idx = torch.arange(logits.size(0), device=logits.device)
                logits_last = logits[row_idx, action_positions, :]
                logits_bins = logits_last[:, reserved_ids_tensor.to(logits.device)]

                scores_1d, _ = score_candidates_from_reserved_logits(
                    logits_bins=logits_bins,
                    label_mode=config.label_mode,
                    score_mode=config.score_mode,
                    bin_softmax_temperature=config.bin_softmax_temperature,
                )
                scores = scores_1d.view(batch_size, num_candidates)

                listwise, metrics = listwise_loss(
                    scores=scores,
                    teacher_abs=batch["teacher_abs"],
                    candidate_mask=batch["candidate_mask"],
                    config=config,
                )

                with torch.no_grad():
                    ref_input_ids = input_ids.to(state.reference_device) if state.reference_device != state.train_device else input_ids
                    ref_attention = attention_mask.to(state.reference_device) if state.reference_device != state.train_device else attention_mask
                    ref_logits = reference(input_ids=ref_input_ids, attention_mask=ref_attention, use_cache=False).logits

                lm_kl = language_model_kl(logits, ref_logits, attention_mask, action_positions)
                loss = listwise + float(config.lm_kl_weight) * lm_kl

                sync_context = contextlib.nullcontext()
                if state.enabled and hasattr(student, "no_sync") and (step_idx % grad_accum != grad_accum - 1):
                    sync_context = student.no_sync()
                with sync_context:
                    (loss / grad_accum).backward()

                should_step = (step_idx + 1) % grad_accum == 0
                if should_step:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), float(config.max_grad_norm))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                running["loss"] += float(loss.detach().item())
                running["listwise"] += float(listwise.detach().item())
                running["lm_kl"] += float(lm_kl.detach().item())
                running["hit1"] += metrics["candidate_hit1"]
                running["teacher_margin"] += metrics["teacher_margin"]
                running["teacher_keep_ratio"] += metrics["teacher_keep_ratio"]
                running_items += 1

                if state.is_main and running_items % 10 == 0:
                    denom = max(1, running_items)
                    progress.set_postfix(
                        loss=f"{running['loss'] / denom:.4f}",
                        hit1=f"{running['hit1'] / denom:.3f}",
                        margin=f"{running['teacher_margin'] / denom:.3f}",
                    )

                if should_step and config.save_steps > 0 and global_step % int(config.save_steps) == 0:
                    if state.is_main:
                        ckpt_dir = os.path.join(config.output_dir, "checkpoints", f"step_{global_step}")
                        ensure_dir(ckpt_dir)
                        model_to_save.save_pretrained(ckpt_dir)
                        log_main(state, f"[S2T-Local] saved checkpoint: {ckpt_dir}")
                    if state.enabled:
                        dist.barrier()

            if state.is_main:
                epoch_dir = os.path.join(config.output_dir, f"epoch_{epoch}")
                ensure_dir(epoch_dir)
                model_to_save.save_pretrained(epoch_dir)
                log_main(state, f"[S2T-Local] saved epoch checkpoint: {epoch_dir}")
            if state.enabled:
                dist.barrier()

    except torch.OutOfMemoryError:
        if state.is_main:
            emergency_dir = os.path.join(config.output_dir, "oom_emergency")
            ensure_dir(emergency_dir)
            model_to_save.save_pretrained(emergency_dir)
            print(f"[S2T-Local] OOM encountered. Emergency checkpoint saved to {emergency_dir}.")
        raise
    finally:
        grad_hook.remove()
        if state.is_main:
            model_to_save.save_pretrained(config.output_dir)
            print(f"[S2T-Local] final LoRA checkpoint saved to {config.output_dir}")
        if state.enabled:
            dist.barrier()
            dist.destroy_process_group()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args() -> S2TLocalTrainingConfig:
    parser = argparse.ArgumentParser(description="Train S2T-local candidate selector.")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--slm_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--slm_device", type=str, default="cuda:0")
    parser.add_argument("--reference_device", type=str, default="cuda:1")
    parser.add_argument("--output_dir", type=str, default="./runs/s2t_local")
    parser.add_argument("--data_glob", type=str, default="./runs/*/data/s2t_samples*.jsonl")

    parser.add_argument("--num_bins", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--expected_candidates", type=int, default=16)
    parser.add_argument("--label_mode", type=str, default="prob_bin", choices=["prob_bin", "topk_binary"])
    parser.add_argument("--score_mode", type=str, default="expected_bin", choices=["expected_bin"])
    parser.add_argument("--bin_softmax_temperature", type=float, default=1.0)
    parser.add_argument("--candidate_softmax_temperature", type=float, default=1.0)
    parser.add_argument("--score_group_norm", action="store_true")

    parser.add_argument("--loss_mode", type=str, default="top1_ce", choices=["kl", "top1_ce", "pairwise_margin"])
    parser.add_argument("--teacher_kl_weight", type=float, default=0.1)
    parser.add_argument("--lm_kl_weight", type=float, default=0.1)
    parser.add_argument("--pairwise_margin", type=float, default=1.0)
    parser.add_argument("--teacher_filter_mode", type=str, default="none", choices=["none", "weight", "drop"])
    parser.add_argument("--teacher_min_margin", type=float, default=0.0)
    parser.add_argument("--teacher_margin_power", type=float, default=1.0)
    parser.add_argument("--teacher_weight_eps", type=float, default=1e-3)

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_ckpt", type=str, default="")
    parser.add_argument("--reference_placement", type=str, default="same", choices=["same", "paired", "cpu"])

    args = parser.parse_args()
    config = S2TLocalTrainingConfig(**vars(args))
    if config.label_mode == "topk_binary":
        config.num_bins = 2
    return config


def main() -> None:
    train_s2t_local(parse_args())


if __name__ == "__main__":
    main()
