"""Shared utilities for Select-to-Think experiments.

This module contains the reusable pieces needed by both evaluation and
S2T-local training: dataset adapters, prompt formatting, model loading,
reserved-token scoring, and listwise training batches.

Answer extraction and mathematical equivalence are in the parser module.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

try:
    from peft import LoraConfig, PeftModel, get_peft_model
except Exception:  # pragma: no cover
    LoraConfig = PeftModel = get_peft_model = None


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Please answer the math problem step by step, "
    "and put the final answer in the format \\boxed{answer}."
)

MULTIPLE_CHOICE_SYSTEM_PROMPT = (
    "You are a helpful assistant. You are solving high-level academic problems."
)

MULTIPLE_CHOICE_SUFFIX = (
    "Please reason step by step to find the correct choice. "
    "At the end of your response, strictly output your final answer in the "
    "exact format: 'The answer is (X)', where X is the letter of the correct choice."
)



@dataclass
class ModelBundle:
    """Loaded models, tokenizers, and tokenizer-derived metadata."""

    slm: Optional[torch.nn.Module]
    slm_tokenizer: Optional[Any]
    slm_formatter: Optional["ChatFormatter"]
    llm: Optional[torch.nn.Module]
    llm_tokenizer: Optional[Any]
    llm_formatter: Optional["ChatFormatter"]
    same_vocab: bool
    s2t_token_ids: Tuple[int, ...]


def set_seed(seed: int) -> None:
    """Make stochastic decoding and data loading reproducible."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert Python or tensor scalars to float without raising."""

    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        return float(value.detach().float().cpu().reshape(-1)[0].item())
    try:
        return float(value)
    except Exception:
        return default


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters."""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_dir(path: Union[str, Path]) -> str:
    """Create a directory if needed and return it as a string."""

    path = str(path)
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path: Union[str, Path], payload: Dict[str, Any]) -> None:
    """Write a deterministic JSON file."""

    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Union[str, Path], record: Dict[str, Any]) -> None:
    """Append one JSON record to a JSONL file."""

    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_jsonl(paths: Iterable[Union[str, Path]]) -> Iterable[Dict[str, Any]]:
    """Yield valid JSON records from one or more JSONL files."""

    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


# ---------------------------------------------------------------------------
# Dataset and prompting helpers
# ---------------------------------------------------------------------------


def load_olympiadbench_split(
    lang: str = "en",
    text_only: bool = True,
    subject: str = "math",
    test_size: float = 0.1,
    seed: int = 42,
):
    """Load and split OlympiadBench text-only examples."""

    split_name = f"test_{'en' if lang.lower().startswith('en') else 'cn'}"
    dataset = load_dataset("lmms-lab/OlympiadBench", split=split_name)
    if text_only:
        dataset = dataset.filter(
            lambda ex: ex.get("images") is None or (isinstance(ex.get("images"), list) and len(ex["images"]) == 0)
        )

    def has_answer(example: Dict[str, Any]) -> bool:
        answer = example.get("final_answer")
        return answer is not None and (not isinstance(answer, str) or bool(answer.strip()))

    dataset = dataset.filter(has_answer)
    if subject:
        key = "math" if subject.lower().startswith("math") else "physics"
        dataset = dataset.filter(
            lambda ex: key in str(ex.get("subject") or ex.get("source") or ex.get("category") or "").lower()
        )

    stratify_col = next((col for col in ["subject", "source", "category"] if col in dataset.column_names), None)
    try:
        split = dataset.train_test_split(test_size=test_size, seed=seed, stratify_by_column=stratify_col)
    except Exception:
        split = dataset.train_test_split(test_size=test_size, seed=seed)
    return split["train"], split["test"]


def load_data_splits(
    dataset_name: str,
    train_num: int,
    test_num: int,
    seed: int = 42,
):
    """Load the train/evaluation split used by S2T experiments."""

    if "gsm8k" in dataset_name:
        dataset = load_dataset("openai/gsm8k", "main")
        train_dataset, test_dataset = dataset["train"], dataset["test"]
    elif "aime" in dataset_name:
        dataset = load_dataset("math-ai/aime25", split="test")
        train_dataset = test_dataset = dataset
    elif dataset_name == "math" or "math" in dataset_name:
        train_dataset = load_dataset("qwedsacf/competition_math", split="train")
        test_dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    elif "olympiadbench" in dataset_name:
        train_dataset, test_dataset = load_olympiadbench_split()
    elif "mmlu_pro" in dataset_name:
        dataset = load_dataset("TIGER-Lab/MMLU-Pro")
        train_dataset, test_dataset = dataset["validation"], dataset["test"]
    elif "arc_challenge" in dataset_name:
        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge")
        train_dataset, test_dataset = dataset["train"], dataset["test"]
    elif "truthfulqa" in dataset_name:
        dataset = load_dataset("truthful_qa", "multiple_choice")
        split_name = "train" if "train" in dataset else "validation"
        train_dataset = test_dataset = dataset[split_name]
    elif "mt_bench" in dataset_name:
        dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        train_dataset = test_dataset = dataset
    else:
        raise NotImplementedError(f"Benchmark is not supported: {dataset_name}")

    train_count = min(int(train_num), len(train_dataset))
    test_count = min(int(test_num), len(test_dataset))
    return train_dataset.select(range(train_count)), test_dataset.select(range(test_count))


class ChatFormatter:
    """Small wrapper around tokenizer chat templates."""

    def __init__(self, tokenizer_name: str):
        self.tokenizer_name = tokenizer_name
        tokenizer_cls = PreTrainedTokenizerFast if "Qwen" in tokenizer_name else AutoTokenizer
        try:
            self.tokenizer = tokenizer_cls.from_pretrained(tokenizer_name, trust_remote_code=True, use_fast=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True, use_fast=True)

    @staticmethod
    def _is_message_list(value: Any) -> bool:
        return isinstance(value, list) and (len(value) == 0 or isinstance(value[0], dict))

    @staticmethod
    def _fallback_render_messages(messages: Sequence[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        chunks = []
        for message in messages:
            role = message["role"].strip().lower()
            content = message["content"].strip()
            chunks.append(f"{role.capitalize()}: {content}")
        if add_generation_prompt:
            chunks.append("Assistant:")
        return "\n\n".join(chunks)

    def build_prompt_from_messages(self, messages: Sequence[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(messages, tokenize=False)
        return self._fallback_render_messages(messages, add_generation_prompt)

    def build_prompt(self, question: Union[str, Sequence[Dict[str, str]]], data_name: str) -> str:
        if self._is_message_list(question):
            return self.build_prompt_from_messages(question, add_generation_prompt=True)

        if any(name in data_name for name in ["mmlu_pro", "truthfulqa", "arc"]):
            system_content = MULTIPLE_CHOICE_SYSTEM_PROMPT
            user_content = f"{question}\n{MULTIPLE_CHOICE_SUFFIX}"
        elif "humaneval" in data_name:
            system_content = "You write Python code. Output only Python code. No markdown, no explanation."
            user_content = str(question)
        else:
            system_content = DEFAULT_SYSTEM_PROMPT
            user_content = str(question)

        if "gemma" in self.tokenizer_name.lower():
            return self.build_prompt_from_messages(
                [{"role": "user", "content": system_content + "\n" + user_content}],
                add_generation_prompt=True,
            )
        return self.build_prompt_from_messages(
            [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}],
            add_generation_prompt=True,
        )

    def build_inputs(self, question: Union[str, Sequence[Dict[str, str]]], data_name: str):
        prompt = self.build_prompt(question, data_name)
        return self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)


def resolve_stop_token_ids(tokenizer: Any) -> List[int]:
    """Resolve EOS and chat-template stop tokens for generation."""

    stop_ids = set()
    if getattr(tokenizer, "eos_token_id", None) is not None:
        stop_ids.add(int(tokenizer.eos_token_id))

    for token in ["<|eot_id|>", "<|im_end|>", "<|endoftext|>"]:
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id != getattr(tokenizer, "unk_token_id", None):
                stop_ids.add(int(token_id))
        except Exception:
            pass
        try:
            encoded = tokenizer.encode(token, add_special_tokens=False)
            if len(encoded) == 1:
                stop_ids.add(int(encoded[0]))
        except Exception:
            pass

    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, str):
        for token in set(re.findall(r"<\|[^|>]+\|>", template)):
            token_lower = token.lower()
            if any(marker in token_lower for marker in ["im_end", "eot", "endoftext", "eos"]):
                try:
                    token_id = tokenizer.convert_tokens_to_ids(token)
                    if isinstance(token_id, int) and token_id != getattr(tokenizer, "unk_token_id", None):
                        stop_ids.add(int(token_id))
                except Exception:
                    pass

    if "qwen" in str(getattr(tokenizer, "name_or_path", "")).lower() or "qwen" in type(tokenizer).__name__.lower():
        stop_ids.update({151645, 151643})
    return sorted(token_id for token_id in stop_ids if token_id is not None)


def get_s2t_reserved_token_ids(
    tokenizer: Any,
    embedding_size: int,
    num_bins: int,
    avoid_ids: Optional[Iterable[int]] = None,
    avoid_special: bool = True,
) -> List[int]:
    """Choose reserved output-token rows used as S2T-local scoring bins."""

    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        raise ValueError("Tokenizer must expose vocab_size.")
    if vocab_size < num_bins:
        raise ValueError(f"vocab_size={vocab_size} is smaller than num_bins={num_bins}.")

    vocab = tokenizer.get_vocab()
    gemma_reserved = sorted(idx for token, idx in vocab.items() if "<unused" in token and ">" in token)
    if len(gemma_reserved) >= num_bins:
        return [int(idx) for idx in gemma_reserved[:num_bins]]

    llama_reserved = sorted(
        (idx for token, idx in vocab.items() if "<|reserved_special_token_" in token),
        reverse=True,
    )
    if len(llama_reserved) >= num_bins:
        return sorted(int(idx) for idx in llama_reserved[:num_bins])

    avoid = set(int(x) for x in (avoid_ids or []) if x is not None)
    if avoid_special:
        avoid.update(int(x) for x in getattr(tokenizer, "all_special_ids", []) if x is not None)
    avoid.update(resolve_stop_token_ids(tokenizer))

    picked: List[int] = []
    for token_id in range(int(embedding_size) - 1, int(vocab_size), -1):
        if token_id in avoid:
            continue
        picked.append(int(token_id))
        if len(picked) == int(num_bins):
            break
    if len(picked) < int(num_bins):
        raise ValueError(f"Could not find {num_bins} reserved token ids above vocab_size={vocab_size}.")
    return sorted(picked)


def score_candidates_from_reserved_logits(
    logits_bins: torch.Tensor,
    label_mode: str,
    score_mode: str = "expected_bin",
    bin_softmax_temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map reserved-bin logits to one scalar preference score per candidate."""

    temperature = max(float(bin_softmax_temperature or 1.0), 1e-6)
    logits_float = logits_bins.float()
    probs = F.softmax(logits_float / temperature, dim=-1)

    if label_mode == "topk_binary":
        if logits_bins.size(-1) != 2:
            raise ValueError(f"topk_binary requires two bins, got {logits_bins.size(-1)}.")
        return logits_float[..., 1], probs
    if label_mode != "prob_bin":
        raise ValueError(f"Unknown label_mode: {label_mode}")

    if score_mode == "expected_bin":
        weights = torch.linspace(0.0, 1.0, logits_bins.size(-1), device=logits_bins.device, dtype=torch.float32)
        return (probs * weights).sum(dim=-1), probs
    raise ValueError(f"Unknown score_mode: {score_mode}")


def masked_group_norm_2d(scores: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize candidate scores row-wise over valid candidates only."""

    mask_f = mask.float()
    denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (scores * mask_f).sum(dim=1, keepdim=True) / denom
    var = ((scores - mean) ** 2 * mask_f).sum(dim=1, keepdim=True) / denom
    return (scores - mean) / torch.sqrt(var + eps)


def sample_indices_without_replacement(
    logits: torch.Tensor,
    num_samples: int,
    top_k: int = 20,
    top_p: float = 0.95,
    temperature: float = 0.6,
) -> torch.Tensor:
    """Sample local token indices from filtered logits without replacement."""

    logits = logits.float().clone()
    if temperature is not None and temperature != 1.0:
        logits = logits / float(temperature)
    if top_k is not None and top_k > 0:
        top_k = min(int(top_k), logits.size(-1))
        threshold = torch.topk(logits, top_k, dim=-1).values[..., -1].unsqueeze(-1)
        logits = logits.masked_fill(logits < threshold, -float("inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove_sorted = cumulative > float(top_p)
        remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
        remove_sorted[..., 0] = False
        remove = torch.zeros_like(remove_sorted, dtype=torch.bool)
        remove.scatter_(dim=-1, index=sorted_indices, src=remove_sorted)
        logits = logits.masked_fill(remove, -float("inf"))

    valid = torch.isfinite(logits)
    count = min(int(num_samples), int(valid.sum().item()))
    if count < 1:
        return torch.argmax(logits).view(1)
    probs = F.softmax(logits, dim=-1)
    if torch.isnan(probs).any() or safe_float(probs.sum()) <= 0.0:
        probs = valid.float() / valid.float().sum().clamp_min(1.0)
    return torch.multinomial(probs, num_samples=count, replacement=False)


def load_model_bundle(
    slm_model: str,
    llm_model: str = "",
    slm_device: str = "cuda:0",
    llm_device: str = "cuda:1",
    load_slm: bool = True,
    load_llm: bool = True,
    s2t_lora_ckpt: str = "",
    use_s2t_lora: bool = False,
    num_bins: int = 16,
    local_files_only: bool = True,
    dtype: torch.dtype = torch.bfloat16,
) -> ModelBundle:
    """Load SLM, optional LLM, tokenizers, and S2T reserved token ids."""

    slm = slm_tokenizer = slm_formatter = None
    llm = llm_tokenizer = llm_formatter = None
    s2t_token_ids: Tuple[int, ...] = ()

    if load_slm:
        slm_formatter = ChatFormatter(slm_model)
        slm_tokenizer = slm_formatter.tokenizer
        slm = AutoModelForCausalLM.from_pretrained(
            slm_model,
            trust_remote_code=True,
            local_files_only=local_files_only,
            torch_dtype=dtype,
            device_map={"": slm_device} if torch.cuda.is_available() else "cpu",
        )
        if use_s2t_lora and s2t_lora_ckpt:
            if PeftModel is None:
                raise ImportError("peft is required to load an S2T-local LoRA checkpoint.")
            slm = PeftModel.from_pretrained(slm, s2t_lora_ckpt)
        slm.eval()
        embedding_size = int(slm.get_input_embeddings().weight.shape[0])
        s2t_token_ids = tuple(get_s2t_reserved_token_ids(slm_tokenizer, embedding_size, num_bins=num_bins))

    if load_llm:
        if not llm_model:
            raise ValueError("llm_model must be provided when load_llm=True.")
        llm_formatter = ChatFormatter(llm_model)
        llm_tokenizer = llm_formatter.tokenizer
        llm = AutoModelForCausalLM.from_pretrained(
            llm_model,
            trust_remote_code=True,
            local_files_only=local_files_only,
            torch_dtype=dtype,
            device_map={"": llm_device} if torch.cuda.is_available() else "auto",
        )
        llm.eval()

    same_vocab = False
    if slm_tokenizer is not None and llm_tokenizer is not None:
        same_vocab = (
            getattr(slm_tokenizer, "vocab_size", None) == getattr(llm_tokenizer, "vocab_size", None)
            and slm_tokenizer.get_vocab() == llm_tokenizer.get_vocab()
        )

    return ModelBundle(
        slm=slm,
        slm_tokenizer=slm_tokenizer,
        slm_formatter=slm_formatter,
        llm=llm,
        llm_tokenizer=llm_tokenizer,
        llm_formatter=llm_formatter,
        same_vocab=same_vocab,
        s2t_token_ids=s2t_token_ids,
    )


# ---------------------------------------------------------------------------
# S2T-local listwise training dataset
# ---------------------------------------------------------------------------


class S2TLocalDataset(Dataset):
    """Group-wise candidate-ranking dataset for S2T-local training."""

    def __init__(self, data_glob: str, num_bins: int, max_len: int, expected_candidates: Optional[int] = 16):
        self.files = sorted(glob.glob(data_glob))
        self.num_bins = int(num_bins)
        self.max_len = int(max_len)
        self.expected_candidates = expected_candidates
        self.groups: List[Dict[str, Any]] = []
        for file_path in self.files:
            self._load_file(file_path)

    @staticmethod
    def _group_key(record: Dict[str, Any]) -> str:
        if record.get("group_id") is not None:
            return str(record["group_id"])
        problem_id = record.get("problem_id", record.get("pid"))
        step = record.get("step", record.get("t", record.get("step_id")))
        if problem_id is not None and step is not None:
            return f"{problem_id}:{step}"
        context = record.get("context_ids", record.get("prefix_ids", record.get("ctx_ids", []))) or []
        context_tail = context[-256:] if isinstance(context, list) else []
        digest = hashlib.md5(",".join(map(str, context_tail)).encode("utf-8")).hexdigest()
        return f"context:{digest}"

    @staticmethod
    def _context_ids(record: Dict[str, Any]) -> List[int]:
        if isinstance(record.get("context_ids"), list):
            return [int(x) for x in record["context_ids"]]
        if isinstance(record.get("prefix_ids"), list):
            return [int(x) for x in record["prefix_ids"]]
        if isinstance(record.get("input_ids"), list) and len(record["input_ids"]) >= 2:
            return [int(x) for x in record["input_ids"][:-1]]
        return []

    @staticmethod
    def _candidate_id(record: Dict[str, Any]) -> Optional[int]:
        for key in ["action_id", "cand_id", "token_id", "action"]:
            if record.get(key) is not None:
                return int(record[key])
        return None

    @staticmethod
    def _teacher_prob(record: Dict[str, Any]) -> Optional[float]:
        for key in ["teacher_prob_abs", "teacher_abs", "teacher_prob_raw", "teacher_prob", "teacher_prob_cond"]:
            if record.get(key) is not None:
                try:
                    return float(record[key])
                except Exception:
                    return None
        if "is_ref" in record:
            return 1.0 if int(record["is_ref"]) == 1 else 0.0
        return None

    def _trim_group(self, group: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        n = min(len(group["cand_ids"]), len(group["teacher_prob_abs"]))
        if n == 0:
            return None
        group["cand_ids"] = list(group["cand_ids"][:n])
        group["teacher_prob_abs"] = list(group["teacher_prob_abs"][:n])
        if self.expected_candidates is not None and n > int(self.expected_candidates):
            k = int(self.expected_candidates)
            group["cand_ids"] = group["cand_ids"][:k]
            group["teacher_prob_abs"] = group["teacher_prob_abs"][:k]
        return group if group["cand_ids"] else None

    def _load_file(self, file_path: str) -> None:
        buffer: Dict[str, Dict[str, Any]] = {}
        with open(file_path, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except Exception:
                    continue

                if isinstance(record.get("cand_ids"), list) and isinstance(record.get("teacher_prob_abs"), list):
                    group = self._trim_group(record)
                    if group:
                        self.groups.append(group)
                    continue

                candidate_id = self._candidate_id(record)
                teacher_prob = self._teacher_prob(record)
                context = self._context_ids(record)
                if candidate_id is None or teacher_prob is None or not context:
                    continue

                key = self._group_key(record)
                group = buffer.setdefault(
                    key,
                    {
                        "group_id": key,
                        "problem_id": record.get("problem_id", record.get("pid")),
                        "step": record.get("step", record.get("t", record.get("step_id"))),
                        "context_ids": context,
                        "cand_ids": [],
                        "teacher_prob_abs": [],
                    },
                )
                if len(context) > len(group["context_ids"]):
                    group["context_ids"] = context
                group["cand_ids"].append(candidate_id)
                group["teacher_prob_abs"].append(float(teacher_prob))

        for group in buffer.values():
            trimmed = self._trim_group(group)
            if trimmed:
                self.groups.append(trimmed)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.groups[index]
        max_prefix = max(1, self.max_len - 1)
        context = record["context_ids"][-max_prefix:]
        return {
            "context_ids": context,
            "cand_ids": record["cand_ids"],
            "teacher_prob_abs": record["teacher_prob_abs"],
        }


def collate_s2t_local_groups(batch: Sequence[Dict[str, Any]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """Flatten grouped candidate examples into [B * K, L] model inputs."""

    batch_size = len(batch)
    num_candidates = max(len(item["cand_ids"]) for item in batch)
    flat_input_ids: List[torch.Tensor] = []
    teacher_abs = torch.zeros(batch_size, num_candidates, dtype=torch.float32)
    candidate_mask = torch.zeros(batch_size, num_candidates, dtype=torch.float32)

    for batch_idx, item in enumerate(batch):
        context = [int(x) for x in item["context_ids"]]
        candidates = [int(x) for x in item["cand_ids"]]
        teacher_probs = [float(x) for x in item["teacher_prob_abs"]]
        for cand_idx in range(num_candidates):
            if cand_idx < len(candidates):
                flat_input_ids.append(torch.tensor(context + [candidates[cand_idx]], dtype=torch.long))
                teacher_abs[batch_idx, cand_idx] = teacher_probs[cand_idx]
                candidate_mask[batch_idx, cand_idx] = 1.0
            else:
                flat_input_ids.append(torch.tensor(context + [pad_token_id], dtype=torch.long))

    max_len = max(ids.numel() for ids in flat_input_ids)
    input_ids = torch.full((batch_size * num_candidates, max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for row_idx, ids in enumerate(flat_input_ids):
        input_ids[row_idx, : ids.numel()] = ids
        attention_mask[row_idx, : ids.numel()] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "teacher_abs": teacher_abs,
        "candidate_mask": candidate_mask,
        "batch_size": torch.tensor(batch_size, dtype=torch.long),
        "num_candidates": torch.tensor(num_candidates, dtype=torch.long),
    }


def add_lora_to_student(model: torch.nn.Module, rank: int, alpha: int, dropout: float) -> torch.nn.Module:
    """Attach LoRA adapters to the transformer blocks and keep lm_head trainable."""

    if LoraConfig is None or get_peft_model is None:
        raise ImportError("peft is required for S2T-local LoRA training.")
    config = LoraConfig(
        r=int(rank),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["lm_head"],
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model
