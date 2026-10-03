from __future__ import annotations

import gzip
import json
import os
import random
import re
from pathlib import Path
from typing import Any, TextIO

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

_ALPACA_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}"""

_MMLU_ANSWER_RE = re.compile(
    r"\bthe\s+correct\s+answer\s+is\s*(?:option\s*)?([A-D])\b",
    flags=re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def _open_text(path: str | os.PathLike[str]) -> TextIO:
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8")
    return path.open(mode="r", encoding="utf-8")


class PackedSFTDataset(Dataset[dict[str, torch.Tensor]]):
    """A packed, next-token-prediction dataset of prompt-response documents."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_path: str | os.PathLike[str],
        seq_length: int,
        shuffle: bool,
    ) -> None:
        if seq_length <= 0:
            raise ValueError("seq_length must be positive")
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer must define an EOS token")

        with _open_text(dataset_path) as source:
            documents = [json.loads(line) for line in source if line.strip()]

        if shuffle:
            random.shuffle(documents)

        token_ids: list[int] = []
        for document in documents:
            text = _ALPACA_TEMPLATE.format(
                instruction=document["prompt"],
                response=document["response"],
            )
            token_ids.extend(tokenizer.encode(text))
            token_ids.append(tokenizer.eos_token_id)

        self._token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.seq_length = seq_length

    def __len__(self) -> int:
        # Each example needs one additional token to construct shifted labels.
        return max(0, (len(self._token_ids) - 1) // self.seq_length)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = index * self.seq_length
        tokens = self._token_ids[start : start + self.seq_length + 1]
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
        }


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike[str],
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    return PackedSFTDataset(tokenizer, dataset_path, seq_length, shuffle)


def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return iterate_batches(dataset, batch_size, shuffle)


def parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    del mmlu_example  # The prescribed output format is independent of the options.
    match = _MMLU_ANSWER_RE.search(model_output)
    if match is not None:
        return match.group(1).upper()

    # Also accept a bare option letter, while avoiding letters occurring in prose.
    bare_response = model_output.strip().rstrip(".").strip()
    if re.fullmatch(r"[A-D]", bare_response, flags=re.IGNORECASE):
        return bare_response.upper()
    return None


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    return parse_mmlu_response(mmlu_example, model_output)


def parse_gsm8k_response(model_output: str) -> str | None:
    matches = _NUMBER_RE.findall(model_output)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def run_parse_gsm8k_response(model_output: str) -> str | None:
    return parse_gsm8k_response(model_output)


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _sequence_log_probability(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
) -> torch.Tensor:
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define an EOS token")

    token_ids = tokenizer.encode(text)
    token_ids.append(tokenizer.eos_token_id)
    if len(token_ids) < 2:
        raise ValueError("formatted sequence must contain at least two tokens")

    device = _model_device(model)
    input_ids = torch.tensor(token_ids[:-1], dtype=torch.long, device=device).unsqueeze(
        0
    )
    labels = torch.tensor(token_ids[1:], dtype=torch.long, device=device).unsqueeze(0)
    logits = model(input_ids=input_ids).logits
    token_log_probs = F.log_softmax(logits, dim=-1).gather(
        dim=-1, index=labels.unsqueeze(-1)
    )
    return token_log_probs.squeeze(-1).sum()


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    chosen_text = _ALPACA_TEMPLATE.format(instruction=prompt, response=response_chosen)
    rejected_text = _ALPACA_TEMPLATE.format(
        instruction=prompt, response=response_rejected
    )

    policy_log_ratio = _sequence_log_probability(
        lm, tokenizer, chosen_text
    ) - _sequence_log_probability(lm, tokenizer, rejected_text)

    with torch.no_grad():
        reference_log_ratio = _sequence_log_probability(
            lm_ref, tokenizer, chosen_text
        ) - _sequence_log_probability(lm_ref, tokenizer, rejected_text)

    reference_log_ratio = reference_log_ratio.to(policy_log_ratio.device)
    return -F.logsigmoid(beta * (policy_log_ratio - reference_log_ratio))


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    return compute_per_instance_dpo_loss(
        lm=lm,
        lm_ref=lm_ref,
        tokenizer=tokenizer,
        beta=beta,
        prompt=prompt,
        response_chosen=response_chosen,
        response_rejected=response_rejected,
    )
