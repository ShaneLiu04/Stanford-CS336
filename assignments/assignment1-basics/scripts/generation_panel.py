"""Generate a controlled qualitative panel and simple degeneration metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.tokenizer import Tokenizer


PROMPTS = [
    "Once upon a time, in a small village",
    "Lily found a tiny dragon in the garden",
    "The little robot wanted to learn",
    "One rainy morning, Tom and Mia",
    "The moon was bright when the fox",
]


def repetition_rate(token_ids: list[int], width: int = 3) -> float:
    ngrams = [tuple(token_ids[index : index + width]) for index in range(len(token_ids) - width + 1)]
    return 1 - len(set(ngrams)) / len(ngrams) if ngrams else 0.0


def load_model(run_dir: Path, device: str) -> tuple[BasicsTransformerLM, dict[str, object]]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    keys = {
        "vocab_size",
        "context_length",
        "d_model",
        "num_layers",
        "num_heads",
        "d_ff",
        "rope_theta",
        "remove_rmsnorm",
        "use_post_norm",
        "ffn_type",
        "tie_embeddings",
        "tied_embedding_init_std",
    }
    model = BasicsTransformerLM(**{key: value for key, value in config.items() if key in keys})
    checkpoint = torch.load(run_dir / "checkpoint-final.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.7, 1.0])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    tokenizer = Tokenizer.from_serialized(
        args.tokenizer_dir / "vocab.json",
        args.tokenizer_dir / "merges.json",
        ["<|endoftext|>"],
    )
    eos_id = tokenizer.special_to_id["<|endoftext|>"]
    records = []
    for run_dir in args.run_dir:
        model, config = load_model(run_dir, args.device)
        for temperature in args.temperatures:
            for prompt_index, prompt in enumerate(PROMPTS):
                torch.manual_seed(336 + prompt_index)
                prompt_ids = torch.tensor(
                    tokenizer.encode(prompt), dtype=torch.long, device=args.device
                )
                generated = model.generate(
                    prompt_ids,
                    args.max_new_tokens,
                    temperature=temperature,
                    top_p=0.95,
                    eos_token_id=eos_id,
                )[0].tolist()
                records.append(
                    {
                        "run": run_dir.name,
                        "temperature": temperature,
                        "prompt": prompt,
                        "text": prompt + tokenizer.decode(generated),
                        "generated_tokens": len(generated),
                        "trigram_repetition_rate": repetition_rate(generated),
                        "tie_embeddings": config.get("tie_embeddings", False),
                    }
                )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = {}
    for run in sorted({str(record["run"]) for record in records}):
        selected = [record for record in records if record["run"] == run]
        summary[run] = {
            "samples": len(selected),
            "mean_generated_tokens": float(np.mean([record["generated_tokens"] for record in selected])),
            "mean_trigram_repetition_rate": float(
                np.mean([record["trigram_repetition_rate"] for record in selected])
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"records": records, "summary": summary}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
