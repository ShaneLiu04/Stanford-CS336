from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.tokenizer import Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    model_keys = {
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
    model = BasicsTransformerLM(**{key: value for key, value in config.items() if key in model_keys})
    checkpoint = torch.load(args.run_dir / "checkpoint-final.pt", map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(args.device).eval()

    tokenizer = Tokenizer.from_serialized(
        args.tokenizer_dir / "vocab.json",
        args.tokenizer_dir / "merges.json",
        ["<|endoftext|>"],
    )
    prompt_ids = torch.tensor(tokenizer.encode(args.prompt), dtype=torch.long, device=args.device)
    eos_id = tokenizer.special_to_id["<|endoftext|>"]
    generated = model.generate(
        prompt_ids,
        args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=eos_id,
    )
    text = args.prompt + tokenizer.decode(generated[0].tolist())
    print(text)


if __name__ == "__main__":
    main()

