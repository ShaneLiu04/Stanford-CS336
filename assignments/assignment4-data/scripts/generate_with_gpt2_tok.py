from __future__ import annotations

import logging

import typer
import torch
from transformers import AutoTokenizer

from cs336_basics.model import BasicsTransformerLM
from cs336_data.modal_utils import VOLUME_MOUNTS, app, build_image

logger = logging.getLogger(__name__)


def generate(
    model_path: str,
    prompt: str = "Linda was on a walk in the park",
    device: str = "cuda:0",
    num_samples: int = 4,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 50,
):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt_ids = torch.tensor(tokenizer.encode(prompt), device=device)
    eos_token_id = tokenizer.eos_token_id
    model = BasicsTransformerLM.from_pretrained(model_path)
    model.eval()
    model.to(device)

    with torch.no_grad():
        for _ in range(num_samples):
            output = model.generate(
                prompt_ids,
                max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                eos_token_id=eos_token_id,
            )
            print("=" * 100)
            print("Prefix: ", prompt)
            print("-" * 100)
            print("Generated: ", tokenizer.decode(output[0].tolist()))
            print("=" * 100)


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu="B200", timeout=60 * 30)
def modal_generate(
    model_path: str,
    prompt: str = "Linda was on a walk in the park",
    num_samples: int = 4,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 50,
):
    generate(
        model_path=model_path,
        prompt=prompt,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )


@app.local_entrypoint()
def modal_main(
    model_path: str,
    prompt: str = "Linda was on a walk in the park",
    num_samples: int = 4,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 50,
):
    modal_generate.remote(
        model_path=model_path,
        prompt=prompt,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )


if __name__ == "__main__":
    """
    Script used to debug that our training script produces a model that generates reasonable text (the validation losses also look good).

    Usage: uv run scripts/generate_with_gpt2_tok.py /path/to/model
    """
    typer.run(generate)
