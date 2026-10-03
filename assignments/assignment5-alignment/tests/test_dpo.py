import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from .adapters import run_compute_per_instance_dpo_loss as compute_per_instance_dpo_loss
from .common import FIXTURES_PATH


def _tokenizer():
    vocab = {
        "<pad>": 0,
        "<eos>": 1,
        "<unk>": 2,
        "###": 3,
        "Instruction": 4,
        ":": 5,
        "Response": 6,
        "The": 7,
        "quick": 8,
        "brown": 9,
        "fox": 10,
        "jumps": 11,
        "over": 12,
        "the": 13,
        "lazy": 14,
        "dog": 15,
        ".": 16,
        "their": 17,
        "crazy": 18,
        "frog": 19,
    }
    word_tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    word_tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=word_tokenizer,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )


def test_per_instance_dpo_loss():
    tokenizer = _tokenizer()
    model = AutoModelForCausalLM.from_pretrained(FIXTURES_PATH / "tiny-gpt2")
    model_ref = AutoModelForCausalLM.from_pretrained(FIXTURES_PATH / "tiny-gpt2-ref")

    prompt = "The quick brown fox jumps over"
    good_response = "the lazy dog."
    bad_response = "their crazy frog."

    loss = compute_per_instance_dpo_loss(
        lm=model,
        lm_ref=model_ref,
        tokenizer=tokenizer,
        beta=0.5,
        prompt=prompt,
        response_chosen=good_response,
        response_rejected=bad_response,
    )

    assert torch.isclose(loss, torch.tensor(0.9104), atol=1e-4)
