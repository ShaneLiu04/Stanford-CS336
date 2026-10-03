from cs336_basics.model import BasicsTransformerLM


def test_embedding_weight_tying_is_optional() -> None:
    untied = BasicsTransformerLM(128, 32, 64, 2, 4, 128)
    tied = BasicsTransformerLM(128, 32, 64, 2, 4, 128, tie_embeddings=True)

    assert untied.lm_head.weight is not untied.token_embeddings.weight
    assert tied.lm_head.weight is tied.token_embeddings.weight
    assert 0.015 < tied.token_embeddings.weight.std().item() < 0.025
    assert sum(parameter.numel() for parameter in tied.parameters()) == (
        sum(parameter.numel() for parameter in untied.parameters()) - 128 * 64
    )
