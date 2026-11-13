"""POST-FREEZE prepend_bos robustness flag: frozen behavior is unchanged by default."""

import json
from dataclasses import replace

import pytest
import torch

from extraction import (
    FINAL_TOKEN_POSITION,
    LAYER_INDEX,
    ResidPostExtractor,
    final_token_position,
    load_corpus_token_ids,
    load_model_and_tokenizer,
    model_input_ids,
)
from manifest.schema import manifest_from_json, manifest_to_json
from tests.test_manifest import synthetic_manifest


def test_default_input_ids_and_position_match_frozen_run():
    ids = list(range(100, 164))

    assert model_input_ids(ids) == ids
    assert final_token_position() == FINAL_TOKEN_POSITION == 63


def test_prepend_bos_keeps_64_content_tokens_and_reads_same_final_token():
    ids = list(range(100, 164))

    with_bos = model_input_ids(ids, bos_token_id=2)

    assert with_bos == [2, *ids]
    assert with_bos[final_token_position(bos_token_id=2)] == ids[FINAL_TOKEN_POSITION]


def test_model_input_ids_rejects_wrong_length():
    with pytest.raises(ValueError):
        model_input_ids(list(range(63)), bos_token_id=2)


def test_manifest_omits_prepend_bos_when_false_and_round_trips_when_true():
    frozen_style = synthetic_manifest(n_base_points=2)
    frozen_style = replace(frozen_style, metadata=replace(frozen_style.metadata, manifest_kind="smoke"))
    bos = replace(frozen_style, metadata=replace(frozen_style.metadata, prepend_bos=True))

    frozen_json = manifest_to_json(frozen_style)
    bos_json = manifest_to_json(bos)

    assert "prepend_bos" not in json.loads(frozen_json)["metadata"]
    assert manifest_from_json(frozen_json).metadata.prepend_bos is False
    assert json.loads(bos_json)["metadata"]["prepend_bos"] is True
    assert manifest_from_json(bos_json).metadata.prepend_bos is True


@pytest.mark.integration
def test_bos_extractor_matches_manual_forward_at_final_content_token():
    model, tokenizer = load_model_and_tokenizer()
    bos = int(tokenizer.bos_token_id)
    token_ids = load_corpus_token_ids(count=1)[0]
    device = next(model.parameters()).device

    captured = {}
    handle = model.model.layers[LAYER_INDEX].register_forward_hook(
        lambda _m, _a, out: captured.__setitem__("h", (out[0] if isinstance(out, tuple) else out).detach())
    )
    try:
        with torch.no_grad():
            model(input_ids=torch.tensor([[bos, *token_ids]], device=device), use_cache=False)
    finally:
        handle.remove()

    with_bos = ResidPostExtractor(model, bos_token_id=bos).extract(token_ids)
    without_bos = ResidPostExtractor(model).extract(token_ids)

    assert with_bos.token_position == 64
    assert without_bos.token_position == 63
    assert torch.allclose(with_bos.activation, captured["h"][0, 64])
    assert not torch.allclose(with_bos.activation, without_bos.activation)
