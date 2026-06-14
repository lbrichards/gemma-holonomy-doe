import torch
import pytest

from extraction import (
    D_MODEL,
    FINAL_TOKEN_POSITION,
    LAYER_INDEX,
    N_LAYERS,
    ResidPostExtractor,
    load_corpus_token_ids,
    load_model_and_tokenizer,
    load_sae,
)


@pytest.fixture(scope="module")
def extraction_stack():
    model, _tokenizer = load_model_and_tokenizer()
    sae = load_sae(device=next(model.parameters()).device)
    extractor = ResidPostExtractor(model)
    token_sequences = load_corpus_token_ids(count=2)
    return extractor, sae, token_sequences


@pytest.mark.integration
def test_resid_post_activation_shape_dtype_finiteness_and_sae_din(extraction_stack):
    extractor, sae, token_sequences = extraction_stack

    extracted = extractor.extract(token_sequences[0])

    assert extracted.activation.shape == (D_MODEL,)
    assert extracted.activation.dtype == torch.float32
    assert torch.isfinite(extracted.activation).all()
    assert sae.d_in == D_MODEL
    assert extracted.layer_index == LAYER_INDEX
    assert extracted.token_position == FINAL_TOKEN_POSITION
    assert extracted.hook_site == "model.model.layers[12].output"


@pytest.mark.integration
def test_position_and_sequence_checks_extract_distinct_final_token_activation(extraction_stack):
    extractor, _sae, token_sequences = extraction_stack

    first_final = extractor.extract(token_sequences[0], token_position=63).activation
    second_final = extractor.extract(token_sequences[1], token_position=63).activation
    first_initial = extractor.extract(token_sequences[0], token_position=0).activation

    assert not torch.equal(first_final, second_final)
    assert not torch.equal(first_final, first_initial)


@pytest.mark.integration
def test_sae_l0_sanity_confirms_resid_post_site(extraction_stack):
    extractor, sae, token_sequences = extraction_stack

    l0_values = []
    for token_ids in token_sequences:
        activation = extractor.extract(token_ids).activation
        code = sae.encode(activation)
        l0_values.append(int(torch.count_nonzero(code > 0).detach().cpu()))

    mean_l0 = sum(l0_values) / len(l0_values)
    print(f"observed_l0_values={l0_values} mean_l0={mean_l0:.2f}")
    assert 10 <= mean_l0 <= 300
    assert all(0 < l0 < sae.n_features for l0 in l0_values)


@pytest.mark.integration
def test_determinism_same_sequence_bitwise_identical_on_backend(extraction_stack):
    extractor, _sae, token_sequences = extraction_stack

    first = extractor.extract(token_sequences[0]).activation
    second = extractor.extract(token_sequences[0]).activation

    assert torch.equal(first, second)


@pytest.mark.integration
def test_layer_correctness_block_index_12_zero_based_and_26_layers(extraction_stack):
    extractor, _sae, _token_sequences = extraction_stack

    assert extractor.layer_index == 12
    assert len(extractor.model.model.layers) == N_LAYERS
