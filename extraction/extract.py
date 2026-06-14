"""Base-point extraction at Gemma 2 2B layer-12 resid_post.

The extraction site is the output of HF ``model.model.layers[12]``, matching the
Gemma Scope residual-stream SAE site ``blocks.12.hook_resid_post``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "google/gemma-2-2b"
SAE_REPO = "google/gemma-scope-2b-pt-res"
SAE_PATH = "layer_12/width_16k/average_l0_82/params.npz"
LAYER_INDEX = 12
D_MODEL = 2304
N_LAYERS = 26
TOKEN_LENGTH = 64
FINAL_TOKEN_POSITION = 63


@dataclass(frozen=True)
class ExtractedActivation:
    """A single base-point activation and the hook metadata that produced it."""

    activation: torch.Tensor
    layer_index: int
    token_position: int
    dtype: str
    device: str
    hook_site: str


@dataclass(frozen=True)
class GemmaScopeSAE:
    """Minimal Gemma Scope JumpReLU SAE encoder used by plane/covariate code."""

    repo: str
    path: str
    w_dec: torch.Tensor
    w_enc: torch.Tensor
    b_dec: torch.Tensor
    b_enc: torch.Tensor
    threshold: torch.Tensor

    @property
    def d_in(self) -> int:
        return int(self.w_enc.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.w_enc.shape[1])

    def encode(self, activation: torch.Tensor) -> torch.Tensor:
        """Encode an activation with Gemma Scope's JumpReLU encoder.

        SAELens metadata for this SAE has ``apply_b_dec_to_input=False``; the
        encoder is therefore ``x @ W_enc + b_enc`` followed by thresholding.
        ``b_dec`` is retained for downstream reconstruction/covariate code but
        is not subtracted here.
        """

        x = activation.detach().to(device=self.w_enc.device, dtype=self.w_enc.dtype)
        pre = x @ self.w_enc + self.b_enc
        return torch.where(pre > self.threshold, pre, torch.zeros_like(pre))


def pick_device() -> torch.device:
    """Return the DOE compute backend."""

    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_tokenizer(
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Any, Any]:
    """Load Gemma 2 2B for MPS/CPU extraction."""

    target_device = pick_device() if device is None else device
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype, attn_implementation="eager").to(target_device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    if len(model.model.layers) != N_LAYERS:
        raise RuntimeError(f"Expected {N_LAYERS} Gemma layers; got {len(model.model.layers)}")
    if int(model.config.hidden_size) != D_MODEL:
        raise RuntimeError(f"Expected hidden size {D_MODEL}; got {model.config.hidden_size}")
    return model, tokenizer


def load_sae(
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    repo: str = SAE_REPO,
    path: str = SAE_PATH,
) -> GemmaScopeSAE:
    """Load the Gemma Scope layer-12 residual SAE params."""

    target_device = pick_device() if device is None else device
    local = hf_hub_download(repo, path)
    params = np.load(local)
    sae = GemmaScopeSAE(
        repo=repo,
        path=path,
        w_dec=torch.from_numpy(params["W_dec"]).to(device=target_device, dtype=dtype),
        w_enc=torch.from_numpy(params["W_enc"]).to(device=target_device, dtype=dtype),
        b_dec=torch.from_numpy(params["b_dec"]).to(device=target_device, dtype=dtype),
        b_enc=torch.from_numpy(params["b_enc"]).to(device=target_device, dtype=dtype),
        threshold=torch.from_numpy(params["threshold"]).to(device=target_device, dtype=dtype),
    )
    if sae.d_in != D_MODEL:
        raise RuntimeError(f"Expected SAE d_in {D_MODEL}; got {sae.d_in}")
    return sae


def _as_input_ids(token_ids: list[int] | torch.Tensor, *, device: torch.device) -> torch.Tensor:
    ids = torch.as_tensor(token_ids, dtype=torch.long, device=device)
    if ids.ndim != 1:
        raise ValueError(f"token_ids must be 1D, got shape {tuple(ids.shape)}")
    if ids.numel() != TOKEN_LENGTH:
        raise ValueError(f"expected {TOKEN_LENGTH} token ids, got {ids.numel()}")
    return ids.unsqueeze(0)


class ResidPostExtractor:
    """Extract base activations from layer-12 resid_post via a forward hook."""

    def __init__(self, model: Any, *, layer_index: int = LAYER_INDEX):
        self.model = model
        self.layer_index = layer_index
        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            raise ValueError("Expected HuggingFace Gemma2ForCausalLM with model.layers")
        if len(model.model.layers) != N_LAYERS:
            raise RuntimeError(f"Expected {N_LAYERS} layers; got {len(model.model.layers)}")

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def hook_site(self) -> str:
        return f"model.model.layers[{self.layer_index}].output"

    def extract(
        self,
        token_ids: list[int] | torch.Tensor,
        *,
        token_position: int = FINAL_TOKEN_POSITION,
    ) -> ExtractedActivation:
        """Run one forward pass and capture the requested resid_post token."""

        input_ids = _as_input_ids(token_ids, device=self.device)
        if token_position < 0:
            token_position = input_ids.shape[1] + token_position
        if not 0 <= token_position < input_ids.shape[1]:
            raise ValueError(f"token_position {token_position} outside sequence length {input_ids.shape[1]}")
        attention_mask = torch.ones_like(input_ids, device=self.device)
        captured: dict[str, torch.Tensor] = {}

        def hook(_module: torch.nn.Module, _args: tuple[Any, ...], output: torch.Tensor) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            captured["h"] = tensor.detach()

        handle = self.model.model.layers[self.layer_index].register_forward_hook(hook)
        try:
            with torch.no_grad():
                self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        finally:
            handle.remove()
        if "h" not in captured:
            raise RuntimeError(f"Forward hook did not fire at layer {self.layer_index}")
        activation = captured["h"][0, token_position].detach().clone()
        if activation.shape != (D_MODEL,):
            raise RuntimeError(f"Expected activation shape ({D_MODEL},), got {tuple(activation.shape)}")
        return ExtractedActivation(
            activation=activation,
            layer_index=self.layer_index,
            token_position=token_position,
            dtype=str(activation.dtype).replace("torch.", ""),
            device=str(activation.device),
            hook_site=self.hook_site,
        )


def load_corpus_token_ids(
    path: Path = Path("run/corpus_draw_n390.json"),
    *,
    partition: str = "experiment",
    count: int | None = None,
) -> list[list[int]]:
    """Load token IDs, not text, from the blind corpus artifact."""

    artifact = json.loads(path.read_text())
    records = artifact["records"]
    if partition == "experiment":
        orders = set(artifact["partitions"]["experiment_draw_orders"])
        selected = [record for record in records if record["draw_order"] in orders]
    elif partition == "reserve":
        orders = set(artifact["partitions"]["reserve_draw_orders"])
        selected = [record for record in records if record["draw_order"] in orders]
    elif partition == "all":
        selected = records
    else:
        raise ValueError(f"unknown partition: {partition}")
    if count is not None:
        selected = selected[:count]
    return [list(record["token_ids"]) for record in selected]
