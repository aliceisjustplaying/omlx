# SPDX-License-Identifier: Apache-2.0
"""DFlash block drafter for the batched Lightning MTP verify path.

The drafter replaces the embedded MTP head as the draft source inside
``omlx.patches.mlx_lm_mtp``: the shared verify forward, acceptance, cache
rollback and emission stay the same, and this module only turns captured
target hidden states into the next draft block for every row.

The draft model is loaded by mlx-vlm: DFlash2, DFlash v1, Laguna or the Muse
Glimmer assistant. Its ``draft_block`` takes the target hidden states of the
tokens committed since the previous call and appends them to a per-row
sliding-window context cache, so each row keeps its own cache here and the
verify path feeds it the accepted positions after every cycle. The forward
here mirrors those classes, including their projection, gating and mask
differences.
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map
from mlx_vlm.speculative.drafters import load_drafter
from mlx_vlm.utils import get_model_path

from ...model_settings import MAX_LIGHTNING_MTP_DRAFT_TOKENS
from ...patches import qwen35_packed_linear
from .drafter import DFlashDrafter
from .variants import apply_partial_rope, load_mask_embedding

logger = logging.getLogger(__name__)

_FP16_MAX = 65504.0


def resolve_block_size(
    model: nn.Module, requested: int | None, max_rows: int | None = None
) -> int:
    """Clamp the block (anchor + proposals) to what Lightning MTP verifies.

    ``max_rows`` is the target's own verify limit, when it has one.
    """
    trained = int(getattr(model.config, "block_size", 0) or 0)
    block = int(requested or trained or 0)
    if block <= 1:
        raise ValueError("DFlash block size must cover the anchor and one proposal")
    limit = MAX_LIGHTNING_MTP_DRAFT_TOKENS + 1
    if max_rows:
        limit = min(limit, int(max_rows))
    if trained and block > trained:
        block = trained
    return min(block, limit)


def resolve_verify_mode(mode: str | None) -> bool:
    """Whether ``dflash_verify_mode`` lets the verified part of a block adapt."""
    if mode in (None, "adaptive"):
        return True
    if mode in ("dflash", "off"):
        return False
    raise ValueError(f"unknown dflash_verify_mode {mode!r}")


def context_bytes_per_row(config: dict, window: int | None = None) -> int:
    """Bytes one request's drafter context can hold, from the draft config.

    A row keeps its K/V ring and the projected history for the window, and
    while it decodes without drafting it queues up to a window of raw layer
    captures. bf16 throughout.
    """
    # A draft bundled with a VLM checkpoint (MiniMax DSpark) nests its config.
    config = config.get("text_config") or config
    hidden = int(config["hidden_size"])
    heads = int(config.get("num_attention_heads") or 1)
    head_dim = int(config.get("head_dim") or hidden // heads)
    kv_heads = int(config.get("num_key_value_heads") or heads)
    dflash_config = config.get("dflash_config") or {}
    captures = len(
        dflash_config.get("target_layer_ids") or config.get("target_layer_ids") or ()
    )
    window = int(
        window
        or dflash_config.get("draft_window_size")
        or config.get("draft_window_size")
        or config.get("sliding_window")
        or 2048
    )
    layers = int(config["num_hidden_layers"])
    per_token = 2 * (layers * 2 * kv_heads * head_dim + hidden + captures * hidden)
    return window * per_token


def quantize_drafter(model: nn.Module, *, bits: int, group_size: int) -> int:
    """Affine-quantize the drafter's linear layers; returns the layer count."""
    quantized = 0

    def predicate(path: str, module: nn.Module) -> bool:
        nonlocal quantized
        if not isinstance(module, nn.Linear):
            return False
        if module.weight.shape[-1] % group_size:
            return False
        quantized += 1
        return True

    nn.quantize(model, group_size=group_size, bits=bits, class_predicate=predicate)
    return quantized


def _target_float_dtype(target_model: Any):
    """Float dtype of the target's token embedding (its scales when quantized)."""
    language_model = getattr(target_model, "language_model", target_model)
    inner = getattr(language_model, "model", language_model)
    embed = getattr(inner, "embed_tokens", None) or getattr(inner, "embed", None)
    for name in ("scales", "weight"):
        value = getattr(embed, name, None)
        if isinstance(value, mx.array) and mx.issubdtype(value.dtype, mx.floating):
            return value.dtype
    return None


def match_target_dtype(model: nn.Module, target_model: Any) -> int:
    """Cast a bf16 drafter's floating parameters to an fp16 target's dtype.

    The drafter's activations follow its own parameters (quantization scales
    included), so a bf16 checkpoint next to an fp16 target computes in bf16.
    An M1 GPU (applegpu_g13) has no bf16 ALUs and runs that at about half the
    fp16 rate, and the drafter then reads the target's fp16 hidden states in a
    narrower mantissa than the target produced them. Parameters that do not
    fit fp16 leave the drafter unchanged. Returns the number of cast arrays.
    """
    target_dtype = _target_float_dtype(target_model)
    if target_dtype is None or target_dtype != mx.float16:
        return 0
    params = [
        p for _, p in tree_flatten(model.parameters()) if p.dtype == mx.bfloat16
    ]
    if not params:
        return 0
    peak = mx.max(mx.stack([mx.abs(p).max().astype(mx.float32) for p in params]))
    if not bool(peak <= _FP16_MAX):
        logger.warning(
            "DFlash drafter kept in bf16: a parameter exceeds the fp16 range"
        )
        return 0
    model.update(
        tree_map(
            lambda p: p.astype(mx.float16)
            if isinstance(p, mx.array) and p.dtype == mx.bfloat16
            else p,
            model.parameters(),
        )
    )
    return len(params)


def _is_inkling2_target(target_model: Any) -> bool:
    language_model = getattr(target_model, "language_model", target_model)
    text = getattr(language_model, "model", None)
    return (
        text is not None
        and hasattr(text, "num_visits")
        and hasattr(text, "embed_norm")
        and hasattr(text, "embed")
    )


def _load_inkling2_dflash2_model(local, raw_config: dict, target_model: Any) -> nn.Module:
    """Load a DFlash2 checkpoint whose target_layer_ids address Inkling 2 visits.

    Inkling 2 loops 24 physical layers through 48 visits plus final post-norm
    tap 48 (``target_layer_ids = [5, 23, 41, 48]`` with ``num_target_layers = 48``),
    which mlx-vlm's ``DFlash2Config.validate`` rejects. Load and validate here
    without patching mlx-vlm.
    """
    from mlx_vlm.speculative.drafters.dflash2.config import DFlash2Config
    from mlx_vlm.speculative.drafters.dflash2.dflash2 import DFlash2DraftModel

    language_model = getattr(target_model, "language_model", target_model)
    text = language_model.model
    num_visits = int(text.num_visits)
    dflash_cfg = dict(raw_config.get("dflash_config") or {})
    real_tids = [int(i) for i in dflash_cfg.get("target_layer_ids") or ()]
    if not real_tids or any(
        i < 0 or i > num_visits or (idx > 0 and real_tids[idx - 1] >= i)
        for idx, i in enumerate(real_tids)
    ):
        raise ValueError(
            f"Inkling 2 DFlash2 target_layer_ids {real_tids!r} invalid for "
            f"{num_visits} visits"
        )
    num_target = int(raw_config.get("num_target_layers") or num_visits)
    if num_target != num_visits:
        raise ValueError(
            f"Inkling 2 DFlash2 expects {num_target} visits, target has {num_visits}"
        )
    target_args = getattr(language_model, "args", None)
    target_text_args = getattr(target_args, "text", target_args)
    if target_text_args is not None:
        if int(raw_config["hidden_size"]) != int(target_text_args.hidden_size):
            raise ValueError(
                f"Inkling 2 DFlash2 hidden_size {raw_config['hidden_size']} != "
                f"target {target_text_args.hidden_size}"
            )
        if int(raw_config["vocab_size"]) != int(target_text_args.vocab_size):
            raise ValueError(
                f"Inkling 2 DFlash2 vocab_size {raw_config['vocab_size']} != "
                f"target {target_text_args.vocab_size}"
            )
    if not hasattr(language_model, "rollback_speculative_cache"):
        raise ValueError("Inkling 2 target does not implement rollback_speculative_cache")

    patched = dict(raw_config)
    patched_dflash = dict(dflash_cfg)
    patched_dflash["target_layer_ids"] = list(range(len(real_tids)))
    patched["dflash_config"] = patched_dflash
    cfg = DFlash2Config.from_dict(patched)
    cfg.target_layer_ids = real_tids

    model = DFlash2DraftModel(cfg)
    shards = sorted(local.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No safetensors shards found in {local}")
    weights = {}
    for shard in shards:
        weights.update(mx.load(str(shard)))
    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()), strict=True)
    return model


def _concat_linear_pair(a: nn.Module, b: nn.Module) -> nn.Module | None:
    """Fuse two bias-free (Quantized)Linear layers with matching input dims."""
    if getattr(a, "bias", None) is not None or getattr(b, "bias", None) is not None:
        return None
    if isinstance(a, nn.QuantizedLinear) and isinstance(b, nn.QuantizedLinear):
        if (
            a.bits != b.bits
            or a.group_size != b.group_size
            or getattr(a, "mode", "affine") != getattr(b, "mode", "affine")
        ):
            return None
        out_dims = int(a.weight.shape[0]) + int(b.weight.shape[0])
        in_dims = int(a.weight.shape[1]) * (32 // int(a.bits))
        fused = nn.QuantizedLinear(
            in_dims,
            out_dims,
            bias=False,
            group_size=a.group_size,
            bits=a.bits,
            mode=getattr(a, "mode", "affine"),
        )
        fused.weight = mx.concatenate([a.weight, b.weight], axis=0)
        fused.scales = mx.concatenate([a.scales, b.scales], axis=0)
        if a.get("biases") is not None and b.get("biases") is not None:
            fused.biases = mx.concatenate([a.biases, b.biases], axis=0)
        return fused
    if isinstance(a, nn.Linear) and isinstance(b, nn.Linear):
        out_dims = int(a.weight.shape[0]) + int(b.weight.shape[0])
        in_dims = int(a.weight.shape[1])
        fused = nn.Linear(in_dims, out_dims, bias=False)
        fused.weight = mx.concatenate([a.weight, b.weight], axis=0)
        return fused
    return None


def _fuse_dflash2_projections(model: nn.Module) -> int:
    """Fuse k_proj+v_proj -> kv_proj and gate_proj+up_proj -> gate_up_proj."""
    fused_count = 0
    for layer in getattr(model, "layers", ()):
        attn = getattr(layer, "self_attn", None)
        if (
            attn is not None
            and hasattr(attn, "k_proj")
            and hasattr(attn, "v_proj")
            and getattr(attn, "kv_proj", None) is None
        ):
            kv = _concat_linear_pair(attn.k_proj, attn.v_proj)
            if kv is not None:
                attn._k_dim = int(attn.n_kv_heads) * int(attn.head_dim)
                attn.kv_proj = kv
                del attn.k_proj
                del attn.v_proj
                fused_count += 1
        mlp = getattr(layer, "mlp", None)
        if (
            mlp is not None
            and hasattr(mlp, "gate_proj")
            and hasattr(mlp, "up_proj")
            and getattr(mlp, "gate_up_proj", None) is None
        ):
            gu = _concat_linear_pair(mlp.gate_proj, mlp.up_proj)
            if gu is not None:
                mlp.gate_up_proj = gu
                del mlp.gate_proj
                del mlp.up_proj
                fused_count += 1
    return fused_count


def _bind_inkling2_dflash2(
    model: nn.Module, raw_config: dict, target_model: Any
) -> None:
    """Bind Inkling 2 normed embeddings, vocab-masked lm_head and post-softcap."""
    language_model = getattr(target_model, "language_model", target_model)
    text = language_model.model
    model.embed_tokens = lambda tokens: text.embed_norm(text.embed(tokens))
    model.embed_scale = 1.0

    lm_head = language_model.lm_head
    vocab_bias = getattr(language_model, "_vocab_bias", None)
    if vocab_bias is not None:
        bias_dtype = _target_float_dtype(target_model) or mx.bfloat16
        bias = vocab_bias.astype(bias_dtype)
        mx.eval(bias)
        model.lm_head = lambda hidden: lm_head(hidden) + bias
    else:
        model.lm_head = lm_head

    dflash_cfg = raw_config.get("dflash_config") or {}
    softcap_mode = dflash_cfg.get("selector_softcap_mode")
    softcap = model.config.final_logit_softcapping
    if softcap_mode == "after_correction" and softcap is not None:
        model.candidate_selector.post_softcap = float(softcap)
        model.config.final_logit_softcapping = None

    fused = _fuse_dflash2_projections(model)
    if fused:
        logger.info("Inkling 2 DFlash2 fused %d projection pairs (kv + gate_up)", fused)


def load_dflash_drafter(
    path: str,
    target_model: Any,
    *,
    tokenizer: Any = None,
    block_size: int | None = None,
    quant_enabled: bool = False,
    quant_bits: int = 4,
    quant_group_size: int = 64,
    verify_mode: str | None = None,
    window_size: int | None = None,
) -> DFlashDrafter:
    """Load a DFlash checkpoint through mlx-vlm and bind it to ``target_model``.

    ``target_model`` is the mlx-vlm model whose ``language_model`` owns the
    embeddings and lm_head the drafter borrows. ``window_size`` overrides the
    checkpoint's context window.
    """
    adaptive = resolve_verify_mode(verify_mode)
    local = get_model_path(path)
    config = json.loads((local / "config.json").read_text())
    inkling2_dflash2 = (
        "DFlash2DraftModel" in (config.get("architectures") or ())
        and _is_inkling2_target(target_model)
    )
    if inkling2_dflash2:
        model = _load_inkling2_dflash2_model(local, config, target_model)
    else:
        model, kind = load_drafter(path, kind="dflash")
        if kind != "dflash":
            raise ValueError(
                f"{path} resolved to drafter kind {kind!r}, expected 'dflash'"
            )
    cast = match_target_dtype(model, target_model)
    if cast:
        logger.info("DFlash drafter cast to fp16 to match the target: %d arrays", cast)
    if quant_enabled and not getattr(model.config, "quantization", None):
        count = quantize_drafter(model, bits=quant_bits, group_size=quant_group_size)
        logger.info(
            "DFlash drafter quantized: %d linear layers at %d bits (group %d)",
            count,
            quant_bits,
            quant_group_size,
        )
    apply_partial_rope(model, config)
    mask_path = local / "mask_embedding.pt"
    mask_embedding = (
        load_mask_embedding(mask_path, int(config["hidden_size"]))
        if mask_path.is_file()
        else None
    )
    if inkling2_dflash2:
        _bind_inkling2_dflash2(model, config, target_model)
    # Laguna checks the target vocabulary through the tokenizer when the
    # target (an mlx-lm model) carries no config.
    elif (
        tokenizer is not None
        and "tokenizer" in inspect.signature(model.bind).parameters
    ):
        model.bind(target_model, tokenizer=tokenizer)
    else:
        model.bind(target_model)
    language_model = getattr(target_model, "language_model", target_model)
    block = resolve_block_size(
        model,
        block_size,
        getattr(language_model, "_omlx_mtp_max_verify_rows", None),
    )
    mx.eval(model.parameters())
    if qwen35_packed_linear.enabled(target_model):
        packed = qwen35_packed_linear.pack_drafter(model)
        logger.info("DFlash drafter packed 4-bit projections: %d layers", packed)
    drafter = DFlashDrafter(
        model,
        block_size=block,
        source_path=path,
        adaptive=adaptive,
        window_size=window_size,
        value_scale=(config.get("dflash_config") or {}).get("attention_value_scale"),
        mask_embedding=mask_embedding,
    )
    logger.info(
        "DFlash drafter loaded: path=%s kind=%s block=%d adaptive=%s target_layers=%s",
        path,
        drafter.kind,
        block,
        drafter.adaptive,
        drafter.target_layer_ids,
    )
    return drafter


def attach_drafter(language_model: Any, drafter: DFlashDrafter) -> None:
    """Expose the drafter to Lightning MTP through the language model markers."""
    language_model._omlx_drafter = drafter
    language_model._omlx_mtp_decode_enabled = True
    language_model._omlx_mtp_multi_request = not getattr(
        language_model, "_omlx_drafter_single_row", False
    )
    language_model._omlx_mtp_batch_rollback = True
    language_model._omlx_mtp_chain = True
    language_model._omlx_mtp_depth = drafter.depth
    language_model._omlx_mtp_head_clone = False


def detach_drafter(language_model: Any) -> DFlashDrafter | None:
    drafter = getattr(language_model, "_omlx_drafter", None)
    if drafter is None:
        return None
    drafter.clear()
    language_model._omlx_drafter = None
    language_model._omlx_mtp_decode_enabled = False
    return drafter


__all__ = [
    "DFlashDrafter",
    "attach_drafter",
    "context_bytes_per_row",
    "detach_drafter",
    "load_dflash_drafter",
    "quantize_drafter",
    "resolve_block_size",
    "resolve_verify_mode",
]
