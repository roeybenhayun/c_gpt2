"""Reload the produced .bin, dequantize each INT8 tensor, and report
reconstruction error against (a) original FP32 source and (b) BF16 cast
of the source.

The split is meaningful for the runtime: at inference time, weights are
held in BF16 in memory once cast at load. So `error_quant_only` (vs BF16)
is what the runtime actually adds on top of the existing BF16 cast loss.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .reader import (
    CTX_LEN,
    MODEL_CONFIGS,
    VOCAB_SIZE,
    load_fp32_weights,
)
from .writer import ALIGNMENT


def _fp32_to_bf16_to_fp32(x: np.ndarray) -> np.ndarray:
    """Round-trip FP32 → BF16 → FP32 (round-to-nearest-even).
    BF16 is the upper 16 bits of FP32 (1s + 8e + 7m).
    """
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    u = x.view(np.uint32).copy()
    rne_bias = ((u >> 16) & 1) + 0x7FFF
    u = (u + rne_bias) & 0xFFFF0000
    return u.view(np.float32)


def _read_arr(fp, count: int, dtype) -> np.ndarray:
    arr = np.fromfile(fp, dtype=dtype, count=count)
    if arr.size != count:
        raise IOError(f"short read: expected {count} {dtype} values, got {arr.size}")
    pad = (-arr.nbytes) % ALIGNMENT
    if pad:
        fp.seek(pad, 1)
    return arr


@dataclass
class TensorError:
    name: str
    layer: int
    rmse_vs_fp32: float
    maxabs_vs_fp32: float
    rmse_vs_bf16: float
    maxabs_vs_bf16: float


def _err(name: str, layer: int, src_fp32: np.ndarray, recon: np.ndarray) -> TensorError:
    src_bf16 = _fp32_to_bf16_to_fp32(src_fp32)
    df = src_fp32 - recon
    db = src_bf16 - recon
    return TensorError(
        name=name, layer=layer,
        rmse_vs_fp32=float(np.sqrt(np.mean(df ** 2))),
        maxabs_vs_fp32=float(np.max(np.abs(df))),
        rmse_vs_bf16=float(np.sqrt(np.mean(db ** 2))),
        maxabs_vs_bf16=float(np.max(np.abs(db))),
    )


def validate(model_size: str, bin_path: Path) -> list[TensorError]:
    cfg = MODEL_CONFIGS[model_size]
    src = load_fp32_weights(model_size)
    d, f = cfg.d_model, cfg.d_ff
    errors: list[TensorError] = []

    with open(bin_path, "rb") as fp:
        # preserved: wte, wpe
        _read_arr(fp, VOCAB_SIZE * d, np.float32)
        _read_arr(fp, CTX_LEN * d, np.float32)

        for li in range(cfg.num_layers):
            layer = src.layers[li]
            _read_arr(fp, d, np.float32)  # ln1_gamma
            _read_arr(fp, d, np.float32)  # ln1_beta

            for name, shape, scale_len in [
                ("W_qkv",     (3 * d, d), 3 * d),
                ("attn_proj", (d, d),     d),
            ]:
                int8 = _read_arr(fp, shape[0] * shape[1], np.int8).reshape(shape)
                if int(int8.min()) < -127 or int(int8.max()) > 127:
                    raise AssertionError(
                        f"{name} layer {li}: int8 out of range "
                        f"[{int(int8.min())}, {int(int8.max())}]"
                    )
                scale = _read_arr(fp, scale_len, np.float32)
                recon = int8.astype(np.float32) * scale[:, None]
                errors.append(_err(name, li, getattr(layer, name), recon))
                if name == "W_qkv":
                    _read_arr(fp, 3 * d, np.float32)  # b_qkv

            _read_arr(fp, d, np.float32)  # attn_proj_bias
            _read_arr(fp, d, np.float32)  # ln2_gamma
            _read_arr(fp, d, np.float32)  # ln2_beta

            for name, shape, scale_len in [
                ("W1", (f, d), f),
                ("W2", (d, f), d),
            ]:
                int8 = _read_arr(fp, shape[0] * shape[1], np.int8).reshape(shape)
                if int(int8.min()) < -127 or int(int8.max()) > 127:
                    raise AssertionError(
                        f"{name} layer {li}: int8 out of range "
                        f"[{int(int8.min())}, {int(int8.max())}]"
                    )
                scale = _read_arr(fp, scale_len, np.float32)
                recon = int8.astype(np.float32) * scale[:, None]
                errors.append(_err(name, li, getattr(layer, name), recon))
                if name == "W1":
                    _read_arr(fp, f, np.float32)  # b1

            _read_arr(fp, d, np.float32)  # b2

        _read_arr(fp, d, np.float32)  # lnf_gamma
        _read_arr(fp, d, np.float32)  # lnf_beta

    return errors


def summarize(errors: list[TensorError]) -> dict[str, dict[str, float]]:
    """Aggregate per-tensor errors across all layers, grouped by tensor name."""
    by_name: dict[str, list[TensorError]] = {}
    for e in errors:
        by_name.setdefault(e.name, []).append(e)
    summary = {}
    for name, lst in by_name.items():
        summary[name] = {
            "n_layers": len(lst),
            "mean_rmse_vs_fp32":   float(np.mean([e.rmse_vs_fp32   for e in lst])),
            "max_rmse_vs_fp32":    float(np.max ([e.rmse_vs_fp32   for e in lst])),
            "max_maxabs_vs_fp32":  float(np.max ([e.maxabs_vs_fp32 for e in lst])),
            "mean_rmse_vs_bf16":   float(np.mean([e.rmse_vs_bf16   for e in lst])),
            "max_rmse_vs_bf16":    float(np.max ([e.rmse_vs_bf16   for e in lst])),
            "max_maxabs_vs_bf16":  float(np.max ([e.maxabs_vs_bf16 for e in lst])),
        }
    return summary
