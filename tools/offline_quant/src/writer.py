"""Sequential writer for the INT8-quantized .bin file.

Layout matches gpt2.c:1361-1404 load_all_weights() order, with the 4 GEMM
tensors per layer replaced by an INT8 buffer immediately followed by its
FP32 per-channel scale vector. All blocks padded to 16-byte alignment
(current model sizes are naturally aligned so padding is a no-op, but
the contract is enforced so future config changes don't silently desync).

The sidecar quant_config.json is a human-readable manifest only — the
runtime loader does not read it; layout is derived from compile-time
model config.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .quantizer import INT8_MAX, QuantizedTensor, QuantizedWeights

ALIGNMENT = 16
TOOL_VERSION = "0.1"

QUANTIZED_TENSOR_NAMES = ["W_qkv", "attn_proj", "W1", "W2"]
PRESERVED_TENSOR_NAMES = [
    "wte", "wpe",
    "ln1_gamma", "ln1_beta", "b_qkv", "attn_proj_bias",
    "ln2_gamma", "ln2_beta", "b1", "b2",
    "lnf_gamma", "lnf_beta",
]


def _write_block(fp, arr: np.ndarray) -> int:
    """Write arr's bytes, then zero-pad to ALIGNMENT. Returns total bytes (incl. padding)."""
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    fp.write(arr.tobytes())
    pad = (-arr.nbytes) % ALIGNMENT
    if pad:
        fp.write(b"\x00" * pad)
    return arr.nbytes + pad


def _write_quantized(fp, qt: QuantizedTensor) -> None:
    _write_block(fp, qt.int8)
    _write_block(fp, qt.scale)


def write_bin(qw: QuantizedWeights, bin_path: Path) -> None:
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bin_path, "wb") as fp:
        _write_block(fp, qw.wte)
        _write_block(fp, qw.wpe)
        for layer in qw.layers:
            _write_block(fp, layer.ln1_gamma)
            _write_block(fp, layer.ln1_beta)
            _write_quantized(fp, layer.W_qkv)
            _write_block(fp, layer.b_qkv)
            _write_quantized(fp, layer.attn_proj)
            _write_block(fp, layer.attn_proj_bias)
            _write_block(fp, layer.ln2_gamma)
            _write_block(fp, layer.ln2_beta)
            _write_quantized(fp, layer.W1)
            _write_block(fp, layer.b1)
            _write_quantized(fp, layer.W2)
            _write_block(fp, layer.b2)
        _write_block(fp, qw.lnf_gamma)
        _write_block(fp, qw.lnf_beta)


def compute_memory_summary(qw: QuantizedWeights) -> dict[str, int]:
    quantized_int8 = 0
    scales = 0
    preserved = qw.wte.nbytes + qw.wpe.nbytes + qw.lnf_gamma.nbytes + qw.lnf_beta.nbytes
    for layer in qw.layers:
        for name in QUANTIZED_TENSOR_NAMES:
            qt: QuantizedTensor = getattr(layer, name)
            quantized_int8 += qt.int8.nbytes
            scales += qt.scale.nbytes
        preserved += (
            layer.ln1_gamma.nbytes + layer.ln1_beta.nbytes
            + layer.b_qkv.nbytes + layer.attn_proj_bias.nbytes
            + layer.ln2_gamma.nbytes + layer.ln2_beta.nbytes
            + layer.b1.nbytes + layer.b2.nbytes
        )
    source_fp32 = preserved + quantized_int8 * 4
    artifact = preserved + quantized_int8 + scales
    MiB = 1024 * 1024
    return {
        "source_fp32_total_mb": round(source_fp32 / MiB),
        "quantized_int8_mb": round(quantized_int8 / MiB),
        "preserved_fp32_mb": round(preserved / MiB),
        "scale_overhead_kb": round(scales / 1024),
        "artifact_total_mb": round(artifact / MiB),
    }


def build_manifest(qw: QuantizedWeights) -> dict:
    cfg = qw.config
    return {
        "tool_version": TOOL_VERSION,
        "model": f"gpt2-{cfg.name}",
        "scheme": "int8_per_channel_symmetric",
        "scale_convention": "amax/127",
        "scale_dtype": "fp32",
        "source_weights_path": f"weights/{cfg.weights_filename}",
        "quantized_tensors": list(QUANTIZED_TENSOR_NAMES),
        "preserved_tensors": list(PRESERVED_TENSOR_NAMES),
        "memory_summary": compute_memory_summary(qw),
    }


def write_quantized(qw: QuantizedWeights, out_dir: Path) -> tuple[Path, Path]:
    """Write gpt2_<size>_quant8.{bin,json} to out_dir. Returns (bin_path, json_path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = qw.config
    bin_path = out_dir / f"gpt2_{cfg.name}_quant8.bin"
    json_path = out_dir / f"gpt2_{cfg.name}_quant8.json"
    write_bin(qw, bin_path)
    with open(json_path, "w") as f:
        json.dump(build_manifest(qw), f, indent=2)
        f.write("\n")
    return bin_path, json_path
