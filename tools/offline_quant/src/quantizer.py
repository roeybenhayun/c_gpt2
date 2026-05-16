"""Per-channel symmetric INT8 quantization for the 4 GEMM weights per layer.

On-disk layout for all 4 quantized tensors is [OUT, IN] (row-major). Per-channel
amax is taken along axis=1 (the IN dim), giving one scale per output channel.

scale = amax / 127  (symmetric, full range [-127, 127] — -128 is left unused
to avoid the abs-overflow trap, matching cuBLAS / TensorRT convention).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .reader import Layer, ModelConfig, Weights

INT8_MAX = 127


@dataclass
class QuantizedTensor:
    int8: np.ndarray   # int8, shape == original FP32 W
    scale: np.ndarray  # float32, length == out-channel count (W.shape[0])


@dataclass
class QuantizedLayer:
    ln1_gamma: np.ndarray
    ln1_beta: np.ndarray
    W_qkv: QuantizedTensor
    b_qkv: np.ndarray
    attn_proj: QuantizedTensor
    attn_proj_bias: np.ndarray
    ln2_gamma: np.ndarray
    ln2_beta: np.ndarray
    W1: QuantizedTensor
    b1: np.ndarray
    W2: QuantizedTensor
    b2: np.ndarray


@dataclass
class QuantizedWeights:
    config: ModelConfig
    wte: np.ndarray
    wpe: np.ndarray
    layers: list[QuantizedLayer]
    lnf_gamma: np.ndarray
    lnf_beta: np.ndarray


def quantize_per_channel(W: np.ndarray) -> QuantizedTensor:
    """W: FP32 [OUT, IN]. Returns int8 buffer + FP32 scale of length OUT."""
    if W.dtype != np.float32:
        raise TypeError(f"expected float32, got {W.dtype}")
    if W.ndim != 2:
        raise ValueError(f"expected 2D weight, got shape {W.shape}")

    amax = np.max(np.abs(W), axis=1)
    # Zero-amax rows would divide by zero. Scale=1 makes the row quantize to all
    # zeros, which is the correct dequant. (HF GPT-2 has no zero rows in practice.)
    scale = np.where(amax > 0, amax / INT8_MAX, np.float32(1.0)).astype(np.float32)
    W_int8 = np.clip(np.rint(W / scale[:, None]), -INT8_MAX, INT8_MAX).astype(np.int8)
    return QuantizedTensor(int8=W_int8, scale=scale)


def dequantize(qt: QuantizedTensor) -> np.ndarray:
    return qt.int8.astype(np.float32) * qt.scale[:, None]


def quantize_layer(layer: Layer) -> QuantizedLayer:
    return QuantizedLayer(
        ln1_gamma=layer.ln1_gamma,
        ln1_beta=layer.ln1_beta,
        W_qkv=quantize_per_channel(layer.W_qkv),
        b_qkv=layer.b_qkv,
        attn_proj=quantize_per_channel(layer.attn_proj),
        attn_proj_bias=layer.attn_proj_bias,
        ln2_gamma=layer.ln2_gamma,
        ln2_beta=layer.ln2_beta,
        W1=quantize_per_channel(layer.W1),
        b1=layer.b1,
        W2=quantize_per_channel(layer.W2),
        b2=layer.b2,
    )


def quantize_weights(weights: Weights) -> QuantizedWeights:
    return QuantizedWeights(
        config=weights.config,
        wte=weights.wte,
        wpe=weights.wpe,
        layers=[quantize_layer(l) for l in weights.layers],
        lnf_gamma=weights.lnf_gamma,
        lnf_beta=weights.lnf_beta,
    )
