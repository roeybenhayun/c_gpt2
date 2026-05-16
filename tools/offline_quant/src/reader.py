"""FP32 weight loader. Mirrors gpt2.c load_all_weights() (gpt2.c:1361-1404)
byte-for-byte: sequential FP32 reads in a fixed order, no metadata.

Trailing bytes past lnf_beta (lm_head, attention causal-mask buffers, etc.)
exist in the source file and are silently ignored — gpt2.c never reads
them either.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

VOCAB_SIZE = 50257
CTX_LEN = 1024


@dataclass(frozen=True)
class ModelConfig:
    name: str
    d_model: int
    num_layers: int
    nof_heads: int
    weights_filename: str

    @property
    def d_ff(self) -> int:
        return self.d_model * 4

    @property
    def head_dim(self) -> int:
        return self.d_model // self.nof_heads


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "small":  ModelConfig("small",  d_model=768,  num_layers=12, nof_heads=12,
                          weights_filename="gpt2_c_weights.bin"),
    "medium": ModelConfig("medium", d_model=1024, num_layers=24, nof_heads=16,
                          weights_filename="gpt2_medium_c_weights.bin"),
    "large":  ModelConfig("large",  d_model=1280, num_layers=36, nof_heads=20,
                          weights_filename="gpt2_large_c_weights.bin"),
}


@dataclass
class Layer:
    ln1_gamma: np.ndarray          # [d_model]
    ln1_beta: np.ndarray           # [d_model]
    W_qkv: np.ndarray              # [3*d_model, d_model]   QUANTIZED
    b_qkv: np.ndarray              # [3*d_model]
    attn_proj: np.ndarray          # [d_model, d_model]     QUANTIZED
    attn_proj_bias: np.ndarray     # [d_model]
    ln2_gamma: np.ndarray          # [d_model]
    ln2_beta: np.ndarray           # [d_model]
    W1: np.ndarray                 # [d_ff, d_model]        QUANTIZED
    b1: np.ndarray                 # [d_ff]
    W2: np.ndarray                 # [d_model, d_ff]        QUANTIZED
    b2: np.ndarray                 # [d_model]


@dataclass
class Weights:
    config: ModelConfig
    wte: np.ndarray                # [vocab_size, d_model]
    wpe: np.ndarray                # [ctx_len, d_model]
    layers: list[Layer]
    lnf_gamma: np.ndarray          # [d_model]
    lnf_beta: np.ndarray           # [d_model]


# tools/offline_quant/src/reader.py  →  c_gpt2/  (four parents up)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def default_weights_dir() -> Path:
    return PROJECT_ROOT / "weights"


def _read_fp32(fp, count: int, shape: tuple[int, ...] | None = None) -> np.ndarray:
    arr = np.fromfile(fp, dtype=np.float32, count=count)
    if arr.size != count:
        raise IOError(
            f"short read: expected {count} float32 values, got {arr.size} "
            f"(file truncated or wrong model size?)"
        )
    if shape is not None:
        arr = arr.reshape(shape)
    return arr


def load_fp32_weights(model_size: str, weights_dir: Path | None = None) -> Weights:
    if model_size not in MODEL_CONFIGS:
        raise ValueError(
            f"unknown model_size {model_size!r}; expected one of {list(MODEL_CONFIGS)}"
        )
    config = MODEL_CONFIGS[model_size]
    weights_dir = weights_dir or default_weights_dir()
    bin_path = weights_dir / config.weights_filename
    if not bin_path.is_file():
        raise FileNotFoundError(
            f"weights file not found: {bin_path}\n"
            f"Run setup.sh to download FP32 weights into {weights_dir}/."
        )

    d = config.d_model
    f = config.d_ff

    with open(bin_path, "rb") as fp:
        wte = _read_fp32(fp, VOCAB_SIZE * d, (VOCAB_SIZE, d))
        wpe = _read_fp32(fp, CTX_LEN * d,    (CTX_LEN, d))

        layers: list[Layer] = []
        for _ in range(config.num_layers):
            ln1_gamma      = _read_fp32(fp, d)
            ln1_beta       = _read_fp32(fp, d)
            W_qkv          = _read_fp32(fp, 3 * d * d, (3 * d, d))
            b_qkv          = _read_fp32(fp, 3 * d)
            attn_proj      = _read_fp32(fp, d * d,     (d, d))
            attn_proj_bias = _read_fp32(fp, d)
            ln2_gamma      = _read_fp32(fp, d)
            ln2_beta       = _read_fp32(fp, d)
            W1             = _read_fp32(fp, f * d,     (f, d))
            b1             = _read_fp32(fp, f)
            W2             = _read_fp32(fp, d * f,     (d, f))
            b2             = _read_fp32(fp, d)
            layers.append(Layer(
                ln1_gamma=ln1_gamma, ln1_beta=ln1_beta,
                W_qkv=W_qkv, b_qkv=b_qkv,
                attn_proj=attn_proj, attn_proj_bias=attn_proj_bias,
                ln2_gamma=ln2_gamma, ln2_beta=ln2_beta,
                W1=W1, b1=b1, W2=W2, b2=b2,
            ))

        lnf_gamma = _read_fp32(fp, d)
        lnf_beta  = _read_fp32(fp, d)

    return Weights(
        config=config, wte=wte, wpe=wpe, layers=layers,
        lnf_gamma=lnf_gamma, lnf_beta=lnf_beta,
    )
