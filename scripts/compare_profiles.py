#!/usr/bin/env python3
"""Compare two nsys profiles by per-kernel total GPU time.

Buckets kernels by family (so cuBLAS gemvx variants and dtype-templated kernels
collapse into one row), then prints a side-by-side table with the BF16/FP32 ratio.

Usage:
    uv run python scripts/compare_profiles.py <baseline.nsys-rep> <target.nsys-rep>
"""

import csv
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def kernel_bucket(name: str) -> str:
    """Collapse dtype-templated and variant-templated kernel names into a family."""
    # cuBLAS GEMV families
    if "internal::gemvx::kernel" in name:
        return "cublas gemvx (all variants)"
    if "gemvNSP_kernel" in name:
        return "cublas gemvNSP"
    if "gemv2N_kernel" in name:
        return "cublas gemv2N"
    if "cublasLt::splitKreduce_kernel" in name:
        return "cublas splitKreduce"
    # cutlass GEMM families — separate tensorop from simt
    if "cutlass" in name and "tensorop" in name:
        return "cutlass tensorop GEMM (tensor cores)"
    if "cutlass" in name and "simt" in name:
        return "cutlass simt GEMM"
    # Our custom kernels — strip the parameter list to normalize across dtypes
    m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", name)
    if m:
        return m.group(1)
    return name.strip()


def parse_nsys_csv(report_path: Path) -> dict:
    """Run nsys stats on a .nsys-rep file and return {bucket: (total_ns, instances)}."""
    cmd = [
        "nsys", "stats",
        "--report", "cuda_gpu_kern_sum",
        "--format", "csv",
        "--output", "-",
        str(report_path),
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode()
    # Strip the leading NOTICE/Processing lines until we hit the CSV header
    lines = out.splitlines()
    header_idx = next(i for i, l in enumerate(lines) if l.startswith("Time (%)"))
    csv_text = "\n".join(lines[header_idx:])
    reader = csv.DictReader(csv_text.splitlines())

    buckets = defaultdict(lambda: [0, 0])  # bucket -> [total_ns, instances]
    for row in reader:
        bucket = kernel_bucket(row["Name"])
        buckets[bucket][0] += int(row["Total Time (ns)"])
        buckets[bucket][1] += int(row["Instances"])
    return {k: tuple(v) for k, v in buckets.items()}


def fmt_ns(ns: int) -> str:
    return f"{ns / 1e6:>8.1f} ms"


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    baseline_path = Path(sys.argv[1])
    target_path = Path(sys.argv[2])

    print(f"Baseline: {baseline_path.name}")
    print(f"Target:   {target_path.name}")
    print()

    base = parse_nsys_csv(baseline_path)
    targ = parse_nsys_csv(target_path)

    all_buckets = sorted(set(base) | set(targ))
    base_total = sum(v[0] for v in base.values())
    targ_total = sum(v[0] for v in targ.values())

    rows = []
    for bucket in all_buckets:
        b_ns, b_inst = base.get(bucket, (0, 0))
        t_ns, t_inst = targ.get(bucket, (0, 0))
        ratio = (t_ns / b_ns) if b_ns > 0 else float("inf") if t_ns > 0 else 0.0
        delta = t_ns - b_ns
        rows.append((bucket, b_ns, t_ns, ratio, delta, b_inst, t_inst))

    # Sort by larger of base/target time (descending)
    rows.sort(key=lambda r: -max(r[1], r[2]))

    name_w = max(len(r[0]) for r in rows)
    name_w = min(name_w, 42)

    header = (
        f"{'kernel family':<{name_w}}  "
        f"{'baseline':>11}  "
        f"{'target':>11}  "
        f"{'ratio':>7}  "
        f"{'Δ time':>11}  "
        f"{'inst (b/t)':>16}"
    )
    print(header)
    print("-" * len(header))
    for bucket, b_ns, t_ns, ratio, delta, b_inst, t_inst in rows:
        name = bucket if len(bucket) <= name_w else bucket[: name_w - 1] + "…"
        ratio_str = f"{ratio:.2f}x" if ratio else "—"
        delta_str = f"{delta / 1e6:+.1f} ms"
        inst_str = f"{b_inst}/{t_inst}"
        print(
            f"{name:<{name_w}}  "
            f"{fmt_ns(b_ns):>11}  "
            f"{fmt_ns(t_ns):>11}  "
            f"{ratio_str:>7}  "
            f"{delta_str:>11}  "
            f"{inst_str:>16}"
        )

    print("-" * len(header))
    overall_ratio = targ_total / base_total if base_total else 0
    print(
        f"{'TOTAL':<{name_w}}  "
        f"{fmt_ns(base_total):>11}  "
        f"{fmt_ns(targ_total):>11}  "
        f"{overall_ratio:.2f}x".rjust(7) + "  "
        f"{(targ_total - base_total) / 1e6:+.1f} ms".rjust(11)
    )


if __name__ == "__main__":
    main()
