# dot_2d GPU vs CPU Benchmark

Benchmarks matrix multiplication (sgemm) using **cuBLAS** (GPU) vs **OpenBLAS** (CPU) across matrix sizes corresponding to real transformer model dimensions.

## Prerequisites

- NVIDIA GPU with CUDA 12.x
- CUDA toolkit (default path: `/usr/local/cuda-12.8`)
- OpenBLAS (`sudo apt install libopenblas-dev`)
- Python 3 with a virtual environment

### Python dependencies

```bash
# Using the project venv
source ../../transformer_env/bin/activate
pip install matplotlib seaborn
```

## Build

```bash
make
```

To override the CUDA path:

```bash
make CUDA_PATH=/usr/local/cuda-12.6
```

## Run the C binary directly

```bash
# Human-readable output
./dot_2d_gpu 4096

# JSON output (for scripting)
./dot_2d_gpu 4096 --json
```

## Run the full benchmark suite

```bash
source ../../transformer_env/bin/activate
python3 benchmark.py
```

This will:
1. Collect hardware info (CPU, GPU, RAM, CUDA version)
2. Run benchmarks for 8 matrix sizes (512 to 16384) mapped to real model dimensions
3. Save results to `benchmark_results/benchmark_results.json`
4. Generate plots in `benchmark_results/`:
   - `benchmark_plot.png` - combined 2x2 overview
   - `plot_execution_time.png` - execution time comparison
   - `plot_throughput.png` - GFLOPS throughput
   - `plot_speedup.png` - GPU speedup factor
   - `plot_reference_table.png` - model reference table

## Output structure

```
benchmark_results/
  benchmark_results.json    # Full JSON report with HW info + all measurements
  benchmark_plot.png        # Combined 2x2 plot
  plot_execution_time.png   # Individual: execution time bars
  plot_throughput.png       # Individual: GFLOPS line chart
  plot_speedup.png          # Individual: speedup bars
  plot_reference_table.png  # Individual: reference table
```
