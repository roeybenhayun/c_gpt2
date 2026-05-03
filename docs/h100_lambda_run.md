# Running GPT-2 on a Lambda Cloud H100 Instance

End-to-end runbook for benchmarking this repo on a Lambda Cloud **on-demand
1× H100 (80 GB SXM5)** — 26 vCPUs, 225 GiB RAM, 2.8 TiB SSD, **$4.29 / GPU / hr**
as of 2026-05. Optimised for **minimum billed minutes**: do as little as possible
on the instance, copy results back, terminate.

> **Quick automated path** (recommended): once the instance is booted and you
> have the IP, scp a single setup-and-run script onto it, SSH in with agent
> forwarding, and let it do steps 2–6:
>
> ```bash
> # from your laptop
> scp scripts/lambda_instance_run.sh ubuntu@<ip>:~
> ssh -A ubuntu@<ip>
>
> # on the instance
> bash ~/lambda_instance_run.sh
> ```
>
> You watch the benchmarks live in your SSH session. When it prints
> "RUN COMPLETE", the script's closing banner templates the exact rsync
> command (with this instance's IP already filled in) for you to run on the
> laptop. Then terminate the instance from the Lambda dashboard.
>
> The manual steps below are the long-form explanation — useful when
> debugging, or when you want to deviate from the recommended profiled benchmark.

> Reference numbers from a local RTX 5080 (Blackwell, sm_120):
> Small ≈ 158 TPS, Medium ≈ 95 TPS, Large ≈ 60 TPS (decode preset, BF16). H100
> has ~3.5× the memory bandwidth of the 5080 (~3.35 TB/s HBM3 vs ~960 GB/s
> GDDR7) and decode is bandwidth-bound on this code, so expect roughly
> **2–3× higher TPS on Large**.

---

## 0. Before launching the instance

Pre-launch checklist (do these on your laptop — they cost no GPU minutes):

- [ ] Latest changes pushed to `master` on GitHub.
- [ ] **Lambda SSH key uploaded to the Lambda dashboard** (Settings → SSH Keys).
      This is the key Lambda injects into the instance at boot — without it you
      literally cannot SSH in, and you cannot add it after launch through the
      UI. If your laptop key isn't already there:
      ```bash
      cat ~/.ssh/id_ed25519.pub        # or id_rsa.pub — paste into Lambda
      ```
- [ ] Decide how the instance will authenticate to **GitHub** for the clone in
      step 2. Easiest is SSH agent forwarding from your laptop (no key material
      ever lands on the cloud box). Confirm your GitHub key is loaded locally:
      ```bash
      ssh-add -l                       # should list your GitHub key
      ssh-add ~/.ssh/id_ed25519        # if missing
      ```
- [ ] Local archive dir for results (already created): `logs/lambda/h100/`.
      This sits inside the project's `logs/` tree, which is gitignored
      (`.gitignore:20` → `logs/*`), so cloud results won't be accidentally committed.

Local sanity check:
```bash
git status                       # should be clean
git log --oneline -1 origin/master
```

Then launch a **1× H100 (80 GB SXM5)** on-demand instance from the Lambda
dashboard, region nearest you. Note the public IP — referenced as `$H100_IP`
below.

---

## 1. SSH in and start a tmux session

Use `-A` to forward your local SSH agent so the GitHub clone in step 2 works
without copying keys to the instance. `tmux` keeps the run going if your SSH
drops (still billed, but not lost):

```bash
ssh -A ubuntu@$H100_IP
tmux new -s gpt2
```

Verify agent forwarding works (should print "Hi <you>! You've successfully
authenticated…" then exit 1, which is fine):

```bash
ssh -T git@github.com
```

Quick sanity check on the GPU:

```bash
nvidia-smi                       # expect: H100 80GB HBM3, CUDA 12.x
nvcc --version
```

H100 is Hopper (sm_90) and works on any CUDA ≥ 11.8, but the Makefile
hardcodes the 12.8 path (see step 4) — easiest if `nvcc --version` reports
12.x and `/usr/local/cuda-12.8/bin/nvcc` exists. Lambda's stock image usually
ships exactly this.

---

## 2. Pull the code

With agent forwarding from step 1, the SSH clone just works:

```bash
cd ~
git clone git@github.com:roeybenhayun/c_gpt2.git
cd c_gpt2
git checkout master
```

Fallbacks if you skipped agent forwarding:

- HTTPS + `gh auth login` (browser device-code flow):
  ```bash
  gh auth login
  git clone https://github.com/roeybenhayun/c_gpt2.git
  ```
- Or generate a keypair on the instance and add it as a GitHub deploy key.

---

## 3. Run the setup script

`setup.sh` installs `uv`, system deps (`libjansson-dev`, `libopenblas-dev`,
build tools), downloads the tokenizer, downloads weights, and runs `uv sync`.

Weight download is interactive — pipe `all` to grab Small/Medium/Large in one
shot:

```bash
yes all | ./setup.sh
```

Expected: ~5 GB of weights download (Small 500 MB + Medium 1.4 GB + Large 3 GB).
On the H100 instance's network this should be a minute or two — plenty of
SSD headroom in the 2.8 TiB volume.

> **Cost tip:** if you'll spin up H100 instances repeatedly, push the
> `weights/` directory to a Lambda persistent volume or to your own S3 bucket
> after the first run. On subsequent runs, `aws s3 sync` is much faster than
> re-pulling from HuggingFace.

---

## 4. Patch the Makefile's NVCC path

The Makefile hardcodes `NVCC = /usr/local/cuda-12.8/bin/nvcc` (see
`Makefile:84`). **Lambda's Ubuntu 24.04 + Lambda Stack image installs `nvcc`
via apt at `/usr/bin/nvcc`** — the `/usr/local/cuda-12.8/` directory does
not exist on this image. Without this patch, the GPU build fails immediately
with "nvcc not found".

Apply the one-line fix:

```bash
sed -i 's|/usr/local/cuda-12.8/bin/nvcc|/usr/bin/nvcc|' Makefile
grep -n 'NVCC =' Makefile         # verify: line 84 should now read /usr/bin/nvcc
```

Sanity-check the toolchain `nvcc` will use:

```bash
which nvcc                        # expect: /usr/bin/nvcc
nvcc --version                    # expect: release 12.8.x
```

> **Why this is needed:** the project Makefile pins the path used on the
> author's local desktop. nvcc resolves its real path to find co-located
> tooling (`cudafe++`, `ptxas`, etc.), all of which Lambda installs
> alongside `/usr/bin/nvcc`, so the build behaves identically once the
> path is corrected. **Don't commit this change** — it would break local
> Linux desktops that *do* have CUDA at `/usr/local/cuda-12.8/`. A more
> robust fix (`NVCC ?= nvcc`) is tracked separately.

---

## 5. Start the tokenizer server

The C binary talks to `tokenizer.py` on `127.0.0.1:65432`. Start it in a
separate tmux pane (Ctrl-b ", then in the new pane):

```bash
cd ~/c_gpt2
uv run python tokenizer.py
```

Leave it running. Switch back with Ctrl-b o.

---

## 6. Build + run the benchmarks

`./scripts/run.sh` does build + run + log together.

### Pre-flight — confirm nsys works before the long benchmark

`--profile` invokes `nsys` directly via `eval`. If `nsys` is missing or
mis-pathed, the benchmark JSON still gets produced (so you don't notice
mid-run) but **no `.nsys-rep` is written** — and you only find out after
spending 8 minutes on profiled runs that produced nothing profilable. The
automated `lambda_instance_run.sh` script does all of the steps below; this
section is the manual / debugging reference.

#### 1. Install the right packages (Lambda Stack 24.04 specifics)

The Lambda Stack image **does not ship nsys pre-installed**. The package name
is `nsight-systems` (NOT `nsight-systems-cli` as on stock Ubuntu — that
package doesn't exist in Lambda's apt repo at all):

```bash
sudo apt-get update
sudo apt-get install -y nsight-systems nsight-systems-target
```

The `-target` package is critical — it contains the `QdstrmImporter` binary
that converts raw nsys captures into `.nsys-rep` reports. Without it,
profiled runs produce only `.qdstrm` files, and `compare_profiles.py` /
`nsys-ui` won't read them without a manual conversion step.

#### 2. Symlink the importer (Lambda packaging quirk)

Even with both packages installed, nsys **can't find the importer** out of
the box on this image — it ships at `/usr/lib/nsight-systems/host-linux-x64/`
but nsys looks for it as a sibling of its own bin/ directory. Symptom:

```
Importer error status: The importer binary and its dependencies were not found.
Unable to retrieve the importer version: skipping importation of the QDSTRM file.
```

The QdstrmImporter binary itself runs fine (`/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter --help` prints help) — nsys just looks in the wrong place. Fix:

```bash
NSYS_REAL=$(readlink -f $(which nsys))
NSYS_PARENT=$(dirname $(dirname $NSYS_REAL))
sudo ln -sf /usr/lib/nsight-systems/host-linux-x64 $NSYS_PARENT/host-linux-x64
```

#### 3. Smoke test that nsys can produce a report

```bash
# logs/ exists with room to write (run.sh creates it on first run, but verify)
ls -ld ~/c_gpt2/logs/ 2>/dev/null || mkdir -p ~/c_gpt2/logs/
df -h ~/c_gpt2/                      # 2.8 TiB SSD; profile artefacts are ~6 GB total

# Smoke test: use `sleep 0.5` rather than `true` — `true` exits too fast for
# nsys 2024+ to attach and capture, producing a "no .nsys-rep" false alarm.
# `--force-overwrite` handles stale files from previous attempts.
rm -f /tmp/nsys_smoke.*
nsys profile --stats=false --force-overwrite=true -o /tmp/nsys_smoke sleep 0.5
ls -lh /tmp/nsys_smoke.nsys-rep      # this file MUST exist
rm -f /tmp/nsys_smoke.*
```

If `/tmp/nsys_smoke.nsys-rep` exists, nsys is fully working. If only
`/tmp/nsys_smoke.qdstrm` exists, the importer symlink in step 2 didn't take
— diagnose with `ldd /usr/lib/nsight-systems/host-linux-x64/QdstrmImporter | grep 'not found'`.

#### What nsys produces alongside benchmark JSON

Where `nsys` lands by default — and where the runner writes — is
`~/c_gpt2/logs/`, the same directory as the JSON metric files. Each profiled
run produces three siblings:

- `gpt2_<size>_<tag>_<preset>_profile_<TS>.nsys-rep` — the report (the file
  `nsys-ui` and `compare_profiles.py` actually consume)
- `gpt2_<size>_<tag>_<preset>_profile_<TS>.sqlite` — query DB (created by
  `--stats=true`; safe to delete or skip on rsync)
- `gpt2_<size>_<tag>_<preset>_profile_<TS>.json` — the normal metric log

After the **first** profiled command starts (next subsection), confirm in a
second tmux pane that the artefacts actually land:

```bash
ls -lh ~/c_gpt2/logs/*.nsys-rep 2>/dev/null
# expect the first .nsys-rep within ~30-60 sec of the small model finishing
```

If only `.json` files appear and no `.nsys-rep` after the small run completes
→ stop the benchmark and diagnose, don't burn cloud minutes on profileless profiled
runs.

### Recommended run — full benchmark with profiling (default)

For the most data per dollar of cloud time, run **FP32 + BF16 across all three
sizes and all three workload presets, with `--profile` enabled** so every run
also emits an `.nsys-rep` for kernel-level analysis later:

```bash
./scripts/run.sh --gpu --bf16 --profile --decode
./scripts/run.sh --gpu --bf16 --profile --prefill
./scripts/run.sh --gpu --bf16 --profile --balanced
```

What this produces:

- 3 presets × 3 sizes × 2 dtypes = **18 JSON logs** with TTFT / TPOT / TPS / E2E
- 18 matching `.nsys-rep` files (plus `.sqlite` siblings) for kernel breakdown
- Total billed time: roughly **6–10 minutes** (the `--profile` flag adds ~30 s
  of nsys overhead per run, on top of the ~3–6 min raw run time)
- Net cost at $4.29/hr: **~$0.50–0.75**

`--profile` is GPU-only — the runner silently skips CPU runs even if `--cpu`
is set. Skipping CPU on the cloud is the right call regardless: there's no
reason to pay $4.29/hr for the OpenBLAS path your laptop already runs for free.

### Faster / cheaper variants

If you've already paid for one full benchmark and just want to re-collect a single
slice, the same flags compose down:

```bash
# BF16 only, all sizes, just decode + prefill, profiled — ~3 min wall, ~$0.25
./scripts/run.sh --bf16 --profile --decode
./scripts/run.sh --bf16 --profile --prefill

# Single size for a fast smoke test
./scripts/run.sh --gpu --bf16 --profile --decode large
```

### Why `--profile` is the default here

On your local desktop, `--profile` is opt-in because nsys writes large `.sqlite`
files (~250 MB per profiled run) and the disk fills up fast. On the cloud
instance you're going to terminate in 10 minutes anyway — disk doesn't matter,
and the profiles are what answer "where did the time actually go?" once you've
seen surprising TPS numbers. Skipping `--profile` on the cloud and then wishing
you had it later costs you another $0.50 instance launch.

---

## 7. Copy logs back to your laptop

**From the laptop**, while the H100 is still running. Each run lands in its
own labelled subdir of `logs/lambda/h100/` — date for chronology, label for
intent. Don't dump multiple runs into the same flat directory: the analyzer's
discovery rule (newest-mtime per `model_<tag>_<preset>` pattern) will silently
mix runs whose scopes differ (e.g. an old FP32+BF16 benchmark with a new BF16-only
profiled run), and you'll only notice when a plot looks subtly wrong.

```bash
cd ~/projects/c_gpt2

# Pick a label that says what the run *is*, not just when it happened.
# Examples: full_benchmark_bf16_fp32, bf16_nsys_profile, decode_only, prefill_benchmark_long_prompt
RUN_LABEL="bf16_nsys_profile"                                      # ← edit per run
RUN_DIR="logs/lambda/h100/$(date +%Y%m%d_%H%M%S)_${RUN_LABEL}"
mkdir -p "$RUN_DIR"

rsync -avz --progress \
    ubuntu@$H100_IP:~/c_gpt2/logs/ \
    "$RUN_DIR/"
```

Sanity check (numbers vary by run scope — e.g. 18 for the full benchmark, 6 for
BF16-only one preset, etc.):

```bash
ls "$RUN_DIR"/*.json     | wc -l    # JSON metric logs
ls "$RUN_DIR"/*.nsys-rep | wc -l    # nsys profile reports (if --profile was used)
```

> **Skip `.sqlite` to halve transfer time.** Every nsys profile produces a
> `.nsys-rep` (the report `nsys-ui` and `compare_profiles.py` both read) plus
> a `.sqlite` sibling (~250 MB each, used only by some advanced nsys queries).
> The sqlite files are regenerable from the rep, so it's safe to skip them on
> the rsync if upstream bandwidth is tight:
> ```bash
> rsync -avz --progress --exclude='*.sqlite' \
>     ubuntu@$H100_IP:~/c_gpt2/logs/ \
>     "$RUN_DIR/"
> ```

---

## 8. Terminate the instance

**Stop billing immediately** — terminate from the Lambda dashboard. SSH
disconnects don't stop billing, and shutting down the OS from inside the
instance doesn't either. Only **Terminate** in the dashboard (or via API)
stops the meter.

---

## 9. Run the analysis locally

Point `performance_analysis.py` at the H100 archive directly with `--log-dir`
— it leaves your local `logs/` results untouched:

```bash
cd ~/projects/c_gpt2
uv run python scripts/performance_analysis.py --gpu --bf16 --decode   --log-dir "$RUN_DIR"
uv run python scripts/performance_analysis.py --gpu --bf16 --prefill  --log-dir "$RUN_DIR"
uv run python scripts/performance_analysis.py --gpu --bf16 --balanced --log-dir "$RUN_DIR"
```

Plots land in `plots/` and a summary table prints to stdout. (Without
`--log-dir`, the analyzer falls back to its `logs/` default — see
`scripts/performance_analysis.py:18`.)

For a kernel-level FP32-vs-BF16 diff (if you profiled in step 6):

```bash
uv run python scripts/compare_profiles.py "$RUN_DIR"/<...>_gpu_*.nsys-rep "$RUN_DIR"/<...>_bf16_*.nsys-rep
```

---

## Cost-minimisation summary

At **$4.29 / GPU / hr** (≈ **$0.072 / minute**):

| Step | What costs minutes | Approx. cost | Mitigation |
|------|-------------------|--------------|------------|
| Setup (`yes all \| ./setup.sh`) | weight download (~5 GB) | ~$0.15 | persist `weights/` to S3 / Lambda volume after first run |
| Building | ~30 s | ~$0.04 | nothing to do |
| Running (18 configs) | ~3–6 min | ~$0.20–0.45 | drop presets you don't care about; skip `--cpu` |
| Profiling (`--profile`) | ~30 s extra per run | ~$0.04 | only profile the size you'll dig into |
| Idle SSH session | full price | $0.072/min | `tmux` so you don't restart on disconnect; **terminate** when done |

**Total billed time for one full-suite run, no dawdling: ~10–15 minutes ≈ $0.75–1.10.**
The most expensive thing you can do is forget to terminate after — leaving the
instance up overnight (~12 h) is ~**$51**. Set a phone timer.

### Reference: actual measured cost (first run)

| Run | Billed time | Total cost | Notes |
|-----|------------:|-----------:|-------|
| 2026-05-02 — full benchmark (FP32 + BF16, decode + prefill + balanced, no profile) | **0.36 hr (~22 min)** | **$1.53** | First-time setup overhead: SSH key registration, Makefile path patch (Lambda installs `nvcc` at `/usr/bin/nvcc`, not the project's pinned path), discovering each step the runbook didn't yet warn about |

So the "no dawdling" estimate above was about **40% optimistic for a first
run** — first-time friction (debugging the build path, looking up commands,
deciding what to do next) is real and worth budgeting for. Subsequent runs
on the same image, with the runbook's fixes already in front of you, should
land much closer to the table's estimate.

Implication for future planning: budget **~$1.50–2.00 for any first-time
experiment on a new instance type** (B200, H200, MI300X, etc.) and **~$0.75–1.00
per repeat run** on a known-good image. Not 4 figures, not 3 figures —
benchmarking modern accelerators is genuinely cheap as long as you remember
to terminate.
