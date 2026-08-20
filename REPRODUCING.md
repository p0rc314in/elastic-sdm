# Reproducing the experiments

The reproduction reruns thirteen unique arms: five WikiText-103 prices, the
same five N = 1 024 adaptive-recall prices, and three additional N = 2 304
recall arms. Every arm uses one SDM memory head and balanced R = W = 16 access.

## Environment and data

Use Python 3.12 or 3.13 on Linux with a CUDA development toolchain and CUDA
GPUs. The checked-in `uv.lock` fixes the Python dependency graph, including
PyTorch 2.11, Triton 3.6, NumPy 2.5.2, PyArrow 25.0.1, and tiktoken 0.14.0.
The build step compiles the two unmodified released-SDM CUDA extensions once
per invocation and reuses them across arms.

```bash
uv sync --frozen
uv run --no-sync ./reproduce.sh prepare
```

Preparation downloads the pinned public WikiText-103 Parquet shards and, on a
cold tiktoken cache, the two public GPT-2 tokenizer assets. It verifies every
download, then builds the language payload twice; the two builds must be
byte-identical. Download caches and prepared data stay under ignored `runs/`.
Preparation also regenerates the deterministic adaptive-recall stream. The
resulting identities are:

```text
WikiText manifest     fc4ef13cbc38070f2d7774dffbfd5be48cab31fe45d6d9995d522fc3bac1dde6
WikiText payload      f430ce52a43a44b88f5a8ec1ec5882866daaa568595bfbd6b765e3369586f85e
WikiText double build 888d5d27b8086223d48b2347364428947603ab4400a0b7c594d9b350baca7aa0
GPT-2 vocab.bpe       1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5
GPT-2 encoder.json    196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783
Recall manifest       b0587d62c3ab709c94e37742892451a39463a7d577212b9379bd76d966f97800
```

Allow about 4 GB of free space during preparation. An existing WikiText
payload is reused only after its manifest, double-build record, and every
training input pass hash verify.

## Training

The recorded runs used NVIDIA A100-SXM4-80GB GPUs. To run each suite with one
arm per GPU:

```bash
GPUS=0,1,2,3,4,5,6,7 uv run --no-sync ./reproduce.sh recall
GPUS=0,1,2,3,4 uv run --no-sync ./reproduce.sh language
```

With fewer GPUs, arms queue automatically. This single command prepares the
data and runs the recall suite followed by the language suite:

```bash
GPUS=0,1,2,3,4,5,6,7 uv run --no-sync ./reproduce.sh all
```

The eight released recall arms account for about 23 aggregate A100 GPU-hours
and finish in roughly 3.5 hours on eight GPUs. The five language arms account
for 65.1 aggregate training-and-evaluation GPU-hours and finish in roughly 13
hours on five GPUs. Their source pods were allocated for 66.64 GPU-hours,
costing $99.30 at $1.49 per GPU-hour. Budget about 90 aggregate A100 GPU-hours
and $135–$145 for all thirteen arms, plus preparation and storage overhead.
Running `all` sequentially on eight GPUs takes about 17 hours.

Outputs are written to `runs/reproduction/{recall,language}/<arm>/`.
Language arms retain a terminal recovery checkpoint; the complete output tree
therefore needs several additional gigabytes.

Each arm checks its exact configuration, balanced access, prefix causality,
the released SDM recurrence against a serial recurrence, packed copy-on-write
state against a dense logical table, and a finite first optimizer step. The
WikiText arms then consume exactly 21 603 optimizer steps and 353 941 347
target presentations across three non-replacement corpus passes. Validation is
scored after each pass, and terminal evaluation scores every test transition
once.

After each suite, fresh compact tables are written under
`runs/reproduction/measurements/` and compared with the recorded values.
Recall query loss should remain around 1.88–1.98, exact-set accuracy around
51–62 %, and total `mean_private_rows` within 5 % of the recorded CSV.
WikiText test NLL should remain around 4.42–4.51, with the same row metric
within 5 %. Released SDM uses BF16 atomic accumulation; these ranges cover
small numerical variation across clean reruns.

## Checking the recorded values

```bash
uv run --no-sync ./reproduce.sh verify-results
```

This command checks arithmetic, file hashes, and provenance for the compact
values committed under `data/`. It does not rerun an experiment. Use
`recall`, `language`, or `all` for reproduction.
