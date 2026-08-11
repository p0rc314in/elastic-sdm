# Reproducing the experiments

The reproduction reruns nine complete seed-0 arms: seven adaptive-recall
prices and two WikiText-103 prices. Every arm uses one SDM memory head,
`N=1,024`, and `R=W=16`.

## Environment and data

The checked-in `uv.lock` fixes the complete Python dependency graph. Python
3.11–3.13 and CUDA GPUs are required.

```bash
uv sync --frozen
uv run --no-sync ./reproduce.sh prepare
```

Preparation downloads checksum-pinned public WikiText-103 Parquet shards and
regenerates the exact language and recall streams. The resulting manifest
hashes must be:

```text
WikiText  b1bb41b7bc8f9c1fe4bb22820e6d242d67d4cc143f3763c371ff6ea6e6fd987d
Recall    b0587d62c3ab709c94e37742892451a39463a7d577212b9379bd76d966f97800
```

Allow roughly 2 GB for downloaded, intermediate, and generated data.

## Training

Use seven identical GPUs for the concurrent recall arms, then two for the
language pair:

```bash
GPUS=0,1,2,3,4,5,6 uv run --no-sync ./reproduce.sh recall
GPUS=0,1 uv run --no-sync ./reproduce.sh language
```

With one GPU, arms queue sequentially. `uv run --no-sync ./reproduce.sh all`
runs preparation and both suites.

The recorded recall sweep consumed 23.83 aggregate billed A100 GPU-hours. Its
seven arms were collected in two matched launches, but a clean seven-GPU rerun
should finish in roughly 3.5–3.7 wall-clock hours. The recorded language pair
consumed 5.04 aggregate billed GPU-hours on two GPUs; that total includes a
serial occupancy export that the included exact batched counter replaces.
Current language reruns should take about 4.8 aggregate GPU-hours, or 2.4
wall-clock hours on two GPUs. At the recorded $1.49–$1.59 per GPU-hour, all nine
arms cost $44.86; budget approximately $45–$48 plus small setup and storage
overhead.

Recall exact-set accuracy should remain around 52–59%. WikiText test NLL should
remain around 5.42–5.48, with the priced arm within 0.03 NLL of the unpriced
control. These are tolerance bands for a complete rerun, not substitutions for
training.

Outputs are written under `runs/reproduction/`. Each arm checks exact prefix
causality, native recurrence, packed copy-on-write equivalence, configuration,
and a finite first optimizer step before continuing. The suite then extracts
fresh tables and enforces the recorded quality and state bounds.

## Recorded values

```bash
uv run --no-sync ./reproduce.sh verify-results
```

This checks arithmetic and provenance for the compact values committed under
`data/`; it does not run training.
