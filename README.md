# Elastic SDM

## Intuition

Sparse Delta Memory (SDM) gives each recurrent layer a large logical table but
touches only a few rows per token. A conventional implementation still gives
every sequence a private copy of every row. For `L` layers with `N` rows of
width `V`, copy-on-write changes retained per-sequence state from `O(LNV)` to
`O(UV + LN)`: learned initial tables are shared, `U` first-written rows across
the stack become private, and an `LN` map records which logical rows have been
materialized.

The allocator can exploit only the sparsity the router produces. Task loss
does not distinguish rewriting a private row from first-writing a new one,
even though only the latter increases retained state. Elastic SDM adds that
missing signal: charge the model for each distinct row it makes private.
Logical capacity remains available when a sequence needs it, but routine
sequences can learn to reuse a much smaller working set.

## Construction

The SDM controller and recurrence are unchanged. Training adds one term:

```text
task loss + λ × unique written rows / logical rows
```

The forward value counts the exact union of hard top-`W` write addresses. Its
gradient uses selected softmax mass scattered into logical rows and combined
over time with a noisy-OR. This is a route-reuse surrogate, not a probabilistic
interpretation of top-`W` selection. The surrogate disappears at inference.

At serving time, an untouched address reads from shared learned initialization.
Its first write allocates one private value row; later writes reuse that row.
The packed allocator is exactly equivalent to a dense logical table for the
same routes, gates, and values.

The experiment used seven SDM blocks followed by one attention block, width
128, four attention heads, and one independently configured SDM memory head.
Every SDM layer had `N=1,024` addresses and balanced sparse access `R=W=16`.
No parameter or state budget was changed between prices.

## Result

On a complete 30,000-step adaptive-recall suite, a seven-price sweep exposed a
quality–state frontier rather than simply trading accuracy for compression:

| λ | Query loss | Exact-set | Private rows | Active rows | Overlay factor | Value + map factor |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.92831 | 56.64% | 3,309.2 | 46.17% | 2.15× | 2.11× |
| 0.012 | 1.93673 | 56.49% | 2,311.5 | 32.25% | 3.08× | 3.00× |
| 0.024 | 1.92957 | **57.70%** | 1,778.6 | 24.81% | 4.00× | 3.88× |
| 0.04 | 1.94265 | 54.45% | 1,385.2 | 19.33% | 5.13× | 4.94× |
| 0.12 | 1.93593 | 57.34% | 1,015.7 | 14.17% | 7.00× | 6.64× |
| 0.24 | 1.95605 | 54.04% | 746.9 | 10.42% | 9.52× | 8.86× |
| 0.40 | 1.96253 | 52.57% | 687.2 | 9.59% | 10.35× | 9.58× |

Relative to unpriced SDM, `λ=0.024` removed 46.25% of private rows while
raising exact-set accuracy by 1.06 percentage points; query loss changed by
only +0.00125. At `λ=0.12`, pricing removed 69.31% of rows and retained a
0.70-point exact-set advantage at a +0.00762 loss cost.

The extra points show where that gain stops. Occupancy decreases at every
stronger price, but quality is nonmonotonic: `λ=0.04` falls below both its
neighbors, and the `0.24` and `0.40` endpoints give up recall for diminishing
additional state reduction. The measured points are shown directly rather
than fit with a smooth trend.

![Exact-set accuracy against private rows; labels show value-overlay compression](figures/frontier.png)

The dense logical FP32 state is 3.5 MiB per sequence at this shape. The
moderate point's compact live overlay is 0.875 MiB, or 0.903 MiB including the
fixed 28 KiB direct map. The shared 917,504-parameter initial memory is model
state, not per-sequence recurrent state. Training activations, kernel
workspace, and peak device allocation are also separate from these retained
state measurements.

A matched 65.536-million-token WikiText-103 pair confirms that the selected
price does not buy recall-specific compression by damaging language modeling.
This is a control at `λ=0.024`, not a second price sweep. Row counts use 64
fixed validation routes at a 2,048-token prefix:

| λ | Test NLL | Private rows | Active rows | Overlay factor | Value + map factor |
|---:|---:|---:|---:|---:|---:|
| 0 | 5.45196 | 3,090.7 | 43.12% | 2.30× | 2.26× |
| 0.024 | **5.44370** | **2,180.7** | **30.42%** | **3.26×** | **3.18×** |

Pricing removed 29.44% of the language working set while improving test NLL
by 0.00826. Its mean loss over all 4,000 training steps was also lower
(5.93872 versus 5.95085), so the final difference is not an isolated
evaluation fluctuation.

## Why it matters

Sparse addressing limits work per token; it does not by itself minimize the
state a request retains over time. A first-touch price turns that hidden
allocation pattern into an explicit training control while preserving SDM's
logical capacity and inference rule. These are seed-0 mechanism results, not
an optimized value of `λ`; the useful price should be calibrated at each
deployment scale and workload.

## Reproduction

The repository contains the complete nine-arm training and evaluation path:

```bash
uv sync --frozen
uv run --no-sync ./reproduce.sh all
```

See [REPRODUCING.md](REPRODUCING.md) for checksums, stages, hardware, runtime,
cost, expected ranges, and output locations. Exact definitions and settings
are in [APPENDIX.md](APPENDIX.md); machine-readable provenance is in
[provenance.json](provenance.json).

## References

- Loïc Cabannes et al., [Sparse Delta Memory: Scaling the State of Linear RNNs through Sparsity](https://arxiv.org/abs/2607.07386), 2026.
- Stephen Merity et al., [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843), 2016.
