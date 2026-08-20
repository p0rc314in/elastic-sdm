# Experimental details

## Objective and router gradient

For each SDM bank, hard occupancy is the terminal fraction of logical rows
selected by at least one top-`W` write. If `q_ti` is the selected softmax mass
scattered from token `t` to logical row `i`, training uses:

```text
C_soft = (1/N) sum_i [1 - product_t(1 - q_ti)]
C_ST   = stopgrad(C_hard - C_soft) + C_soft
L      = L_task + lambda * mean_bank(C_ST)
```

The straight-through value equals the exact hard union in the forward pass and
uses the soft term only in the backward pass. For one selected row, its soft
marginal cost is:

```text
dC_soft / dq_ti = (1/N) product_{s != t}(1 - q_si)
```

A row with no routing mass elsewhere in the sequence has a large marginal
cost. A row reused by other tokens has a small one because the product has
begun to saturate.
Through the selected-score softmax, gradient descent shifts write mass away
from relatively novel selected rows and toward selected rows with lower reuse
cost. With only one write position, all selected routes have the same marginal
cost and the selected weights sum to one, so there is no distributional
signal. The useful signal comes from route overlap across time; a token's
gradient can depend on reuse both earlier and later in the training sequence.

This construction makes four deliberate choices:

- The forward statistic is the exact terminal write union because first writes
  allocate copy-on-write state; reads and repeated writes reuse existing state.
- The soft union is a noisy-OR over time, so the cost naturally saturates as a
  row is reused.
- The surrogate follows physical routing by using the selected write-softmax
  weights.
- Each bank and memory head receives equal weight through a mean of normalized
  occupancies, keeping the objective independent of model depth at the tested
  equal-capacity shape.

The selected top-`W` indices remain discrete. The occupancy gradient reshapes
selected weights and their product-key factors; those score changes can alter
later top-`W` boundaries during training. Selected softmax mass supplies the
directional route gradient, while the hard union supplies the allocation
statistic. The auxiliary term is absent at inference.

## Shared model

Both experiments use the same local-SDM shell:

- layout `BBBBBBBA`: seven SDM blocks followed by one attention block;
- width 128 and expansion-four MLPs;
- four attention heads, independently configured from one SDM memory head;
- balanced sparse access `R=W=16`;
- learned initial memory, raw product-key routing, selected softmax, scalar SDM
  controller, read-after-write recurrence, per-head normalization, channelwise
  output gate, and output projection;
- BF16 activations and role-keyed initialization.

Matched arms vary the occupancy price and, in the capacity experiment, the
number of logical rows.

## Language experiment

The WikiText-103 curve runs λ in {0, 0.024, 0.12, 0.24, 0.40} with N = 1 024
at seed 0. Each arm trains for three complete corpus passes: 21 603 optimizer
steps at effective batch 8 and context length 2 048, totaling 353 941 347
scored target presentations. Each pass presents all 117 980 449 training
targets once, without replacement; the final partial record is retained and
padding is excluded from the loss.

AdamW uses a peak learning rate of 3e-4, 540 linear warmup steps, cosine decay
to 0.1 of the peak, β = (0.9, 0.95), weight decay 0.01, and gradient clipping
at 1.0.

Validation scores all 247 416 transitions after each pass. Terminal test
evaluation scores all 283 426 test transitions exactly once with a 512-token
stride and up to 2 048 tokens of context. Working-set measurements use 64
fixed full-context validation windows at a 2 048-token prefix.

## Recall canary

The fixed-capacity recall canary uses the same five prices as the language
experiment: `lambda` in `{0, 0.024, 0.12, 0.24, 0.40}` with `N=1,024` at seed
0. Each arm trains for 30,000 steps at effective batch 32. Evaluation covers
30 deterministic pointer-chase, span-recall, and overwrite-recall conditions:
61,440 examples and 983,040 queries per arm.

## Capacity experiment

The adaptive-recall comparison is a matched 2 by 3 matrix at seed 0:

| Logical rows per bank | Occupancy prices |
|---:|---:|
| 1,024 | 0, 0.024, 0.12 |
| 2,304 | 0, 0.054, 0.27 |

The nonzero prices hold λ/N fixed: `0.024/1,024 = 0.054/2,304` and
`0.12/1,024 = 0.27/2,304`. This is the objective coefficient on the reported
mean private rows per bank. With seven banks, one additional private row in one
bank changes the objective by λ/(7N). Each arm trains for 30,000 steps at
effective batch 32 and uses the same recall evaluation. The three `N=1,024`
rows reuse anchors from the fixed-capacity recall canary, and the experiment
adds three `N=2,304` arms.

Both capacities are square product-key tables: `1,024=32^2` and
`2,304=48^2`.

## State accounting

With seven banks and value width 128, dense FP32 mutable state contains
`7*N*128*4` bytes. A compact private row uses 512 value bytes and four bytes
of row metadata; the direct logical-to-physical map uses `7*N*4` bytes.

The machine-readable `mean_private_rows`, overlay bytes, and dense-state totals
sum over all seven banks. The README's private-row ratios and row-growth values
derive from that stack-wide mean. Dividing the row count by seven changes
neither ratio.

| `N` | Logical rows | Dense mutable bytes | Direct-map bytes | Shared initial-memory parameters |
|---:|---:|---:|---:|---:|
| 1,024 | 7,168 | 3,670,016 | 28,672 | 917,504 |
| 2,304 | 16,128 | 8,257,536 | 64,512 | 2,064,384 |

At `N=1,024`, the recall model has 2,608,668 trainable parameters. At
`N=2,304`, it has 3,813,340. Of the 1,204,672 added parameters, 1,146,880 are
learned initial-memory values and 57,792 are the larger read and write
product-key projections. Exact-set accuracy records quality preservation for
the larger architecture; private-row growth records its request-state response.

The README's per-request state percentage is
`value_plus_map_bytes / dense_state_bytes`: it charges both private value plus
row-metadata bytes and the fixed direct map. The machine-readable
`value_plus_map_savings` field stores the complementary reduction. Reported
overlay compression omits the map; value-plus-map compression includes it.
Shared learned initialization belongs to model-state accounting. Private rows
and the direct map belong to per-request accounting. Parameters, training
activations, kernel workspace, and peak device allocation are separate from
this state accounting.

## Reproduction checks

All arms consume deterministic checksum-addressed streams and use identical
optimizer settings within each task. Every run checks exact prefix causality,
the SDM recurrence against its serial oracle, and packed copy-on-write state
against a dense logical table before training.

The recorded `N=1,024` recall arms used an accepted oracle-equivalent CUDA
recurrence; the `N=2,304` arms used the released fused recurrence after forward,
state, and gradient equivalence checks. The included reproduction uses the
released fused recurrence at both capacities. Rerun acceptance compares quality
and state within declared numerical tolerances that cover BF16 atomic
accumulation variation.
