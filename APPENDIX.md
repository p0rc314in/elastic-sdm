# Experimental details

## Objective

For each SDM bank, hard occupancy is the terminal number `U_T` of distinct
top-`W` write addresses divided by `N`. The stack averages that fraction over
its seven SDM banks. If `q_ti` is the selected top-`W` softmax mass scattered
to logical row `i`, the training surrogate is:

```text
C_soft = (1/N) Σᵢ [1 − Πₜ(1 − qₜᵢ)]
C_ST   = stopgrad(C_hard − C_soft) + C_soft
L      = L_task + λ × mean_b(C_ST,b)
```

The hard forward value is exact allocator occupancy. The soft term only carries
router gradient. Since top-`W` selection is discrete, `q` is not an inclusion
probability; its useful gradient comes from route overlap across positions.

## Shared model

Both experiments used the same local-SDM shell:

- layout `BBBBBBBA`: seven SDM blocks followed by one attention block;
- width 128 and expansion-four MLPs;
- four attention heads, independently configured from one SDM memory head;
- `N=1,024` logical rows per SDM layer;
- balanced sparse access `R=W=16`;
- learned initial memory, raw product-key routing, selected softmax, scalar SDM
  controller, read-after-write recurrence, per-head normalization, channelwise
  output gate, and output projection;
- BF16 activations, role-keyed initialization, and seed 0.

The model has 917,504 learned initial-memory parameters. Those parameters are
shared model state, not mutable per-request state. The recall model has
1,691,164 other active parameters; the language model has 14,712,348.

## Recall experiment

The recall experiment ran λ in
`{0, 0.012, 0.024, 0.04, 0.12, 0.24, 0.40}` for 30,000 steps at batch 32.
Evaluation covered 30 deterministic pointer-chase, span-recall, and
overwrite-recall conditions: 61,440 examples and 983,040 queries. The complete
condition set, rather than a selected slice, defines the reported mean state
and accuracy.

## Language experiment

The WikiText-103 confirmation ran λ in `{0, 0.024}` for 4,000 steps at batch 8
and sequence length 2,048: 65.536 million GPT-2 tokens per arm. Final validation
and test evaluation each used the same 128 checksum-addressed sequences.
Working-set measurements use 64 fixed validation routes at prefix lengths 16,
64, 256, 1,024, and 2,048.

## State accounting

Seven `N=1,024`, `V=128` FP32 tables contain 917,504 mutable values, or 3.5
MiB per sequence if materialized densely. The compact live-state accounting
charges 516 bytes per touched address: 512 bytes for 128 FP32 values and four
bytes of row-management metadata. The reported dense-to-overlay factor is
therefore:

```text
3,670,016 / (private rows × 516)
```

This is retained mutable value state. The current direct logical-to-physical
map adds a fixed 28 KiB per sequence at this seven-bank shape, and the result
files report both overlay-only and value-plus-map factors. Tiny counters and
template indices, unused reserved slab headroom, shared learned initialization,
model parameters, training activations, kernel workspace, and peak device
allocation remain separate. The allocator's `storage_bytes()` method reports
its actual reserved tensors when a concrete capacity policy is selected.

## Reproduction boundary

All arms consume deterministic checksum-addressed streams and use identical
optimizer settings. Each run checks exact prefix causality, native SDM against
its serial recurrence, and packed copy-on-write state against a dense logical
table before training. Results are descriptive single-seed mechanism evidence;
no multi-seed statistical inference is claimed.
