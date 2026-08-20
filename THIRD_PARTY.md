# Third-party sources

## Sparse Delta Memory

The experiments use the Sparse Delta Memory semantics described by Cabannes
et al. The CUDA training path vendors the official released sparse WY
recurrence used by the reproduction.

- Paper: <https://arxiv.org/abs/2607.07386>
- Official repository: <https://github.com/facebookresearch/sparse-delta-memory>
- Revision: `183e7df809131b80ad4393741029d0f20fc3640b`
- Vendored tree: `third_party/released_sdm/lingua/sparse_delta_memory/`
- Vendored code license: [CC BY-NC 4.0](third_party/released_sdm/LICENSE)
  ([upstream copy](https://github.com/facebookresearch/sparse-delta-memory/blob/183e7df809131b80ad4393741029d0f20fc3640b/LICENSE))
- Paper license: CC BY 4.0

The vendored tree and `third_party/released_sdm/lingua/__init__.py` are
byte-for-byte copies of that revision. The checkout contains source code; model
weights and generated artifacts are excluded. The root MIT license applies to
the original code and documentation in this repository; it does not relicense
the vendored subtree. The vendored files remain under CC BY-NC 4.0.

## WikiText-103

The reproduction script downloads the public WikiText-103 raw-v1 Parquet
conversion from `Salesforce/wikitext` at revision
`5fddba447aa4e75996922ea0d6b18b42f0a81cc4`, then verifies every shard before
local preprocessing. The dataset card lists CC BY-SA and GFDL licensing for
the retrieved source corpus.
