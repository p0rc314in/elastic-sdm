#!/usr/bin/env python3
"""Build the two unmodified released-SDM CUDA extensions once per host."""

from __future__ import annotations

import json

import torch


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("a CUDA development host is required")
    from lingua.sparse_delta_memory.cuda.sparse_ip_cuda import (
        _get_cuda_module as build_sparse_inner_product,
    )
    from lingua.sparse_delta_memory.cuda.warp_cooperative_gather_cuda import (
        _get_cuda_module as build_warp_gather,
    )

    build_sparse_inner_product()
    build_warp_gather()
    print(
        json.dumps(
            {
                "status": "complete",
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "extensions": [
                    "sparse_ip_sorted",
                    "warp_cooperative_gather",
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
