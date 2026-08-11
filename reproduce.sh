#!/usr/bin/env bash
set -euo pipefail

stage="${1:-all}"
gpus="${GPUS:-}"
gpu_args=()
if [[ -n "$gpus" ]]; then
  gpu_args=(--gpus "$gpus")
fi

case "$stage" in
  prepare)
    python3 scripts/prepare_wikitext.py --output-root runs/data
    python3 benchmarks/prepare_recall_suite.py \
      --output-dir runs/data/adaptive_recall_seed102337_v1 \
      --steps 30000 --batch-size 32 --eval-examples 2048 \
      --stream-seed 102337 \
      --expected-manifest-sha256 \
      b0587d62c3ab709c94e37742892451a39463a7d577212b9379bd76d966f97800
    ;;
  recall|language)
    python3 scripts/run_experiments.py "$stage" "${gpu_args[@]}"
    ;;
  verify-results)
    python3 scripts/verify_results.py
    ;;
  all)
    "$0" prepare
    python3 scripts/run_experiments.py all "${gpu_args[@]}"
    ;;
  *)
    echo "usage: ./reproduce.sh [prepare|recall|language|verify-results|all]" >&2
    exit 2
    ;;
esac
