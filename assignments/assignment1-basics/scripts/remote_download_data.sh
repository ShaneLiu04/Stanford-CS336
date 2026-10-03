#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-data/raw}"
HF_BASE="${HF_ENDPOINT:-https://huggingface.co}"
mkdir -p "$ROOT"
cd "$ROOT"

download() {
  local output="$1"
  local url="$2"
  if [[ ! -s "$output" ]]; then
    wget --continue --tries=8 --timeout=60 --output-document="$output" "$url"
  fi
}

download TinyStoriesV2-GPT4-train.txt \
  "$HF_BASE/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
download TinyStoriesV2-GPT4-valid.txt \
  "$HF_BASE/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
if [[ ! -s owt_train.txt ]]; then
  download owt_train.txt.gz \
    "$HF_BASE/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz"
  gzip --decompress --force owt_train.txt.gz
fi
if [[ ! -s owt_valid.txt ]]; then
  download owt_valid.txt.gz \
    "$HF_BASE/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz"
  gzip --decompress --force owt_valid.txt.gz
fi
sha256sum TinyStoriesV2-GPT4-*.txt owt_*.txt > SHA256SUMS
wc -c -l TinyStoriesV2-GPT4-*.txt owt_*.txt > DATASET_SIZES.txt
