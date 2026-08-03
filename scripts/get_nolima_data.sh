#!/usr/bin/env bash
# Download official NoLiMa needlesets + shuffled haystacks (Adobe Research / HF).
#
# Source:
#   https://github.com/adobe-research/NoLiMa
#   https://huggingface.co/datasets/amodaresi/NoLiMa
#
# Usage:
#   bash scripts/get_nolima_data.sh [DEST_DIR]
# Default DEST_DIR: ./nolima_data
set -euo pipefail

DEST="${1:-./nolima_data}"
HF="https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main"
mkdir -p "$DEST/needlesets" "$DEST/haystack/rand_shuffle"

echo "==> NoLiMa data -> $DEST"

cd "$DEST/needlesets"
for f in \
  needle_set.json \
  needle_set_hard.json \
  needle_set_MC.json \
  needle_set_ONLYDirect.json \
  needle_set_w_CoT.json \
  needle_set_w_Distractor.json
do
  echo "  needlesets/$f"
  wget -q -c "$HF/needlesets/$f" -O "$f"
done

cd "$DEST/haystack/rand_shuffle"
for i in 1 2 3 4 5; do
  echo "  haystack/rand_shuffle/rand_book_$i.txt"
  wget -q -c "$HF/haystack/rand_shuffle/rand_book_$i.txt" -O "rand_book_$i.txt"
done

echo
echo "Done. Example runs:"
echo "  python -m eval.nolima.run --task onehop --exp 0 --seq 2048 --size nolima-smoke"
echo "  python -m eval.nolima.run --task twohop --exp 1 --seq 4096 --depth 0.5"
echo "  python -m eval.nolima.sweep --tasks onehop,twohop,hard --exps 0,1 --size nolima-smoke"
