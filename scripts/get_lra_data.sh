#!/usr/bin/env bash
# Fetch Long Range Arena (LRA) data for the long-context evaluation track.
#
# Tasks covered by this repo's LRA track:
#   - listops      : generated on the fly (NO download).
#   - text         : byte-level IMDb via HF `stanfordnlp/imdb` (NO download).
#   - image        : CIFAR-10 via HF `cifar10` (auto-download on first run).
#   - pathfinder   : synthetic 32x32 generator (NO download).
#   - pathfinder_x : synthetic 128x128 generator (NO download).
#   - retrieval    : ACL Anthology (AAN) document matching. Requires the official
#                    LRA id pairs PLUS the original AAN texts.
#
# This script only needs to be run for the RETRIEVAL task.
#
# Usage:
#   bash scripts/get_lra_data.sh [DEST_DIR]
# Default DEST_DIR: ./lra_data
set -euo pipefail

DEST="${1:-./lra_data}"
mkdir -p "$DEST"

echo "==> LRA data setup"
echo "    listops / pathfinder / pathfinder_x : generated in-process."
echo "    text    : stanfordnlp/imdb via Hugging Face."
echo "    image   : cifar10 via Hugging Face (downloaded on first run)."
echo "    retrieval (AAN): see steps below."
echo

RETR_DIR="$DEST/retrieval"
mkdir -p "$RETR_DIR"

# 1) Official LRA release (contains the retrieval id-pair tsv files under matching/).
#    Large (~7GB). Only the retrieval id files are needed here.
LRA_GZ_URL="https://storage.googleapis.com/long-range-arena/lra_release.gz"
echo "==> Step 1: download the LRA release containing AAN id pairs:"
echo "      curl -L -o '$DEST/lra_release.gz' '$LRA_GZ_URL'"
echo "      tar -xzf '$DEST/lra_release.gz' -C '$DEST'"
echo "    Then copy the retrieval id-pair tsv files (new_aan_pairs.*.tsv) into:"
echo "      $RETR_DIR/"
echo

# 2) Original AAN corpus (the actual paper texts), from the AAN project.
echo "==> Step 2: download the AAN corpus texts from http://aan.how/download/"
echo "    Extract per-paper texts and place them under one of:"
echo "      $RETR_DIR/papers/<paper_id>.txt        (one file per paper), OR"
echo "      $RETR_DIR/aan_texts.tsv                (rows of '<paper_id>\\t<text>')"
echo

echo "Once both pieces are in place, run e.g.:"
echo "  python -m eval.lra.run --task retrieval --exp 0 --seq 4096 --data-dir '$RETR_DIR'"
echo
echo "Other full-suite examples:"
echo "  python -m eval.lra.run --task image --exp 0 --seq 1025 --size lra-smoke"
echo "  python -m eval.lra.run --task pathfinder --exp 0 --seq 1025 --size lra-smoke"
echo "  python -m eval.lra.run --task pathfinder_x --exp 0 --seq 4097 --size lra-oom"
echo "  python -m eval.ruler.run --task niah_single_1 --exp 0 --seq 2048 --size ruler-smoke"
echo "  python -m eval.ruler.sweep --tasks vt,cwe,fwe,qa_1 --exps 0 --size ruler-smoke"
echo "  bash scripts/get_nolima_data.sh && python -m eval.nolima.run --task onehop --exp 0 --seq 2048 --size nolima-smoke"
