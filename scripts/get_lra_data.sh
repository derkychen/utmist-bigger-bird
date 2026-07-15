#!/usr/bin/env bash
# Fetch Long Range Arena (LRA) data for the long-context evaluation track.
#
# Tasks covered by this repo's LRA track:
#   - listops   : generated on the fly by shared/lra_dataset.py (NO download needed).
#   - text      : byte-level IMDb via the HF `stanfordnlp/imdb` dataset (NO download needed).
#   - retrieval : ACL Anthology (AAN) document matching. Requires the official LRA id
#                 pairs PLUS the original AAN texts (not redistributable by Google).
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
echo "    listops : generated in-process (nothing to download)."
echo "    text    : uses stanfordnlp/imdb via Hugging Face (nothing to download)."
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
