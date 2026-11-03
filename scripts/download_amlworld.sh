#!/usr/bin/env bash
# Download IBM AMLworld (HI-Small and LI-Small by default) from Kaggle.
# Needs the Kaggle CLI and an API token in ~/.kaggle/kaggle.json
#   pip install kaggle
# Usage: scripts/download_amlworld.sh [data_dir] [variants...]
set -euo pipefail
DIR="${1:-data/amlworld}"
shift || true
VARIANTS=("${@:-HI-Small LI-Small}")
mkdir -p "$DIR"
for v in ${VARIANTS[@]}; do
  for f in "${v}_Trans.csv" "${v}_Patterns.txt"; do
    if [ ! -f "$DIR/$f" ]; then
      kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml -f "$f" -p "$DIR"
      [ -f "$DIR/$f.zip" ] && unzip -o "$DIR/$f.zip" -d "$DIR" && rm "$DIR/$f.zip"
    fi
  done
done
ls -lh "$DIR"
