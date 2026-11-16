#!/usr/bin/env bash
# Download only the files listed in files.txt from the OpenNeuro datasets.
# Each line of files.txt is <dataset>/<path inside the dataset>.
# Usage: ./download_data.sh OUTPUT_DIR [files.txt]
# Needs: git, git-annex
set -euo pipefail

declare -A VERSION=(
  [ds004386]=1.1.2
  [ds004616]=1.1.1
  [ds005075]=1.0.1
  [ds006729]=1.0.0
  [ds007932]=1.1.0
)

OUT="${1:?usage: $0 OUTPUT_DIR [files.txt]}"
FILES="$(realpath "${2:-$(dirname "$0")/files.txt}")"
mkdir -p "$OUT"
cd "$OUT"

# 1) clone each dataset and check out its version
for ds in "${!VERSION[@]}"; do
  [ -d "$ds" ] || git clone "https://github.com/OpenNeuroDatasets/$ds.git"
  git -C "$ds" checkout "${VERSION[$ds]}"
done

# 2) download every file listed in the text file
while IFS= read -r line || [ -n "$line" ]; do
  line="${line%$'\r'}"
  [ -z "$line" ] && continue
  ds="${line%%/*}"
  path="${line#*/}"
  git -C "$ds" annex get "$path" || echo "FAILED: $line" >&2
done < "$FILES"