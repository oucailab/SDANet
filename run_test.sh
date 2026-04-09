#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OPT="options/test/HSI/test_SDAnetChux4.yml"
GPU_ID="0"

if [[ $# -ge 1 ]]; then
	if [[ "$1" =~ ^[0-9]+$ ]]; then
		GPU_ID="$1"
		[[ $# -ge 2 ]] && OPT="$2"
	else
		OPT="$1"
		[[ $# -ge 2 ]] && GPU_ID="$2"
	fi
fi

echo "Running: CUDA_VISIBLE_DEVICES=$GPU_ID python ./basicsr/test.py -opt ./$OPT"
echo

CUDA_VISIBLE_DEVICES="$GPU_ID" python ./basicsr/test.py -opt "./$OPT"
