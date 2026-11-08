#!/usr/bin/env bash
set -euo pipefail

NGRAM_ORDER=3

# Pruning:
# 1-gram: none
# 2-gram: >=2
# 3-gram: >=2
PRUNE="0 2 2"

DISCOUNT="--discount_fallback"

# memory limit for lmplz
MEMORY="50%"

# model names
ARPA_OUT="model_3gram.arpa"
BINARY_OUT="model_3gram.binary"


if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <directory1> [directory2 ... directoryN]"
  exit 1
fi


echo "Reading files .txt from directories: $*"
echo "Creating ${NGRAM_ORDER}-gram model with KenLM"

find "$@" \
  -type f \
  -name "*.txt" \
  -print0 \
| xargs -0 cat \
| lmplz \
    -o "${NGRAM_ORDER}" \
    --memory "${MEMORY}" \
    --prune ${PRUNE} \
    ${DISCOUNT} \
> "${ARPA_OUT}"

echo "✔ Model ARPA saved as ${ARPA_OUT}"

echo "Binary conversion"
build_binary "${ARPA_OUT}" "${BINARY_OUT}"

echo "✔ Binary model asved as ${BINARY_OUT}"
echo "🎉 All done!"