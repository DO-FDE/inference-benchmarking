#!/usr/bin/env bash
#
# prep_aiperf_corpus.sh
#
# Prerequisite for running aiperf against the GSM8K arm.
# Swaps aiperf's built-in prompt corpus (Shakespeare) for a GSM8K corpus,
# backing up the original exactly once so the swap is reversible and re-runnable.
#
# Usage:
#   ./prep_aiperf_corpus.sh [GSM8K_CORPUS]   # swap in GSM8K (default: ./gsm8k_corpus.txt)
#   ./prep_aiperf_corpus.sh --restore        # put the original Shakespeare corpus back
#
set -euo pipefail

# Robust pip: works with system python, venvs, PEP-668 externally-managed envs
# (Homebrew), and rootless containers. Uses python3 -m pip (bare `pip` is often
# absent on macOS/Homebrew).
pip_install() {
  local pkg="$1"
  python3 -m pip install -q $pkg 2>/dev/null && return 0
  python3 -m pip install -q --user $pkg 2>/dev/null && return 0
  python3 -m pip install -q --break-system-packages $pkg 2>/dev/null && return 0
  python3 -m pip install -q --break-system-packages --user $pkg 2>/dev/null && return 0
  pip3 install -q $pkg 2>/dev/null && return 0
  pip install -q $pkg 2>/dev/null && return 0
  return 1
}

GSM8K_CORPUS="${1:-gsm8k_corpus.txt}"
BACKUP="./shakespeare.orig.txt"

# --- Locate aiperf's corpus asset -------------------------------------------
# Resolved from the installed package so it tracks whatever venv/image is active.
locate_asset() {
  python3 - <<'PY'
from pathlib import Path
import aiperf.dataset.generator.prompt as p
print(Path(p.__file__).parent / p.DEFAULT_CORPUS_FILE)
PY
}

# --- Build the GSM8K corpus from the dataset (only if missing) --------------
build_gsm8k_corpus() {
  local out="$1"
  echo "Building GSM8K corpus -> ${out}"
  pip_install datasets || { echo "ERROR: could not install datasets" >&2; exit 1; }
  OUT="${out}" python3 - <<'PY'
import os
from datasets import load_dataset
out = os.environ["OUT"]
ds = load_dataset("openai/gsm8k", "main", split="test")
with open(out, "w") as f:
    for row in ds:
        f.write(row["question"].strip().replace("\n", " ") + "\n")
print(f"wrote {out}")
PY
}

if ! python3 -c "import aiperf" 2>/dev/null; then
  echo "aiperf not importable; installing it."
  pip_install aiperf || pip_install "aiperf --ignore-installed typing_extensions" \
    || { echo "ERROR: could not install aiperf" >&2; exit 1; }
fi
if ! python3 -c "import aiperf" 2>/dev/null; then
  echo "ERROR: aiperf still not importable after install attempt." >&2
  echo "       Check pip output above (network access, wheel availability)." >&2
  exit 1
fi

ASSET="$(locate_asset)"
if [[ -z "${ASSET}" || ! -f "${ASSET}" ]]; then
  echo "ERROR: could not locate aiperf corpus asset (got: '${ASSET}')." >&2
  echo "       The package layout may have changed in this aiperf version." >&2
  exit 1
fi
echo "aiperf corpus asset: ${ASSET}"

# --- Restore mode -----------------------------------------------------------
if [[ "${GSM8K_CORPUS}" == "--restore" ]]; then
  if [[ ! -f "${BACKUP}" ]]; then
    echo "ERROR: no backup at ${BACKUP}; nothing to restore." >&2
    exit 1
  fi
  cp "${BACKUP}" "${ASSET}"
  rm -rf ~/.cache/aiperf
  echo "Restored original Shakespeare corpus and cleared aiperf token cache."
  exit 0
fi

# --- Ensure the GSM8K corpus exists (build it if not) -----------------------
if [[ ! -f "${GSM8K_CORPUS}" ]]; then
  echo "GSM8K corpus '${GSM8K_CORPUS}' not found; building it from the dataset."
  build_gsm8k_corpus "${GSM8K_CORPUS}"
fi
if [[ ! -s "${GSM8K_CORPUS}" ]]; then
  echo "ERROR: GSM8K corpus '${GSM8K_CORPUS}' is missing or empty after build." >&2
  echo "       Check network access and that 'datasets' installed correctly." >&2
  exit 1
fi

# --- Back up the original ONCE ----------------------------------------------
# Guard on backup existence: on a re-run the asset already holds GSM8K content,
# so copying it again would overwrite the real original. Only the first run backs up.
if [[ ! -f "${BACKUP}" ]]; then
  cp "${ASSET}" "${BACKUP}"
  echo "Backed up original corpus -> ${BACKUP}"
else
  echo "Backup already exists at ${BACKUP}; leaving it untouched."
fi

# --- Swap the corpus in -----------------------------------------------------
cp "${GSM8K_CORPUS}" "${ASSET}"
echo "Swapped in GSM8K corpus from ${GSM8K_CORPUS}"

# aiperf keys its token cache on the tokenizer, not corpus content, so a stale
# cache would serve Shakespeare-derived tokens against the new corpus. Clear it.
rm -rf ~/.cache/aiperf
echo "Cleared aiperf token cache (~/.cache/aiperf)"

echo "Done. GSM8K arm is ready — you can now run aiperf."