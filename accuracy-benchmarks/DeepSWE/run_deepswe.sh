#!/usr/bin/env bash
# Usage: ./run_deepswe.sh <provider/model> <api_key_env_var> <api_key_value> <endpoint_url> <job_name> [n_concurrent]
set -euo pipefail

MODEL="$1"
KEY_ENV_VAR="$2"
KEY_VALUE="$3"
ENDPOINT="$4"
JOB_NAME="$5"
N_CONCURRENT="${6:-4}"

export "$KEY_ENV_VAR"="$KEY_VALUE"
export OPENAI_BASE_URL="$ENDPOINT"

pier run -p tasks --agent mini-swe-agent \
  --model "$MODEL" \
  --ak model_kwargs="{\"api_base\": \"$ENDPOINT\"}" \
  --n-concurrent "$N_CONCURRENT" \
  --job-name "$JOB_NAME"
