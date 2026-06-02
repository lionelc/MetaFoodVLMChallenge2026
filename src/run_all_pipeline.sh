#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-optimization/experiment_results/final_reproduction_run_${RUN_STAMP}}"
mkdir -p "$RUN_DIR"
RUN_DIR_ABS="$(cd "$RUN_DIR" && pwd)"

CACHE_MANIFEST="${CACHE_MANIFEST:-${RUN_DIR_ABS}/cache_manifest.json}"
CACHE_LOG_DIR="${CACHE_LOG_DIR:-${RUN_DIR_ABS}/cache_worker_logs}"
CACHE_CONSOLE_LOG="${CACHE_CONSOLE_LOG:-${RUN_DIR_ABS}/01_cache_final_4models_auto.log}"
SETUP_LOG="${SETUP_LOG:-${RUN_DIR_ABS}/00_environment_setup.log}"
REPRO_LOG="${REPRO_LOG:-${RUN_DIR_ABS}/02_reproduce_final_submission.log}"
SUBMISSION_OUT="${SUBMISSION_OUT:-${RUN_DIR_ABS}/submission.csv}"
REPRO_MANIFEST="${REPRO_MANIFEST:-${RUN_DIR_ABS}/reproduction_manifest.json}"
PATHS_FILE="${RUN_DIR_ABS}/pipeline_paths.txt"

CONCURRENCY="${CONCURRENCY:-2}"
QWEN_DTYPE="${QWEN_DTYPE:-bfloat16}"
DESCRIPTION_SAVE_EVERY="${DESCRIPTION_SAVE_EVERY:-50}"
BATCH_OVERRIDE="${BATCH_OVERRIDE:-qwen3_8b=192 qwen3_2b=192 siglip2=1024 clip_h=512}"
FORCE_CACHE="${FORCE_CACHE:-0}"
SKIP_DESCRIPTION_CACHE_BUILD="${SKIP_DESCRIPTION_CACHE_BUILD:-0}"
PIPELINE_DRY_RUN="${PIPELINE_DRY_RUN:-0}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
PIP_CHECK="${PIP_CHECK:-1}"
VERIFY_IMPORTS="${VERIFY_IMPORTS:-1}"
UNINSTALL_UNUSED_TORCHAUDIO="${UNINSTALL_UNUSED_TORCHAUDIO:-1}"
HF_HOME="${HF_HOME:-${SCRIPT_DIR}/.hf_home}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-${SCRIPT_DIR}/secrets/huggingface_token}"
REQUIRE_HF_TOKEN="${REQUIRE_HF_TOKEN:-0}"
CHECK_HF_ACCESS="${CHECK_HF_ACCESS:-1}"

export HF_HOME
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

on_error() {
  local exit_code=$?
  echo
  echo "Pipeline failed with exit code ${exit_code}."
  echo "Run folder: ${RUN_DIR_ABS}"
  echo "Setup log: ${SETUP_LOG}"
  echo "Cache log: ${CACHE_CONSOLE_LOG}"
  echo "Reproduction log: ${REPRO_LOG}"
}
trap on_error ERR

print_command() {
  printf 'Command:'
  printf ' %q' "$@"
  printf '\n'
}

log_setup() {
  printf '%s\n' "$*" | tee -a "$SETUP_LOG"
}

run_setup_command() {
  print_command "$@" | tee -a "$SETUP_LOG"
  if [[ "$PIPELINE_DRY_RUN" == "1" ]]; then
    printf 'PIPELINE_DRY_RUN=1; command not executed.\n' | tee -a "$SETUP_LOG"
    return 0
  fi
  "$@" 2>&1 | tee -a "$SETUP_LOG"
}

run_and_log() {
  local log_file="$1"
  shift

  {
    printf 'Started UTC: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'Working directory: %s\n' "$SCRIPT_DIR"
    print_command "$@"
    printf '\n'
  } | tee "$log_file"

  if [[ "$PIPELINE_DRY_RUN" == "1" ]]; then
    printf 'PIPELINE_DRY_RUN=1; command not executed.\n' | tee -a "$log_file"
    return 0
  fi

  "$@" 2>&1 | tee -a "$log_file"
}

install_dependencies() {
  log_setup "Dependency setup:"
  if [[ "$INSTALL_DEPS" != "1" ]]; then
    log_setup "  INSTALL_DEPS=${INSTALL_DEPS}; skipping pip install."
    return 0
  fi
  if [[ ! -f requirements.txt ]]; then
    log_setup "  requirements.txt is missing."
    return 1
  fi

  if [[ "$UNINSTALL_UNUSED_TORCHAUDIO" == "1" ]]; then
    log_setup "  Removing unused torchaudio if present to avoid preinstalled torch/torchaudio version conflicts."
    run_setup_command "$PYTHON_BIN" -m pip uninstall -y torchaudio
  fi

  local pip_install_args=()
  if [[ -n "${PIP_INSTALL_ARGS:-}" ]]; then
    read -r -a pip_install_args <<< "$PIP_INSTALL_ARGS"
  fi
  run_setup_command "$PYTHON_BIN" -m pip install "${pip_install_args[@]}" -r requirements.txt

  if [[ "$PIP_CHECK" == "1" ]]; then
    run_setup_command "$PYTHON_BIN" -m pip check
  fi
}

configure_huggingface_auth() {
  log_setup "Hugging Face setup:"
  mkdir -p "$HF_HOME"
  chmod 700 "$HF_HOME" 2>/dev/null || true

  local token_source=""
  if [[ -n "${HF_TOKEN:-}" ]]; then
    token_source="HF_TOKEN environment variable"
  elif [[ -f "$HF_TOKEN_FILE" ]]; then
    HF_TOKEN="$(tr -d '\r\n' < "$HF_TOKEN_FILE")"
    token_source="$HF_TOKEN_FILE"
  elif [[ -f "${HF_HOME}/token" ]]; then
    HF_TOKEN="$(tr -d '\r\n' < "${HF_HOME}/token")"
    token_source="${HF_HOME}/token"
  fi

  if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
    printf '%s' "$HF_TOKEN" > "${HF_HOME}/token"
    chmod 600 "${HF_HOME}/token" 2>/dev/null || true
    log_setup "  Token configured from ${token_source}; stored for Hugging Face Hub at ${HF_HOME}/token."
  else
    log_setup "  No token found. Set HF_TOKEN, create ${HF_TOKEN_FILE}, or pre-create ${HF_HOME}/token."
    if [[ "$REQUIRE_HF_TOKEN" == "1" ]]; then
      log_setup "  REQUIRE_HF_TOKEN=1, so stopping before model download."
      return 1
    fi
  fi

  log_setup "  HF_HOME=${HF_HOME}"
}

check_huggingface_model_access() {
  if [[ "$CHECK_HF_ACCESS" != "1" ]]; then
    log_setup "Hugging Face model access check skipped: CHECK_HF_ACCESS=${CHECK_HF_ACCESS}."
    return 0
  fi
  if [[ -z "${HF_TOKEN:-}" ]]; then
    log_setup "Hugging Face model access check skipped because no token is configured."
    return 0
  fi

  log_setup "Checking Hugging Face access for required models:"
  if [[ "$PIPELINE_DRY_RUN" == "1" ]]; then
    log_setup "  PIPELINE_DRY_RUN=1; model access check not executed."
    return 0
  fi

  "$PYTHON_BIN" - <<'PY' 2>&1 | tee -a "$SETUP_LOG"
import os
from huggingface_hub import HfApi

models = [
    "Qwen/Qwen3-VL-Embedding-8B",
    "Qwen/Qwen3-VL-Embedding-2B",
    "Qwen/Qwen3-VL-8B-Instruct",
    "google/gemma-4-E4B-it",
]

api = HfApi(token=os.environ.get("HF_TOKEN"))
for model_id in models:
    api.model_info(model_id)
    print(f"  OK: {model_id}")
PY
}

verify_python_imports() {
  if [[ "$VERIFY_IMPORTS" != "1" ]]; then
    log_setup "Import/version check skipped: VERIFY_IMPORTS=${VERIFY_IMPORTS}."
    return 0
  fi
  log_setup "Checking Python imports and package versions:"
  if [[ "$PIPELINE_DRY_RUN" == "1" ]]; then
    log_setup "  PIPELINE_DRY_RUN=1; import/version check not executed."
    return 0
  fi

  "$PYTHON_BIN" - <<'PY' 2>&1 | tee -a "$SETUP_LOG"
import importlib
import importlib.metadata as md

checks = [
    ("numpy", "numpy"),
    ("PIL", "Pillow"),
    ("tqdm", "tqdm"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("pandas", "pandas"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("transformers", "transformers"),
    ("sentence_transformers", "sentence-transformers"),
    ("open_clip", "open_clip_torch"),
    ("accelerate", "accelerate"),
    ("huggingface_hub", "huggingface-hub"),
    ("safetensors", "safetensors"),
    ("tokenizers", "tokenizers"),
    ("timm", "timm"),
    ("ftfy", "ftfy"),
    ("regex", "regex"),
    ("qwen_vl_utils", "qwen-vl-utils"),
    ("av", "av"),
    ("peft", "peft"),
    ("kagglehub", "kagglehub"),
]

for import_name, package_name in checks:
    importlib.import_module(import_name)
    print(f"  OK: {package_name}=={md.version(package_name)}")
PY
}

cache_args=(
  cache_final_4models_auto.py
  --qwen-dtype "$QWEN_DTYPE"
  --manifest-out "$CACHE_MANIFEST"
  --log-dir "$CACHE_LOG_DIR"
)

if [[ "$FORCE_CACHE" == "1" ]]; then
  cache_args+=(--force)
fi

if [[ -n "${BATCH_OVERRIDE:-}" ]]; then
  read -r -a batch_override_args <<< "$BATCH_OVERRIDE"
  cache_args+=(--batch-override "${batch_override_args[@]}")
fi

repro_args=(
  reproduce_final_submission.py
  --output "$SUBMISSION_OUT"
  --manifest-out "$REPRO_MANIFEST"
  --description-save-every "$DESCRIPTION_SAVE_EVERY"
)

if [[ "$SKIP_DESCRIPTION_CACHE_BUILD" == "1" ]]; then
  repro_args+=(--skip-description-cache-build)
fi

{
  printf 'run_dir=%s\n' "$RUN_DIR_ABS"
  printf 'setup_log=%s\n' "$SETUP_LOG"
  printf 'cache_manifest=%s\n' "$CACHE_MANIFEST"
  printf 'cache_worker_logs=%s\n' "$CACHE_LOG_DIR"
  printf 'cache_console_log=%s\n' "$CACHE_CONSOLE_LOG"
  printf 'submission=%s\n' "$SUBMISSION_OUT"
  printf 'reproduction_manifest=%s\n' "$REPRO_MANIFEST"
  printf 'reproduction_log=%s\n' "$REPRO_LOG"
  printf 'concurrency=%s\n' "$CONCURRENCY"
  printf 'qwen_dtype=%s\n' "$QWEN_DTYPE"
  printf 'install_deps=%s\n' "$INSTALL_DEPS"
  printf 'hf_home=%s\n' "$HF_HOME"
  printf 'hf_token_file=%s\n' "$HF_TOKEN_FILE"
  printf 'hf_token_configured=%s\n' "$([[ -n "${HF_TOKEN:-}" || -f "$HF_TOKEN_FILE" || -f "${HF_HOME}/token" ]] && echo yes || echo no)"
  if [[ -n "${BATCH_OVERRIDE:-}" ]]; then
    printf 'batch_override=%s\n' "$BATCH_OVERRIDE"
  fi
} | tee "$PATHS_FILE"

echo
echo "Step 1/3: install dependencies and configure Hugging Face access."
: > "$SETUP_LOG"
log_setup "Started UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
log_setup "Working directory: ${SCRIPT_DIR}"
install_dependencies
configure_huggingface_auth
check_huggingface_model_access
verify_python_imports

echo
echo "Step 2/3: build or validate the four embedding caches."

embedding_cache_complete=true
for cache_file in \
  siglip2_cache/test1_text_features.npy siglip2_cache/test1_image_features.npy \
  siglip2_cache/test2_text_features.npy siglip2_cache/test2_image_features.npy \
  qwen3_cache/qwen3vl8b_test1_text.npy qwen3_cache/qwen3vl8b_test1_img.npy \
  qwen3_cache/qwen3vl8b_test2_text.npy qwen3_cache/qwen3vl8b_test2_img.npy \
  clip_cache/vit_h_14_dfn5b/test1_text_features.npy clip_cache/vit_h_14_dfn5b/test1_image_features.npy \
  clip_cache/vit_h_14_dfn5b/test2_text_features.npy clip_cache/vit_h_14_dfn5b/test2_image_features.npy \
  qwen3vl2b_cache/test1_text_features.npy qwen3vl2b_cache/test1_image_features.npy \
  qwen3vl2b_cache/test2_text_features.npy qwen3vl2b_cache/test2_image_features.npy; do
  if [[ ! -f "$SCRIPT_DIR/$cache_file" ]]; then
    embedding_cache_complete=false
    break
  fi
done

if [[ "$embedding_cache_complete" == "true" && "$FORCE_CACHE" != "1" ]]; then
  echo "All 16 embedding cache files found — skipping cache build."
else
  echo "Wave 1/3: qwen3_8b + qwen3_2b (parallel)"
  run_and_log "$CACHE_CONSOLE_LOG" "$PYTHON_BIN" "${cache_args[@]}" --models qwen3_8b qwen3_2b --concurrency 2
  echo "Wave 2/3: siglip2"
  run_and_log "$CACHE_CONSOLE_LOG" "$PYTHON_BIN" "${cache_args[@]}" --models siglip2 --concurrency 1
  echo "Wave 3/3: clip_h"
  run_and_log "$CACHE_CONSOLE_LOG" "$PYTHON_BIN" "${cache_args[@]}" --models clip_h --concurrency 1
fi

echo
echo "Step 3/3: reproduce the final submission from the completed caches."
run_and_log "$REPRO_LOG" "$PYTHON_BIN" "${repro_args[@]}"

echo
echo "Pipeline complete."
echo "Run folder: ${RUN_DIR_ABS}"
echo "Submission: ${SUBMISSION_OUT}"
echo "Reproduction manifest: ${REPRO_MANIFEST}"
