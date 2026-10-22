#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./models/convert-hf-whisper-model.sh [options] <huggingface-repo-or-local-dir>

Convert a Hugging Face Whisper-family fine-tuned model into whisper.cpp artifacts:
  - ggml-<output-name>.bin
  - ggml-<output-name>-encoder.mlmodelc

Options:
  -o, --output-name NAME         Output stem. Default: derived from repo or directory name.
      --base-model NAME          Override the inferred base Whisper model
                                 (tiny, tiny.en, base, base.en, small, small.en,
                                  medium, medium.en, large-v1, large-v2, large-v3, large-v3-turbo)
      --models-dir DIR           Output directory. Default: <repo>/models
      --hf-cache-dir DIR         Hugging Face cache directory for snapshot downloads
      --python PYTHON            Python executable. Default: python3
      --whisper-assets-root DIR  Directory whose child path whisper/assets contains mel_filters.npz
      --skip-coreml              Only create the ggml .bin file
      --install-deps             Install missing Python dependencies into the selected Python environment
      --keep-temp                Keep temporary .pt and .mlpackage files
  -h, --help                     Show this help text

Examples:
  ./models/convert-hf-whisper-model.sh MediaTek-Research/Breeze-ASR-25
  ./models/convert-hf-whisper-model.sh ./Breeze-ASR-25 --output-name breeze-asr-25
  ./models/convert-hf-whisper-model.sh MediaTek-Research/Breeze-ASR-25 --base-model large-v2
EOF
}

script_dir=$(
    cd -- "$(dirname "$0")" >/dev/null 2>&1 || exit 1
    pwd -P
)
repo_root=$(
    cd -- "$script_dir/.." >/dev/null 2>&1 || exit 1
    pwd -P
)

python_bin="python3"
models_dir="$repo_root/models"
hf_cache_dir=""
output_name=""
base_model=""
source_arg=""
skip_coreml=0
install_deps=0
keep_temp=0
whisper_assets_root=""

while [ $# -gt 0 ]; do
    case "$1" in
        -o|--output-name)
            output_name=${2:-}
            shift 2
            ;;
        --base-model)
            base_model=${2:-}
            shift 2
            ;;
        --models-dir)
            models_dir=${2:-}
            shift 2
            ;;
        --hf-cache-dir)
            hf_cache_dir=${2:-}
            shift 2
            ;;
        --python)
            python_bin=${2:-}
            shift 2
            ;;
        --whisper-assets-root)
            whisper_assets_root=${2:-}
            shift 2
            ;;
        --skip-coreml)
            skip_coreml=1
            shift
            ;;
        --install-deps)
            install_deps=1
            shift
            ;;
        --keep-temp)
            keep_temp=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            if [ -n "$source_arg" ]; then
                echo "Only one model source may be provided" >&2
                usage >&2
                exit 1
            fi
            source_arg=$1
            shift
            ;;
    esac
done

if [ -z "$source_arg" ]; then
    usage >&2
    exit 1
fi

mkdir -p "$models_dir"

temp_dir=$(mktemp -d "${TMPDIR:-/tmp}/whisper-hf-convert.XXXXXX")
cleanup() {
    if [ "$keep_temp" -eq 0 ]; then
        rm -rf "$temp_dir"
    fi
}
trap cleanup EXIT

require_python_module() {
    local module="$1"
    if ! "$python_bin" - <<PY >/dev/null 2>&1
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("$module") else 1)
PY
    then
        return 1
    fi
}

install_required_deps() {
    "$python_bin" -m pip install --upgrade pip
    "$python_bin" -m pip install -r "$repo_root/models/requirements-coreml.txt"
}

missing_deps=0

for module in whisper transformers huggingface_hub torch numpy; do
    if ! require_python_module "$module"; then
        missing_deps=1
    fi
done

if [ "$skip_coreml" -eq 0 ]; then
    for module in coremltools ane_transformers; do
        if ! require_python_module "$module"; then
            missing_deps=1
        fi
    done
fi

if [ "$missing_deps" -eq 1 ]; then
    if [ "$install_deps" -eq 1 ]; then
        install_required_deps
    else
        cat >&2 <<EOF
Missing Python dependencies.

Install them with:
  $python_bin -m pip install -r "$repo_root/models/requirements-coreml.txt"

Or re-run this script with:
  --install-deps
EOF
        exit 1
    fi
fi

if [ -d "$source_arg" ]; then
    model_dir=$(
        cd -- "$source_arg" >/dev/null 2>&1 || exit 1
        pwd -P
    )
else
    model_dir=$("$python_bin" - "$source_arg" "$hf_cache_dir" <<'PY'
import os
import sys
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
cache_dir = sys.argv[2] or None

path = snapshot_download(
    repo_id=repo_id,
    cache_dir=cache_dir,
)
print(os.path.realpath(path))
PY
)
fi

if [ -z "$output_name" ]; then
    output_name=$("$python_bin" - "$source_arg" "$model_dir" <<'PY'
import os
import re
import sys

source_arg = sys.argv[1]
model_dir = sys.argv[2]
name = os.path.basename(source_arg.rstrip("/")) if os.path.isdir(source_arg) else source_arg.split("/")[-1]
slug = re.sub(r"[^0-9A-Za-z._-]+", "-", name).strip("-").lower()
print(slug)
PY
)
fi

if [ -z "$base_model" ]; then
    base_model=$("$python_bin" - "$model_dir/config.json" <<'PY'
import json
import os
import sys

config_path = sys.argv[1]
with open(config_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

name_hint = " ".join(
    str(cfg.get(key, "")) for key in ("_name_or_path", "name_or_path")
).lower()

known_names = [
    "large-v3-turbo",
    "large-v3",
    "large-v2",
    "large-v1",
    "medium.en",
    "medium",
    "small.en",
    "small",
    "base.en",
    "base",
    "tiny.en",
    "tiny",
]

for candidate in known_names:
    if candidate in name_hint:
        print(candidate)
        raise SystemExit(0)

signature = (
    cfg.get("d_model"),
    cfg.get("encoder_layers"),
    cfg.get("decoder_layers"),
    cfg.get("encoder_attention_heads"),
    cfg.get("decoder_attention_heads"),
    cfg.get("vocab_size"),
)

mapping = {
    (384, 4, 4, 6, 6, 51865): "tiny",
    (384, 4, 4, 6, 6, 51864): "tiny.en",
    (512, 6, 6, 8, 8, 51865): "base",
    (512, 6, 6, 8, 8, 51864): "base.en",
    (768, 12, 12, 12, 12, 51865): "small",
    (768, 12, 12, 12, 12, 51864): "small.en",
    (1024, 24, 24, 16, 16, 51865): "medium",
    (1024, 24, 24, 16, 16, 51864): "medium.en",
    (1280, 32, 32, 20, 20, 51865): "large-v2",
}

if signature in mapping:
    print(mapping[signature])
    raise SystemExit(0)

raise SystemExit(
    "Could not infer the base Whisper model from config.json. "
    "Re-run with --base-model <name>."
)
PY
)
fi

case "$base_model" in
    tiny|tiny.en|base|base.en|small|small.en|medium|medium.en|large-v1|large-v2|large-v3|large-v3-turbo)
        ;;
    *)
        echo "Unsupported base model: $base_model" >&2
        exit 1
        ;;
esac

if [ -z "$whisper_assets_root" ]; then
    whisper_assets_root=$("$python_bin" - <<'PY'
import os
import whisper

print(os.path.dirname(os.path.dirname(os.path.realpath(whisper.__file__))))
PY
)
fi

if [ ! -f "$whisper_assets_root/whisper/assets/mel_filters.npz" ]; then
    echo "Could not find whisper assets at: $whisper_assets_root/whisper/assets/mel_filters.npz" >&2
    echo "Pass --whisper-assets-root to point at an OpenAI Whisper install or repository root." >&2
    exit 1
fi

bin_path="$models_dir/ggml-$output_name.bin"
coreml_dir="$models_dir/ggml-$output_name-encoder.mlmodelc"
tmp_bin_dir="$temp_dir/ggml"
tmp_coreml_package="$temp_dir/coreml-encoder-$output_name.mlpackage"
coreml_compile_log="$temp_dir/coreml-compile.log"

mkdir -p "$tmp_bin_dir"

echo "Model source     : $model_dir"
echo "Output name      : $output_name"
echo "Base model       : $base_model"
echo "Whisper assets   : $whisper_assets_root"
echo "Output bin       : $bin_path"
if [ "$skip_coreml" -eq 0 ]; then
    echo "Output Core ML   : $coreml_dir"
fi

(
    cd -- "$repo_root" || exit 1
    "$python_bin" models/convert-h5-to-ggml.py "$model_dir" "$whisper_assets_root" "$tmp_bin_dir"
)

mv -f "$tmp_bin_dir/ggml-model.bin" "$bin_path"

if [ "$skip_coreml" -eq 0 ]; then
    (
        cd -- "$repo_root" || exit 1
        "$python_bin" models/convert-h5-to-coreml.py \
            --model-name "$base_model" \
            --model-path "$model_dir" \
            --encoder-only True
    )

    mv -f "$repo_root/models/coreml-encoder-$base_model.mlpackage" "$tmp_coreml_package"
    rm -rf "$coreml_dir"

    if ! "$python_bin" - "$tmp_coreml_package" "$coreml_dir" >"$coreml_compile_log" 2>&1 <<'PY'
from coremltools.models.utils import compile_model
import sys

compile_model(sys.argv[1], sys.argv[2])
PY
    then
        if command -v xcrun >/dev/null 2>&1 && xcrun --find coremlc >/dev/null 2>&1; then
            xcrun coremlc compile "$tmp_coreml_package" "$models_dir" >/dev/null
            mv -f "$models_dir/$(basename "${tmp_coreml_package%.mlpackage}").mlmodelc" "$coreml_dir"
        else
            cat "$coreml_compile_log" >&2
            echo "Failed to compile the Core ML model with both coremltools and xcrun." >&2
            exit 1
        fi
    fi
fi

if [ "$keep_temp" -eq 0 ]; then
    rm -f "$repo_root/models/hf-$base_model.pt"
    rm -rf "$repo_root/models/coreml-encoder-$base_model.mlpackage"
fi

echo
echo "Done."
echo "  $bin_path"
if [ "$skip_coreml" -eq 0 ]; then
    echo "  $coreml_dir"
fi
