# Shared Apertus runtime resolution, sourced by both harness job scripts.
# The Emu3.5 vision-tokenizer codebase: baked at /workspace/Emu3.5 in pre-Aug-11
# images, dropped from newer ones; fall back to the shared capstor copy.
if [[ -z "${VLLM_APERTUS_EMU35_CODEBASE:-}" ]]; then
  if [[ -d /workspace/Emu3.5 ]]; then
    VLLM_APERTUS_EMU35_CODEBASE=/workspace/Emu3.5
  else
    VLLM_APERTUS_EMU35_CODEBASE=/capstor/store/cscs/swissai/infra01/multimodal-eval/Emu3.5
  fi
fi
export VLLM_APERTUS_EMU35_CODEBASE

# Called only after the harness has selected an Apertus model. Launchers stage
# before submission; jobs repeat the file check for direct sbatch entrypoints.
prefetch_emu35_vision_tokenizer() {
  case "${PREFETCH_EMU35_VISION_TOKENIZER:-true}" in
    1|[Tt][Rr][Uu][Ee]|[Tt]|[Yy][Ee][Ss]|[Yy]|[Oo][Nn]) ;;
    *) return 0 ;;
  esac
  local directory="${1:?Expected models cache root}/BAAI/Emu3.5-VisionTokenizer"
  [[ -f "$directory/config.yaml" && -f "$directory/model.ckpt" ]] && return 0
  "${2:-python3}" - "$directory" <<'PY'
from pathlib import Path
import sys
from huggingface_hub import snapshot_download

directory = Path(sys.argv[1]).expanduser()
required = ["config.yaml", "model.ckpt"]
if not all((directory / name).is_file() for name in required):
    snapshot_download(repo_id="BAAI/Emu3.5-VisionTokenizer", local_dir=str(directory),
                      allow_patterns=required)
missing = [name for name in required if not (directory / name).is_file()]
if missing:
    raise SystemExit(f"Missing Apertus vision tokenizer files in {directory}: {', '.join(missing)}")
print(f"Staged Apertus vision tokenizer: {directory}")
PY
}
