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
