"""vLLM logits processor restricting Apertus generation to the text vocab.

The symmetric multimodal head lets visual/audio tokens win the argmax and
derail generations (measured: gsm8k 74.2 vs 79.8 with a text-only head, same
weights). SamplingParams.allowed_token_ids caps at 1024 ids, so the mask is
applied engine-side instead: ids at and above APERTUS_TEXT_ONLY_OUTPUT_VOCAB
are set to -inf on every step. Enabled only when that env var is set; the
wrapper passes this class to the engine's logits_processors.
"""

import os

import torch
from vllm.v1.sample.logits_processor import LogitsProcessor


class ApertusTextOnlyLogits(LogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        self.text_vocab = int(os.environ.get("APERTUS_TEXT_ONLY_OUTPUT_VOCAB", "0"))

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update) -> None:
        pass

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.text_vocab and logits.shape[-1] > self.text_vocab:
            logits[:, self.text_vocab:] = float("-inf")
        return logits
