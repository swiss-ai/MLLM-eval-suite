"""Suite-side lm-eval extension: Apertus-aware vLLM model, harness untouched.

Registers ``vllm_apertus`` — upstream's VLLM backend plus the text-only output
mask (shared/apertus_text_only_logits.py) when APERTUS_TEXT_ONLY_OUTPUT_VOCAB
is set. lm-eval's model_args string cannot carry a list, so the processor is
injected here instead. Run as ``python -m apertus_lm_eval_ext`` — it delegates
straight to lm-eval's CLI after registration.
"""

import os

from lm_eval.api.registry import register_model
from lm_eval.models.vllm_causallms import VLLM


@register_model("vllm_apertus")
class VLLMApertus(VLLM):
    def __init__(self, *args, **kwargs):
        if os.environ.get("APERTUS_TEXT_ONLY_OUTPUT_VOCAB"):
            kwargs.setdefault(
                "logits_processors",
                ["apertus_text_only_logits:ApertusTextOnlyLogits"],
            )
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    from lm_eval.__main__ import cli_evaluate

    cli_evaluate()
