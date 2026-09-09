import json

from suite.preflight import check_text_view
from suite.result_selection import ineligible_reason, protocol_mismatch


def test_text_result_must_match_declared_prompting():
    assert protocol_mismatch({"chat_template": None}, "gsm8k")
    assert not protocol_mismatch({"chat_template": "{{ messages }}"}, "gsm8k")
    assert not protocol_mismatch({"chat_template": None}, "mmvp"), "image tasks declare no text protocol"
    assert ineligible_reason(None, {"chat_template": None, "n-samples": {"gsm8k": {"original": 3, "effective": 3}}}, "gsm8k") == "protocol-mismatch"
    assert ineligible_reason(None, {"chat_template": "x", "n-samples": {"gsm8k": {"original": 3, "effective": 3}}}, "gsm8k") is None


def _view(tmp_path, cfg):
    d = tmp_path / "view"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(cfg))
    return d


def test_text_lane_refuses_the_unpruned_multimodal_head(tmp_path):
    (bad,) = check_text_view(_view(tmp_path, {"model_type": "apertus", "vocab_size": 266752}))
    assert not bad.ok and "extract_text_backbone" in bad.detail
    (good,) = check_text_view(_view(tmp_path / "g", {"model_type": "apertus", "vocab_size": 131072, "output_vocab_size": 131072}))
    assert good.ok
    assert check_text_view(_view(tmp_path / "o", {"model_type": "llama", "vocab_size": 128256})) == []
