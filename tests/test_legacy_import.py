import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_dashboard import import_legacy  # noqa: E402


def _legacy(tmp_path):
    legacy = {"table": [
        {"task": "mmvp", "metric": "mmvp_accuracy", "framework": "lmms-eval", "cells": {"m1": {"v": 69.0, "run": "r0"}}},
        {"task": "gsm8k", "metric": "exact_match", "framework": "lm-eval", "cells": {"m1": {"v": 80.97, "run": "r0"}}},
    ]}
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps(legacy))
    return p


def test_legacy_import_fills_image_rows_but_never_text_rows(tmp_path):
    merged, models = {}, ["m1"]
    n = import_legacy(merged, models, [_legacy(tmp_path)])
    assert n == 1
    assert merged[("mmvp", "mmvp_accuracy")]["cells"]["m1"] == {"v": 69.0, "run": "r0", "legacy": True}
    assert ("gsm8k", "exact_match") not in merged


def test_legacy_import_never_overrides_a_current_cell(tmp_path):
    merged = {("mmvp", "mmvp_accuracy"): {"task": "mmvp", "metric": "mmvp_accuracy", "framework": "lmms-eval", "cells": {"m1": {"v": 62.0, "run": "r1"}}}}
    assert import_legacy(merged, ["m1"], [_legacy(tmp_path)]) == 0
    assert merged[("mmvp", "mmvp_accuracy")]["cells"]["m1"]["run"] == "r1"
