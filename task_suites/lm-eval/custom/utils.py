"""math_verify scoring for the suite's custom math tasks.

Upstream's minerva/leaderboard math utils hard-require antlr4 4.11 for a
legacy sympy.parse_latex path; math_verify itself works with the 4.9.3 pin
VLMEvalKit's scorers need, so these tasks import only math_verify.
"""

import logging

from math_verify import parse, verify

eval_logger = logging.getLogger(__name__)


def _last_boxed(text):
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    return None


def process_results(doc, results):
    try:
        gold = parse("$" + str(doc["answer"]) + "$")
        pred = parse(results[0])
        ok = bool(verify(gold, pred))
    except Exception as exc:
        eval_logger.debug("math_verify failed: %s", exc)
        ok = False
    return {"exact_match": 1.0 if ok else 0.0}


def process_docs_math_lvl5(dataset):
    def _extract(doc):
        doc["answer"] = _last_boxed(doc["solution"]) or ""
        return doc

    return dataset.filter(lambda d: d["level"] == "Level 5").map(_extract)
