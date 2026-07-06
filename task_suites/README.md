# Suites

Suite files define task groups; each named `--suite` maps to one file here. These files are the
source of truth (stale copies inside the submodules are not used by the launchers).

- `lmms-eval/`: visual, geospatial (needs staged RS imagery via `RS_DATASETS_ROOT`), and audio suites.
- `VLMEvalKit/`: smoke, full, spatial (EASI set), and llm_judge (OpenAI-judge-scored; key required).
