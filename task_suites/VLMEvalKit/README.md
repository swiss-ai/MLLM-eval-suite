# VLMEvalKit Suites

Pass these files to the launchers with `--tasks`, or by suite name with `--suite`. One dataset name
per line. `llm_judge.txt` lists the benchmarks whose eval stage calls an OpenAI judge — the launcher
refuses to submit them without `OPENAI_API_KEY` (env or `third_party/VLMEvalKit/.env`).
