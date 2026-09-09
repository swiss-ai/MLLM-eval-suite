# Requested text benchmark coverage

The `text-requested` suite launches the 31 requested benchmark names using the
pinned Swiss AI `lm-eval-harness` branch based on upstream v0.4.13. Existing
`text-smoke`, `text-full`, and `math` suites retain their task lists. Use the
separate text launcher; the production dispatcher still serves the two
multimodal frameworks.

The harness's personal integration branch is `yxu/main` in
`swiss-ai/lm-evaluation-harness`. Submodule updates track that branch, while
each suite commit retains an exact harness commit for reproducible evaluations.
Creating the baseline branch does not change the evaluated submodule pin.

```bash
export ORCH_REPO_ROOT="$PWD"
bash launchers/lm-eval/eval.sh /path/to/model --tasks mmlu_pro
bash launchers/lm-eval/eval.sh /path/to/model --suite text-smoke
```

The harness checkout and suite-local `task_suites/lm-eval/custom/` definitions are
loaded from disk. Jobs do not install dependencies. IFBench remains the existing
suite-local task, backed by `ifbench_verifiers`; the other requested names come
from the harness. See the harness's `docs/swiss_task_ports.md` for source revision,
dataset definitions, extraction rules, and optional dependency requirements.

## Launch controls

AlpacaEval requires an explicitly exported `ALPACA_EVAL_ANNOTATORS_CONFIG` naming
an installed annotator configuration or a configuration path accessible inside
the container. Preflight requires this setting. Credentials and endpoints come
from that configuration and the official `alpaca_eval` package; a local scorer
can run without `OPENAI_API_KEY`. The launcher exports the submission environment
to Slurm. Configure any scorer-specific credentials in that environment, and
ensure any configuration files are available on the container's mounted paths.
Do not put secrets in the annotator configuration name.

`ALPACA_EVAL_OUTPUT_DIR` optionally retains scorer artifacts, including judge
configuration. Choose a separate output directory for each model/run. Judge
identity changes the evaluation protocol and must accompany any comparison.
The adapter reports length-controlled win rate as a fraction; the dashboard
shows its percentage. No default endpoint or judge is selected by this suite.

HumanEval and MBPP execute model-produced Python when scored. Enable this only
for an evaluation environment intended to run generated code:

```bash
export ORCH_REPO_ROOT="$PWD"
bash launchers/lm-eval/eval.sh /path/to/model \
  --tasks humaneval_instruct,mbpp_instruct --confirm-run-unsafe-code
```

The opt-in is also accepted after `--` by the job script. It passes
`--confirm_run_unsafe_code` to the harness and sets `HF_ALLOW_CODE_EVAL=1` inside
the job. Ordinary launches add neither setting. The harness retains its own
unsafe-code check; requesting these tasks without the opt-in cannot score them.
This flag is permission to execute code, not a sandbox implementation.

To run the entire requested set, configure the judge and code opt-in explicitly:

```bash
export ORCH_REPO_ROOT="$PWD"
export ALPACA_EVAL_ANNOTATORS_CONFIG=/mounted/path/to/annotators.yaml
bash launchers/lm-eval/eval.sh /path/to/model \
  --suite text-requested --confirm-run-unsafe-code
```

Text evaluations apply the model's chat template by default, matching the
release instruction-model protocol. The registry sets
`[defaults].lm_eval_chat_template = true`; a task's `chat_template` field can
override that default. The launcher queries `suite.tasks --chat-template` for
each task. `LM_EVAL_CHAT_TEMPLATE=0` forces completion prompts;
`LM_EVAL_CHAT_TEMPLATE=1` forces chat formatting, and an explicit job argument
`-- --apply-chat-template` remains supported. Finalization and collection also
check the harness result against the registry: an override that disagrees with
the declaration is marked invalid and excluded from published results. For a
base-model protocol, declare `chat_template = false` for the task (or change the
registry default for that evaluation checkout), and retain the same declaration
when collecting results.

Each run manifest records the actual setting in
`generation.apply_chat_template` (`1` enabled, `0` disabled), together with the
tokenizer template path and hash. Compare scores only with the intended prompt
protocol and template identity: missing chat formatting caused the earlier text
evaluation gap documented in [C10](superpowers/specs/2026-09-08-eval-suite-hardening-design.md#1210-correction-the-text-gap-was-the-chat-template-and-c10-makes-the-protocol-declared).
The suite does not impose common few-shot counts or decoding settings.
The pinned harness selectively ports Swiss report task configurations and math
extraction from `4ac31da`, while retaining upstream's newer correctness fixes.
MMLU-Flan and MMLU-Pro use ordered extraction for the headline. Both MATH tasks
report Math Verify; their exact-match scores remain in raw results. MathQA
reports raw accuracy, so old `acc_norm` headlines must not be compared as the
same metric. These are protocol revisions, not retrospective changes to old run
artifacts. Preserve the harness pin, metric key and task version when comparing.

Upstream BBQ answer remapping remains intact. The Swiss report harness's mapping
can credit incorrect concrete answers when the gold answer is unknown; the
suite does not reproduce that inflated score. Multi-IF remains first-turn-only,
and `mgsm_en_cot_en` still evaluates English rather than the full language group.

## Dashboard metrics

Names below are exact serialized metric/filter keys. Fractions are displayed as
percentages. SQuAD v2's F1 is already in percent and has an explicit unit marker,
so a score of `0.5` stays `0.5%`. Other existing task mappings are preserved.

| Requested benchmark | Dashboard metric | Scope |
| --- | --- | --- |
| `mmlu_flan_cot_zeroshot` | `exact_match,ordered-extract` | Zero-shot Flan CoT, subject-weighted aggregate |
| `mmlu_pro` | `exact_match,ordered-extract` | Subject-weighted aggregate |
| `truthfulqa_mc2` | `acc,none` | MC2 probability score |
| `commonsense_qa` | `acc,none` | Multiple-choice accuracy |
| `squadv2` | `f1,none` | Answer F1, native percent |
| `hellaswag` | `acc_norm,none` | Length-normalized accuracy |
| `ifeval` | `prompt_level_strict_acc,none` | All instructions satisfied per prompt |
| `ifbench` | `prompt_level_strict_acc,none` | Existing suite-local verifier |
| `multi-if` | `prompt_level_strict_acc,none` | **Single first turn only** |
| `alpaca_eval` | `length_controlled_winrate,none` | Explicit judge configuration |
| `gsm8k_cot` | `exact_match,flexible-extract` | Eight-shot CoT |
| `hendrycks_math` | `math_verify,none` | Six-shot, 2048 tokens, subject-weighted aggregate |
| `minerva_math` | `math_verify,none` | Four-shot, 1024 tokens, subject-weighted aggregate |
| `mathqa` | `acc,none` | Raw multiple-choice accuracy; assistant answer prefix |
| `humaneval_instruct` | `pass@1,create_test` | Generated-code execution opt-in |
| `mbpp_instruct` | `pass_at_1,extract_code` | Generated-code execution opt-in |
| `bbh` | `exact_match,get-answer` | CoT few-shot aggregate |
| `acp_bench` | 14 component rows; see below | Upstream tag, no unified aggregate |
| `drop` | `f1,none` | Three-shot answer-only prompt; answer F1, native fraction |
| `global_mmlu_gen_0shot` | `exact_match,extract-answer` | Ported language/subject aggregate |
| `mgsm_en_cot_en` | `exact_match,flexible-extract` | English language, English CoT prompt |
| `truthfulqa_multilingual_mc2` | `acc,none` | Multilingual MC2 aggregate |
| `include_base_44_gen_0shot` | `exact_match,extract-answer` | Ported 44-language aggregate |
| `include_base_new_45_gen_0shot` | `exact_match,extract-answer` | Ported new 45-language aggregate |
| `xnli` | `acc,none` | Language-weighted aggregate |
| `blend_sample` | `acc,none` | Ported sampled country subsets |
| `cultural_bench` | `acc,none` | Ported easy/hard and country aggregate |
| `switzerland_qa_0shot` | `exact_match,extract-answer` | Ported five-language aggregate |
| `bbq` | `acc,none` | Answer accuracy; bias metrics remain in raw results |
| `toxigen` | `acc,none` | Hateful-statement classification accuracy |
| `wmdp` | `acc,none` | Bio/chem/cyber knowledge accuracy, not a safety score |

The imported `multi-if` implementation reads and scores only `turn_1`. It does
not evaluate later turns, instruction retention across turns, or full multi-turn
Multi-IF performance. Its existing task name is preserved for reproducibility.

ACPBench's upstream `acp_bench` tag expands to seven boolean and seven
multiple-choice components: `areach`, `app`, `just`, `land`, `prog`, `reach`, and
`val`. Dashboard rows `acp_<component>_bool` use
`exact_match,extract-yes-no`; `acp_<component>_mcq` rows use
`exact_match,mcq-extract`. The registry records all 14 expected task results, and
manifest finalization requires a numeric result for every component. There is
no synthesized ACPBench headline or cross-filter aggregate.

## Validation boundary

Offline tests exercise task registration, launcher argument and environment
propagation, judge preflight, manifest completeness, and dashboard metric keys
and units. A fake scheduler and inference process keep the real shell launchers,
preflight, and manifests under test. These checks execute no generated code,
make no live judge calls, download no benchmark datasets, and use no GPUs.
Dataset accessibility, image dependencies, model-specific chat templates, full
benchmark scores, and live judge operation require separate runtime validation.
