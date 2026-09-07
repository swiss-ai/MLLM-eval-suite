# Handover to Claude: evaluation image upgrade

Prepared 2026-09-08 for `/iopsstor/scratch/cscs/xyixuan/apertus/MLLM-eval-suite`.
Paths below are relative to that repository unless absolute.

The new ARM64/CUDA 13 evaluation image is built and has passed a small Apertus
70B text smoke test. **Production launch defaults still select the old image.**
Other model families, multimodal execution, and performance have not been
validated on the new image. The immediate user request was to write this
handover; the last recommendation was a staged replacement after compatibility
and matched performance checks.

## User intent and decisions

The user asked which vLLM version we used, whether newer commits could improve
speed, whether a newer snapshot offered binaries, and then authorized proceeding
and improving the Dockerfile/build. That work produced the trial below.

The user subsequently emphasized that the suite supports other models as well as
Apertus. Preserve that scope. The dashboard registry lists Qwen2.5-VL, Qwen3-VL,
Molmo/Molmo2, Gemma3/4, Pixtral, EuroVLM, and Emu3, alongside Apertus. See
`scripts/dashboard_models.txt`. Registry entries and wrapper availability are
not evidence that those models pass in the new image.

An earlier description of the smaller image as "better" was too broad and was
corrected: the smaller size is measured; broad compatibility and a speedup are
unproven. No global promotion has occurred. No model training was done as part
of this image task.

## Artifacts and versions

Production image:

```text
/capstor/store/cscs/swissai/infra01/multimodal-eval/MLLM-eval-suite/sqsh/apertus-vllm-vision-eval-prod.sqsh
```

The earlier inspection of this August 13 image recorded:

- vLLM `0.26.1rc1.dev687+g324f452f6`.
- Source commit `324f452f640015e9616b5ad5615fae6c25eb6353`.
- Size 21,908,488,192 bytes.

Trial directory:

```text
cache/image-builds/vllm-51da0ca66-20260907/
```

Trial image, inside that directory:

```text
apertus-vllm-51da0ca66-cu130.sqsh
```

- Size: **16,279,797,760 bytes**, 25.7% smaller than the inspected production image.
- Recorded SHA256: `2f388196c5bea7b397d5277e053e709425f841681c7f187202c59825ad62dbf4`.
- vLLM: `0.28.1rc1.dev500+g51da0ca66`.
- vLLM source commit: `51da0ca66c8065619c79e35dff97aa99aeaf5644`.
- This is a pinned development snapshot, not a stable release.
- PyTorch `2.13.0+cu130`, torchvision `0.28.0+cu130`, torchaudio
  `2.11.0+cu130`, CUDA 13.0, FlashInfer Python `0.6.18`, Transformers
  `5.15.0.dev0`, TorchCodec `0.15.0`.

The build uses the official ARM64 vLLM wheel directly:

```text
https://wheels.vllm.ai/51da0ca66c8065619c79e35dff97aa99aeaf5644/vllm-0.28.1rc1.dev500%2Bg51da0ca66-cp38-abi3-manylinux_2_28_aarch64.whl
```

Wheel SHA256:
`569881fd0678d3cc6d54ba762743fdc6cce4489df9dcb36f439cf02e2f4ebc00`.
The Dockerfile verifies it during installation.

Other source pins in the recipe:

| Component | Commit |
| --- | --- |
| swiss-ai/transformers | `3596b056f73c7f44849c042cc02413c747636eab` |
| lmms-eval dependency metadata | `84947e886131d912eac1aaab1c3dcb5908a35c92` |
| VLMEvalKit dependency metadata | `a8b0a456f2807831736e686e34b061c7ae8d5fd6` |
| TorchCodec | `dc0f10da1fa2c807dedb7ed97de7b36d762b968b` |

The CUDA base is pinned by digest in the Dockerfile. Full runtime constraints
are in `dockerfiles/runtime-constraints.txt`; installed package manifests are
inside the image under `/opt/apertus/`.

## Changes made

Modified build files:

- `dockerfiles/Dockerfile.vllm-multimodal-eval-prod-cu130`
- `dockerfiles/build_image.sbatch`

Added build helpers, validation, and documentation:

- `dockerfiles/build_trial_image.sh`
- `dockerfiles/export_squashfs.sh`
- `dockerfiles/runtime-constraints.txt`
- `dockerfiles/image_requirements.py`
- `dockerfiles/verify_image.py`
- `dockerfiles/test_image_requirements.py`
- `dockerfiles/test_verify_image.py`
- `dockerfiles/BUILDING.md`

The recipe installs the hash-verified wheel without a second vLLM source tree,
pins core sources and runtime packages, checks dependency resolution before
compiling TorchCodec, removes temporary source trees in the same build layer,
and records build/package provenance. Unexpected dependency conflicts fail
validation. The one explicit metadata exception is the existing legacy
`latex2sympy2==1.9.1` declaration of ANTLR 4.7.2 while the installed runtime is
4.9.3; actual legacy and current parser execution passed.

The dependency selector retains lmms-eval's declared `all` extras recursively
and VLMEvalKit requirements. It is not an Apertus-only dependency list. Decord
and cd-fvd remain excluded; prior production inspection found neither installed.
Other exclusions are replacements or separately installed pinned packages.

Standard suite launchers put checked-out harness code on the runtime Python
path. The image contains dependencies rather than baked harness repositories.
Validate through those launchers: direct standalone harness invocation inside
the bare image is not the same execution path. Also record actual harness
checkout revisions in comparisons; they can differ from the dependency pins.

The build launcher snapshots its own script and an immutable build context,
records input hashes, uses a private Podman store and uv cache in `/dev/shm`,
and publishes the SquashFS without overwriting an existing output. It never
deletes another build's store. The build lock is not inherited by container
helper processes.

The installed CSCS Enroot exporter had a cleanup trap that hard-coded
`docker rm` during Podman export and ran after its working directory/container
had been removed. `export_squashfs.sh` patches the exact known trap in a private
library copy, preserves the original exit status, and leaves host Enroot
unchanged. Successful and deliberately failing exports were both checked.

## Why the build took time and why the image is smaller

There were dependency-resolution/build retries, slow unpacking of an optional
FlashInfer bundle with about 85,000 files on shared/FUSE storage, and the Enroot
export cleanup failure. The final build moved package/build storage into private
node memory and corrected export handling. These were image-build problems,
not training runs. Logs and immutable contexts remain in the trial directory.

Both images use the same inspected SquashFS compression settings: zstd with
uncompressed data blocks and compressed metadata. The reduction comes from
content cleanup, including source/build/cache leftovers and omission of the
optional `flashinfer-cubin` package. Model weights live outside the image and
the tested model remained BF16.

The production cubin 0.6.17 package occupied about 4.7 GB of apparent file data.
The candidate 0.6.18 bundle contained about 6.6 GB and its inspected cubins
targeted Blackwell; no Hopper benefit was identified in that exact bundle.
FlashInfer Python, nvcc, and Ninja remain installed, and Hopper sampling/JIT
worked in the smoke test. This does not establish offline readiness for every
backend or compatibility on other GPU architectures. The primary package
documentation explains optional binaries and first-use compilation/downloads:
https://docs.flashinfer.ai/installation.html#package-options

Do not add apparent directory sizes to explain compressed image savings:
SquashFS deduplication means they are not additive.

## Validation completed and limits

The trial directory contains `VALIDATION.md`, `gpu-runtime-validation.json`,
`gpu-validation.log`, `apertus-smoke.json`, `apertus-smoke.log`, and
`smoke_apertus.py`. These records were reread for this handover; GPU tests were
not rerun just to write it.

Completed on 2026-09-07:

- Six focused Python unit tests and shell syntax checks.
- Tiny Enroot export with content verification, plus an intentionally invalid
  SquashFS option that correctly produced a nonzero result.
- Installed-version/dependency checks with the exact ANTLR exception above.
- GPU imports for the core libraries and the vLLM CUDA extension.
- Actual fused embedding CUDA operator output checked against expected values,
  including owned and masked token IDs.
- Actual arithmetic through both math parser implementations.
- Apertus 70B BF16, TP4, FA3, Model Runner V2: four short text generations
  returned `4`, `Bern`, `4`, `Bern`. Greedy used temperature 0/top-p 1;
  sampling used temperature 0.6/top-p 0.95. Each response was two tokens.
  The command exited zero.

Model path used:

```text
cache/models/textview/70B-Final-fast
```

Cold initialization took 428.7804 seconds, including compilation. The tiny
generation timings are not throughput evidence. No matched old/new speed
benchmark, complete suite evaluation, non-Apertus model test, or image/video/
audio model inference test has completed as part of this work.

Additional observations to preserve:

- Ordinary all-reduce dispatch was CUSTOM/PYNCCL with
  `VLLM_ALLREDUCE_USE_FLASHINFER=0`. Independently, the newer default compiled
  pass `fuse_allreduce_rms=true` used FlashInfer for eligible patterns. A
  comparison intending to keep communication backends fixed must control both
  settings in both arms. Do not describe the trial as completely FlashInfer-free.
- The optional experimental CUDA xIELU package was absent; vLLM used its Python
  implementation.
- A tokenizer-only directory caused nonfatal model-config probe errors.
- Teardown logged forced worker termination after responses completed. The
  command exited zero, and GPU memory was checked as freed afterward.
- `math-verify` is now 0.9.0; the earlier production inspection found 0.1.0.
  A whole-image comparison includes metric/dependency changes and cannot isolate
  a vLLM-only effect. Separate raw generation comparisons from scoring changes.
- The recipe pins important components but is not a complete reproducible lock:
  transitive packages and Ubuntu updates resolve at build time and are recorded.

## Launch defaults: checked on 2026-09-08

All three launchers source `slurm/shared/sbatch_overrides.sh`:

- `launchers/lm-eval/eval.sh`
- `launchers/lmms-eval/eval.sh`
- `launchers/VLMEvalKit/eval.sh`

Unless overridden, `EVAL_ENVIRONMENT` selects
`toml/shared/apertus-vllm-vision-eval-prod.toml`. That file still points to the
old production SquashFS above. The new image was not enabled globally.

Only the dedicated trial EDF selects the new build:

```text
cache/image-builds/vllm-51da0ca66-20260907/trial.toml
```

To exercise a normal launcher with the trial, set `EVAL_ENVIRONMENT` to that
file's absolute path for that invocation and retain the evaluation's existing
arguments. The EDF includes writable runtime caches and a FlashInfer all-reduce
override, so compare its environment with production deliberately. Changing the
shared production TOML later covers all three default launchers; explicit EDF
overrides still select their own images.

## Suggested continuation

1. Inspect available cached models, existing evaluation arguments/results, and
   current allocations before launching work. Reuse completed validation.
2. Run small evaluations through the actual launchers for representative
   Qwen-VL, Molmo, and Gemma models, covering image inputs and the video/audio
   paths actually used. Track remaining families and unsupported paths
   explicitly rather than declaring the entire registry validated.
3. Compare old/new images on identical representative workloads and hardware.
   Fix model weights, harness revisions, prompts, generation settings, batch
   sizes, TP, attention/communication choices, and CPU settings. Check valid
   outputs, peak GPU memory, warmed throughput, and fresh-cache startup.
4. If compatibility and performance are acceptable, publish/retain the new
   image at a durable versioned path, verify its checksum, and update the shared
   production TOML. Keep the old image for rollback. Do not overwrite the old
   artifact or imply that queued/running checks have passed.

## CSCS execution notes

Read `dockerfiles/BUILDING.md` for build commands. The existing image does not
need rebuilding simply to continue validation. Any rebuild must use a fresh
output directory because the launcher refuses to overwrite an image.

Earlier work used allocation `3318237` on `nid007076`, with four GH200 GPUs,
account `infra01`, reservation `SD-69241-apertus-1-5-0`. This is historical
context, not a live allocation claim; verify scheduler state before reuse.

When invoking host Slurm tools from inside Pyxis, remove inherited environment
variables containing `SPANK`, including names starting with `_SLURM_SPANK`.
For a different EDF, also clear `OCI_ANNOTATION_*` and inherited
`SRUN_CONTAINER_IMAGE`, `SRUN_CONTAINER_NAME`, `SRUN_ENVIRONMENT`, and
`ENROOT_OVERLAYFS_PROGRAM`. Inherited annotations previously caused a second
SSH daemon to attempt to bind the coding container's existing port.

Use host `srun --overlap --cpu-bind=none` with a verified allocation; attach the
trial EDF to the intended GPU step. Do not cancel the enclosing coding
allocation. Node-local `/dev/shm` caches may disappear with that allocation;
published artifacts and records are on shared storage.

The user's separate question about job `3320410` concerned a Megatron text-SFT
recompute validation job in another repository. It was unrelated to the vLLM
build and was not launched or modified by this work. Its state was only
inspected historically; consult Slurm/logs for current status if asked.

## Workspace preservation

No commits, pushes, or production promotion were made for this image work.
The repository already contained many unrelated deletions, modified submodules,
and other edits. In particular, `slurm/VLMEvalKit/eval_job.slurm` appears modified
in current status; do not assume every dirty file belongs to this task.
Preserve other work and review scoped diffs before any integration. Some Git
objects were unavailable during earlier inspection, so broad restore/reset is
especially inappropriate.

Useful records in the trial directory also include `build.log`,
`build-stamp.txt`, `source-sha256.txt`, `podman-image.json`, the image checksum,
and the exact `context.*` snapshots. Two failed temporary SquashFS exports were
removed after final success; their logs were retained.

For build-code changes, the existing local checks are:

```sh
python3 -m unittest discover -s dockerfiles -p 'test_*.py'
bash -n dockerfiles/build_trial_image.sh dockerfiles/build_image.sbatch dockerfiles/export_squashfs.sh
```

Inside a GPU-enabled trial container, the existing runtime verifier is:

```sh
/opt/venv/bin/python /opt/apertus/verify_image.py --gpu --output /writable/path/gpu-validation.json
```

Choose a new output path when adding evidence so the original successful
validation records remain available.
