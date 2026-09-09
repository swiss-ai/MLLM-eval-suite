The CUDA 13 recipe builds a separate ARM64 Apertus evaluation runtime from an official vLLM wheel. The production SquashFS and production EDF are not changed by a build.

The selected wheel is `0.28.1rc1.dev500+g51da0ca66`, from commit `51da0ca66c8065619c79e35dff97aa99aeaf5644`. Its SHA256 is checked during installation. NVIDIA's CUDA base is pinned by digest, and Transformers, TorchCodec and the two harness dependency manifests are pinned by commit.

From a CSCS host, submit from the suite root:

```sh
sbatch --account=infra01 dockerfiles/build_image.sbatch "$PWD" "$PWD/cache/image-builds/vllm-trial"
```

When submitting from inside a Pyxis container, first remove all inherited environment variables containing `SPANK`, including `_SLURM_SPANK_OPTION_pyxis_environment`. Otherwise the host build can inadvertently start inside the calling container. When using an existing allocation, launch `build_trial_image.sh` with `srun --overlap --cpu-bind=none` and the same clean environment.

When launching GPU validation with a different EDF, also remove inherited `OCI_ANNOTATION_*` variables so the target EDF supplies its own hooks. Inheriting the coding container's SSH annotation otherwise starts a second SSH daemon on the same port.

The launcher snapshots the recipe, hashes the inputs, uses a private Podman store and uv cache in `/dev/shm`, and exports the image in the same allocation. Keeping unpacked wheels off Lustre avoids slow installation of packages such as FlashInfer, which contains about 85,000 files. Build layers and the dependency cache survive retries within the allocation; final images and build records are written to shared storage. The launcher refuses to overwrite an existing output image and never deletes another build's store.

The output directory contains the `.sqsh`, its SHA256, source hashes, the build stamp and Podman inspection metadata. Inside the image, `/opt/apertus/` contains exact installed Python/system package lists, source revisions, selected harness requirements and validation results.
Before publishing the versioned candidate, the builder uses `unsquashfs` to
read `/etc/apertus_image_version` and requires it to match this build's stamp.
Unreadable or mismatched exports fail without publishing an image or checksum.

`generate_docker.sh [output-directory]` remains as a compatibility entrypoint to
the same builder. It preserves the site APT sources/proxy and the persistent
wheel-cache default formerly used by that command. It now creates a versioned
candidate in the output directory; it does not rotate the production SquashFS.

For direct builds, `UV_CACHE_DIR` overrides the default allocation-local uv cache.
Set `APERTUS_APT_CONFIG_DIR` to a directory containing `empty.sources.list`,
`my-sources.d`, and `99-jfrog-proxy` to use site APT configuration. Those inputs
are copied into the build snapshot and included in its source hashes. File and
directory symlinks are dereferenced, including nested links, so later changes
to the live APT configuration cannot alter the captured inputs.

Promotion remains explicit: validate the candidate with the required model and
modality workloads, then update the intended EDF to its versioned path. Retain
the previous image and EDF path for rollback. Building a candidate does not
change either production default.

The exporter uses a private copy of Enroot's library to correct a known cleanup trap in the installed CSCS version. That trap hard-codes `docker rm` for Podman and runs after removing its working directory and export container. The correction uses the selected engine from a valid working directory and preserves the original export exit status. Export failures still stop publication; the host Enroot installation is unchanged.

Every package install uses `runtime-constraints.txt`. Unexpected dependency conflicts fail the build. The sole exception is the existing `latex2sympy2==1.9.1` declaration of ANTLR 4.7.2 while the suite uses 4.9.3. MathVision needs this legacy parser. Both legacy and current parsers are checked by the GPU smoke command. Decord and cd-fvd remain excluded: neither was present in the existing certified image, and Decord has no ARM64 distribution.

Build validation checks metadata, protected versions, CUDA version and Apertus config support. Driver-backed imports and the new fused embedding operator are checked afterward, on a GPU node:

```sh
/opt/venv/bin/python /opt/apertus/verify_image.py --gpu --output /writable/path/gpu-validation.json
```

The base recipe is still a combined evaluation environment. Transitive dependencies and Ubuntu updates are resolved at build time and recorded; it is not a complete reproducible dependency lock. A future exact rebuild should consume the recorded package manifests. The current recipe also requests math-verify 0.9.0, whereas the August 13 production image contains 0.1.0, so a full-image comparison does not isolate vLLM alone.

The optional `flashinfer-cubin` bundle is omitted for the GH200/BF16/FA3 target. Version 0.6.18 contains about 85,000 files and 6.6 GB of payload, predominantly Blackwell kernels. `flashinfer-python==0.6.18`, nvcc and Ninja remain installed for Hopper JIT kernels. This is not a promise of offline support for every FlashInfer backend: other models/backends may compile or download kernels on first use. See [FlashInfer package options](https://docs.flashinfer.ai/installation.html#package-options).

Local checks:

```sh
python3 -m unittest discover -s dockerfiles -p 'test_*.py'
bash -n dockerfiles/build_trial_image.sh dockerfiles/build_image.sbatch dockerfiles/export_squashfs.sh
```
