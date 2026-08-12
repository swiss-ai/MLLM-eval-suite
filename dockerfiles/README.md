# Dockerfiles

Store Dockerfiles and build documentation for evaluation environments. Keep image-specific instructions close to the Dockerfile that needs them.

## Layer cache

Cross-job layer caching is not currently wired: podman's --cache-to/--cache-from
require a registry reference, so every build re-executes all layers (~55 min).
The planned fix is a small local OCI registry on scratch; until then the
capstor build-cache directory is unused.

