# Evaluation images

Use [BUILDING.md](BUILDING.md) for the build commands, cache settings, validation,
and promotion requirements. `build_trial_image.sh` is the shared implementation;
`build_image.sbatch` and the compatibility entrypoint `generate_docker.sh` call it.
