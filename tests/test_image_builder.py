"""Image entrypoints share immutable build inputs and never rotate production."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import pytest


REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def build_environment(tmp_path):
    recipe = tmp_path / "suite/dockerfiles"
    shutil.copytree(REPO / "dockerfiles", recipe)
    output = tmp_path / "candidate"
    tools = tmp_path / "bin"
    tools.mkdir()
    for name in ("podman", "enroot", "uname", "unsquashfs"):
        path = tools / name
        path.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
from pathlib import Path
name = Path(sys.argv[0]).name
with open(os.environ['BUILD_COMMANDS'], 'a') as log:
    log.write(json.dumps([name, *sys.argv[1:]]) + '\\n')
if name == 'uname':
    print('aarch64')
elif name == 'podman' and sys.argv[1:3] == ['image', 'inspect']:
    print('{}')
elif name == 'enroot':
    flag = '--output' if '--output' in sys.argv else '-o'
    Path(sys.argv[sys.argv.index(flag) + 1]).write_bytes(b'candidate-image')
    sys.exit(int(os.environ.get('EXPORT_STATUS', '0')))
elif name == 'unsquashfs':
    assert sys.argv[1] == '-cat' and sys.argv[3] == '/etc/apertus_image_version'
    if os.environ.get('UNSQUASHFS_STATUS'):
        sys.exit(int(os.environ['UNSQUASHFS_STATUS']))
    commands = [json.loads(line) for line in Path(os.environ['BUILD_COMMANDS']).read_text().splitlines()]
    build = next(command for command in commands if command[:2] == ['podman', 'build'])
    stamp = next(arg.split('=', 1)[1] for arg in build if arg.startswith('BUILD_STAMP='))
    print(os.environ.get('IMAGE_STAMP', stamp))
''')
        path.chmod(0o755)
    library = tmp_path / "enroot-library"
    library.mkdir()
    (library / "docker.sh").write_text("# fixture library\n")
    job_id = "cpu-test-" + uuid.uuid4().hex
    env = dict(os.environ, PATH=f"{tools}:{os.environ['PATH']}", SLURM_JOB_ID=job_id,
               BUILD_COMMANDS=str(tmp_path / "commands.jsonl"), ENROOT_LIBRARY_PATH=str(library),
               UV_CACHE_DIR=str(tmp_path / "wheel-cache"))
    for key in ("APERTUS_RECIPE_DIR", "APERTUS_BUILD_SNAPSHOT", "APERTUS_APT_CONFIG_DIR",
                "EXPORT_STATUS", "UNSQUASHFS_STATUS", "IMAGE_STAMP"):
        env.pop(key, None)
    # The unmodified builder owns this unique store; no container command is real.
    key = hashlib.sha256(str(output).encode()).hexdigest()[:12]
    store = Path(f"/dev/shm/apertus-build-{os.getuid()}-{job_id}-{key}")
    try:
        yield recipe, output, env
    finally:
        if store.exists():
            shutil.rmtree(store)


def invoke(recipe, output, env, entrypoint="generate_docker.sh"):
    return subprocess.run(["bash", str(recipe / entrypoint), str(output)],
                          env=env, capture_output=True, text=True)


def test_legacy_entrypoint_builds_versioned_candidate_preserving_site_options(build_environment):
    recipe, output, env = build_environment
    production = recipe / "apertus-vllm-vision-eval-prod.sqsh"
    production.write_bytes(b"certified-production")
    process = invoke(recipe, output, env)
    assert process.returncode == 0, process.stdout + process.stderr
    assert production.read_bytes() == b"certified-production"
    images = list(output.glob("*.sqsh"))
    assert len(images) == 1 and images[0].read_bytes() == b"candidate-image"
    assert images[0].with_suffix(".sqsh.sha256").is_file()
    context, = output.glob("context.*")
    commands = [json.loads(line) for line in Path(env["BUILD_COMMANDS"]).read_text().splitlines()]
    build, = [line for line in commands if line[:2] == ["podman", "build"]]
    assert build[-1] == str(context)
    assert f"{env['UV_CACHE_DIR']}:/root/.cache/uv:rw" in build
    for name, destination in (("empty.sources.list", "/etc/apt/sources.list"),
                              ("my-sources.d", "/etc/apt/sources.list.d"),
                              ("99-jfrog-proxy", "/etc/apt/apt.conf.d/99-jfrog-proxy")):
        assert f"{context}/apt/{name}:{destination}:ro,z" in build
    hashes = (output / "source-sha256.txt").read_text()
    assert "./apt/99-jfrog-proxy" in hashes and "./apt/my-sources.d/ubuntu.sources" in hashes
    # A published candidate is never overwritten by retrying either entrypoint.
    for entrypoint in ("generate_docker.sh", "build_trial_image.sh"):
        retry = invoke(recipe, output, env, entrypoint)
        assert retry.returncode != 0 and "Refusing to overwrite" in retry.stderr
        assert images[0].read_bytes() == b"candidate-image"


def test_export_failure_never_publishes_candidate_or_replaces_production(build_environment):
    recipe, output, env = build_environment
    production = recipe / "apertus-vllm-vision-eval-prod.sqsh"
    production.write_bytes(b"certified-production")
    env["EXPORT_STATUS"] = "17"
    process = invoke(recipe, output, env)
    assert process.returncode == 17, process.stdout + process.stderr
    assert not list(output.glob("apertus-vllm-*.sqsh"))
    assert production.read_bytes() == b"certified-production"


@pytest.mark.parametrize("failure", ["unreadable", "wrong_stamp"])
def test_invalid_exported_stamp_never_publishes_candidate(build_environment, failure):
    recipe, output, env = build_environment
    production = recipe / "apertus-vllm-vision-eval-prod.sqsh"
    production.write_bytes(b"certified-production")
    if failure == "unreadable":
        env["UNSQUASHFS_STATUS"] = "19"
    else:
        env["IMAGE_STAMP"] = "stamp from a different build"
    process = invoke(recipe, output, env)
    assert process.returncode != 0, process.stdout + process.stderr
    assert not list(output.glob("apertus-vllm-*.sqsh"))
    assert not list(output.glob("*.sqsh.sha256"))
    assert production.read_bytes() == b"certified-production"


def test_apt_symlinks_are_snapshotted_and_hashed_independently(build_environment):
    recipe, output, env = build_environment
    external = recipe.parent / "live-apt"
    external.mkdir()
    original_proxy = 'Acquire::http::Proxy "http://proxy.example:8080";\n'
    original_sources = "Types: deb\nURIs: https://example.invalid/ubuntu\n"
    proxy = external / "proxy"
    proxy.write_text(original_proxy)
    sources = external / "sources"
    sources.write_text(original_sources)
    sources_dir = external / "sources.d"
    sources_dir.mkdir()
    (sources_dir / "ubuntu.sources").symlink_to(sources)
    (recipe / "99-jfrog-proxy").unlink()
    (recipe / "99-jfrog-proxy").symlink_to(proxy)
    shutil.rmtree(recipe / "my-sources.d")
    (recipe / "my-sources.d").symlink_to(sources_dir, target_is_directory=True)

    process = invoke(recipe, output, env)
    assert process.returncode == 0, process.stdout + process.stderr
    context, = output.glob("context.*")
    proxy.write_text("changed proxy")
    sources.write_text("changed sources")
    copied = {"apt/99-jfrog-proxy": original_proxy,
              "apt/my-sources.d/ubuntu.sources": original_sources}
    hashes = (output / "source-sha256.txt").read_text()
    for name, expected in copied.items():
        assert (context / name).read_text() == expected
        assert not (context / name).is_symlink()
        assert f"{hashlib.sha256(expected.encode()).hexdigest()}  ./{name}" in hashes
    assert not (context / "apt/my-sources.d").is_symlink()
