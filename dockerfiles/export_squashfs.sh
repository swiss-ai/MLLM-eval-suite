#!/usr/bin/env bash
# Keep the installed Enroot unchanged; patch only the known cleanup trap locally.
set -euo pipefail
uri=${1:?Expected podman image URI}
output=${2:?Expected output SquashFS path}
temporary_root=${3:?Expected private temporary directory}
[[ $uri == podman://* ]] || { echo 'Expected a Podman image' >&2; exit 1; }
library_path=${ENROOT_LIBRARY_PATH:-/usr/lib/enroot}
compat_library=$(mktemp -d "$temporary_root/enroot-library.XXXXXX")
trap 'export_status=$?; rm -rf -- "$compat_library"; exit "$export_status"' EXIT
cp -R "$library_path/." "$compat_library/"
python3 - "$compat_library/docker.sh" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text()
broken = '''trap 'common::rmall "${tmpdir}" 2> /dev/null; docker rm -f -v "${tmpdir##*/}" > /dev/null 2>&1' EXIT'''
fixed = '''trap 'export_status=$?; cd /; common::rmall "${tmpdir}" 2> /dev/null || :; "${engine}" rm -f -v "${tmpdir##*/}" > /dev/null 2>&1 || :; exit "${export_status}"' EXIT'''
if source.count(broken) > 1:
    raise SystemExit('Unexpected Enroot cleanup implementation')
if broken in source:
    path.write_text(source.replace(broken, fixed, 1))
    print('Applied private Enroot cleanup fix; export errors retain their exit status.', flush=True)
PY
ENROOT_LIBRARY_PATH="$compat_library" enroot import --output "$output" "$uri"
