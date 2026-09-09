"""Fail the image build on runtime drift or unreviewed dependency conflicts."""
from importlib import metadata
import argparse
import json
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def check_dependencies(installed):
    errors = []
    exceptions = []
    for name, dist in installed.items():
        for text in dist.requires or []:
            req = Requirement(text)
            if req.marker and not req.marker.evaluate({"extra": ""}):
                continue
            dep = installed.get(canonicalize_name(req.name))
            if dep is not None and req.specifier.contains(dep.version, prereleases=True):
                continue
            message = f"{name}=={dist.version} requires {req}; installed {dep.version if dep else 'missing'}"
            # Retain the original mathvision parser while using the ANTLR
            # runtime needed by the current metric stack. No blanket ignore.
            if (name == "latex2sympy2" and dist.version == "1.9.1"
                    and canonicalize_name(req.name) == "antlr4-python3-runtime"
                    and str(req.specifier) == "==4.7.2" and dep
                    and dep.version == "4.9.3"):
                exceptions.append(message)
            else:
                errors.append(message)
    return errors, exceptions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true", help="Also validate driver-backed imports and the CUDA operator")
    parser.add_argument("--output", type=Path, default=Path("/opt/apertus/runtime-validation.json"))
    args = parser.parse_args()
    root = Path("/opt/apertus")
    expected = root / "runtime-constraints.txt"
    errors = []
    for line in expected.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        req = Requirement(line)
        actual = metadata.version(req.name)
        if not req.specifier.contains(actual, prereleases=True):
            errors.append(f"runtime drift: {req.name} {actual}, expected {req.specifier}")

    installed = {canonicalize_name(d.metadata["Name"]): d for d in metadata.distributions()}
    conflicts, exceptions = check_dependencies(installed)
    errors.extend(conflicts)
    if errors:
        raise SystemExit("Image validation failed:\n" + "\n".join(errors))

    import torch
    import transformers

    assert torch.version.cuda == "13.0", torch.version.cuda
    assert transformers.AutoConfig.for_model("apertus").model_type == "apertus"
    record = {
        "vllm": metadata.version("vllm"), "torch": torch.__version__,
        "cuda": torch.version.cuda, "torchvision": metadata.version("torchvision"),
        "torchaudio": metadata.version("torchaudio"), "torchcodec": metadata.version("torchcodec"),
        "transformers": transformers.__version__, "dependency_exceptions": exceptions,
        "gpu_imports_verified": False, "fused_embedding_operator_verified": False,
    }
    if args.gpu:
        import torchaudio  # noqa: F401
        import torchvision  # noqa: F401
        import torchcodec  # noqa: F401
        import vllm._C_stable_libtorch  # Requires the host libcuda.so.1.
        from latex2sympy2 import latex2sympy as legacy_parse
        from latex2sympy2_extended import latex2sympy as current_parse
        assert torch.cuda.is_available(), "CUDA is unavailable"
        assert hasattr(torch.ops._C, "vocab_parallel_embedding"), "missing fused embedding kernel"
        # Parsers intentionally preserve unevaluated expression structure.
        assert legacy_parse("1+1").doit() == 2
        assert current_parse("1+1").doit() == 2
        record.update(gpu_imports_verified=True, fused_embedding_operator_verified=True,
                      math_parser_smoke_verified=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
