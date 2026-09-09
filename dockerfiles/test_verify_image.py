"""Installed-package checks must accept compatible prereleases and reject drift."""
import importlib.metadata
from pathlib import Path
import tempfile
import unittest

from verify_image import check_dependencies


class DependencyCheckTest(unittest.TestCase):
    def check(self, packages):
        with tempfile.TemporaryDirectory() as root:
            installed = {}
            for name, version, requirements in packages:
                info = Path(root) / f"{name}-{version}.dist-info"
                info.mkdir()
                (info / "METADATA").write_text(
                    f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
                    + "".join(f"Requires-Dist: {r}\n" for r in requirements))
                installed[name] = importlib.metadata.Distribution.at(info)
            return check_dependencies(installed)

    def test_installed_compatible_prerelease(self):
        errors, _ = self.check([("vllm", "0.28.1rc1", ["transformers>=5.10.4"]),
                                ("transformers", "5.15.0.dev0", [])])
        self.assertEqual(errors, [])

    def test_incompatible_version_is_rejected(self):
        errors, _ = self.check([("vllm", "0.28.1rc1", ["transformers>=5.10.4"]),
                                ("transformers", "5.9.0", [])])
        self.assertEqual(len(errors), 1)

    def test_only_exact_legacy_exception_is_allowed(self):
        packages = [("latex2sympy2", "1.9.1", ["antlr4-python3-runtime==4.7.2"]),
                    ("antlr4-python3-runtime", "4.9.3", [])]
        errors, exceptions = self.check(packages)
        self.assertEqual(errors, [])
        self.assertEqual(len(exceptions), 1)
        packages[1] = ("antlr4-python3-runtime", "4.13.2", [])
        errors, exceptions = self.check(packages)
        self.assertEqual(len(errors), 1)
        self.assertEqual(exceptions, [])


if __name__ == "__main__":
    unittest.main()
