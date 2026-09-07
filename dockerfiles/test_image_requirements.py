"""Check dependency selection without installing packages or running Docker."""
import importlib.util
from pathlib import Path
import unittest


class RequirementsTest(unittest.TestCase):
    def load_selector(self):
        path = Path(__file__).with_name("image_requirements.py")
        self.assertTrue(path.is_file(), "image dependency selector is missing")
        spec = importlib.util.spec_from_file_location("image_requirements", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.select_requirements

    def test_expands_only_requested_extras(self):
        select = self.load_selector()
        project = {"name": "lmms_eval", "dependencies": ["requests>=2"],
                   "optional-dependencies": {"all": ["lmms_eval[video]", "rich"],
                                             "video": ["torchcodec>=0.3", "decord"],
                                             "emu3": ["torch==2.2.1"]}}
        result = select(project, ["all"], [])
        self.assertIn("torchcodec>=0.3", result)
        self.assertIn("requests>=2", result)
        self.assertNotIn("decord", result)
        self.assertNotIn("torch==2.2.1", result)

    def test_preserves_markers_and_filters_legacy_conflicts(self):
        select = self.load_selector()
        project = {"name": "lmms_eval", "dependencies": ["latex2sympy2", "math-verify"]}
        result = select(project, [], ["# comment", "antlr4-python3-runtime==4.11.1",
                                     "datasets", 'foo>=2; python_version >= "3.12"'])
        self.assertNotIn("latex2sympy2", result)
        self.assertNotIn("antlr4-python3-runtime==4.11.1", result)
        self.assertIn("math-verify", result)
        self.assertIn('foo>=2; python_version >= "3.12"', result)

    def test_unknown_extra_fails(self):
        select = self.load_selector()
        with self.assertRaisesRegex(ValueError, "missing"):
            select({"name": "lmms_eval"}, ["missing"], [])


if __name__ == "__main__":
    unittest.main()
