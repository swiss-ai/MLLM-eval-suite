"""Select pinned harness dependency sets; harness code is mounted at run time."""
import argparse
from pathlib import Path
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

# decord has no ARM64 distribution; cd-fvd depends on it. Both were absent
# from the certified image. latex2sympy2 is installed separately with a
# documented ANTLR metadata exception. The remaining replacements are pinned
# explicitly in runtime-constraints.txt / the Dockerfile.
SKIP = {"decord", "cd-fvd", "dotenv", "latex2sympy2", "antlr4-python3-runtime"}


def select_requirements(project, extras, vlmeval_lines):
    own_name = canonicalize_name(project["name"])
    optional = project.get("optional-dependencies", {})
    pending = list(project.get("dependencies", []))
    seen_extras = set()

    def add_extra(extra):
        if extra not in optional:
            raise ValueError(f"missing requested {own_name} extra: {extra}")
        if extra not in seen_extras:
            seen_extras.add(extra)
            pending.extend(optional[extra])

    for extra in extras:
        add_extra(extra)
    pending.extend(line.strip() for line in vlmeval_lines
                   if line.strip() and not line.lstrip().startswith("#"))
    selected = []
    for text in pending:
        requirement = Requirement(text)
        if canonicalize_name(requirement.name) == own_name:
            if requirement.marker is None or requirement.marker.evaluate():
                for extra in sorted(requirement.extras):
                    add_extra(extra)
        elif canonicalize_name(requirement.name) not in SKIP:
            if text not in selected:
                selected.append(text)
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lmms_pyproject", type=Path)
    parser.add_argument("vlmeval_requirements", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    project = tomllib.loads(args.lmms_pyproject.read_text())["project"]
    selected = select_requirements(project, ["all"],
                                   args.vlmeval_requirements.read_text().splitlines())
    args.output.write_text("\n".join(selected) + "\n")
    print(f"Selected {len(selected)} harness dependencies; exclusions: {sorted(SKIP)}")
