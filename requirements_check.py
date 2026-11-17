"""Offline check of flat pip requirements against this interpreter's metadata.

Uses pip's bundled PEP 508 parser, available in installer-created venvs.
No imports of application/native packages and no network access.
"""
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def main(paths):
    try:
        from pip._vendor.packaging.requirements import Requirement
    except ImportError:
        print("NOT CURRENT: pip requirement parser unavailable; recreate/repair venv", file=sys.stderr)
        return 1
    failures = []
    for path in paths:
        for number, line in enumerate(Path(path).read_text().splitlines(), 1):
            text = line.split(" #", 1)[0].strip()
            if not text or text.startswith("#"):
                continue
            try:
                requirement = Requirement(text)
                if requirement.url or requirement.extras:
                    raise ValueError("URLs/extras are not supported by the offline checker")
                if requirement.marker and not requirement.marker.evaluate():
                    continue
                installed = version(requirement.name)
                if not requirement.specifier.contains(installed, prereleases=True):
                    failures.append(f"{text}: installed {installed}")
            except PackageNotFoundError:
                failures.append(f"{text}: missing")
            except Exception as exc:
                failures.append(f"{path}:{number}: cannot check {text!r}: {exc}")
    for failure in failures:
        print(f"NOT CURRENT: requirement {failure}", file=sys.stderr)
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
