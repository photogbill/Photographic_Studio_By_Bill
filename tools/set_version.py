"""Release.bat helper: show or set the Studio's version number (studio/__init__.py and pyproject.toml).

    python tools/set_version.py --show      prints the current version
    python tools/set_version.py --bump      raises the last number by one (0.2.0 -> 0.2.1); prints it
    python tools/set_version.py 0.2.0       sets it (three numbers, higher than now); prints it
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INIT = ROOT / "studio" / "__init__.py"
PYPROJECT = ROOT / "pyproject.toml"
_INIT_RE = re.compile(r'^__version__ = "([^"]*)"', re.M)
_TOML_RE = re.compile(r'^version = "([^"]*)"', re.M)


def current() -> str:
    m = _INIT_RE.search(INIT.read_text(encoding="utf-8"))
    return m.group(1) if m else "0.0.0"


def _nums(v: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3]) + (0,) * (3 - len(re.findall(r"\d+", v)[:3]))


def set_version(new: str) -> str:
    new = new.strip().lstrip("vV")
    if not re.fullmatch(r"\d+\.\d+\.\d+", new):
        raise ValueError(f"{new!r} is not three numbers like 0.2.0")
    if _nums(new) <= _nums(current()):
        raise ValueError(f"{new} is not higher than the current {current()}")
    for path, rx, fmt in ((INIT, _INIT_RE, '__version__ = "{}"'), (PYPROJECT, _TOML_RE, 'version = "{}"')):
        text = path.read_text(encoding="utf-8")
        if not rx.search(text):
            raise ValueError(f"no version line in {path.name}")
        path.write_text(rx.sub(fmt.format(new), text, count=1), encoding="utf-8", newline="")
    return new


def bumped() -> str:
    a, b, c = _nums(current())
    return f"{a}.{b}.{c + 1}"


def main(argv) -> int:
    if not argv or argv[0] == "--show":
        print(current())
        return 0
    if argv[0] == "--bump":
        argv = [bumped()]
    try:
        print(set_version(argv[0]))
    except ValueError as err:
        print(f"PROBLEM: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
