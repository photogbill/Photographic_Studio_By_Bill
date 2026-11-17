"""What changed, written down as it is built, so nobody has to remember it at push time.

Bill (2026-11-16): push.bat stays "just push", with the comment about the changes filled in by itself after any part
is built. So:

    docs/NEXT_UPDATE.md   the waiting list: one plain line per change, added by whoever builds it, as it is built
                          (rule 0.3.13 in BUILD_PLAN.md; CLAUDE.md says the same)
    CHANGELOG.md          every update's lines under its version number, newest first

    python tools/whats_new.py --show                 print the waiting lines
    python tools/whats_new.py --add "Fixed: ..."     add a line to the waiting list (builders may also edit the file)
    python tools/whats_new.py --message FILE         push.bat: write the git comment from the waiting lines into FILE;
                                                     exit code 1 when nothing is waiting
    python tools/whats_new.py --release 0.1.6        Release.bat: move the waiting lines into CHANGELOG.md under that
                                                     version and empty the waiting list
    python tools/whats_new.py --notes 0.1.6 --out F  GitHub: write that version's lines into F (the release page,
                                                     which "What's new" in the Studio opens); empty when it has none

Everything is plain UTF-8 text. Nothing here ever fails a push: a problem prints a line and returns.
"""
from __future__ import annotations

import datetime as _dt
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PENDING = ROOT / "docs" / "NEXT_UPDATE.md"
CHANGELOG = ROOT / "CHANGELOG.md"

PENDING_HEADER = """# What is in the next update

<!--
  The waiting list. One plain line per change, written for photographers, added as each part is built:
      - New: Focus Stack merges a focus bracket into one sharp photo.
      - Fixed: exports no longer replace the camera's own JPEG.
  Start each line with "- ". push.bat sends these lines as the comment; Release.bat then moves them into
  CHANGELOG.md under the new version number and empties this list. Leave this header where it is.
-->
"""

CHANGELOG_HEADER = """# Photographic Studio By Bill: what changed, update by update

Newest first. Written by tools/whats_new.py from docs/NEXT_UPDATE.md each time Release.bat makes an update.
"""


def pending_lines(path: Path = PENDING) -> list:
    """The waiting changes, one string each (without the leading "- "). Indented lines continue the one above."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    out: list = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = re.match(r"^\s{0,1}[-*]\s+(.*)$", line)
        if m:
            out.append(m.group(1).strip())
        elif out and raw[:1] in (" ", "\t"):
            out[-1] += " " + line.strip()
    return [x for x in out if x]


def add_line(text: str, path: Path = PENDING) -> None:
    text = " ".join(text.split())
    if not text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    body = path.read_text(encoding="utf-8") if path.exists() else PENDING_HEADER
    if not body.endswith("\n"):
        body += "\n"
    path.write_text(body + f"- {text}\n", encoding="utf-8", newline="\n")


def commit_message(lines: list) -> str:
    """The git comment: a short first line, then every change."""
    first = lines[0]
    head = first if len(first) <= 66 else first[:63].rstrip() + "..."
    more = f" (and {len(lines) - 1} more)" if len(lines) > 1 else ""
    return f"Update: {head}{more}\n\n" + "\n".join(f"- {x}" for x in lines) + "\n"


def release(version: str, pending: Path = PENDING, changelog: Path = CHANGELOG, today: str | None = None) -> int:
    """Move the waiting lines into the changelog under ``version``. Returns how many lines moved."""
    lines = pending_lines(pending)
    if not lines:
        return 0
    version = version.strip().lstrip("vV")
    today = today or _dt.date.today().isoformat()
    section = f"## {version} ({today})\n\n" + "\n".join(f"- {x}" for x in lines) + "\n"
    old = changelog.read_text(encoding="utf-8") if changelog.exists() else CHANGELOG_HEADER
    m = re.search(r"(?m)^## ", old)
    head, rest = (old[:m.start()], old[m.start():]) if m else (old, "")
    if not head.endswith("\n\n"):
        head = head.rstrip("\n") + "\n\n"
    changelog.write_text(head + section + ("\n" + rest if rest else ""), encoding="utf-8", newline="\n")
    pending.write_text(PENDING_HEADER, encoding="utf-8", newline="\n")
    return len(lines)


def notes(version: str, changelog: Path = CHANGELOG) -> str:
    """One version's lines from the changelog ('' when it has none)."""
    version = version.strip().lstrip("vV")
    try:
        text = changelog.read_text(encoding="utf-8")
    except OSError:
        return ""
    m = re.search(rf"(?m)^## {re.escape(version)}\b[^\n]*\n(.*?)(?=^## |\Z)", text, flags=re.S)
    return m.group(1).strip() + "\n" if m and m.group(1).strip() else ""


def main(argv) -> int:
    try:
        if not argv or argv[0] == "--show":
            for x in pending_lines():
                print(f"- {x}")
            return 0
        if argv[0] == "--add" and len(argv) > 1:
            add_line(" ".join(argv[1:]))
            return 0
        if argv[0] == "--message" and len(argv) > 1:
            lines = pending_lines()
            if not lines:
                return 1
            Path(argv[1]).write_text(commit_message(lines), encoding="utf-8", newline="\n")
            return 0
        if argv[0] == "--release" and len(argv) > 1:
            n = release(argv[1])
            print(f"{n} line(s) moved into CHANGELOG.md under {argv[1]}" if n else "nothing was waiting")
            return 0
        if argv[0] == "--notes" and len(argv) > 1:
            text = notes(argv[1])
            if "--out" in argv and argv.index("--out") + 1 < len(argv):
                Path(argv[argv.index("--out") + 1]).write_text(text, encoding="utf-8", newline="\n")
            else:
                sys.stdout.write(text)
            return 0
        print(__doc__)
        return 0
    except Exception as err:                              # noqa: BLE001  (a notes problem must never stop a push)
        print(f"whats_new: {err}")
        return 1 if argv and argv[0] == "--message" else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
