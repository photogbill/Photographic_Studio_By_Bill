#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  start-paint.sh - opens Photographic Studio By Bill straight into Paint (Linux).
#  Everything else (first-run setup, updates, logs) is start-studio.sh's job:
#  this only asks it to open on the Paint page. If the pen's pressure does not
#  reach Paint, open Paint > Tablet check: it says what is wrong in plain words.
# ---------------------------------------------------------------------------

main() {
    local here
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
    exec bash "$here/start-studio.sh" --paint "$@"
}

main "$@"; exit $?
