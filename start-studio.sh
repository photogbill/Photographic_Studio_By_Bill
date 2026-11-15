#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  start-studio.sh - opens Photographic Studio By Bill (Linux).
#  The first run builds the private Python inside this folder (tools/ensure_env.sh).
#  An update the Studio downloaded (Help > Check for updates) is put in place here,
#  before the Studio starts. Settings, calibrations, logs and photos are never
#  touched: an update holds only software and shared profiles, and nothing is
#  deleted. Developer copies (a .git folder) are never overwritten.
#  The Studio's own log is logs/studio.log; this script's output and anything the
#  window prints before that log opens go to logs/launcher.log (the last start).
# ---------------------------------------------------------------------------

note() {     # a message that is seen whether the Studio was started from a terminal or from the menu
    printf ' %s\n' "$*"
    [ -t 1 ] && return 0
    if command -v notify-send >/dev/null 2>&1; then
        notify-send "Photographic Studio" "$*" >/dev/null 2>&1
    elif command -v zenity >/dev/null 2>&1; then
        zenity --error --title="Photographic Studio" --text="$*" >/dev/null 2>&1 &
    elif command -v kdialog >/dev/null 2>&1; then
        kdialog --title "Photographic Studio" --error "$*" >/dev/null 2>&1 &
    fi
}

apply_update() {
    local app="$1" newv="?" upd log="$1/logs/update.log"
    [ -f "$app/_update/ready.txt" ] && newv="$(head -n 1 "$app/_update/ready.txt")"
    upd="$(find "$app/_update/pending" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -n 1)"
    if [ ! -e "$app/.git" ] && [ -n "$upd" ] && [ -f "$upd/studio/__init__.py" ]; then
        echo " Installing Photographic Studio $newv..."
        printf '[%s] applying update %s from %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$newv" "$upd" >> "$log"
        sleep 2                                    # give a Studio that is just closing a moment to finish
        local opt="--remove-destination"           # new files, so this running script is never rewritten in place
        cp --help 2>&1 | grep -q -- "$opt" || opt=""
        # shellcheck disable=SC2086
        if cp -a $opt "$upd/." "$app/" >> "$log" 2>&1; then
            echo "$newv" > "$app/.installed_version"
            chmod +x "$app/start-studio.sh" "$app/install-linux.sh" 2>/dev/null
            printf '[%s] update %s is in place\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$newv" >> "$log"
        else
            note "The update could not be put in place; the current version starts instead. Details: logs/update.log"
            printf '[%s] update %s failed (cp)\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$newv" >> "$log"
        fi
    fi
    rm -rf "$app/_update/pending"
    rm -f "$app/_update/ready.txt"
}

main() {
    local app rc
    app="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
    cd "$app" || exit 1
    mkdir -p logs
    if [ "${1:-}" != "--no-update" ] && [ -f "$app/_update/ready.txt" ]; then
        apply_update "$app"
        exec bash "$app/start-studio.sh" --no-update     # the new version of this script carries on
    fi
    {
        printf '[%s] start-studio.sh in %s (%s)\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$app" "$(uname -srm)"
        printf '  session: %s, DISPLAY=%s, WAYLAND_DISPLAY=%s\n' "${XDG_SESSION_TYPE:-?}" "${DISPLAY:-}" "${WAYLAND_DISPLAY:-}"
    } > logs/launcher.log
    if [ ! -f "$app/tools/ensure_env.sh" ]; then
        note "tools/ensure_env.sh is missing; run install-linux.sh again."
        exit 1
    fi
    # shellcheck source=tools/ensure_env.sh
    . "$app/tools/ensure_env.sh"
    if ! studio_env "$app" 2>&1 | tee -a logs/launcher.log; then :; fi
    if [ -x "$app/python/bin/python3" ]; then PY="$app/python/bin/python3"; else PY="$app/.venv/bin/python"; fi
    if [ ! -x "$PY" ] || ! "$PY" -c "$STUDIO_IMPORTS" >/dev/null 2>&1; then
        note "The Studio could not start: its Python is not ready. Run install-linux.sh again (details: logs/setup.log)."
        exit 1
    fi
    if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ] && [ -z "${QT_QPA_PLATFORM:-}" ]; then
        note "No desktop session was found (DISPLAY and WAYLAND_DISPLAY are empty): start the Studio from your desktop."
        exit 1
    fi
    "$PY" -m studio.app >> logs/launcher.log 2>&1
    rc=$?
    printf '[%s] the Studio closed (exit %s)\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$rc" >> logs/launcher.log
    if [ "$rc" -ne 0 ]; then
        if grep -q -i "platform plugin\|libxcb\|xcb-cursor\|cannot open shared object" logs/launcher.log; then
            note "The Studio's window could not open: system libraries are missing. Run install-linux.sh again; it lists them and can add them."
        else
            note "The Studio stopped with an error. Details: logs/launcher.log and logs/studio.log (Help > Copy diagnostic report)."
        fi
    fi
    return "$rc"
}

main "$@"; exit $?
