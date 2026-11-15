#!/usr/bin/env bash
# ===========================================================================
#  install-linux.sh - Photographic Studio By Bill on Linux. The only file you need.
#
#  Run it in a terminal:      bash install-linux.sh
#
#  It asks where to put the Studio (~/Photographic-Studio unless you choose
#  another folder), then downloads the newest version from GitHub and sets up a
#  private Python environment inside that one folder. No root for the Studio
#  itself: the one thing that may need your password is adding a few system
#  libraries the window needs, and it asks first.
#
#  Run it again at any time to UPDATE. Your settings, calibrations, logs and
#  photos are kept. Outside the folder there is only an applications-menu entry,
#  and only if you say yes. To remove the Studio, delete the folder (and
#  ~/.local/share/applications/photographic-studio-by-bill.desktop).
#
#  Inside a developer's git copy it does not download anything: it only sets up
#  Python there (the same as Install.bat on Windows).
#
#  Options:  --dir FOLDER   install or update there without asking for a folder
#            --yes          take every default, ask nothing
#            --no-menu      no applications-menu entry
#            --no-start     do not offer to start the Studio at the end
#  Needs: bash, curl or wget, tar, and a glibc Linux on x86_64 or ARM64
#  (Ubuntu, Mint, Debian, Fedora, openSUSE, Arch and the like; Bazzite and the
#  other Fedora Atomic systems too: everything goes in your home folder).
# ===========================================================================

REPO="photogbill/Photographic_Studio_By_Bill"
BRANCH="main"
MENU_FILE="$HOME/.local/share/applications/photographic-studio-by-bill.desktop"

say() { printf ' %s\n' "$*"; [ -n "${LOG:-}" ] && printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$LOG"; }
logonly() { [ -n "${LOG:-}" ] && printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$LOG"; }
have_tty() { { : < /dev/tty; } 2>/dev/null; }

# ask_line PROMPT: one line from the keyboard (empty with no keyboard or --yes)
ask_line() {
    REPLY=""
    if [ "${STUDIO_YES:-0}" = "1" ] || ! have_tty; then return 0; fi
    printf ' %s ' "$1" > /dev/tty
    read -r REPLY < /dev/tty || REPLY=""
}

fetch() {   # fetch URL FILE
    rm -f "$2"
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 --retry-delay 2 -sS -A "PhotographicStudio-Installer" -o "$2" "$1" >> "$LOG" 2>&1 && [ -s "$2" ]
    elif command -v wget >/dev/null 2>&1; then
        wget -q --tries=3 -U "PhotographicStudio-Installer" -O "$2" "$1" >> "$LOG" 2>&1 && [ -s "$2" ]
    else
        return 1
    fi
}

fail() { echo; say "PROBLEM: $*"; [ -n "${LOG:-}" ] && say "Details: $LOG"; echo; exit 1; }

main() {
    local dest="" no_menu=0 no_start=0 here src tag url mode menu_made=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --dir) dest="$2"; shift 2 ;;
            --dir=*) dest="${1#--dir=}"; shift ;;
            --yes|-y) export STUDIO_YES=1; shift ;;
            --no-menu) no_menu=1; shift ;;
            --no-start) no_start=1; shift ;;
            -h|--help) sed -n '2,32p' "${BASH_SOURCE[0]}" 2>/dev/null | sed 's/^# \{0,1\}//'; exit 0 ;;
            *) echo " Unknown option: $1 (try --help)"; exit 1 ;;
        esac
    done

    echo
    echo " Photographic Studio By Bill - Linux installer"
    echo " ============================================="
    echo
    if [ "$(uname -s)" != "Linux" ]; then
        echo " This installer is for Linux. On Windows, use Install-Photographic-Studio.bat."
        exit 1
    fi
    if [ "$(id -u)" = "0" ] && [ "${STUDIO_ALLOW_ROOT:-0}" != "1" ]; then
        echo " Please run this as yourself, not as root or with sudo: the Studio belongs in your own"
        echo " folders. (It asks for your password itself if it needs to add system libraries.)"
        exit 1
    fi
    command -v tar >/dev/null 2>&1 || { echo " PROBLEM: 'tar' is not installed."; exit 1; }
    if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
        echo " PROBLEM: neither curl nor wget is installed. Install one (e.g. sudo apt install curl) and run this again."
        exit 1
    fi

    # Run from inside a Studio folder: update (or, in a git copy, set up) that folder.
    here=""
    if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
        here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
    fi
    if [ -z "$dest" ]; then
        if [ -n "$here" ] && [ -f "$here/studio/__init__.py" ]; then
            dest="$here"
        else
            dest="$HOME/Photographic-Studio"
        fi
        while :; do
            echo " Install folder:  $dest"
            if [ -e "$dest/.git" ]; then
                echo "                  (a developer's git copy: only its Python is set up, nothing is downloaded)"
            elif [ -f "$dest/studio/__init__.py" ]; then
                echo "                  (the Studio is already here: it will be UPDATED,"
                echo "                   your settings, calibrations and photos stay)"
            else
                echo "                  (a new folder)"
            fi
            echo
            ask_line "Press Enter to continue, type a different folder, or q to quit >"
            case "$REPLY" in
                "") break ;;
                q|Q) exit 1 ;;
                *)
                    local pick="${REPLY/#\~/$HOME}"
                    pick="${pick%/}"
                    case "$pick" in /*) ;; *) pick="$PWD/$pick" ;; esac
                    if [ -f "$pick/studio/__init__.py" ] || [ "$(basename "$pick")" = "Photographic-Studio" ]; then
                        dest="$pick"
                    else
                        dest="$pick/Photographic-Studio"
                    fi
                    echo ;;
            esac
        done
    else
        dest="${dest/#\~/$HOME}"
        case "$dest" in /*) ;; *) dest="$PWD/$dest" ;; esac
    fi
    dest="${dest%/}"

    mkdir -p "$dest/logs" 2>/dev/null || { echo " PROBLEM: cannot create $dest - pick a folder you can write to."; exit 1; }
    [ -w "$dest" ] || { echo " PROBLEM: cannot write to $dest - pick a folder you can write to."; exit 1; }
    dest="$(cd "$dest" && pwd -P)"
    LOG="$dest/logs/installer.log"
    logonly "installer started: $dest (bash $BASH_VERSION, $(uname -srm))"
    if [ -r /etc/os-release ]; then logonly "system: $(. /etc/os-release && echo "${PRETTY_NAME:-unknown}")"; fi

    if [ -e "$dest/.git" ]; then
        mode=git
        echo
        say "This is a git copy: setting up Python only (update it with git, not this installer)."
    else
        mode=download
        local work="$dest/_installer"
        mkdir -p "$work"
        echo
        if [ -n "${STUDIO_SOURCE_TARBALL:-}" ]; then          # tests and offline installs: a local copy of the source
            say "1/4  Using the Studio from $STUDIO_SOURCE_TARBALL..."
            cp "$STUDIO_SOURCE_TARBALL" "$work/studio.tar.gz" || fail "could not read $STUDIO_SOURCE_TARBALL"
            tag="${STUDIO_SOURCE_TAG:-}"
        else
            # Which version: the newest release on GitHub, or the newest code if there is no release yet.
            tag=""
            if fetch "https://api.github.com/repos/$REPO/releases/latest" "$work/release.json"; then
                tag="$(grep -o '"tag_name": *"[^"]*"' "$work/release.json" | head -n 1 | sed 's/.*"\([^"]*\)"$/\1/')"
            fi
            rm -f "$work/release.json"
            if [ -n "$tag" ]; then
                url="https://github.com/$REPO/archive/refs/tags/$tag.tar.gz"
                say "1/4  Downloading Photographic Studio $tag from GitHub..."
            else
                url="https://github.com/$REPO/archive/refs/heads/$BRANCH.tar.gz"
                say "1/4  Downloading the newest Photographic Studio from GitHub..."
            fi
            logonly "download $url"
            fetch "$url" "$work/studio.tar.gz" || fail "the download did not work. Check the internet connection and run the installer again."
        fi

        say "2/4  Unpacking..."
        rm -rf "$work/src"
        mkdir -p "$work/src"
        tar -xzf "$work/studio.tar.gz" -C "$work/src" >> "$LOG" 2>&1 || fail "the download could not be unpacked. Run the installer again."
        src="$(find "$work/src" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
        [ -n "$src" ] && [ -f "$src/studio/__init__.py" ] || fail "the download does not look like the Studio. Run the installer again."
        if [ ! -f "$src/start-studio.sh" ] && [ -z "${STUDIO_SOURCE_TARBALL:-}" ]; then
            # a release from before the Linux version: take the newest code instead
            say "      Release $tag has no Linux files yet; downloading the newest code instead..."
            logonly "release $tag has no start-studio.sh: falling back to branch $BRANCH"
            tag=""
            fetch "https://github.com/$REPO/archive/refs/heads/$BRANCH.tar.gz" "$work/studio.tar.gz" || fail "the download did not work. Check the internet connection and run the installer again."
            rm -rf "$work/src"; mkdir -p "$work/src"
            tar -xzf "$work/studio.tar.gz" -C "$work/src" >> "$LOG" 2>&1 || fail "the download could not be unpacked. Run the installer again."
            src="$(find "$work/src" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
            [ -n "$src" ] && [ -f "$src/start-studio.sh" ] || fail "the newest code has no Linux files either."
        fi

        say "3/4  Copying it into place (your settings, calibrations and photos are kept)..."
        # The download holds only software and shared profiles, so nothing of yours is overwritten, and nothing is
        # deleted. --remove-destination writes new files instead of rewriting old ones in place, so a script that is
        # running (this one included) is never changed under its own feet.
        if cp --help 2>&1 | grep -q -- '--remove-destination'; then
            cp -a --remove-destination "$src/." "$dest/" >> "$LOG" 2>&1 || fail "the files could not be copied into $dest. If the Studio is open, close it and run the installer again."
        else
            cp -a "$src/." "$dest/" >> "$LOG" 2>&1 || fail "the files could not be copied into $dest. If the Studio is open, close it and run the installer again."
        fi
        [ -n "$tag" ] && echo "$tag" > "$dest/.installed_version"
        echo "sh" > "$dest/.install_kind"
        rm -rf "$work/src" "$work/studio.tar.gz"
        logonly "copied $src into $dest"
    fi
    chmod +x "$dest/start-studio.sh" "$dest/install-linux.sh" 2>/dev/null

    [ "$mode" = download ] && say "4/4  Setting up Python and the packages (the first time takes a few minutes)..."
    echo
    [ -f "$dest/tools/ensure_env.sh" ] || fail "tools/ensure_env.sh is missing from $dest."
    # shellcheck source=tools/ensure_env.sh
    . "$dest/tools/ensure_env.sh"
    studio_env "$dest" || fail "setting up Python did not finish. Run the installer again; it carries on where it stopped."

    echo
    say "Checking the engine..."
    local out
    out="$(cd "$dest" && "$PY" -m studio version 2>&1)" || { printf '%s\n' "$out" >> "$LOG"; fail "the engine did not start."; }
    printf '%s\n' "$out" | tee -a "$LOG"
    "$PY" -c "import rawpy; print('  LibRaw', '.'.join(map(str, rawpy.libraw_version)), '- reads CR2 and CR3')" 2>&1 | tee -a "$LOG"

    echo
    say "Checking the window can open..."
    local window_ok=1 probe
    if _studio_ostree; then
        # image-based (Bazzite / Fedora Atomic): adding libraries means a system layer and a reboot, so only when the
        # window really cannot open (a Wayland session needs none of the X11 ones)
        studio_qt_probe; probe=$?
        if [ "$probe" != 0 ]; then
            studio_qt_check "$dest" || window_ok=0
            studio_qt_probe; probe=$?
        else
            _studio_log "image-based system: the window opens; missing X11-only libraries (if any) not offered: $(_studio_missing_libs | tr '\n' ' ')"
        fi
    else
        studio_qt_check "$dest" || window_ok=0
        studio_qt_probe; probe=$?
    fi
    case $probe in
        0) say "The window can open here." ; window_ok=1 ;;
        1) window_ok=0
           say "The window could not open yet. Details: logs/setup.log"
           say "If libraries were listed above, install them and run this installer again." ;;
        2) say "(No desktop session here, so the window was not tried; it is checked again when the Studio starts.)" ;;
    esac

    # Applications-menu entry (asked; kept up to date if it is already there).
    if [ "$no_menu" = 0 ]; then
        case "$dest" in
            *'"'*|*'`'*|*'$'*|*\\*)
                say "The folder name has characters a menu entry cannot hold; start the Studio with start-studio.sh." ;;
            *)
                if [ -f "$MENU_FILE" ] || { echo; _studio_ask "Add Photographic Studio to your applications menu?" y; }; then
                    mkdir -p "$(dirname "$MENU_FILE")"
                    cat > "$MENU_FILE" <<DESKTOP
[Desktop Entry]
Type=Application
Version=1.0
Name=Photographic Studio
GenericName=Raw photo editor
Comment=Photographic Studio By Bill - a free raw studio. Photos never leave this PC.
Exec=bash "$dest/start-studio.sh"
Path=$dest
Icon=$dest/assets/icon.svg
Terminal=false
Categories=Graphics;Photography;RasterGraphics;
Keywords=raw;photo;denoise;CR2;CR3;develop;
StartupWMClass=photographic-studio
DESKTOP
                    command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$(dirname "$MENU_FILE")" >/dev/null 2>&1
                    say "Menu entry: $MENU_FILE"
                    menu_made=1
                fi ;;
        esac
    fi
    logonly "installer finished (mode $mode, window ok $window_ok)"

    echo
    echo " ====================================================================="
    echo "  All set. Photographic Studio is in:"
    echo "     $dest"
    if [ "$menu_made" = 1 ]; then
        echo "  Start it from your applications menu, or in a terminal:"
    else
        echo "  Start it in a terminal with:"
    fi
    echo "     \"$dest/start-studio.sh\""
    if [ "$mode" = git ]; then
        echo "  This is a git copy: update it with git."
    else
        echo "  To update later: turn on Help > Check for updates in the Studio, or run"
        echo "  this installer again (\"$dest/install-linux.sh\")."
    fi
    echo " ====================================================================="
    echo
    if _studio_ostree && [ -n "${SteamGamepadUI:-}${GAMESCOPE_WAYLAND_DISPLAY:-}" ]; then
        echo "  (This looks like Game Mode: start the Studio from Desktop Mode.)"
        echo
    fi
    if [ "$no_start" = 0 ] && [ "$window_ok" = 1 ] && { [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; }; then
        if _studio_ask "Start the Studio now?" y; then
            nohup bash "$dest/start-studio.sh" >/dev/null 2>&1 &
            disown 2>/dev/null
        fi
    fi
    return 0
}

main "$@"; exit $?
