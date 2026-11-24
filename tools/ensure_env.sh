# shellcheck shell=bash
# ---------------------------------------------------------------------------
#  tools/ensure_env.sh - used by start-studio.sh and install-linux.sh (Linux).
#  Do not run it directly: it is sourced, then studio_env APP is called.
#
#  Makes sure the app's private Python exists and has every package in
#  requirements.txt, and sets PY to its python. Two kinds, both in the app folder:
#    .venv/    a virtual environment of the PC's own Python (3.10 or newer)
#    python/   a private copy of Python (python-build-standalone, the builds uv
#              uses), downloaded when the PC's Python is too old, has no venv
#              module, or the packages will not install into it. No root needed.
#  Nothing is installed outside the app folder, and pip's cache is turned off so
#  nothing lands in ~/.cache either. Every step is written to logs/setup.log.
#
#  studio_qt_check APP is used by the installer: it finds the system libraries
#  the window needs (Qt's xcb and Wayland plugins) that are missing, and offers to
#  install them with the system's package manager (that one step needs sudo).
# ---------------------------------------------------------------------------

STUDIO_PY_SERIES="3.12"            # the private Python's version (the Windows release is built on 3.12 too)
STUDIO_IMPORTS="import numpy, scipy, rawpy, tifffile, PIL, pytest, PySide6"

export PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUTF8=1

_studio_log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$STUDIO_SETUPLOG"; }
_studio_say() { printf ' %s\n' "$*"; _studio_log "$*"; }

# _studio_ask "question" default(y|n): true for yes. Reads the keyboard even when the script came through a pipe;
# with no keyboard at all (or STUDIO_YES=1) the default is taken.
_studio_ask() {
    local q="$1" def="$2" ans="" hint="[Y/n]"
    [ "$def" = "n" ] && hint="[y/N]"
    if [ "${STUDIO_YES:-0}" = "1" ] || ! { : < /dev/tty; } 2>/dev/null; then
        [ "$def" = "y" ]
        return
    fi
    printf ' %s %s ' "$q" "$hint" > /dev/tty
    read -r ans < /dev/tty || ans=""
    case "$ans" in
        [Yy]*) return 0 ;;
        [Nn]*) return 1 ;;
        *) [ "$def" = "y" ] ;;
    esac
}

# _studio_fetch URL FILE: curl, else wget. Quiet on screen, errors to the log.
_studio_fetch() {
    rm -f "$2"
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 --retry-delay 2 -sS -A "PhotographicStudio-Installer" -o "$2" "$1" >> "$STUDIO_SETUPLOG" 2>&1 && [ -s "$2" ]
    elif command -v wget >/dev/null 2>&1; then
        wget -q --tries=3 -U "PhotographicStudio-Installer" -O "$2" "$1" >> "$STUDIO_SETUPLOG" 2>&1 && [ -s "$2" ]
    else
        _studio_log "neither curl nor wget is installed"
        return 1
    fi
}

_studio_arch() {
    case "$(uname -m)" in
        x86_64|amd64) echo x86_64 ;;
        aarch64|arm64) echo aarch64 ;;
        *) return 1 ;;
    esac
}

# _studio_latest_tag URL: the tag a GitHub "releases/latest" page redirects to (no API: no rate limit).
_studio_latest_tag() {
    local eff=""
    if command -v curl >/dev/null 2>&1; then
        eff="$(curl -fsSL -o /dev/null -w '%{url_effective}' -A "PhotographicStudio-Installer" "$1" 2>> "$STUDIO_SETUPLOG")"
    elif command -v wget >/dev/null 2>&1; then
        eff="$(wget -S --spider --max-redirect=5 -U "PhotographicStudio-Installer" "$1" 2>&1 | awk '/^ *Location: / {l=$2} END {print l}')"
    fi
    eff="${eff%%[[:space:]]*}"
    case "$eff" in */releases/tag/*) echo "${eff##*/releases/tag/}" ;; *) return 1 ;; esac
}

# _studio_get_python APP: python-build-standalone into APP/python, checked against the release's SHA256SUMS.
# STUDIO_PYTHON_TARBALL=/path/file.tar.gz uses a local copy instead of downloading (tests, offline installs);
# STUDIO_PBS_BASE points the download somewhere else (tests).
_studio_get_python() {
    local app="$1" arch base tag name want got tmp
    tmp="$app/_installer"
    mkdir -p "$tmp"
    if [ -n "${STUDIO_PYTHON_TARBALL:-}" ]; then
        _studio_say "Using a private copy of Python from $STUDIO_PYTHON_TARBALL"
        cp "$STUDIO_PYTHON_TARBALL" "$tmp/python.tar.gz" || return 1
    else
        if ldd --version 2>&1 | grep -qi musl; then
            _studio_say "No private Python for this Linux: it uses musl (Alpine or similar), and the Studio's packages need glibc."
            return 1
        fi
        if ! arch="$(_studio_arch)"; then
            _studio_say "No private Python is made for this processor ($(uname -m))."
            return 1
        fi
        base="${STUDIO_PBS_BASE:-https://github.com/astral-sh/python-build-standalone}"
        _studio_say "Downloading a private copy of Python $STUDIO_PY_SERIES into this folder (about 30 MB, one time only)..."
        if ! tag="$(_studio_latest_tag "$base/releases/latest")" || [ -z "$tag" ]; then
            _studio_say "Could not reach GitHub for the private Python (no internet?)."
            return 1
        fi
        _studio_log "python-build-standalone release $tag"
        if ! _studio_fetch "$base/releases/download/$tag/SHA256SUMS" "$tmp/SHA256SUMS"; then
            _studio_say "Could not download the private Python's checksums."
            return 1
        fi
        # the list names every file of the release with its checksum: take the plain build for this processor
        name="$(grep -E -o "cpython-${STUDIO_PY_SERIES//./\\.}\.[0-9]+\+[0-9]+-${arch}-unknown-linux-gnu-install_only\.tar\.gz\$" "$tmp/SHA256SUMS" | sort -V | tail -n 1)"
        if [ -z "$name" ]; then
            _studio_say "Release $tag has no Python $STUDIO_PY_SERIES for $arch."
            rm -f "$tmp/SHA256SUMS"
            return 1
        fi
        want="$(awk -v n="$name" '$2 == n || $2 == "*" n {print $1; exit}' "$tmp/SHA256SUMS")"
        rm -f "$tmp/SHA256SUMS"
        _studio_log "python: $name (sha256 $want)"
        if ! _studio_fetch "$base/releases/download/$tag/${name//+/%2B}" "$tmp/python.tar.gz"; then
            _studio_say "The private Python download did not finish."
            return 1
        fi
        got="$(sha256sum "$tmp/python.tar.gz" | awk '{print $1}')"
        if [ -z "$want" ] || [ "$want" != "$got" ]; then
            _studio_say "The Python download does not match its published checksum; it was not used."
            _studio_log "sha256 wanted '$want', got '$got'"
            rm -f "$tmp/python.tar.gz"
            return 1
        fi
        _studio_log "sha256 checked: $got"
    fi
    rm -rf "$app/python"
    if ! tar -xzf "$tmp/python.tar.gz" -C "$app" >> "$STUDIO_SETUPLOG" 2>&1 || [ ! -x "$app/python/bin/python3" ]; then
        _studio_say "The private Python could not be unpacked."
        rm -rf "$app/python" "$tmp/python.tar.gz"
        return 1
    fi
    rm -f "$tmp/python.tar.gz"
    _studio_log "private Python: $("$app/python/bin/python3" -VV 2>&1 | head -n 1)"
    return 0
}

# _studio_offer_system_python: when neither works, offer the system's own Python with venv (needs sudo; asked).
_studio_offer_system_python() {
    local pm pkgs cmd tty=/dev/tty
    pm="$(_studio_pkg_manager)" || return 1
    [ "$pm" = rpm-ostree ] && return 1           # image-based: its own Python is always there, nothing to add
    case "$pm" in
        apt) pkgs="python3 python3-venv" ;;
        dnf) pkgs="python3" ;;
        zypper) pkgs="python3" ;;
        pacman) pkgs="python" ;;
    esac
    case "$pm" in
        apt) cmd="sudo apt-get install -y $pkgs" ;;
        dnf) cmd="sudo dnf install -y $pkgs" ;;
        zypper) cmd="sudo zypper install -y $pkgs" ;;
        pacman) cmd="sudo pacman -S --needed --noconfirm $pkgs" ;;
    esac
    { : < /dev/tty; } 2>/dev/null || tty=/dev/null
    [ "$(id -u)" = "0" ] && cmd="${cmd#sudo }"
    [ "$tty" = /dev/null ] && [ "$(id -u)" != "0" ] && return 1
    _studio_say "The Studio can use this PC's own Python instead; it needs: $pkgs"
    _studio_ask "Install it now? (runs: $cmd - it may ask for your password)" y || return 1
    _studio_log "running: $cmd"
    # shellcheck disable=SC2086
    $cmd < "$tty" 2>&1 | tee -a "$STUDIO_SETUPLOG"
    return 0
}

# _studio_make_venv APP: a venv of the PC's own Python 3.10+ (the first that can make one with pip in it).
_studio_make_venv() {
    local app="$1" cand
    for cand in python3.13 python3.12 python3.11 python3.10 python3 python3.14; do
        command -v "$cand" >/dev/null 2>&1 || continue
        if ! "$cand" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' >/dev/null 2>&1; then
            _studio_log "$cand is older than 3.10: skipped"
            continue
        fi
        _studio_log "creating .venv with $(command -v "$cand") ($("$cand" -V 2>&1))"
        rm -rf "$app/.venv"
        if "$cand" -m venv "$app/.venv" >> "$STUDIO_SETUPLOG" 2>&1 && "$app/.venv/bin/python" -m pip --version >> "$STUDIO_SETUPLOG" 2>&1; then
            return 0
        fi
        _studio_log "$cand could not make a venv with pip (on Debian and Ubuntu that needs the python3-venv package)"
        rm -rf "$app/.venv"
    done
    return 1
}

_studio_install_packages() {
    local app="$1" mark="$2"
    _studio_say "Installing the packages the Studio needs (numpy, scipy, rawpy, tifffile, Pillow, PySide6, pytest)..."
    _studio_say "This takes a minute or two the first time."
    _studio_log "pip install -r requirements.txt with $PY"
    "$PY" -m pip install --upgrade pip >> "$STUDIO_SETUPLOG" 2>&1
    if "$PY" -m pip install -r "$app/requirements.txt" >> "$STUDIO_SETUPLOG" 2>&1 && "$PY" -c "$STUDIO_IMPORTS" >> "$STUDIO_SETUPLOG" 2>&1; then
        cp "$app/requirements.txt" "$mark/requirements.installed"
        _studio_say "Packages installed."
        return 0
    fi
    _studio_log "package install failed with $PY"
    return 1
}

_studio_optional() {
    local mark="$1"
    _studio_optional_lensfun "$mark"
    _studio_optional_astro "$mark"
    return 0
}

_studio_optional_lensfun() {
    local mark="$1"
    [ -f "$mark/lensfun_tried.txt" ] && return 0
    "$PY" -c "import lensfunpy" >/dev/null 2>&1 && return 0
    _studio_say "Adding the lens database (lensfun, optional, one time only)..."
    "$PY" -m pip install lensfunpy >> "$STUDIO_SETUPLOG" 2>&1
    date > "$mark/lensfun_tried.txt"
    if "$PY" -c "import lensfunpy" >/dev/null 2>&1; then
        _studio_say "Lens database added."
    else
        _studio_say "The lens database could not be added. Everything else works; modern-lens profiles are unavailable."
    fi
    return 0
}

# astropy + photutils (BSD-3): FITS files and the astronomers' star finder. Optional, tried once.
_studio_optional_astro() {
    local mark="$1"
    [ -f "$mark/astro_tried.txt" ] && return 0
    "$PY" -c "import astropy.io.fits, photutils.detection" >/dev/null 2>&1 && return 0
    _studio_say "Adding the astronomy tools (astropy, optional, one time only)..."
    "$PY" -m pip install astropy photutils >> "$STUDIO_SETUPLOG" 2>&1
    date > "$mark/astro_tried.txt"
    if "$PY" -c "import astropy.io.fits" >/dev/null 2>&1; then
        _studio_say "Astronomy tools added."
    else
        _studio_say "The astronomy tools could not be added. Everything else works; FITS files are unavailable."
    fi
    return 0
}

# studio_env APP: sets PY. 0 when the Studio's Python is ready.
studio_env() {
    local app="$1" mark kind
    mkdir -p "$app/logs"
    STUDIO_SETUPLOG="$app/logs/setup.log"
    PY=""
    if [ -x "$app/python/bin/python3" ]; then
        PY="$app/python/bin/python3"; mark="$app/python"; kind=private
    elif [ -x "$app/.venv/bin/python" ]; then
        PY="$app/.venv/bin/python"; mark="$app/.venv"; kind=venv
    else
        _studio_say "First run: building a private Python environment inside this folder (one time only)..."
        if [ "${STUDIO_PRIVATE_PYTHON:-0}" != "1" ] && _studio_make_venv "$app"; then
            PY="$app/.venv/bin/python"; mark="$app/.venv"; kind=venv
        elif _studio_get_python "$app"; then
            PY="$app/python/bin/python3"; mark="$app/python"; kind=private
        elif _studio_offer_system_python && _studio_make_venv "$app"; then
            PY="$app/.venv/bin/python"; mark="$app/.venv"; kind=venv
        else
            _studio_say "PROBLEM: no Python could be set up. Details: logs/setup.log"
            _studio_say "Either install Python 3.10 or newer with venv (Debian, Ubuntu, Mint:"
            _studio_say "  sudo apt install python3 python3-venv), or check the internet connection, and run this again."
            return 1
        fi
    fi
    if "$PY" -c "$STUDIO_IMPORTS" >/dev/null 2>&1 && cmp -s "$app/requirements.txt" "$mark/requirements.installed"; then
        _studio_optional "$mark"
        return 0
    fi
    if ! _studio_install_packages "$app" "$mark"; then
        if [ "$kind" = "venv" ]; then
            # e.g. a Python too new for a package on this system: a private Python of a known-good version instead
            _studio_say "The packages would not install into this PC's Python; using a private copy of Python instead."
            rm -rf "$app/.venv"
            if _studio_get_python "$app"; then
                PY="$app/python/bin/python3"; mark="$app/python"
                _studio_install_packages "$app" "$mark" && { _studio_optional "$mark"; return 0; }
            fi
        fi
        _studio_say "PROBLEM: the package install did not finish."
        _studio_say "Check the internet connection and run this again. Details: logs/setup.log"
        return 1
    fi
    _studio_optional "$mark"
    return 0
}

# ------------------------------------------------------------------------------------------ the window's libraries
# _studio_pkg_for LIB MANAGER: the package that holds a library, per package manager (apt, dnf, zypper, pacman).
_studio_pkg_for() {
    local a="" d="" z="" p=""
    case "$1" in
        libxcb-cursor.so.0)      a=libxcb-cursor0;      d=xcb-util-cursor;      z=libxcb-cursor0;      p=xcb-util-cursor ;;
        libxcb-icccm.so.4)       a=libxcb-icccm4;       d=xcb-util-wm;          z=libxcb-icccm4;       p=xcb-util-wm ;;
        libxcb-image.so.0)       a=libxcb-image0;       d=xcb-util-image;       z=libxcb-image0;       p=xcb-util-image ;;
        libxcb-keysyms.so.1)     a=libxcb-keysyms1;     d=xcb-util-keysyms;     z=libxcb-keysyms1;     p=xcb-util-keysyms ;;
        libxcb-render-util.so.0) a=libxcb-render-util0; d=xcb-util-renderutil;  z=libxcb-render-util0; p=xcb-util-renderutil ;;
        libxcb-util.so.1)        a=libxcb-util1;        d=xcb-util;             z=libxcb-util1;        p=xcb-util ;;
        libxcb-xkb.so.1)         a=libxcb-xkb1;         d=libxcb;               z=libxcb-xkb1;         p=libxcb ;;
        libxcb-randr.so.0)       a=libxcb-randr0;       d=libxcb;               z=libxcb-randr0;       p=libxcb ;;
        libxcb-shape.so.0)       a=libxcb-shape0;       d=libxcb;               z=libxcb-shape0;       p=libxcb ;;
        libxcb-shm.so.0)         a=libxcb-shm0;         d=libxcb;               z=libxcb-shm0;         p=libxcb ;;
        libxcb-sync.so.1)        a=libxcb-sync1;        d=libxcb;               z=libxcb-sync1;        p=libxcb ;;
        libxcb-xfixes.so.0)      a=libxcb-xfixes0;      d=libxcb;               z=libxcb-xfixes0;      p=libxcb ;;
        libxcb-render.so.0)      a=libxcb-render0;      d=libxcb;               z=libxcb-render0;      p=libxcb ;;
        libxcb-xinput.so.0)      a=libxcb-xinput0;      d=libxcb;               z=libxcb-xinput0;      p=libxcb ;;
        libxcb.so.1)             a=libxcb1;             d=libxcb;               z=libxcb1;             p=libxcb ;;
        libX11-xcb.so.1)         a=libx11-xcb1;         d=libX11-xcb;           z=libX11-xcb1;         p=libx11 ;;
        libX11.so.6)             a=libx11-6;            d=libX11;               z=libX11-6;            p=libx11 ;;
        libxkbcommon-x11.so.0)   a=libxkbcommon-x11-0;  d=libxkbcommon-x11;     z=libxkbcommon-x11-0;  p=libxkbcommon-x11 ;;
        libxkbcommon.so.0)       a=libxkbcommon0;       d=libxkbcommon;         z=libxkbcommon0;       p=libxkbcommon ;;
        libEGL.so.1)             a=libegl1;             d=libglvnd-egl;         z=libEGL1;             p=libglvnd ;;
        libGL.so.1)              a=libgl1;              d=libglvnd-glx;         z=libGL1;              p=libglvnd ;;
        libOpenGL.so.0)          a=libopengl0;          d=libglvnd-opengl;      z=libOpenGL0;          p=libglvnd ;;
        libfontconfig.so.1)      a=libfontconfig1;      d=fontconfig;           z=libfontconfig1;      p=fontconfig ;;
        libfreetype.so.6)        a=libfreetype6;        d=freetype;             z=libfreetype6;        p=freetype2 ;;
        libdbus-1.so.3)          a=libdbus-1-3;         d=dbus-libs;            z=libdbus-1-3;         p=dbus ;;
        libglib-2.0.so.0|libgthread-2.0.so.0)
                                 a=libglib2.0-0;        d=glib2;                z=libglib-2_0-0;       p=glib2 ;;
        libwayland-client.so.0)  a=libwayland-client0;  d=libwayland-client;    z=libwayland-client0;  p=wayland ;;
        libwayland-cursor.so.0)  a=libwayland-cursor0;  d=libwayland-cursor;    z=libwayland-cursor0;  p=wayland ;;
        libwayland-egl.so.1)     a=libwayland-egl1;     d=libwayland-egl;       z=libwayland-egl1;     p=wayland ;;
        libSM.so.6)              a=libsm6;              d=libSM;                z=libSM6;              p=libsm ;;
        libICE.so.6)             a=libice6;             d=libICE;               z=libICE6;             p=libice ;;
        *) return 1 ;;
    esac
    case "$2" in apt) echo "$a" ;; dnf|rpm-ostree) echo "$d" ;; zypper) echo "$z" ;; pacman) echo "$p" ;; *) return 1 ;; esac
}

# Fedora Atomic systems (Bazzite, Silverblue, Kinoite, Aurora, Bluefin): the system is a read-only image; packages
# are layered with rpm-ostree and take effect after a reboot (Bazzite's dnf refuses to change the host).
_studio_ostree() { [ -e /run/ostree-booted ] || [ "${STUDIO_FAKE_OSTREE:-0}" = "1" ]; }

_studio_pkg_manager() {
    if _studio_ostree && command -v rpm-ostree >/dev/null 2>&1; then echo rpm-ostree
    elif _studio_ostree; then return 1
    elif command -v apt-get >/dev/null 2>&1; then echo apt
    elif command -v dnf >/dev/null 2>&1; then echo dnf
    elif command -v zypper >/dev/null 2>&1; then echo zypper
    elif command -v pacman >/dev/null 2>&1; then echo pacman
    else return 1
    fi
}

_studio_missing_libs() {
    local qt f
    qt="$("$PY" -c 'import os, PySide6; print(os.path.join(os.path.dirname(PySide6.__file__), "Qt"))' 2>/dev/null)" || return 0
    for f in "$qt"/plugins/platforms/libqxcb.so "$qt"/plugins/platforms/libqwayland*.so \
             "$qt"/lib/libQt6Gui.so.6 "$qt"/lib/libQt6Widgets.so.6 "$qt"/lib/libQt6XcbQpa.so.6; do
        [ -e "$f" ] && ldd "$f" 2>/dev/null
    done | awk '/not found/ {print $1}' | sort -u
}

# studio_qt_check APP: 0 when nothing is missing (or it was installed).
studio_qt_check() {
    local app="$1" missing pm pkgs="" unknown="" lib pkg cmd
    STUDIO_SETUPLOG="$app/logs/setup.log"
    missing="$(_studio_missing_libs)"
    if [ -z "$missing" ]; then
        _studio_log "window libraries: nothing missing"
        return 0
    fi
    _studio_log "window libraries missing: $(echo "$missing" | tr '\n' ' ')"
    pm="$(_studio_pkg_manager)" || pm=""
    for lib in $missing; do
        if [ -n "$pm" ] && pkg="$(_studio_pkg_for "$lib" "$pm")"; then
            case " $pkgs " in *" $pkg "*) ;; *) pkgs="$pkgs $pkg" ;; esac
        else
            unknown="$unknown $lib"
        fi
    done
    pkgs="${pkgs# }"
    echo
    _studio_say "The Studio's window needs a few system libraries this PC does not have yet:"
    _studio_say "  $(echo "$missing" | tr '\n' ' ')"
    if [ -n "$pkgs" ]; then
        case "$pm" in
            apt) cmd="sudo apt-get install -y $pkgs" ;;
            dnf) cmd="sudo dnf install -y $pkgs" ;;
            zypper) cmd="sudo zypper install -y $pkgs" ;;
            pacman) cmd="sudo pacman -S --needed --noconfirm $pkgs" ;;
            rpm-ostree) cmd="sudo rpm-ostree install $pkgs" ;;
        esac
        _studio_say "They come with your system's packages: $pkgs"
        if [ "$pm" = rpm-ostree ]; then
            # an image-based system: a layer on the system image, used after a reboot. Bazzite asks for few layers,
            # so this is never done without a yes.
            _studio_say "This system is image-based (Bazzite / Fedora Atomic): they would be added as a layer with"
            _studio_say "  $cmd"
            _studio_say "and are used after a restart of the PC."
            if _studio_ask "Add them now? (it may ask for your password; restart the PC afterwards)" n; then
                _studio_log "running: $cmd"
                # shellcheck disable=SC2086
                $cmd < /dev/tty 2>&1 | tee -a "$STUDIO_SETUPLOG"
                _studio_say "Restart the PC, then start the Studio."
            else
                _studio_say "Skipped. Run that command yourself if the Studio's window does not open."
            fi
            return 1
        fi
        local def=y tty=/dev/tty
        { : < /dev/tty; } 2>/dev/null || tty=/dev/null
        if [ "$(id -u)" = "0" ]; then
            cmd="${cmd#sudo }"
        elif [ "$tty" = /dev/null ]; then
            def=n                                  # sudo would have nowhere to ask for the password
        fi
        if _studio_ask "Install them now? (runs: $cmd - it may ask for your password)" "$def"; then
            _studio_log "running: $cmd"
            # shellcheck disable=SC2086
            $cmd < "$tty" 2>&1 | tee -a "$STUDIO_SETUPLOG"
            missing="$(_studio_missing_libs)"
            if [ -n "$missing" ] && [ "$pm" = apt ]; then
                # a package list that was never fetched (fresh systems) does not know the package yet
                _studio_say "Refreshing the package list and trying again..."
                local upd="sudo apt-get update"
                [ "$(id -u)" = "0" ] && upd="apt-get update"
                # shellcheck disable=SC2086
                $upd < "$tty" 2>&1 | tail -n 3 | tee -a "$STUDIO_SETUPLOG"
                # shellcheck disable=SC2086
                $cmd < "$tty" 2>&1 | tee -a "$STUDIO_SETUPLOG"
                missing="$(_studio_missing_libs)"
            fi
            if [ -z "$missing" ]; then
                _studio_say "System libraries installed."
                return 0
            fi
            _studio_say "Still missing: $(echo "$missing" | tr '\n' ' ')"
        else
            _studio_say "Skipped. To add them later, run:"
            _studio_say "  $cmd"
        fi
    fi
    if [ -n "$unknown" ]; then
        _studio_say "Install the package that provides each of these with your system's software manager:"
        _studio_say " $unknown"
    fi
    return 1
}

# studio_qt_probe: opens Qt for a moment (no window shown) to prove the Studio's window can start here.
studio_qt_probe() {
    if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
        _studio_log "no desktop session (no DISPLAY or WAYLAND_DISPLAY): window check skipped"
        return 2
    fi
    local out
    if out="$(timeout 60 "$PY" -c 'import sys
from PySide6.QtWidgets import QApplication
app = QApplication(sys.argv[:1])
print(app.platformName())' 2>&1)"; then
        _studio_log "window check: Qt started on '$(echo "$out" | tail -n 1)'"
        return 0
    fi
    _studio_log "window check failed:"
    printf '%s\n' "$out" >> "$STUDIO_SETUPLOG"
    return 1
}
