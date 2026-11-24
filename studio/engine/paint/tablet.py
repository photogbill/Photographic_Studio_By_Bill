"""What the pen reports, and a plain-words diagnosis (BUILD_PLAN_PAINT 0.2, P0.1).

The Tablet check panel feeds every pen and mouse event here. ``diagnose()`` turns what was seen into short sentences:
what works, what is missing, and what to do. It is written first for a Huion HS64 on Linux (no tilt, 8192 levels,
the kernel's own hid-uclogic driver, Linux 5.1 or newer), and covers Windows too.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class PenStats:
    tablet_events: int = 0
    mouse_events: int = 0          # mouse events while no pen event came (the pen acting as a mouse)
    pressures: list = field(default_factory=list)
    tilts: int = 0                 # events with a non-zero tilt
    rotations: int = 0
    eraser_seen: bool = False
    device: str = ""
    times: list = field(default_factory=list)
    keys: list = field(default_factory=list)

    def add_tablet(self, pressure: float, tilt_x: float, tilt_y: float, t: float, eraser: bool = False,
                   device: str = "", rotation: float = 0.0) -> None:
        self.tablet_events += 1
        self.pressures.append(float(pressure))
        if abs(tilt_x) > 0.01 or abs(tilt_y) > 0.01:
            self.tilts += 1
        if abs(rotation) > 0.01:
            self.rotations += 1
        self.eraser_seen |= bool(eraser)
        if device:
            self.device = device
        self.times.append(float(t))
        if len(self.pressures) > 5000:
            self.pressures = self.pressures[-5000:]
            self.times = self.times[-5000:]

    def add_mouse(self) -> None:
        self.mouse_events += 1

    def add_key(self, name: str) -> None:
        self.keys = (self.keys + [name])[-8:]

    def rate(self) -> float:
        """Pen reports per second while the pen moves (gaps over 0.1 s are pauses, not slow reports)."""
        if len(self.times) < 10:
            return 0.0
        gaps = [b - a for a, b in zip(self.times, self.times[1:]) if 0 < b - a < 0.1]
        return 1.0 / (sum(gaps) / len(gaps)) if gaps else 0.0

    def pressure_levels(self) -> int:
        """Distinct pressure values seen (a working pen shows many; a pen without pressure shows one or two)."""
        return len({round(p, 5) for p in self.pressures if p > 0})


def diagnose(st: PenStats, platform: str = "linux") -> list[tuple[str, str]]:
    """Returns (level, text) pairs; level is 'ok', 'warn' or 'bad' (shown green, amber, red)."""
    out: list = []
    linux = platform.startswith("linux")
    if st.tablet_events == 0:
        if st.mouse_events == 0:
            return [("warn", "Draw a few strokes in the box above with the pen, pressing lightly, then hard.")]
        out.append(("bad", "Only mouse events have arrived: the pen is moving the pointer like a mouse, so Paint "
                           "gets no pressure."))
        if linux:
            out += [
                ("warn", "Unplug the tablet, wait five seconds and plug it back in, then try again."),
                ("warn", "Check that the desktop sees it as a tablet: KDE System Settings > Drawing Tablet, or "
                         "GNOME Settings > Wacom Tablet. If it is not listed there, the system is not using it as a "
                         "tablet yet."),
                ("warn", "The Huion HS64 needs no Huion driver on Linux: the kernel's own driver (hid-uclogic) has "
                         "supported it since Linux 5.1. If Huion's own Linux driver is installed, remove it: the two "
                         "fight over the pen."),
                ("warn", "On a very old system (Linux older than 5.1) update the system, or install the DIGImend "
                         "drivers."),
            ]
        else:
            out += [
                ("warn", "Unplug the tablet and plug it back in, then try again."),
                ("warn", "Open the tablet's own settings program and turn on Windows Ink for this program (Huion: "
                         "'Enable Windows Ink'). Paint reads Windows Ink and WinTab."),
                ("warn", "If two tablet drivers are installed (an old Wacom driver and a Huion one), remove the "
                         "one you do not use."),
            ]
        return out

    levels = st.pressure_levels()
    ps = [p for p in st.pressures if p > 0]
    if levels <= 2:
        out.append(("bad", "The pen is seen, but its pressure is not arriving (always "
                           f"{ps[0]:.2f})." if ps else "The pen is seen, but its pressure is always zero."))
        if linux:
            out.append(("warn", "Check the pressure in the desktop's tablet settings (its test area). If it works "
                                "there but not here, start Paint from the start-paint.sh launcher, which uses the "
                                "desktop's own tablet support."))
        else:
            out.append(("warn", "Turn on Windows Ink in the tablet's settings program, or turn it off if it is on "
                                "(some drivers give pressure only one way), then restart Paint."))
    else:
        out.append(("ok", f"Pressure arrives: {levels} different levels seen so far."))
        if max(ps) < 0.65:
            out.append(("warn", f"Your hardest press so far reached {max(ps):.0%}. Press harder once, or choose "
                                "the 'Soft' pressure curve so a lighter hand gets the full range."))
        if min(ps) > 0.3 and len(ps) > 50:
            out.append(("warn", f"Your lightest touch is already {min(ps):.0%}: thin lines may be hard. The "
                                "'Firm' pressure curve gives more control at the light end."))
    if st.tilts:
        out.append(("ok", "Tilt arrives (brushes can follow the pen's lean)."))
    else:
        out.append(("ok", "No tilt reported. That is normal for pens without tilt, such as the Huion HS64: "
                          "brushes that use tilt simply stay upright."))
    out.append(("ok" if st.eraser_seen else "warn",
                "The pen's eraser end works." if st.eraser_seen else
                "Eraser end not seen yet (turn the pen over to try it; many pens, the HS64 included, have none). "
                "Press E to switch to the eraser instead."))
    r = st.rate()
    if r:
        if r < 100:
            out.append(("warn", f"The pen reports {r:.0f} times a second, which is low (the HS64 can do 233 or "
                                "more). Use a USB port on the computer itself rather than a hub."))
        else:
            out.append(("ok", f"The pen reports {r:.0f} times a second."))
    if st.keys:
        out.append(("ok", "Keys arrive: " + ", ".join(st.keys) + ". Tablet press keys send ordinary keys; set "
                          "them in the desktop's tablet settings."))
    return out


def tilt_angle(tilt_x: float, tilt_y: float) -> tuple[float, float]:
    """(lean in degrees from upright, direction in degrees) from Qt's x / y tilt."""
    lean = min(90.0, math.hypot(tilt_x, tilt_y))
    return lean, math.degrees(math.atan2(tilt_y, tilt_x)) if lean else 0.0
