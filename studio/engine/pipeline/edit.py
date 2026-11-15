"""The Edit: every parameter that turns a raw into a picture. Serialised into the sidecar (section 5).

Each stage has a version number stored with the edit, so an old sidecar renders the same after an update
(determinism rule 2.3). When a stage's maths changes in a way that changes pixels, bump its version and
keep the old code path reachable for old sidecars.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field

# Bump a number when that stage's output changes for the same parameters.
STAGE_VERSIONS = {
    "linearize": 1,
    "banding": 1,
    "lens_ca": 1,
    "defringe": 1,
    "denoise": 1,
    "white_balance": 1,
    "highlights": 1,
    "demosaic": 1,
    "camera_matrix": 1,
    "exposure": 1,
    "tone": 1,
    "gamut": 1,
    "output": 1,
    "sharpen": 1,
    "falloff": 1,
    "geometry": 1,
    "microcontrast": 1,
    "light": 1,
    "smart_lighting": 1,
    "colour": 1,
    "dehaze": 1,
    "texture": 1,
    "grain": 1,
    "profile": 1,
    "masks": 1,
    "retouch": 1,
    "glow": 1,
    "portrait": 1,
}


@dataclass
class WhiteBalance:
    mode: str = "as_shot"            # as_shot | daylight | temp_tint | multipliers
    temperature: float = 5500.0
    tint: float = 0.0
    multipliers: list | None = None  # [R, G, B] with G = 1 (mode 'multipliers')


@dataclass
class Tone:
    mode: str = "standard"           # standard | camera_match | linear
    contrast: float = 1.35
    mid_grey: float = 0.11           # raw-relative value rendered as 18% grey (baseline exposure)
    shoulder: float = 1.0
    white: float = 3.0               # stops above mid grey where the output reaches white
    toe: float = 0.0                 # -100..100 film toe: + opens the deepest shadows softly, - crushes them
    match_camera_brightness: bool = False   # OFF by default (Bill: no automatic brightness changes unless
                                            # turned on). On: per-body offset so mid grey matches the camera JPEG.


@dataclass
class Denoise:
    enabled: bool = False            # off by default until Bill's blind test (Phase 3 acceptance)
    method: str = "hybrid"           # hybrid (NLM luminance + starlet colour) | starlet | nlm | mppca
    luma: float = 1.0                # luminance strength (1 = matched to the measured noise)
    chroma: float = 1.5              # colour-noise strength
    grain: float = 0.0               # keep-grain: fraction of removed luminance noise put back


@dataclass
class LensFix:
    auto_ca: bool = True             # lateral colour fringing, measured from the photo (no profile needed)
    defringe: bool = True            # purple / green fringe removal: ON by default (plan decision 0.4)
    defringe_strength: float = 1.0
    vignette: bool = False           # vignette correction OFF by default (Bill likes vignetting)
    vignette_amount: float = 1.0     # fraction of the measured falloff removed, in stops (1 = all of it)
    vignette_model: dict | None = None   # frozen falloff model (lensfun / estimated / lens wizard), see falloff.py
    tag: str = ""                    # lens picked by the user for contactless glass ("" = what the camera recorded)


@dataclass
class Geometry:
    """Stage 13, all in the upright (as displayed) frame. Defaults change nothing."""
    distortion: dict | None = None   # frozen lens distortion model (lensfun or measured from straight lines)
    distortion_amount: float = 1.0   # fraction of that model's correction applied
    distortion_manual: float = 0.0   # slider -100..100: + fixes barrel, - fixes pincushion
    rotate: float = 0.0              # straighten, degrees, + = counter-clockwise
    vertical: float = 0.0            # perspective -100..100: + widens the top (leaning-back buildings)
    horizontal: float = 0.0          # perspective -100..100: + makes the right side taller
    crop: list | None = None         # [x0, y0, x1, y1] fractions of the frame left after "constrain"; None = all
    aspect: str = "original"         # remembered crop aspect for the interface (original | free | 3:2 | 4:5 ...)
    constrain: bool = True           # zoom in just enough that warps leave no empty corners


@dataclass
class Sharpen:
    """Stage 12, capture sharpening by deconvolution ("Recover lens sharpness")."""
    enabled: bool = False            # off by default: a look decision for Bill (vintage character lenses)
    amount: float = 0.5              # 0..1.5: how far toward the 0.5 px target the measured blur is undone
    radius: float = 0.0              # blur sigma in px; 0 = measured from this photo
    threshold: float = 1.0           # noise gate: higher sharpens only crisp edges


@dataclass
class Microcontrast:
    """Stage 16: more shades of shadow (tonal separation), not edge sharpening. See detail/microcontrast.py."""
    amount: float = 0.0              # 0..1
    shadows: float = 1.0
    midtones: float = 0.7
    highlights: float = 0.3
    scale: float = 0.0               # -1 finer .. +1 broader
    range_ev: float = 2.0            # differences up to this many stops are "shades"; larger ones are edges


@dataclass
class Light:
    """Stage 15 controls beyond exposure and contrast (color/light.py). Defaults change nothing."""
    highlights: float = 0.0          # -100..100: - brings back bright areas, + brightens them
    shadows: float = 0.0             # -100..100: + lifts the shadows along the tone curve
    whites: float = 0.0              # -100..100: moves the white point (+ = brighter top end)
    blacks: float = 0.0              # -100..100: - deepens the blacks, + lifts them into a matte
    smart: float = 0.0               # 0..100 Smart lighting: local shadow lift that keeps detail (edge-aware)
    curve: dict | None = None        # point curves {"master"|"r"|"g"|"b": [[x, y], ...]}, 0..1 display values
    param: dict | None = None        # parametric curve {"highlights"|"lights"|"darks"|"shadows": -100..100}
    dehaze: float = 0.0              # -100..100: + removes haze (dark-channel prior), - adds a soft haze


@dataclass
class Colour:
    """Stage 17 (color/grade.py), in OKLCh. Defaults change nothing."""
    saturation: float = 0.0          # -100..100 (-100 = greyscale)
    vibrance: float = 0.0            # -100..100: muted colours first, skin protected
    hsl: dict | None = None          # {"red": [hue, sat, lum], ...} 8 named bands, each -100..100
    ranges: list | None = None       # colour wheel: [{"hue", "width", "falloff", "shift", "sat", "lum"}]
    grading: dict | None = None      # {"shadows": [hue, strength, lum], "midtones": .., "highlights": ..,
                                     #  "global": .., "balance": -100..100, "blending": 0..100}
    bw: bool = False                 # black and white
    bw_mix: dict | None = None       # {"red": -100..100, ...}: how bright each colour turns in B&W
    lut: str = ""                    # a .cube look (profiles/looks/luts or settings/luts)
    lut_amount: float = 1.0


@dataclass
class Finish:
    """Texture (stage 16) and film grain (end of stage 17), detail/finish.py. Defaults change nothing."""
    texture: float = 0.0             # -100..100: fine detail up (+) or smoothed (-), never edges
    grain: float = 0.0               # 0..100
    grain_size: float = 1.0          # grain width in full-size pixels (0.5 fine .. 3 coarse)
    grain_roughness: float = 50.0    # 0..100: even .. clumpy
    grain_colour: bool = False       # False = monochrome grain (like B&W film)


@dataclass
class Edit:
    exposure_ev: float = 0.0
    white_balance: WhiteBalance = field(default_factory=WhiteBalance)
    highlight_mode: str = "clip"     # clip (neutral) | rebuild (colour from the edges of clipped areas) |
                                     # glow (measured from the lens glow, raw/glow.py) | none
    glow_profile: str = ""           # lens glow profile for "glow" (settings/glow/<name>.json); "" = from the photo only
    profile: str = ""                # camera profile (.dcp file name); "" = the camera matrix (LibRaw / Adobe DNG)
    demosaic: str = "menon"          # menon | mhc | bilinear | half
    tone: Tone = field(default_factory=Tone)
    denoise: Denoise = field(default_factory=Denoise)
    lens: LensFix = field(default_factory=LensFix)
    geometry: Geometry = field(default_factory=Geometry)
    sharpen: Sharpen = field(default_factory=Sharpen)
    microcontrast: Microcontrast = field(default_factory=Microcontrast)
    light: Light = field(default_factory=Light)
    colour: Colour = field(default_factory=Colour)
    finish: Finish = field(default_factory=Finish)
    masks: list | None = None        # local adjustments (local/masks.py): shapes attached to the photo + settings
    retouch: list | None = None      # heal / clone / red-eye spots (local/retouch.py), attached to the photo
    portrait: dict | None = None     # faces, face sliders and body sculpting (local/portrait.py), attached to the photo
    use_optical_black: bool = True
    banding: bool = True             # row banding from the masked border, Wiener-gated (never adds bands)
    gamut_compress: bool = True      # smooth gamut compression before the tone curve (gamut.py)
    stage_versions: dict = field(default_factory=lambda: dict(STAGE_VERSIONS))

    # ------------------------------------------------------------------------------- (de)serialisation
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Edit":
        d = copy.deepcopy(d or {})
        wb = WhiteBalance(**{k: v for k, v in d.pop("white_balance", {}).items()
                             if k in WhiteBalance.__dataclass_fields__})
        tone = Tone(**{k: v for k, v in d.pop("tone", {}).items() if k in Tone.__dataclass_fields__})
        dn = Denoise(**{k: v for k, v in d.pop("denoise", {}).items() if k in Denoise.__dataclass_fields__})
        ln = LensFix(**{k: v for k, v in d.pop("lens", {}).items() if k in LensFix.__dataclass_fields__})
        ge = Geometry(**{k: v for k, v in (d.pop("geometry", None) or {}).items() if k in Geometry.__dataclass_fields__})
        sh = Sharpen(**{k: v for k, v in (d.pop("sharpen", None) or {}).items() if k in Sharpen.__dataclass_fields__})
        mc = Microcontrast(**{k: v for k, v in (d.pop("microcontrast", None) or {}).items()
                              if k in Microcontrast.__dataclass_fields__})
        li = Light(**{k: v for k, v in (d.pop("light", None) or {}).items() if k in Light.__dataclass_fields__})
        co = Colour(**{k: v for k, v in (d.pop("colour", None) or {}).items() if k in Colour.__dataclass_fields__})
        fi = Finish(**{k: v for k, v in (d.pop("finish", None) or {}).items() if k in Finish.__dataclass_fields__})
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        e = cls(white_balance=wb, tone=tone, denoise=dn, lens=ln, geometry=ge, sharpen=sh, microcontrast=mc,
                light=li, colour=co, finish=fi, **known)
        # stages missing from an old sidecar are version 1
        for k in STAGE_VERSIONS:
            e.stage_versions.setdefault(k, 1)
        return e

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "Edit":
        return cls.from_dict(json.loads(s))

    def raw_key(self) -> str:
        """Cache key for everything up to and including the camera matrix (stages 1-11)."""
        d = self.to_dict()
        keep = {k: d[k] for k in ("white_balance", "highlight_mode", "demosaic", "use_optical_black", "denoise", "banding",
                                          "lens", "sharpen", "profile")}
        if d.get("highlight_mode") == "glow":
            keep["glow_profile"] = d.get("glow_profile", "")
            keep["glow_v"] = d["stage_versions"].get("glow", 1)
        keep["lens"] = {k: v for k, v in keep["lens"].items() if k != "tag"}      # the tag changes no pixels
        if not keep["sharpen"]["enabled"]:
            keep["sharpen"] = False
        keep["v"] = {k: d["stage_versions"][k] for k in
                     ("linearize", "banding", "denoise", "lens_ca", "defringe", "white_balance", "highlights", "demosaic",
                      "camera_matrix", "sharpen", "falloff", "profile")}
        return json.dumps(keep, sort_keys=True)
