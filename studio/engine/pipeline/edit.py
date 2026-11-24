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
    "denoise": 4,                    # 2: the safeguard exists (version 1 sidecars render without it)
    #                                  3: the detail guard exists (older sidecars render without it)
    #                                  4: automatic strength and shadow-lift awareness exist (likewise)
    "white_balance": 1,
    "highlights": 1,
    "demosaic": 1,
    "camera_matrix": 2,              # 2: two-light (tungsten + daylight) matrices when the camera has them
    #                                  (color/spectral.py); version 1 sidecars keep the daylight-only matrix
    "exposure": 1,
    "tone": 3,                       # 2: photos below the lowest native ISO (Sony ISO 50/64/80, Canon L, Nikon Lo)
    #                                  are darkened as the camera does (tone.extended_low_iso_ev)
    #                                  3: curve colour "natural" by default (pipeline.natural_curve)
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
    "detail": 1,
    "hdr": 1,
    "film": 1,                       # film negatives -> positives (local/film.py)
    "halftone": 1,                   # printing dots on scans of printed photos (local/halftone.py)
    "stains": 1,                     # stains and corrosion, unmixed (local/stains.py, the Plate Lab)
    "scratches": 1,                  # dust and scratches on scans and old photos (local/scratches.py)
    "depth": 1,                      # depth from one photo: haze by distance (local/depth.py)
    "jpeg": 1,
    "relight": 1,                    # the light moved, from the depth map's surfaces (local/relight.py)
    "composite": 1,                  # subjects from other photos laid on the finished picture (composite.py)                       # JPEG repair inside the file's quantisation intervals (local/jpegfix.py)
    "colourize": 2,                  # black and white -> colour (local/colourize.py); 2: faded tones evened first
    "infrared": 1,                   # infrared / full spectrum: IR matrix step, mono raw, palettes (color/infrared.py)
    "flare": 1,                      # lens-flare ghosts subtracted (local/flare.py)
}


@dataclass
class WhiteBalance:
    mode: str = "as_shot"            # as_shot | daylight | temp_tint | multipliers
    temperature: float = 5500.0
    tint: float = 0.0
    multipliers: list | None = None  # [R, G, B] with G = 1 (mode 'multipliers')


@dataclass
class Tone:
    mode: str = "standard"           # standard | camera_match | linear | lightroom (pipeline.lightroom_tone)
    contrast: float = 1.35
    mid_grey: float = 0.11           # raw-relative value rendered as 18% grey (baseline exposure)
    shoulder: float = 1.0
    white: float = 3.0               # stops above mid grey where the output reaches white
    toe: float = 0.0                 # -100..100 film toe: + opens the deepest shadows softly, - crushes them
    curve_colour: str = ""           # "" = by tone stage version | "natural" (keeps colours) | "rich" (per channel)
    match_camera_brightness: bool = False   # OFF by default (Bill: no automatic brightness changes unless
                                            # turned on). On: per-body offset so mid grey matches the camera JPEG.


@dataclass
class Denoise:
    enabled: bool = False            # off by default until Bill's blind test (Phase 3 acceptance)
    method: str = "hybrid"           # hybrid (NLM luminance + starlet colour) | starlet | nlm | mppca |
                                     # learned (this camera's own trained network, when there is one)
    luma: float = 1.0                # luminance strength (1 = matched to the measured noise)
    chroma: float = 1.5              # colour-noise strength
    grain: float = 0.0               # keep-grain: fraction of removed luminance noise put back
    auto: bool = True                # strength chosen from the photo itself (raw/auto_strength.py); luma and
                                     # chroma above are then not used. Off = the sliders decide.
    lean: float = 1.0                # with auto: the photo's own choice times this (1 = as chosen)
    lift: bool = True                # clean harder what the edit lifts out of the shadows
    guard: bool = True               # the detail guard (BUILD_PLAN_NOISE E): whatever Clean removed that is not
                                     # noise is given back, and nothing may be added that noise could not hide
    safeguard: bool = True           # "restore" step (23.12): a photosite the denoiser moved by more than 4.5x
                                     # its noise was not noise (a star, a spark): the original is kept there


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
class Detail:
    """Stage 16, "Detail by scale" (detail/scales.py, plan 23.11): every microcontrast method with its own
    amount, stackable. Percent. "Local contrast" is the first engine and keeps its place in ``microcontrast``
    (amount 0..1, which may now also go down to -0.5); its shadows / midtones / highlights weights are shared by
    every method. Defaults change nothing."""
    tonal: float = 0.0               # 0..100: medium scales up, the finest slightly relaxed (Bill's meaning)
    crisp: float = 0.0               # -50..100: the finest scale up; negative softens gently
    separation: float = 0.0          # 0..100: crowded tones spread apart, region by region
    veil: float = 0.0                # -100..100: + takes off the stray-light haze floor, - adds a soft haze
    broad: float = 0.0               # -50..100: large shapes, overall punch
    off: list | None = None          # methods switched off without losing their amounts (may hold "local")


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
class AutoHDR:
    """Auto HDR on one raw (color/hdr.py, plan 23.6). Off until the user turns it on: it changes brightness."""
    enabled: bool = False
    strength: str = "medium"         # light | medium | heavy | highlights | lowkey | shadows
    amount: float = 100.0            # 0..150 percent of that strength
    engine: str = "fusion"           # fusion (exposure fusion) | laplacian (local Laplacian) | retinex (MSR)


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
    demosaic: str = "rcd"            # rcd (default from 2026-11-15) | menon | lmmse | dual | mhc | bilinear | half
                                     # (sidecars store their algorithm, so older edits keep "menon")
    green_eq: bool = False           # even out the two greens of each 2x2 quad before demosaicing (plan 23.8)
    tone: Tone = field(default_factory=Tone)
    denoise: Denoise = field(default_factory=Denoise)
    lens: LensFix = field(default_factory=LensFix)
    geometry: Geometry = field(default_factory=Geometry)
    sharpen: Sharpen = field(default_factory=Sharpen)
    microcontrast: Microcontrast = field(default_factory=Microcontrast)
    detail: Detail = field(default_factory=Detail)
    hdr: AutoHDR = field(default_factory=AutoHDR)
    light: Light = field(default_factory=Light)
    colour: Colour = field(default_factory=Colour)
    finish: Finish = field(default_factory=Finish)
    masks: list | None = None        # local adjustments (local/masks.py): shapes attached to the photo + settings
    retouch: list | None = None      # heal / clone / red-eye spots (local/retouch.py), attached to the photo
    portrait: dict | None = None     # faces, face sliders and body sculpting (local/portrait.py), attached to the photo
    film: dict | None = None         # film negative -> positive (local/film.py): {"kind", "base", "dmax", ...}
    halftone: dict | None = None     # printing dots (local/halftone.py): {"on", "amount"}
    stains: dict | None = None       # stains and corrosion (local/stains.py): {"on", "amount", "sensitivity"}
    scratches: dict | None = None    # dust and scratch removal (local/scratches.py): {"on", "amount", ...}
    colourize: dict | None = None    # black and white -> colour (local/colourize.py): {"on", "amount", ...}
    depth: dict | None = None        # depth from one photo (local/depth.py): {"haze"}
    jpeg: dict | None = None         # JPEG repair (local/jpegfix.py): {"on"}
    relight: dict | None = None      # move the light (local/relight.py): {"amount", "azimuth", "elevation", "soft"}
    composite: list | None = None    # subjects from other photos (pipeline/composite.py): [{"src", "mask", "x", ...}]
    flare: dict | None = None        # lens-flare ghosts (local/flare.py): {"on", "amount", "ghosts": [...], "text"}
    infrared: dict | None = None     # infrared / full spectrum (color/infrared.py): filter, IR matrix step, B&W from
                                     # the channels, mono raw, palette, hot spot; set in the Infrared tab
    use_optical_black: bool = True
    banding: bool = True             # row banding from the masked border, Wiener-gated (never adds bands)
    calibrated: bool = True          # take out what this camera's calibration measured (raw/calibrated.py): hot
                                     # and warm photosites that show, the fixed banding, the black offset. Does
                                     # nothing for a camera with no profile.
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
        dn_d = d.pop("denoise", {})
        dn = Denoise(**{k: v for k, v in dn_d.items() if k in Denoise.__dataclass_fields__})
        if "safeguard" not in dn_d and (d.get("stage_versions") or {}).get("denoise", 2) < 2:
            dn.safeguard = False     # written before the safeguard existed: renders exactly as it did
        if "guard" not in dn_d and (d.get("stage_versions") or {}).get("denoise", 3) < 3:
            dn.guard = False         # written before the detail guard existed: renders exactly as it did
        if (d.get("stage_versions") or {}).get("denoise", 4) < 4:
            dn.auto = bool(dn_d.get("auto", False))     # written before these existed: renders exactly as it did
            dn.lift = bool(dn_d.get("lift", False))
        if "calibrated" not in d and "stage_versions" in d:
            d["calibrated"] = False  # a sidecar written before the calibration was used: renders exactly as it did
        ln = LensFix(**{k: v for k, v in d.pop("lens", {}).items() if k in LensFix.__dataclass_fields__})
        ge = Geometry(**{k: v for k, v in (d.pop("geometry", None) or {}).items() if k in Geometry.__dataclass_fields__})
        sh = Sharpen(**{k: v for k, v in (d.pop("sharpen", None) or {}).items() if k in Sharpen.__dataclass_fields__})
        mc = Microcontrast(**{k: v for k, v in (d.pop("microcontrast", None) or {}).items()
                              if k in Microcontrast.__dataclass_fields__})
        de = Detail(**{k: v for k, v in (d.pop("detail", None) or {}).items() if k in Detail.__dataclass_fields__})
        hd = AutoHDR(**{k: v for k, v in (d.pop("hdr", None) or {}).items() if k in AutoHDR.__dataclass_fields__})
        li = Light(**{k: v for k, v in (d.pop("light", None) or {}).items() if k in Light.__dataclass_fields__})
        co = Colour(**{k: v for k, v in (d.pop("colour", None) or {}).items() if k in Colour.__dataclass_fields__})
        fi = Finish(**{k: v for k, v in (d.pop("finish", None) or {}).items() if k in Finish.__dataclass_fields__})
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        e = cls(white_balance=wb, tone=tone, denoise=dn, lens=ln, geometry=ge, sharpen=sh, microcontrast=mc,
                detail=de, hdr=hd, light=li, colour=co, finish=fi, **known)
        # stages missing from an old sidecar are version 1
        for k in STAGE_VERSIONS:
            e.stage_versions.setdefault(k, 1)
        return e

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "Edit":
        return cls.from_dict(json.loads(s))

    def shadow_lift(self) -> float:
        """Roughly how many stops this edit lifts the shadows, to the nearest half (0 to 4): from exposure, the
        Shadows and Blacks sliders, Smart lighting and Auto HDR. Clean uses it to clean the lifted parts harder."""
        lift = max(float(self.exposure_ev), 0.0) + 2.0 * max(float(self.light.shadows), 0.0) / 100.0 \
            + 2.0 * max(float(self.light.smart), 0.0) / 100.0 + max(float(self.light.blacks), 0.0) / 100.0 \
            + (1.0 if self.hdr.enabled else 0.0)
        return round(min(lift, 4.0) * 2.0) / 2.0

    def raw_key(self) -> str:
        """Cache key for everything up to and including the camera matrix (stages 1-11)."""
        d = self.to_dict()
        keep = {k: d[k] for k in ("white_balance", "highlight_mode", "demosaic", "green_eq", "use_optical_black", "denoise", "banding",
                                          "calibrated",
                                          "lens", "sharpen", "profile")}
        if self.denoise.enabled and self.denoise.lift:      # Clean follows the lift (in half-stop steps)
            keep["lift"] = (self.shadow_lift(), round(float(self.exposure_ev) * 2.0) / 2.0)
        if d.get("film"):                          # a negative is turned positive in stages 1-12
            keep["film"] = d["film"]
            keep["film_v"] = d["stage_versions"].get("film", 1)
        if (d.get("halftone") or {}).get("on"):    # printing dots are taken out there
            keep["halftone"] = d["halftone"]
            keep["halftone_v"] = d["stage_versions"].get("halftone", 1)
        if (d.get("stains") or {}).get("on"):      # stains and corrosion are unmixed there
            keep["stains"] = d["stains"]
            keep["stains_v"] = d["stage_versions"].get("stains", 1)
        if (d.get("scratches") or {}).get("on"):   # dust and scratches are taken out there too
            keep["scratches"] = d["scratches"]
            keep["scratches_v"] = d["stage_versions"].get("scratches", 1)
        if (d.get("colourize") or {}).get("on") or (d.get("colourize") or {}).get("hints"):   # and colour
            keep["colourize"] = d["colourize"]
            keep["colourize_v"] = d["stage_versions"].get("colourize", 1)
        if float((d.get("relight") or {}).get("amount", 0)) > 1e-6:     # the light moved
            keep["relight"] = d["relight"]
            keep["relight_v"] = d["stage_versions"].get("relight", 1)
        if (d.get("jpeg") or {}).get("on"):                 # JPEG repair changes the decoded pixels
            keep["jpeg"] = d["jpeg"]
            keep["jpeg_v"] = d["stage_versions"].get("jpeg", 1)
        if d.get("depth") and abs(float((d.get("depth") or {}).get("haze", 0))) > 1e-6:    # haze by distance
            keep["depth"] = d["depth"]
            keep["depth_v"] = d["stage_versions"].get("depth", 1)
        fl = d.get("flare") or {}
        if fl.get("on") and fl.get("ghosts") and float(fl.get("amount", 0)) > 0:     # flare ghosts subtracted
            keep["flare"] = {"amount": fl.get("amount"), "ghosts": fl.get("ghosts")}
            keep["flare_v"] = d["stage_versions"].get("flare", 1)
        ir = d.get("infrared") or {}
        if ir.get("on"):                                   # the infrared matrix step, mono raw and hot spot
            keep["infrared"] = {k: v for k, v in ir.items() if k not in ("guess", "palette", "wb_source")}
            keep["infrared_v"] = d["stage_versions"].get("infrared", 1)
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
