"""Paint's brush pack (BUILD_PLAN_PAINT P2, the Brush Atlas 0.8): over a hundred presets, each grounded in a real
brush, a real medium and a real technique, each saying what it is best for and weak at, sorted into categories for
the brush list's filter. Not yet measured with the brushwork meter (0.8): the values are set by eye from how each
tool behaves.
"""
from __future__ import annotations

from .brush import BrushSettings, presets
from .curves import Curve

CATEGORIES = ["Pencil and graphite", "Ink and pens", "Markers", "Oil and acrylic", "Watercolour and gouache",
              "Pastel, charcoal and chalk", "Airbrush", "Blend and smudge", "Texture and effects", "East Asian ink",
              "Erasers", "Tools", "Imported", "MyPaint", "Mine"]

LIN = Curve()
SOFT = Curve.preset("Soft")


def _dyn(size=None, opacity=None, flow=None):
    return {"size": {} if size is None else {"pressure": size}, "opacity": {} if opacity is None else
            {"pressure": opacity}, "flow": {} if flow is None else {"pressure": flow}}


def B(name, tag, best, weak, **kw):
    kw.setdefault("dynamics", _dyn(LIN))
    return BrushSettings(name, best_for=best, weak_at=weak, tags=tag, **kw)


_CORE_TAGS = {"Pencil (HB)": "Pencil and graphite", "Ink pen": "Ink and pens", "Round brush": "Oil and acrylic",
              "Soft airbrush": "Airbrush", "Marker": "Markers", "Flat wash": "Watercolour and gouache",
              "Charcoal (rough paper)": "Pastel, charcoal and chalk", "Soft pastel": "Pastel, charcoal and chalk",
              "Dry brush (canvas)": "Oil and acrylic", "Spray": "Texture and effects",
              "Leaves (colour jitter)": "Texture and effects", "Gouache (pigment)": "Watercolour and gouache",
              "Smudge (smear)": "Blend and smudge", "Blender (dull)": "Blend and smudge",
              "Oil (loaded brush)": "Oil and acrylic", "Eraser (hard)": "Erasers", "Eraser (soft)": "Erasers"}


def _pencils():
    out = []
    grades = [("6H", 0.25, 2.5, 0.35), ("4H", 0.32, 2.8, 0.45), ("2H", 0.42, 3.0, 0.55), ("H", 0.5, 3.2, 0.6),
              ("B", 0.65, 4.0, 0.75), ("2B", 0.72, 4.5, 0.8), ("4B", 0.82, 5.5, 0.85), ("6B", 0.9, 6.5, 0.9),
              ("8B", 0.96, 7.5, 0.95)]
    for g, op, size, grain in grades:
        hard = "H" in g
        out.append(B(f"Pencil ({g})", "Pencil and graphite",
                     "light construction lines and fine detail that erase cleanly" if hard else
                     "dark, rich shading that builds with pressure",
                     "dark values (it stays pale, as a hard lead does)" if hard else
                     "fine precise lines; it smears into the paper's grain",
                     size=size, min_size=0.4, hardness=0.9, opacity=op, flow=0.8, spacing=0.07,
                     grain="fine", grain_strength=grain * 0.8, dynamics=_dyn(SOFT, LIN)))
    out += [
        B("Mechanical pencil 0.5", "Pencil and graphite", "even, thin technical lines", "tone and shading",
          size=2.5, min_size=0.85, hardness=1.0, opacity=0.8, spacing=0.05, dynamics=_dyn(None, LIN)),
        B("Silverpoint", "Pencil and graphite", "the Renaissance's fine grey drawing: delicate, even hatching on a "
          "prepared ground", "any dark value; it cannot be built up much", size=1.6, min_size=0.8, hardness=1.0,
          opacity=0.35, spacing=0.05, grain="fine", grain_strength=0.3, dynamics=_dyn(None, LIN)),
        B("Coloured pencil", "Pencil and graphite", "layered colour that lets the paper sparkle through",
          "flat solid colour", size=5, min_size=0.5, hardness=0.85, opacity=0.7, spacing=0.06, grain="cold_press",
          grain_strength=0.75, dynamics=_dyn(SOFT, LIN)),
        B("Graphite stick (side)", "Pencil and graphite", "broad tonal areas with the stick laid on its side",
          "lines", size=30, min_size=0.6, hardness=0.6, opacity=0.6, spacing=0.05, roundness=0.25, angle=10,
          grain="rough", grain_strength=0.8, dynamics=_dyn(None, LIN)),
        B("Woodless graphite", "Pencil and graphite", "bold, expressive drawing with a thick soft core",
          "fine detail", size=9, min_size=0.4, hardness=0.75, opacity=0.9, spacing=0.06, grain="cold_press",
          grain_strength=0.6, dynamics=_dyn(LIN, LIN)),
    ]
    return out


def _inks():
    out = [B(f"Technical pen {w}", "Ink and pens", "even lines for diagrams, comics and architecture",
             "any line that should swell or taper", size=max(1.2, w * 10), min_size=1.0, hardness=1.0,
             spacing=0.04, dynamics=_dyn()) for w in (0.1, 0.3, 0.5, 0.8)]
    out += [
        B("Dip pen (G nib)", "Ink and pens", "comic inking: lines swell strongly with pressure, thin hairlines",
          "even lines; fast pressure changes make blobs", size=9, min_size=0.08, hardness=1.0, spacing=0.04,
          dynamics=_dyn(Curve([(0, 0), (0.6, 0.45), (1, 1)]))),
        B("Crow quill", "Ink and pens", "the finest pen lines and delicate hatching", "bold lines",
          size=4, min_size=0.15, hardness=1.0, spacing=0.04),
        B("Brush pen", "Ink and pens", "lively lines with thick-thin rhythm, lettering", "even technical lines",
          size=14, min_size=0.05, hardness=0.95, spacing=0.04, roundness=0.8, dynamics=_dyn(SOFT)),
        B("Fineliner", "Ink and pens", "clean outlines and writing", "variation in line", size=3, min_size=0.9,
          hardness=1.0, spacing=0.04, dynamics=_dyn()),
        B("Ballpoint", "Ink and pens", "sketchbook drawing with soft, buildable lines", "solid black",
          size=2.5, min_size=0.7, hardness=0.9, opacity=0.55, flow=0.8, mode="buildup", spacing=0.05,
          grain="fine", grain_strength=0.3, dynamics=_dyn(None, LIN)),
        B("Calligraphy (broad nib)", "Ink and pens", "italic and gothic lettering: thick and thin by direction",
          "the nib angle is fixed at 30 degrees", size=16, min_size=0.7, hardness=1.0, spacing=0.03,
          roundness=0.12, angle=30, dynamics=_dyn(Curve([(0, 0.7), (1, 1)]))),
        B("Ink wash", "Ink and pens", "grey tonal washes over ink drawings", "crisp edges", size=60, hardness=0.3,
          opacity=0.35, flow=0.6, spacing=0.06, grain="cold_press", grain_strength=0.3, dynamics=_dyn(LIN, LIN)),
        B("Stippling pen", "Ink and pens", "tone made of dots, as engravers and illustrators do",
          "speed: tone builds slowly", size=4, min_size=0.6, hardness=1.0, spacing=3.0, scatter=2.0,
          dynamics=_dyn(None)),
        B("Hatching pen", "Ink and pens", "parallel-line shading that thickens with pressure", "curved hatching",
          size=60, hardness=0.9, mode="hatching", hatch_angle=45, hatch_spacing=6, crosshatch=1.0, spacing=0.1,
          dynamics=_dyn(None)),
        B("Cross-hatching pen", "Ink and pens", "deep pen shading: a second set of lines crosses as you press",
          "light tones (lines show at once)", size=60, hardness=0.9, mode="hatching", hatch_angle=45,
          hatch_spacing=5, crosshatch=0.55, spacing=0.1, dynamics=_dyn(None)),
        B("Sketch web", "Ink and pens", "loose gesture sketching: lines join the stroke to itself",
          "clean line art", size=3, mode="sketch", sketch_reach=12, sketch_density=0.3, spacing=0.3,
          opacity=0.7, dynamics=_dyn(None, LIN)),
    ]
    return out


def _markers():
    return [
        B("Alcohol marker (chisel)", "Markers", "flat even colour that darkens where strokes overlap",
          "blending without streaks", size=22, hardness=0.95, opacity=0.45, flow=0.6, mode="buildup",
          roundness=0.3, angle=40, spacing=0.05, dynamics=_dyn()),
        B("Alcohol marker (brush tip)", "Markers", "marker colouring with tapered strokes", "large flat areas",
          size=18, min_size=0.2, hardness=0.9, opacity=0.45, flow=0.6, mode="buildup", spacing=0.05),
        B("Highlighter", "Markers", "see-through bright strokes over text or drawings", "dark colours",
          size=26, hardness=1.0, opacity=0.3, mode="buildup", roundness=0.25, angle=0, spacing=0.04,
          dynamics=_dyn()),
        B("Felt tip", "Markers", "bold, slightly soft lines for cartoons", "fine detail", size=7, min_size=0.7,
          hardness=0.85, spacing=0.05, dynamics=_dyn(LIN)),
        B("Paint marker", "Markers", "opaque colour over dark grounds", "soft edges", size=10, min_size=0.8,
          hardness=1.0, spacing=0.04, dynamics=_dyn()),
    ]


def _oils():
    k = dict(tag="Oil and acrylic")

    def O(name, best, weak, **kw):
        return B(name, k["tag"], best, weak, **kw)
    return [
        O("Hog bristle round", "energetic alla prima strokes that grab thick paint", "smooth glazes",
          size=34, min_size=0.3, hardness=0.7, spacing=0.05, dual=True, dual_size=0.3, dual_count=7,
          dual_scatter=0.75, space="pigment", dynamics=_dyn(LIN)),
        O("Hog bristle flat", "blocking in planes of colour with crisp edges", "fine lines", size=40,
          min_size=0.5, hardness=0.75, roundness=0.3, angle=-20, spacing=0.05, dual=True, dual_size=0.25,
          dual_count=8, dual_scatter=0.8, space="pigment", dynamics=_dyn(LIN)),
        O("Hog bristle filbert", "soft-edged shapes and flesh: a flat with rounded corners", "sharp corners",
          size=38, min_size=0.4, hardness=0.6, roundness=0.45, angle=-20, spacing=0.05, dual=True,
          dual_size=0.3, dual_count=7, dual_scatter=0.7, space="pigment", dynamics=_dyn(LIN)),
        O("Bright (short flat)", "short, controlled, thick strokes", "long fluid strokes", size=30, min_size=0.5,
          hardness=0.8, roundness=0.45, angle=-20, spacing=0.05, opacity=1.0, space="pigment"),
        O("Sable round", "smooth glazes and fine points: a springy brush that holds its point",
          "thick impasto", size=20, min_size=0.08, hardness=0.75, opacity=0.8, flow=0.6, spacing=0.05,
          space="pigment", dynamics=_dyn(LIN, None, LIN)),
        O("Sable flat", "clean edges and smooth even passages", "textured strokes", size=30, min_size=0.4,
          hardness=0.8, opacity=0.85, roundness=0.25, angle=-20, spacing=0.05, space="pigment"),
        O("Fan brush", "foliage, grass and softening: bristles fanned apart", "solid colour", size=60,
          min_size=0.5, hardness=0.8, roundness=0.25, spacing=0.05, dual=True, dual_size=0.08, dual_count=10,
          dual_scatter=1.0, opacity=0.8),
        O("Rigger (liner)", "long thin lines: rigging, branches, hair", "anything broad", size=5, min_size=0.2,
          hardness=0.95, spacing=0.04, smudge_length=0.0, dynamics=_dyn(SOFT)),
        O("Dagger (striper)", "lines that turn from thin to wide in one stroke", "even width",
          size=18, min_size=0.1, hardness=0.9, roundness=0.4, angle=45, spacing=0.04),
        O("Mop", "soft, wide passages and softening edges", "detail", size=110, min_size=0.5, hardness=0.25,
          opacity=0.6, flow=0.4, spacing=0.05, dynamics=_dyn(LIN, None, LIN)),
        O("Palette knife", "scraping thick paint flat with sharp edges, mixing on the canvas", "soft edges",
          size=46, min_size=0.6, hardness=1.0, roundness=0.12, angle=-30, spacing=0.03, mode="smudge",
          smudge_kind="smear", smudge_length=0.75, colour_rate=0.45, dynamics=_dyn(LIN)),
        O("Impasto knife", "heavy knife marks that drag colours together", "control; it always mixes",
          size=56, min_size=0.6, hardness=1.0, roundness=0.2, angle=-30, spacing=0.03, mode="smudge",
          smudge_kind="smear", smudge_length=0.85, colour_rate=0.3, dynamics=_dyn(LIN)),
        O("Glazing brush", "thin transparent layers that shift colour without hiding form", "covering",
          size=70, hardness=0.4, opacity=0.25, flow=0.5, spacing=0.05, space="pigment", dynamics=_dyn(LIN)),
        O("Scumble", "broken, dry, light colour dragged over dark: atmosphere and air", "smooth colour",
          size=60, min_size=0.5, hardness=0.6, opacity=0.6, spacing=0.05, grain="canvas", grain_strength=0.85,
          grain_pressure=False, dynamics=_dyn(LIN)),
        O("Alla prima (loaded)", "wet into wet in one sitting: the colour mixes with what is there",
          "clean colour over wet paint", size=36, min_size=0.4, hardness=0.6, mode="smudge", smudge_kind="dull",
          smudge_length=0.45, colour_rate=0.35, spacing=0.05, dynamics=_dyn(LIN)),
        O("Impressionist tache", "the 1860s plein-air stroke: short, flat, broad, even dabs of mixed colour",
          "blending; that is the point", size=26, min_size=0.7, hardness=0.85, roundness=0.4, angle=-25,
          spacing=0.05, space="pigment", jitter_v=0.04, jitter_per_stroke=True, dynamics=_dyn(LIN)),
        O("Pointillist dot", "Seurat's separate dots of colour", "speed", size=8, min_size=0.8, hardness=0.9,
          spacing=2.5, scatter=0.6, jitter_h=0.03, jitter_v=0.05, dynamics=_dyn()),
        O("Sfumato", "Leonardo's smoky transitions: layers so soft no edge shows", "anything crisp",
          size=140, hardness=0.0, opacity=0.5, flow=0.05, mode="airbrush", space="oklab", spacing=0.05,
          dynamics=_dyn(None, None, LIN)),
        O("Egg tempera hatching", "tempera's fine, short, buildable strokes", "large areas", size=4,
          min_size=0.6, hardness=0.95, opacity=0.45, mode="buildup", spacing=0.05, dynamics=_dyn(None, LIN)),
    ]


def _water():
    W = "Watercolour and gouache"
    return [
        B("Watercolour flat wash", W, "even washes of transparent colour on cold-press paper", "opaque cover",
          size=80, min_size=0.6, hardness=0.4, opacity=0.4, flow=0.7, spacing=0.05, space="pigment",
          grain="cold_press", grain_strength=0.35, dynamics=_dyn(LIN)),
        B("Watercolour round", W, "general watercolour painting with a point", "opaque cover", size=30,
          min_size=0.15, hardness=0.5, opacity=0.45, flow=0.6, spacing=0.05, space="pigment",
          grain="cold_press", grain_strength=0.3, dynamics=_dyn(LIN, None, LIN)),
        B("Wet in wet", W, "soft blooms of colour into wet paper", "control of edges", size=70, hardness=0.1,
          opacity=0.35, flow=0.4, spacing=0.05, space="pigment", dynamics=_dyn(LIN, None, LIN)),
        B("Watercolour dry brush", W, "sparkling broken strokes on rough paper", "smooth colour", size=34,
          min_size=0.5, hardness=0.7, opacity=0.6, spacing=0.05, space="pigment", grain="rough",
          grain_strength=0.9, grain_pressure=False, dynamics=_dyn(LIN)),
        B("Watercolour glaze", W, "transparent layers that deepen colour", "lightening", size=60, hardness=0.35,
          opacity=0.2, flow=0.6, spacing=0.05, space="pigment", dynamics=_dyn(LIN)),
        B("Gouache flat", W, "flat opaque designs and posters", "transparent effects", size=36, min_size=0.5,
          hardness=0.85, opacity=0.95, roundness=0.3, angle=-20, spacing=0.05, space="pigment"),
        B("Gouache round", W, "opaque matte painting with a point", "transparent effects", size=24,
          min_size=0.2, hardness=0.8, opacity=0.95, spacing=0.05, space="pigment"),
        B("Squirrel mop wash", W, "big soft washes: skies and backgrounds", "detail", size=160, hardness=0.15,
          opacity=0.3, flow=0.5, spacing=0.05, space="pigment", dynamics=_dyn(LIN)),
        B("Splatter", W, "flicked droplets of paint", "control", size=120, mode="spray", count=2,
          particle_size=0.07, hardness=0.95, spacing=0.4, dynamics=_dyn(None, None, LIN)),
    ]


def _dry():
    P = "Pastel, charcoal and chalk"
    return [
        B("Vine charcoal", P, "light, easily changed lines and tone for laying in", "deep blacks", size=10,
          min_size=0.4, hardness=0.6, opacity=0.6, spacing=0.06, grain="rough", grain_strength=0.85,
          dynamics=_dyn(SOFT, LIN)),
        B("Compressed charcoal", P, "deep velvety blacks", "light delicate tone", size=14, min_size=0.5,
          hardness=0.75, opacity=1.0, spacing=0.06, grain="rough", grain_strength=0.6, dynamics=_dyn(LIN, LIN)),
        B("Charcoal pencil", P, "controlled charcoal lines and detail", "large areas", size=5, min_size=0.4,
          hardness=0.85, opacity=0.85, spacing=0.06, grain="cold_press", grain_strength=0.7,
          dynamics=_dyn(SOFT, LIN)),
        B("Charcoal (side of the stick)", P, "big tonal masses", "lines", size=40, min_size=0.7, hardness=0.6,
          roundness=0.2, angle=15, opacity=0.7, spacing=0.05, grain="rough", grain_strength=0.85,
          dynamics=_dyn(None, LIN)),
        B("Hard pastel", P, "crisp pastel lines and early layers", "rich final layers", size=12, min_size=0.5,
          hardness=0.8, opacity=0.8, spacing=0.06, roundness=0.6, grain="cold_press", grain_strength=0.75,
          dynamics=_dyn(LIN, LIN)),
        B("Oil pastel", P, "waxy, bold colour that smears a little", "fine detail", size=18, min_size=0.6,
          hardness=0.7, opacity=0.95, spacing=0.05, grain="cold_press", grain_strength=0.45,
          dynamics=_dyn(LIN)),
        B("Pastel pencil", P, "detail in pastel paintings", "broad areas", size=6, min_size=0.5, hardness=0.85,
          opacity=0.85, spacing=0.06, grain="cold_press", grain_strength=0.7, dynamics=_dyn(SOFT, LIN)),
        B("Chalk", P, "sparkling, crumbly strokes", "smooth tone", size=18, min_size=0.5, hardness=0.8,
          mode="chalk", spacing=0.12, grain="rough", grain_strength=0.5, dynamics=_dyn(LIN)),
        B("Conté crayon", P, "square-stick drawing: lines on the edge, tone on the flat", "very dark blacks",
          size=12, min_size=0.5, hardness=0.85, roundness=0.4, angle=20, opacity=0.85, spacing=0.06,
          grain="laid", grain_strength=0.6, dynamics=_dyn(LIN, LIN)),
        B("Sanguine", P, "red-chalk figure drawing in the old masters' way", "cool colours (pick a red earth)",
          size=10, min_size=0.4, hardness=0.75, opacity=0.8, spacing=0.06, grain="laid", grain_strength=0.65,
          dynamics=_dyn(SOFT, LIN)),
        B("Wax crayon", P, "children's-book texture with gaps and wax", "smooth colour", size=14, min_size=0.6,
          hardness=0.8, mode="chalk", spacing=0.1, grain="rough", grain_strength=0.75, dynamics=_dyn(LIN)),
    ]


def _air():
    A = "Airbrush"
    return [
        B("Fine airbrush", A, "small soft details and highlights", "large areas", size=24, hardness=0.0,
          flow=0.1, mode="airbrush", spacing=0.05, dynamics=_dyn(None, None, LIN)),
        B("Wide soft airbrush", A, "backgrounds and broad gradients", "anything crisp", size=320, hardness=0.0,
          flow=0.05, mode="airbrush", spacing=0.05, dynamics=_dyn(None, None, LIN)),
        B("Grainy airbrush", A, "soft shading with a fine tooth, less digital", "perfectly smooth gradients",
          size=120, hardness=0.0, flow=0.08, mode="airbrush", spacing=0.05, grain="fine", grain_strength=0.5,
          dynamics=_dyn(None, None, LIN)),
        B("Glow", A, "light that adds up like light: glows, fire, magic", "darkening", size=100, hardness=0.0,
          flow=0.1, mode="airbrush", space="light", spacing=0.05, dynamics=_dyn(None, None, LIN)),
        B("Shadow glaze", A, "soft cast shadows built in thin layers", "lightening", size=150, hardness=0.0,
          opacity=0.4, flow=0.06, mode="airbrush", space="oklab", spacing=0.05, dynamics=_dyn(None, None, LIN)),
    ]


def _blend():
    S = "Blend and smudge"
    return [
        B("Finger smear", S, "rubbing charcoal and pastel with a fingertip", "keeping texture", size=30,
          hardness=0.3, mode="smudge", smudge_kind="smear", smudge_length=0.85, opacity=0.9, spacing=0.05,
          dynamics=_dyn(None, LIN)),
        B("Large soft blender", S, "evening out big areas of colour", "detail", size=120, hardness=0.2,
          mode="smudge", smudge_kind="dull", smudge_length=0.7, opacity=0.6, spacing=0.05,
          dynamics=_dyn(None, LIN)),
        B("Bristle blender", S, "blending that leaves brush marks, as a dry hog brush does", "smooth blends",
          size=46, hardness=0.6, mode="smudge", smudge_kind="smear", smudge_length=0.8, spacing=0.05,
          dual=True, dual_size=0.2, dual_count=8, dual_scatter=0.8, dynamics=_dyn(None, LIN)),
        B("Wet knife blend", S, "pulling two wet colours together with a knife edge", "subtle blends",
          size=40, hardness=1.0, roundness=0.15, angle=-30, mode="smudge", smudge_kind="smear",
          smudge_length=0.9, colour_rate=0.1, spacing=0.03, dynamics=_dyn(LIN)),
    ]


def _fx():
    F = "Texture and effects"
    return [
        B("Grass blades", F, "fields of grass: thin upright strokes that vary in colour", "anything else",
          size=30, min_size=0.4, hardness=0.9, roundness=0.08, angle=85, spacing=0.5, scatter=0.8, count=2,
          jitter_h=0.05, jitter_v=0.15, dynamics=_dyn(LIN, None, None)),
        B("Stars", F, "starfields of tiny points", "big stars", size=160, mode="spray", count=1,
          particle_size=0.012, hardness=1.0, spacing=0.6, jitter_v=0.3, dynamics=_dyn()),
        B("Fur", F, "hair and fur pulled out from an edge", "smooth surfaces", size=26, hardness=0.4,
          mode="smudge", smudge_kind="smear", smudge_length=0.93, spacing=0.04, dual=True, dual_size=0.07,
          dual_count=10, dual_scatter=1.0, dynamics=_dyn(Curve([(0, 1), (1, 0.4)]))),
        B("Clouds", F, "soft billowing cloud shapes", "hard edges", size=180, mode="spray", count=2,
          particle_size=0.3, hardness=0.0, opacity=0.25, spacing=0.3, dynamics=_dyn(None, None, LIN)),
        B("Rock texture", F, "rough stone and cliff faces", "smooth surfaces", size=60, hardness=0.8,
          spacing=0.1, dual=True, dual_size=0.18, dual_count=6, dual_scatter=0.9, grain="rough",
          grain_strength=0.7, jitter_v=0.12, dynamics=_dyn(LIN)),
        B("Bark", F, "tree bark: rough vertical texture", "smooth wood", size=40, hardness=0.8, roundness=0.3,
          angle=90, spacing=0.06, grain="canvas", grain_strength=0.6, grain_scale=0.6, dual=True,
          dual_size=0.15, dual_count=6, dual_scatter=0.9, dynamics=_dyn(LIN)),
        B("Rain streaks", F, "falling rain", "anything still", size=50, hardness=0.6, roundness=0.04, angle=75,
          opacity=0.4, spacing=1.2, scatter=1.5, dynamics=_dyn()),
        B("Sparkle", F, "glints and sparks that add up like light", "darks", size=60, mode="spray", count=1,
          particle_size=0.04, hardness=0.3, space="light", spacing=0.5, dynamics=_dyn()),
        B("Grit", F, "dirt, noise and weathering", "clean areas", size=60, mode="chalk", spacing=0.4,
          opacity=0.5, dynamics=_dyn(None, LIN)),
        B("Foam and spray", F, "sea spray and foam", "solid water", size=90, mode="spray", count=3,
          particle_size=0.05, hardness=0.6, opacity=0.8, spacing=0.3, jitter_v=0.1, dynamics=_dyn(None, None, LIN)),
    ]


def _asian():
    E = "East Asian ink"
    return [
        B("Sumi brush", E, "ink painting: one stroke from hair-thin to full body", "even lines", size=40,
          min_size=0.04, hardness=0.85, spacing=0.03, dynamics=_dyn(Curve([(0, 0), (0.5, 0.35), (1, 1)]))),
        B("Lanhao (soft)", E, "soft orchid leaves and flowing lines", "crisp texture", size=30, min_size=0.05,
          hardness=0.6, opacity=0.9, spacing=0.03, dynamics=_dyn(SOFT, None, LIN)),
        B("Jianhao (stiff)", E, "firm, springy strokes for bamboo and rocks", "soft washes", size=26,
          min_size=0.1, hardness=0.9, spacing=0.03, grain="laid", grain_strength=0.3, dynamics=_dyn(LIN)),
        B("Hemp-fibre cun", E, "the long, soft, parallel texture strokes of southern landscapes (pima cun)",
          "angular rock", size=16, min_size=0.3, hardness=0.6, opacity=0.6, spacing=0.04, dual=True,
          dual_size=0.2, dual_count=5, dual_scatter=0.8, dynamics=_dyn(LIN, LIN)),
        B("Axe-cut cun", E, "broad angular side strokes for hard rock (fupi cun)", "soft hills", size=50,
          min_size=0.5, hardness=0.8, roundness=0.3, angle=-35, spacing=0.04, grain="rough",
          grain_strength=0.6, dynamics=_dyn(LIN)),
        B("Raindrop cun", E, "dotted texture for hills and trees (yudian cun)", "lines", size=10, min_size=0.6,
          hardness=0.8, spacing=1.8, scatter=1.0, dynamics=_dyn(LIN)),
        B("Flying white", E, "the dry streaked stroke where paper shows through (feibai)", "solid ink",
          size=36, min_size=0.5, hardness=0.9, roundness=0.6, spacing=0.03, dual=True, dual_size=0.1,
          dual_count=9, dual_scatter=1.0, grain="laid", grain_strength=0.5, grain_pressure=False,
          dynamics=_dyn(LIN)),
    ]


def _erasers_tools():
    return [
        B("Kneaded eraser", "Erasers", "lifting tone gently from charcoal and graphite", "removing completely",
          size=50, hardness=0.3, flow=0.25, eraser=True, spacing=0.05, dynamics=_dyn(None, None, LIN)),
        B("Precision eraser", "Erasers", "small clean corrections and highlights", "large areas", size=6,
          hardness=1.0, eraser=True, spacing=0.05, dynamics=_dyn()),
        B("Textured eraser", "Erasers", "lifting paint in the paper's grain, for texture", "clean removal",
          size=40, hardness=0.8, eraser=True, spacing=0.05, grain="rough", grain_strength=0.8,
          dynamics=_dyn(None, LIN)),
        B("Shape fill", "Tools", "flat shapes: draw the outline, it fills when you lift the pen",
          "nothing shows until you lift the pen", mode="shape", dynamics=_dyn()),
        B("Clone", "Tools", "copying part of the picture (Ctrl+click sets where to copy from)", "blending",
          size=40, hardness=0.6, mode="clone", spacing=0.05, dynamics=_dyn(None, LIN)),
        B("Tangent normal", "Tools", "painting normal maps for games: colour follows the pen's tilt",
          "ordinary painting", size=30, hardness=0.6, mode="normal", spacing=0.05, dynamics=_dyn(None, LIN)),
    ]


def _natural():
    W, O = "Watercolour and gouache", "Oil and acrylic"
    wet = dict(mode="watercolour", spacing=0.08)
    return [
        B("Wet wash (flows and dries)", W, "a real watercolour wash: it flows, dries with darker edges and "
          "granulates", "opaque cover; it needs its own layer", size=90, min_size=0.5, hardness=0.4, water=0.8,
          pigment=0.6, granulation=0.4, edge_darkening=0.6, grain="cold_press", **wet),
        B("Wet round (flows and dries)", W, "watercolour painting with a point, drying as real paint does",
          "opaque cover", size=30, min_size=0.2, hardness=0.5, water=0.6, pigment=0.7, granulation=0.3,
          edge_darkening=0.5, grain="cold_press", **wet),
        B("Wet-in-wet bloom", W, "blooms and soft runs: drop colour into a wash that is still wet",
          "crisp edges", size=70, hardness=0.2, water=1.0, pigment=0.5, granulation=0.2, edge_darkening=0.8,
          grain="cold_press", **wet),
        B("Granulating wash", W, "earth and mineral colours that settle in the paper's valleys",
          "smooth, even colour", size=90, hardness=0.4, water=0.7, pigment=0.7, granulation=1.0,
          edge_darkening=0.4, grain="rough", **wet),
        B("Lifting brush (clean water)", W, "taking watercolour back off the paper: soft highlights, clouds",
          "adding colour (it only removes)", size=40, hardness=0.4, water=0.9, pigment=0.0, lift=0.6, **wet),
        B("Wet dry brush", W, "a nearly dry watercolour brush dragged over rough paper", "smooth washes",
          size=36, hardness=0.7, water=0.2, pigment=0.8, granulation=0.8, edge_darkening=0.2, grain="rough",
          grain_strength=0.6, **wet),
        B("Oil bristle (loaded)", O, "oil strokes with every bristle showing; colour runs out along the stroke "
          "and picks up the paint beneath", "flat even colour", size=40, min_size=0.3, mode="bristle",
          bristles=28, paint_load=1.0, pickup=0.35, impasto=0.25, space="pigment", spacing=0.06,
          dynamics=_dyn(LIN)),
        B("Thin oil bristle", O, "thinned oil: long transparent strokes with fine streaks", "thick paint",
          size=40, min_size=0.3, mode="bristle", bristles=36, paint_load=2.0, pickup=0.2, opacity=0.6,
          space="pigment", spacing=0.06, dynamics=_dyn(LIN)),
        B("Acrylic flat bristle", O, "fast-drying flat strokes with crisp bristle marks", "blending (acrylic "
          "dries fast)", size=44, min_size=0.4, mode="bristle", bristles=30, paint_load=0.8, pickup=0.1,
          roundness=0.3, impasto=0.15, spacing=0.06, dynamics=_dyn(LIN)),
        B("Gouache bristle", W, "matte opaque bristle strokes that run dry into dry-brush texture",
          "thick relief", size=34, min_size=0.3, mode="bristle", bristles=24, paint_load=0.6, pickup=0.15,
          space="pigment", spacing=0.06, dynamics=_dyn(LIN)),
        B("Impasto bristle", O, "thick paint with bristle furrows that catch the light (Layers > Paint relief)",
          "fine detail", size=46, min_size=0.4, mode="bristle", bristles=26, paint_load=1.5, pickup=0.3,
          impasto=0.8, space="pigment", spacing=0.05, dynamics=_dyn(LIN)),
        B("Impasto round", O, "thick, raised strokes in one colour", "glazes", size=34, min_size=0.4,
          hardness=0.8, impasto=0.7, space="pigment", spacing=0.05, dynamics=_dyn(LIN)),
        B("Impasto palette knife", O, "pushing thick paint into ridges with a knife edge", "soft edges",
          size=50, min_size=0.6, hardness=1.0, roundness=0.15, angle=-30, mode="smudge", smudge_kind="smear",
          smudge_length=0.85, colour_rate=0.4, impasto=0.6, spacing=0.03, dynamics=_dyn(LIN)),
    ]


def pack() -> list:
    """Every built-in brush: the first presets, then the pack, each with its category."""
    core = presets()
    for b in core:
        b.tags = _CORE_TAGS.get(b.name, b.tags or "Tools")
    out = core + _pencils() + _inks() + _markers() + _oils() + _water() + _dry() + _air() + _blend() + _fx() + \
        _asian() + _natural() + _erasers_tools()
    seen, uniq = set(), []
    for b in out:
        if b.name not in seen:
            seen.add(b.name)
            uniq.append(b)
    return uniq
