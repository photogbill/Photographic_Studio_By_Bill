# BUILD_PLAN_PAINT.md: Paint, a digital painting section for the Studio

Bill, 2026-11-23 evening: "a comprehensive digital painting section, with support for drawing tablets", first for a
friend in Brazil who paints on **Linux** with a **Huion HS64**. "I want to match it and beat it" (Krita), and the
new Moko app too. "It may take many sessions." Language: English is fine.

This plan says what Krita and Moko do, what Paint will do, in what order, and how we will know each part works.
Read section 0 before building anything.

---

## 0. Decisions, facts and rules

### 0.1 What we are matching (researched 2026-11-23)

**Krita** (krita.org/en/features, docs.krita.org; GPL-3.0, KDE, 20+ years, ~67,000 commits):
- Interface: movable dockers and saveable workspaces, custom shortcuts, 30+ dockers, dark and light themes.
- Brushes: 100+ presets, three stabilizer methods plus a Dynamic Brush tool with drag and mass. **18 brush
  engines:** Pixel, Color Smudge, Bristle, Chalk, Clone, Curve, Deform, Dyna, Filter, Grid, Hatching, MyPaint,
  Particle, Quick, Shape, Sketch, Spray, Tangent Normal. Brushes are organised with tags.
- Wrap-around mode for seamless textures; resource manager with bundles.
- Vector and text: vector library, comic word bubbles, editable anchor points, text tool, SVG-based vectors.
- 2D animation: timeline (thousands of frames), audio, onion skin, opacity/position tweening, FPS, drop frames,
  export to video or image sequence.
- Drawing assistants: vanishing point, straight line, ellipse, fisheye (curvilinear perspective).
- Layers: paint, vector, filter, group and file layers. Selections: add/subtract, feather, invert, global
  selection mask.
- Colour: LCMS ICC management, OpenColorIO for EXR, HDR and scene-referred painting.
- GPU canvas (OpenGL / Direct3D 11) for rotation and zoom. PSD open and save. Python scripting.

**Moko** (Cubebrush / Marc Brunet; beta late 2026, full release planned late Nov / early Dec 2026, Windows and Mac
first, Linux "may follow"):
- Live on-canvas preview of a brush stroke once a brush and colour are picked; brush edits preview live too.
- Pencil, ink, standard brushes, bristle oils, thin oils, acrylics. Custom brushes from images, with pressure
  response, texture and scatter.
- **Natural Paint:** paint tracked on each bristle. The brush picks up colour, carries it, puts it down and runs
  dry. Palette knives push raised paint, and the light can be moved to show the surface.
- A pigment mixing mode (a compact mathematical model, GPU).
- Imports Photoshop `.abr` and Clip Studio `.sut` / `.sutg` brushes.
- Painting-first, but with photo editing in the same app. No subscription, no AI art tools.

**What neither has, and the Studio does:** a full raw developer and photo engine (Develop, Retouch, depth, relight,
subject masks, colour-managed screen and soft proof for prints), and it already runs on Linux. That is where we
beat them, besides matching the painting.

### 0.2 Her tablet: Huion HS64 on Linux (checked)

- Pen: battery-free EMR, **8192 pressure levels**, 5080 LPI, ≥233 reports per second, 160 x 102 mm, 4 press keys.
  Huion's page leaves tilt blank: plan for **no tilt** on her pen. Tilt is supported when a pen has it.
- Linux: the kernel's own `hid-uclogic` driver has supported the HS64 (USB 256c:006d) since mid-2019 (Linux 5.1
  stable and later), so **no Huion driver is needed** on any current distribution. Pressure reaches programs
  through libinput (Wayland and X11).
- Qt (what the Studio is built on) gives every program `QTabletEvent`: pressure 0..1, x/y tilt (-60..60 degrees),
  rotation, tangential pressure (airbrush wheel), eraser end. On Windows it reads Windows Ink and WinTab (Wacom),
  on Linux libinput/XInput2/Wayland tablet. Nothing extra to install.
- Express keys arrive as ordinary key presses (set in the desktop's tablet settings), so Paint maps keys, not
  "Huion buttons".
- **Build a "Tablet check" panel first** (P0): it draws pressure, tilt and speed live and says what the pen
  reports. If pressure does not reach Paint, it says why and what to do. That alone saves her an evening.

### 0.3 Licences: what we may use

The Studio is GPL-3.0. So:
- **Krita** is GPL-3.0: we may read it, port its algorithms and use its file-format knowledge, with credit in
  `THIRD_PARTY.md` and the file's docstring. We port ideas and maths, not the C++ wholesale; Python + NumPy needs its
  own shapes anyway.
- **libmypaint** (MyPaint's brush engine) is ISC: we may port it. MyPaint brush settings (`.myb`) are public
  domain by MyPaint's policy, so MyPaint brush packs can be read and shipped.
- **spectral.js** (Kubelka-Munk pigment mixing, Ronald van Wijnen) is MIT: we may port it. Krita has a draft
  port of the same thing.
- **Mixbox** (real-pigment mixing) is CC BY-NC: **not usable** (non-commercial clashes with GPL). We do our own
  Kubelka-Munk instead, and measure it against spectral.js's results.
- **PSD:** Adobe's specification licence (since 2006) only covers working with Adobe software. Krita and GIMP read
  and write PSD from reverse-engineered knowledge. We do the same through Krita's GPL code as reference, never
  Adobe's spec text. **Bill's call before we write PSD** (P5); OpenRaster (`.ora`) is the open layered format and
  comes first.
- `.abr` (Photoshop brushes) and `.sut` (Clip Studio, an SQLite file) readers: from Krita/GIMP's GPL readers and
  public notes. Tips only (the sampled images and basic spacing); their dynamics differ from ours.

### 0.4 Rules that make Paint good (and keep it Bill-proof)

1. **The pen comes first.** Latency from pen to pixel under 16 ms on a normal laptop; never drop a pen event;
   smooth curves between events (not straight segments); pressure curve per person.
2. **Never lose a painting.** Autosave every minute to beside-the-app storage (`studio/paths.py`, never C:\ or the
   user profile), crash recovery on start, unlimited undo by memory budget, history kept on disk.
3. **Every brush and method says what it is best for and weak at** (plan 23.1).
4. **Low-end machines matter.** Her PC may be modest. Everything has a CPU path that works on 2 cores and 4 GB;
   the GPU path is a speed-up, never a requirement.
5. **Linux is first-class**, not a port: `start-studio.sh` can open straight into Paint; every test runs on the
   Linux test machine.
6. **No AI art.** Painters care. Nothing generates or "finishes" a painting. Studio helpers that only measure or
   select (a subject mask, depth for a reference) are offered as tools, off by default. **Bill's call** (0.6).
7. File names: a painting is the artist's own file (`NAME.ora`), saved where she chooses; exports follow
   `NAME_Studio.ext` and `free_target` as everywhere else.
8. Errors report themselves (`report_problem`); tests on synthetic strokes with known truth.

### 0.5 Architecture (decided here, checked in P0)

- **Canvas:** tiled (256 x 256), each layer a sparse dict of float32 RGBA tiles (premultiplied), so huge canvases
  and undo cost only what was touched. Undo stores the tiles a stroke changed.
- **Colour while painting:** three blend spaces, chosen per brush, said plainly:
  - *Classic* (perceptual, gamma-encoded, how Photoshop, Krita and Clip Studio mix: what painters expect);
  - *Light* (linear: glows and light mix like real light);
  - *Pigment* (Kubelka-Munk: blue + yellow = green, like paint).
- **Brush engine core** in NumPy: dab stamping with sub-pixel placement, spacing by distance and time, a dab cache
  per brush and size. Physics-heavy engines (bristles, watercolour) run on a small working region around the pen.
- **Speed gate (P0):** a benchmark draws a 2,000-dab stroke at 60 px with each engine. If the CPU path cannot keep
  under 16 ms per pen event on the Linux test machine, the core moves to OpenGL shaders through Qt (PySide6 has
  QOpenGLWidget), with NumPy kept as the fallback. No C extension: Bill cannot build one, and it would break his
  one-double-click install.
- **Display:** the canvas is shown through the Studio's colour-managed screen (`color/display.py`), so what she
  sees is what she prints, with soft proof.
- **Code:** `studio/engine/paint/` (no Qt: canvas, tiles, brushes, layers, files), `studio/app/paint/` (canvas
  widget, panels). CLI commands for scripted use and the MCP server, as for every tool (`docs/CLI.md`).

### 0.7 Research: papers that shape how Paint works (Bill, 2026-11-23: "no generative stuff")

Bill's four links, judged:
- **Birth of a Painting** (Jiang et al., arXiv 2511.13191, 2025): single- and dual-colour Bezier strokes drawn by a
  differentiable paint renderer, plus a smudge operator for blending and shading. **Use:** the stroke model
  (two-colour strokes, smudge as an operator) for P2's smudge and the live stroke preview. **Not used:** fitting
  strokes to an image and its texture-generation module (that is reconstruction/generation).
- **Quantum Brush** (Ferreira et al., arXiv 2509.01442, 2025): four open-source brushes whose strokes drive quantum
  algorithms. **Use, as a curiosity:** "Quantum" experimental brushes run on a classical simulator (P2, last).
  Nothing generative; Bill likes the unexpected.
- **Animated Chinese still-life paintings with global effects** (Deng & Akleman, arXiv 2312.05401, CC BY 4.0):
  moving light over a painting by compositing lit renderings. **Use:** P3's impasto light and P10's relight of a
  finished painting with the Studio's relight engine.
- **Loomis Painter** (arXiv 2511.17344): a video diffusion model that invents painting steps. **Generative: not
  used.**

Bill's second batch (21:20), judged so nobody checks them again:
- Parameterized Brushstroke Style Transfer (arXiv 2603.07776): neural style transfer with Bezier strokes. Not used
  (generative); the two-colour Bezier stroke idea is already taken from Birth of a Painting.
- Vectorized Region Based Brush Strokes (arXiv 2506.09969, CC BY-NC-ND): turns a photo into a painting sequence.
  Not used (it paints for you).
- IMPASTO, robotic oil painting (arXiv 2603.29315, CC BY 4.0): a robot arm with a learned canvas model. Not used;
  only its note that a stroke is start, length, bend, angle and **force** confirms pressure as a first-class input.
- Sketch & Paint (arXiv 2502.20119): invents a stroke order for a finished image. Not used; Paint records her real
  process instead (P9 timelapse).
- TIJER 2304449: the results sections are unfilled templates. Not used.

Found in the search (all simulation or measurement, none generative):
- **Watercolour:** Curtis et al., "Computer-Generated Watercolor" (SIGGRAPH 1997): shallow-water flow on a paper
  model, three layers, Kubelka-Munk compositing, edge darkening, granulation, glazing; the base for P3. Van
  Laerhoven & Van Reeth, "Real-time simulation of watery paint" (2005): the same effects interactively (wet-on-wet,
  dark edges, glazing). Sun et al., i3D 2018: GPU diffusion of mineral vs. vegetable pigments (Fick's law and
  Brownian motion) on Xuan paper, for ink and colour-ink painting. Chu & Tai, **MoXi** (SIGGRAPH 2005): ink
  dispersion in absorbent paper.
- **Thick paint:** Baxter et al., **IMPaSTo** (2004): simplified paint physics with Kubelka-Munk and live lighting
  of the paint surface, on the GPU; the base for impasto. Chen et al., **Wetbrush** (SIGGRAPH Asia 2015): bristles
  plus paint as particles near the brush and a density field away from it; the reference for P3's bristle engine
  (we keep its ideas in 2D for speed).
- **Pigment mixing:** Sochorová & Jamriška, "Practical Pigment Mixing for Digital Painting" (SIGGRAPH Asia 2021):
  Kubelka-Munk mixing while staying RGB-in / RGB-out. Its code (Mixbox) is non-commercial, so we build our own
  RGB-in/RGB-out Kubelka-Munk (spectral.js's single-constant model, MIT) and compare against the paper's figures.
  Lu et al., **RealPigment** (Princeton): layering learned from a physical colour chart the artist paints.
- **Brushes from her own paint:** Lu et al., **RealBrush** (SIGGRAPH 2013): new strokes made from scans of the
  artist's real strokes, smears and overlaps, by patch synthesis (no neural network). **Use:** P3 "Scan your own
  brush": she paints a few strokes on paper, photographs them, and paints with that medium.
- **Line art fill:** Sýkora et al., **LazyBrush** (Eurographics 2009): colour scribbles fill hand-drawn line art
  even with gaps (an optimisation, built from what ink-and-paint artists asked for); P1's fill and P7 comics.
- **Smudge done better:** "Region-Aware Color Smudging" (arXiv 2405.02759): a study of how professionals and
  novices actually use smudge tools, then a smudge that respects regions. P2.
- **How the pen should feel:** Ng et al., "In the Blink of an Eye" (CHI 2014): commercial stylus latency was 65-120
  ms; Annett et al., "How low should we go?" (GI 2014): while inking, people notice delay from about 50 ms, judged
  against the pen and hand. **Target:** pen to pixel under 16 ms (one screen frame), never above 50 ms.
- **How artists draw:** Grimm, "Results of an observational study on sketching" (2011): how and why artists make
  the marks they do. PaintCopilot (arXiv 2605.20941) logs real stylus traces (big brushes early, small late);
  we use only the observation for defaults (brush size history, quick size keys), never its continuation model.

**Patents to keep clear of** (check again before P3): Adobe's procedural polygon watercolour (DiVerdi et al.,
US 8,917,282 and 8,917,283, active) and Adobe's erodible "stateful brush tip" (US 10,217,253). Our watercolour
follows the 1997 grid method (Curtis et al.) and Van Laerhoven, not polygons; our pencil tooth is a paper height
map, not an erodible tip mesh.

### 0.6 Bill's answers (2026-11-23, 21:34)

1. **Yes:** Paint is a sidebar section of the Studio, plus a `Start Paint` launcher (.bat and .sh) that opens straight
   into it.
2. **Yes:** no AI that paints. Subject masks and depth only as tools, off by default.
3. **Yes:** PSD read and write from Krita's reverse-engineered knowledge, after `.ora`.
4. Her sample painting and brushes: not known yet. Bill: **"include every brush type we can track down"**, and
   build the brushes on **studies of traditional brush strokes over the centuries** (section 0.8).

### 0.8 The Brush Atlas: brushes built from how painting was really done (Bill, 2026-11-23)

Every preset in Paint is grounded in a **real brush, a real medium and a real technique**, with a short note on
where it comes from in the history of painting. It is the brush library's spine and something neither Krita nor
Moko has: their brushes are named by feel, ours by what painters actually used.

**A brush = shape x hair x medium x technique.** Each is a separate, measurable layer in the engine:
- *Shapes* (from brush makers' references, e.g. Blick's shape guide): round, pointed round, flat, bright (short
  flat), filbert, angle, fan, rigger/liner, dagger/striper, cat's tongue, mop, mottler/spalter, hake, Chinese and
  Japanese brushes (lanhao, jianhao, sumi), quill, plus palette and painting knives.
- *Hair:* hog bristle (stiff, grabs and carries thick paint, energetic marks), sable (springy, holds a point, smooth
  glazes), squirrel and mop hairs (absorbent, big washes), badger and mongoose (soft blending), synthetics, mixed.
  Classical painters chose stiffness on purpose: stiff hog for sculpting thick paint and alla prima, soft hair for
  blending, glazing and clean edges.
- *Media:* oil, egg tempera, fresco, encaustic, watercolour, gouache, acrylic, ink (Western and East Asian),
  pastel, charcoal, graphite, silverpoint, conte.
- *Techniques:* glazing, scumbling, impasto, alla prima, sfumato, dry brush, wet-in-wet, tempera hatching,
  pointillism, the Impressionist *tache*, stippling, sgraffito, and East Asian texture strokes (*cun*).

**History that changes how brushes behave** (facts the presets follow):
- Before the mid-19th century artists' brushes were mostly round. The metal ferrule could be flattened, which made
  flats, brights and filberts possible; the Impressionists used flat brushes for the *tache*, a broad, flat, even
  stroke. The 1841 collapsible paint tube made outdoor oil sketching easy. So an "1860s plein air" set offers flats
  and filberts with fast, loaded strokes; a "Renaissance" set offers rounds, quills and glazes.
- Chinese painters codified *cun* (texture strokes) into many types, including 25 basic ones, for trees, rocks,
  raindrops, mountains and more. Each named *cun* becomes a preset with its ink load and paper behaviour.

**Measured, not guessed.** Art historians and image scientists have measured real brushwork:
- Li, Yao, Hendriks & Wang (IEEE PAMI 2012) extracted van Gogh's brushstrokes automatically and found them strongly
  rhythmic: regularly shaped strokes tightly arranged, with traits that changed across his periods. Earlier work
  (Johnson et al., IEEE Signal Processing 2008, on 101 scans from the Van Gogh and Kroller-Muller museums) measured
  each stroke's curvature, orientation and size, and their averages and spread, to model an artist's brushwork.
- Conservation scans measure paint relief: 3D topography of Rembrandt and Van Gogh surfaces (TU Delft), RTI and
  photogrammetry of Rembrandt's *Man with a Sheet of Music* (National Gallery of Art, 2025), roughness studies of
  brushstroke surfaces.
- **How we use it:** a "brushwork meter" in the test kit measures strokes made with our presets (length, width,
  curvature, orientation spread, rhythm/spacing, relief height) and compares them with the published measurements
  and with public-domain museum scans (open-access collections, CC0). A preset is tuned until its strokes measure
  like the real ones. Measurement only, never a model that paints.
- The impasto height channel takes its range and edge shapes from the relief scans, so a knife ridge or a hog
  bristle furrow looks right under a moving light.

**Era and technique checklist** (Bill's links 2026-11-23: gordonbruceart.com "Brushstrokes of History" and
tresejudd.com "The Evolution of Brushstrokes". Both are uncited blog posts, and one has illustrations labelled "AI
rendered" and doubtful claims (e.g. Egyptians using no brushes). So they are a **checklist of what to cover**; each
item is checked against museum and conservation sources before its preset is tuned):
- Prehistoric: fingers, twigs, reeds, blown pigment -> finger, reed and spray-mouth tools.
- Egypt: fresco on wet plaster, plant-fibre brushes and reed pens.
- Greece: encaustic (pigment in hot beeswax, smooth transitions, built texture) -> a wax medium that cools as you
  paint and can be re-melted with a "heat" tool.
- Byzantine/medieval: egg tempera on panel, fine hatching, gold leaf -> tempera hatching brush, gold-leaf layer that
  catches the moving light (P3 lighting).
- Renaissance: buon fresco (paint only while the day's plaster is wet: a "fresco day" mode where only today's patch
  takes paint), sfumato (thin layered glazes, hazy edges).
- Baroque: visible, textured strokes; thick fluid strokes and bold colour (Rubens); strong light and dark.
- 19th-century academic (Bouguereau): thin layers, no visible brushmarks, edges sharp at key features and soft where
  forms turn into shadow -> a dedicated **edge tool** (sharpen/lose an edge along a stroke), useful to everyone.
- Impressionism: broken colour placed side by side instead of mixed on the palette; the flat *tache*.
- Modern: any object as a brush (Picasso) -> stamp tips from any photo; Pollock's drip and fling -> a drip tool where
  paint leaves the brush on a trajectory and lands with splatter.

**Every brush type we can track down** (licence-checked packs to ship or import):
- MyPaint brush packs: raw brush settings are public domain by MyPaint's policy.
- David Revoy's Krita bundles 2023-01 and 2025-01: CC0 (his 2021 and Krita 4 Extras packs are CC BY 4.0: shippable
  with credit).
- Krita's shipped gbr/gih tips come from David Revoy's Chaos and Evolutions set and Ramon Miranda's Gimp Paint
  Studio; licences checked file by file before shipping.
- GIMP brushes (GPL), and import of her own `.abr`, `.sut`, `.kpp`, `.myb`, `.gbr`/`.gih`.
- Our own Atlas presets from the shapes x hair x media x techniques above (target: 300+ by the end of P3).

### 0.9 Bill's research notes judged (2026-11-23, 22:00-23:12; three pasted surveys, partly from Gemini)

Bill: "a lot of this could be BS from gemini ... let me know if anything is worthwhile." The papers are real; the
hype was in the promises ("zero latency", "120 FPS"). Verdicts, so nobody checks them again:

- **In, P1-P2:** 1-euro filter as the "Speed-adaptive" smoothing mode (**built**, smoothing.py); Oklab as a blend space
  for airbrush, gradients and smudge (the Studio already has Oklab code); **branching undo** (fork a painting, compare,
  cherry-pick layers: cheap because undo already stores changed tiles); our own Kubelka-Munk (Mixbox stays out,
  CC BY-NC); coherence-enhancing diffusion (Weickert 1999) for a smudge that follows contours; Gabor noise (Lagae et al.
  2009) for directional texture brushes; starlet multi-scale frequency separation (the Studio already has starlet);
  paper tooth (Sousa & Buchanan 1999, Van Haevre) for pencil, charcoal and pastel.
- **In, later:** P3 Curtis 1997 watercolour, MoXi ink, IMPaSTo, Wetbrush ideas in 2D; P4 MLS (Schaefer 2006) and bounded
  biharmonic warps (Jacobson 2011), ARAP, Poisson / mean-value cloning, PatchMatch fill, and **keep a layer's original
  pixels plus its transform and resample once** (the real fix for repeated-transform blur, not "Drizzle"); P6 Euler
  spiral / clothoid curves for ink and rulers (Raph Levien's work); P7 LazyBrush flats and Vector Graphics Complexes
  (Dalstein 2014, shared edges); P10 sliced optimal transport in Oklab for palette transfer (Pitie 2005, Rabin 2011),
  2D Radiance Cascades light (GPU only), Frangi line extraction from sketchbook photos, Jakob-Hanika spectral
  upsampling to show a painting under daylight vs. tungsten (metamerism).
- **LPM, Lorentz Pigment Mixing** (Ivkovic et al., IntechOpen 2025, DOI 10.5772/acrt.20250065; code GPL-3.0,
  github.com/RatkoIV/LPM, one MATLAB/Octave file): a shaped curve between RGB channels, not physics, evaluated only by
  eye. **In as a cheap extra blend style**; our Kubelka-Munk stays the "mixes like paint" mode, and the brushwork meter
  decides which looks better.
- **Out:** pen-tip prediction / Kalman lead (**Bill, 22:14: "a nogo, that would piss me off"**: Paint draws only where
  the pen has been); "Drizzle" for transforms (wrong tool); a CAD constraint solver for perspective (overkill);
  EbSynth-style propagation (it paints for the artist); StrokeAggregator sketch cleanup and "assistive vision" only as
  optional tools, off by default, if Bill asks.
- **Native libraries (Bill, 23:12: "Why can't we include C libraries and rust libraries?"):** we can. Three ways that
  never make Bill build anything: packages already on PyPI as prebuilt wheels; our own wheels built by a free GitHub
  Actions job on every push (maturin for Rust, ctypes or cffi for C) and fetched by the installers; and a Python
  fallback for each, so a missing native file only makes Paint slower. Costs: a build job, two platforms to test, the
  MSI, licences (Apache-2.0, MIT, BSD, ISC, LGPL all mix with GPL-3). Worth it: libmypaint (C, ISC) for the MyPaint
  engine, kurbo (Rust) for curves, libigl (C++) for warps. Vello and Graphite would mean rebuilding the app around
  them: out. Rule: stay Python + GPU until the speed gate or a feature proves Python cannot keep up; libmypaint in P2
  is the likely first.

---

## P0. Ground work (first session)

- P0.1 **Tablet check** panel: live pressure / tilt / speed / eraser / keys, with a plain diagnosis.
- P0.2 Canvas widget: zoom, pan, rotate, mirror view, tablet + mouse input, event coalescing switched off (Qt
  compresses tablet events by default: turn `AA_CompressTabletEvents` off), timestamps kept.
- P0.3 Tile store and undo; the speed gate benchmark (0.5); decide CPU or GPU core. Write the numbers in Status.
- Tests: synthetic tablet events (pressure ramp) give the stroke widths the pressure curve says; undo restores
  bit-exact tiles; autosave + recovery round trip.

## P1. A canvas worth painting on

- Pixel brush engine: round and image tips, hardness, spacing, size/opacity/flow/hardness by pressure (and tilt,
  speed, rotation, random, fade, distance), each through an editable curve. Build-up (wash) vs. airbrush.
- Stroke smoothing: Basic, Weighted, Stabilizer (lag with catch-up) and Dynamic (drag and mass, as Krita).
  Plus a pulled-string mode (the pen drags the brush on a string).
- Eraser (pen's eraser end too), colour picker (Alt), fill (tolerance, gap closing for line art, fill by
  reference layer), gradient.
- Layers: add, delete, order, opacity, lock, alpha lock, hide, group, merge, the Photoshop/Krita blend modes.
- Colour: HSV wheel and triangle, wide-gamut picker, swatches, recent colours, foreground/background, value-only
  view.
- Files: OpenRaster `.ora` (layered, open, Krita/GIMP/MyPaint read it), PNG/TIFF/JPEG/WebP export, Studio
  colour profiles.
- Canvas-only mode (Tab), pop-up palette on right click (brushes + colours, as Krita), shortcuts, pen key mapping.
- Acceptance: she paints a sketch-to-finish piece without hitting a missing basic. Tests for every brush input.

## P2. Brushes that feel real

- Smudge / blend engine (Krita's Color Smudge: smearing, dulling, colour rate, overlay mode).
- **Pigment mixing:** our Kubelka-Munk (spectral.js ported, MIT), per brush and in the colour mixer.
- Texture/grain (paper texture under the stroke), dual brush, scatter, count, colour jitter.
- Spray, Chalk, Sketch (web lines), Hatching, Curve, Grid, Particle, Shape (filled shapes), Clone, Filter
  (paint a Studio filter), Deform (liquify by brush), Tangent Normal (paint normal maps: for the RPG Forge too).
- **MyPaint engine:** port libmypaint (ISC), read `.myb` packs.
- Brush editor with a **live preview stroke** (Moko's feature), brush tags and search, bundles in and out.
- Imports: `.abr` sampled tips, `.sut`/`.sutg` tips, Krita `.kpp` presets where they map, GIMP `.gbr`/`.gih`.
- Acceptance: a brush pack of 100+ good presets; each says what it is best for.

## P3. Natural media (where we beat Krita)

- **Bristle engine with paint load:** each bristle carries colour, picks up, deposits, runs dry (Moko's Natural
  Paint). Oils, thin oils, acrylic, gouache.
- **Impasto:** a height channel per paint layer; palette knife pushes it; a light you can move shows the surface,
  live (Moko). Exports with or without the relief.
- **Watercolour:** a wet layer that flows and dries (edge darkening, back-runs, granulation, lifting with a damp
  brush). Krita has no true watercolour; this is a clear win.
- Pencil and charcoal on paper texture with tooth that fills as you press.
- Acceptance: side-by-side pieces against Krita and Moko pictures; Bill and his friend judge.

## P4. Editing, selecting and transforming

- Selections: rectangle, ellipse, lasso, polygon, magic wand, colour range, from layer, feather, grow, shrink,
  invert, a paintable selection mask. Optional (0.6): **subject select** from the Studio's own models.
- Transform: free, perspective, warp, cage, mesh and liquify (Krita has all six), on layers and selections.
- Layer masks, clipping masks, filter and fill layers, file layers (a linked image), and **adjustment layers
  driven by the Studio's photo engine** (curves, colour grading, Natural curve, relight).
- Filters from the Studio (blur, sharpen, noise, Clean) applied to layers.

## P5. Files and colour

- PSD open and save with layers, masks, groups and blend modes (Bill's call, 0.3).
- Krita `.kra` open (it is a zip of OpenRaster-like layers), so she can bring her old work.
- Colour-managed painting in any ICC working space, CMYK soft proof for print (the Studio already does this),
  16-bit and float canvases, HDR / EXR.

## P6. Drawing aids

- Assistants: 1/2/3-point perspective, vanishing point, parallel rulers, ellipse and concentric ellipse, fisheye,
  spline; snapping with a strength slider.
- Symmetry: mirror X/Y, radial/mandala with N spokes; wrap-around (seamless tiles) for textures.
- Reference images panel (pin photos from the Library, colour pick from them, greyscale toggle), grid, guides.
- Colour tools: harmonies, gamut masks, palette from a photo, value check (greyscale preview), colour blindness
  preview.

## P7. Vector, text and comics

- Vector layers (SVG), shapes, pen tool, editable anchors; text tool with fonts and text on path.
- Comic panels (gutters), word bubbles, a page template, export for print with bleed.

## P8. Animation

- Timeline, onion skin, tweened opacity and position, FPS, playback, audio scrubbing, export to video (ffmpeg
  where present) and image sequences / sprite sheets (useful for the RPG Forge).

## P9. Working faster

- Workspaces and movable panels, timelapse recording of the painting, snapshots, history panel, scripting from the
  Studio's CLI and MCP (Claude can drive Paint), resource manager.
- GPU canvas if P0 did not already need it; very large canvases (20k px) stay smooth.

## P10. Photo meets paint (what only the Studio can do)

- Open any photo from the Library or Develop as a paint layer, and send a painting back to Develop.
- Photo to underpainting (value block-in, posterised colour masses) for painters who start from reference. A
  measuring tool, not an AI painter.
- Depth and light from the Studio for a reference (where light comes from, planes of the face).
- Print path: soft proof against her printer's profile, as Kent's prints showed matters.

---

## Order (M)

P0 → P1 → P2 (smudge, pigment, MyPaint first) → P3 → P4 → P5 → P6 → P10 → P7 → P8 → P9. P0 and P1 are the
first one or two sessions; each later part is one to three sessions. Every session: tests, a `NEXT_UPDATE.md`
line for each part, a Status note here.

## Status

- 2026-11-23: plan written, Bill's papers judged and a paper search (0.7), Bill's answers in (0.6), the Brush Atlas
  added (0.8). Nothing built yet. Next: P0.
- 2026-11-23 night (session with Bill; he asked to keep tests small and cheap until the whole app is built): **P0 built
  and most of P1.** Engine `studio/engine/paint/` (no Qt): `tiles.py` (256 px float32 premultiplied tiles, layers,
  document, seven blend modes), `undo.py` (changed tiles, bit-exact; layer changes as snapshots; memory budget),
  `curves.py` (monotone pressure curves, presets), `smoothing.py` (none, Basic, Weighted, Stabilizer, Dynamic, Pulled
  string, Speed-adaptive 1-euro; no prediction), `brush.py` (round / elliptical / image tips with sub-pixel dabs on
  quadratic curves through the samples, spacing, size / opacity / flow by pressure, speed, tilt, random, fade; wash,
  build-up, airbrush; eraser; alpha lock; locked layers; eight presets with best for / weak at), `colour.py`, `ora.py`
  (OpenRaster save / open, atomic), `autosave.py` (`paint/recovery` beside the app, `STUDIO_PAINT_DIR`), `tablet.py`
  (pen statistics and the plain-words diagnosis), `bench.py`. Window `studio/app/paint/`: `canvas.py` (zoom around the
  pointer, pan, rotate, mirror, value view, pen + mouse, eraser end, Alt pick, brush outline), `tabletcheck.py`,
  `widgets.py` (HSV wheel, curve box), `page.py` (Brush / Colour / Layers / Tablet check, files, export as
  NAME_Studio via `free_target`, autosave and recovery offer, Tab canvas-only, shortcuts). `main.py`: the Paint row,
  `AA_CompressTabletEvents` off, `--paint`. Launchers: `Start Paint.bat` (CRLF), `start-paint.sh` (LF), and
  `start-studio.sh` now passes options through.
- **Speed gate (0.5), measured** on the Linux test VM (2 cores of an i7-12800H, NumPy), a fast circular stroke at 233
  reports/s and 2,000 px/s, pressure varying, each event timed with its screen composite: 60 px wash 4,451 dabs,
  median 5.5 ms, p95 12.6 ms, max 16.6 ms (passes); 60 px build-up median 4.1, p95 10.5, max 29.2 (passes); 200 px wash
  median 5.5, p95 16.8, max 63.3 (just over). **Decision: CPU core** (it passes at painting sizes on 2 cores); the GPU
  path (QOpenGLWidget) is kept for big brushes and the display, P9 or sooner if her PC is slower. Not yet measured: pen
  to pixel on a real screen, her PC.
- Tests (quick, synthetic): `tests/test_paint_engine.py` (11: pressure ramp widths follow the curve, wash cap vs.
  build-up, undo bit-exact, layer undo, eraser / alpha lock / locked, picker, .ora round trip, autosave + recovery,
  smoothing, tablet diagnosis, presets), `tests/test_paint_page.py` (4), `test_linux_scripts` covers start-paint.sh.
  All pass on the Linux copy. Not run on Windows; no real pen tested yet (her HS64, Bill's tablet).
- 2026-11-24, just after midnight (Bill: "Continue building"): `fill.py` (fill by tolerance from the picture, the
  layer or a reference layer; gap closing by dilating the lines, finding the region, growing back through similar
  pixels; grow under lines; linear / radial gradients to the second colour or to see-through), blend spaces in
  `colour.py` (Classic, Light, Oklab; `over()` used by the brush, `mix()` by gradients), `brush.load_tip` (any picture
  as a tip), the Fill and Gradient tools and Fill tab, blend space and "Tip from a picture" in the Brush tab, the
  right-click / barrel-button pop-up palette, and the canvas through `color/display.py` (sRGB -> linear ProPhoto ->
  the monitor's profile, soft proof included; unmanaged screens shown as sRGB). Tests: 19 quick ones pass (fill with
  and without gap closing on a broken ring, gradient ends, Oklab vs. Classic, Light vs. Classic, image tip, the tools
  in the window).
- 2026-11-24 00:16 (Bill: "build the next part"): **branching undo** (0.9). `undo.py` is now a tree: undo moves to the
  parent, redo follows the branch last used, painting after an undo adds a branch, `jump()` goes to any step (undo up
  to the shared step, redo down). Memory budget: branches away from the current path go first (oldest first), then
  the oldest steps; the last step can always be undone. The History tab lists the tree (↳ marks a branch, the current
  step in cyan, other branches dimmed); a click jumps. Test: two ideas kept on two branches, bit-exact jumps, trimming.
- 2026-11-24 night, from 00:25 (Bill: groups, then the rest of P1, then P2): **layer groups** (`tiles.py`: the list
  stays flat and bottom first; each layer names its group in `parent`, a group sits directly above its contents;
  pass-through and isolated groups with opacity and mode; hidden / locked groups act on what is inside;
  `paint_layer()` passes a stroke on a selected group to its top layer; move up / down goes into open groups and out
  at the edges, closed groups are stepped over; group, ungroup, merge a group into one layer, delete with contents;
  `check()` repairs a broken tree), groups in .ora both ways (nested `<stack>`, `isolation`, our own
  `studio-pass-through` so a faded pass-through group survives), the Layers panel as a tree. **The rest of P1:** the
  triangle picker and the wide-gamut OKLCH picker (sRGB and Display P3 edges drawn; past sRGB the chroma is lowered at
  the same lightness and hue, since the canvas is sRGB), swatch sets (`swatches.py`: .gpl both ways, Krita .kpl read,
  three built-in sets, palette from the painting by k-means in Oklab), the Keys tab and `keymap.py` (every action, two
  keys each, tablet keys arrive as ordinary keys, a key given to a new action is taken from the old one and said),
  `export.py` (PNG / TIFF / JPEG / **WebP**, ICC profile in every export, sRGB / Display P3 / Adobe RGB / ProPhoto,
  16-bit PNG and TIFF, the Studio's mark so `free_target` replaces only our own files; `.webp` added to
  `is_own_export`), **16-bit layers in .ora** (`Document.bits`, our own small PNG codec since Pillow reads 16-bit
  colour PNGs as 8-bit; recovery copies always 16-bit; MyPaint and Krita read 16-bit layers, GIMP unknown), the **LPM**
  blend style (`colour.lpm_filter` ported from the GPL-3 MATLAB code, `lpm_mix`; finding: the published code is an
  image filter whose transitions never reach the end colours, so it acts only in the middle of a mix, scaled by
  colour difference, lightness kept; the effect is small, about 1 %), and **side by side** of two branches
  (`UndoStack.picture_at`, `other_tip`; History tab button and right-click). Tests: 32 quick ones pass
  (tests/test_paint_engine.py 22, tests/test_paint_page.py 10).
- Same night, **P2 begun:** **smudge** (mode "smudge" in `brush.py`: smear carries a patch of pixels, dull one averaged
  colour; smudge length, colour rate, read this layer or every layer; presets Smudge (smear), Blender (dull), Oil
  (loaded brush); speed gate passes, p95 8-14 ms at 60 px). **Pigment mixing** (`colour.pigment_mix`, blend style
  "pigment", data `spectral_data.py` from spectral.js, MIT): blue + yellow at half = #3d933e as spectral.js; fast path
  (one brush colour cached, each distinct canvas colour once, each (colour, amount in 1/2048) pair once; a first
  version rounded the K/S weight and lost light touches of strong dark pigments, found by the test against the exact
  path and fixed); preset Gouache (pigment). **Speed: pigment fails the gate** at 30-60 px on detailed areas (p95 30-50
  ms per pen event on 2 cores, 12-17 ms on empty canvas); NumPy's per-call overhead on small dab pieces is the limit.
  Numba measured as the native path: 7x faster than the NumPy worst case but about 200 MB to install; Bill to decide
  (asked). **The MyPaint engine** (`mypaint.py`, libmypaint ISC; settings table `mypaint_settings.json`): settings and
  input curves, .myb v3 read and written, stroke and dab maths as listed in its header; not yet: spectral paint mode
  (WGM), smudge buckets, gridmap, directional offsets, colorize, posterize, snap to pixel. `brush.make_stroke` picks
  the engine; "Load MyPaint brushes" in the Brush tab, kept in `paint/brushes`. Speed gate passes (p95 13-16 ms at
  60 px). Not yet tried on real MyPaint packs (none in the workspace). Tests: 39 quick ones pass.
- **Not built yet:** history kept on disk (P1); from P2: texture / grain, dual brush, scatter and colour jitter
  options, the other engines (spray, chalk, sketch, hatching, ...), the brush editor with a live preview stroke,
  brush tags, imports (.abr, .sut, .kpp, .gbr / .gih), the 100-preset pack; the native pigment kernel if Bill agrees.