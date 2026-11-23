# Build plan: Portrait, second pass (one skin model, a light map, shine, pits, mixed-light white balance, the gaps)

Written 2026-11-18 with Bill, after an audit of what Portrait does today and a paper hunt for each gap. For the
builder (Claude Opus). Read section 0 before writing code.

**This is the next build** (Bill, 2026-11-18, 20:00: "add them to the build plan as the next build"). Another
build was running that evening (`BUILD_PLAN_NOISE.md`). **Start only when that work is committed.** The files it
was changing: `studio/app/main.py`, `studio/engine/pipeline/pipeline.py`, `studio/engine/pipeline/edit.py`,
`studio/cli.py`, `studio/cli_tools.py`, `docs/PROGRESS.md`, `docs/NEXT_UPDATE.md`.

**Priorities, set by Bill the same evening (20:11).** "The face-shape sliders, wide-angle faces, and glasses are
high priority. I know glasses glare is the absolute bane of photographers everywhere." And: "all the shaping and
sculpting stuff should be a priority because getting it to auto-detect or user select and having it work is
difficult and will take us time to figure out." So the **Shape track** (block H, with B.5, B.7, B.8) and
**Glasses** (block I) are built first, right after the bench, in small steps Bill can try early and often.
"Skip CSEC for now" (K.4 is parked). Section M is the order.

**Bill's goals, in his words:**
- "Contouring, I'm famously bad at it. I would love both a set of automatic contouring tools and manual ones that
  I can use to selectively dodge and burn."
- "We need a solution for scars and pitted skin."
- "Skin based white balance in mixed light is an awesome idea."
- "We could use shine control, that can be a problem."

**The idea in one paragraph.** Portrait today works on brightness and on OKLab colour. Skin is not an arbitrary
colour: almost all of its colour comes from two pigments, melanin (brown) and haemoglobin (red), plus shading
(light and shadow on the shape of the face) and a mirror-like shine on top that has the colour of the light, not of
the skin. Split a face into those four parts once and four gaps close with one model: redness and pigment become
separate sliders, freckles can be softened without being cloned over, shine can be taken off because we know what
skin colour lies under it, and the light's colour can be read from skin because we know what skin should look
like. The shading part is where contouring, dodge and burn, and pitted skin all live, so they share one **light
map**: a brightness-only layer that the automatic tools and Bill's brush both write into. Nothing in this plan
draws a pixel that was not in the photo.

---

## 0. Read first

### 0.1 Rules (all of `BUILD_PLAN.md` section 0 and `CLAUDE.md` apply; these bite here)

1. **Nothing redraws a face.** Bill: "avoid changing facial details without intention" (no CodeFormer, no GFPGAN,
   and so no GAN, diffusion or face-prior method here either). Every tool in this plan is a gain (brightness or
   colour multiplier), a warp of existing pixels, or a Heal / Fill spot the person can see and delete. Inpainting
   only ever happens inside a spot the person asked for.
2. **The gain-map form stays** (decision 45): face work is computed on a copy of the face at most 640 px wide
   and applied as a smooth multiplier, so pores are untouched and a 100% view equals the export. New tools use it.
3. **Every slider starts at 0 and is subtle at 100** (decision 42). New tools are off by default. Old edits render
   exactly as they did: new keys default to "off"; bump a stage version only if an old edit would change.
4. **More than one method where it helps, each with "best for / weak at"** (23.1), and results colour-coded by how
   they were produced: cyan measured, amber estimated, violet shared.
5. **Models:** permissive licence only (Apache-2.0, MIT, BSD), and not trained on research-only or non-commercial
   data (the choice already made for the composition models) unless Bill says yes for that model. Nothing here
   *needs* a model; where one is optional it says so.
6. **The candle rule.** White balance tools are buttons, never defaults, and honour "Keep the mood". A candle-lit
   face must stay candle-lit unless Bill moves the slider.
7. **New interface code goes in new files.** `studio/app/main.py` is about 7,600 lines. Do not grow it.
8. Nothing on C:\. Bill runs `.bat` files only. Do not push. One plain line per change in `docs/NEXT_UPDATE.md`.
   Errors report themselves. Tests on synthetic data with known truth, run before saying something works.
9. **Say what was not done.** Each block ends with numbers in `docs/RESULTS.md`. A tool that does not beat doing
   nothing on the bench is switched off and the result written down.

### 0.2 What exists (use it, do not rebuild)

- `studio/engine/local/portrait.py`: `Face`, `zones()` (skin, lines, under, smile, eye, teeth, blem, cheek, ext,
  chin, jawshade), `_skin_gate()`, `line_layers()` (oriented morphology: long dark grooves, round spots cancel),
  `face_gain()` (the gain map), `paste_gain()`, `find_blemishes()` (dark and red blobs, mole guard), `warps()`,
  `chin_skin()`, `_disp()`, `source_of()`, `apply_warps()` (capped, exactly inverted warps).
- `studio/engine/local/eyes.py`: `find_irises()` (mesh v2 iris points, the iris model, or the classical fit, each
  finished by an edge search), `eye_masks()`, `apply_eyes()`, `find_red_pupils()`, `pupil_fix()`.
- `studio/engine/local/masks.py`: `skin_weight()`, `smooth_skin()` / `_smooth_skin()` (two guided-filter layers,
  finest kept), `_GuideStats`, brush strokes stored as points (`"brush"` shapes), the `person` shape.
- `studio/engine/local/relight.py`: `normals()`, `fit_light()` (first-order spherical harmonics: where the light
  comes from), `gain()`. `studio/engine/local/depth.py`: `predict()` (the depth model in `models/depth`).
- `studio/engine/local/people.py`: person, skin, face, hair, clothes, arms, legs, torso; the hair segmenter and the
  multiclass model when present; BlazePose's 33 body points.
- `studio/engine/local/faces.py`: BlazeFace, the 468-point mesh, Face Landmarker v2 (478 points, irises).
- `studio/engine/local/retouch.py` (Heal, Clone, spots as data), `fill.py` (`fill()`, PatchMatch, LaMa when
  installed, `match_grain()`), `colourmatch.py` (settings solved to match a reference), `composition.py`.
- `studio/engine/color/autowb.py`: `estimates()` (grey edge, Grayness Index with its two-light split, Night),
  `keep_mood()`, `default_mood()`. `studio/engine/lens/geometry.py`, `lens/lines.py` (straight lines found).
- `studio/engine/dualpixel/depth.py` (depth from Dual Pixel files). `studio/app/process.py: process_files()`.
- Tests: `tests/test_portrait.py` (13), `tests/test_eyes.py` (10), `tests/test_people.py` (7).

### 0.3 What the audit found on 2026-11-18 (checked in the code unless marked "expected")

Built and on screen: Find faces, Click the eyes, Remove blemishes, Younger, Lines, Under eyes, Smile lines, Even
skin, Jawline, Double chin, Eyes, Teeth, Eye colour, the iris tools, Fix red eye, and Body sculpting (Push, Shrink,
Grow; breast, butt, waist, tummy, arm, thigh shapes). **None of it has been tuned or judged on a real portrait of
Bill's** ("awaiting a real portrait" in `docs/PROGRESS.md`). That is gap number one.

1. No shine control anywhere in the portrait code.
2. Even skin evens brightness and OKLab colour together: redness and pigment cannot be moved apart.
3. `find_blemishes()` looks only inside the face outline (`blem` zone). Shoulders, back, chest are never searched.
4. Inside the mesh the skin gate never falls below 0.5 (`gate = 0.5 + 0.5 * gate`), and the `lines` zone includes
   the skin round the mouth. **Expected, untested:** Lines fades moustache hairs and stray hairs on a forehead.
5. **Expected, untested:** Remove blemishes heals freckles as dark spots (up to its limit of 60).
6. The iris is a circle clipped by the eyelid outline. Lashes over the iris are inside the mask; a turned eye is an
   ellipse. **Expected:** brown to blue at 100% tints lashes slightly. An iris placed by hand is not built.
7. The face entry stores 468 points as x, y only. No depth (z) is kept and there is no triangle list.
8. One set of portrait sliders serves every face in a photo. A face can only be switched on or off.
9. `process_files()` never finds faces, so Portrait sliders pasted or applied across a shoot do nothing on photos
   where Find faces was not pressed by hand. (The command line can: `find --what faces`.)
10. Body warps move whatever is under them. Only Double chin is weighted by skin (`chin_skin()`).
11. A scan or JPEG has its colour profile undone, then the Studio's **own** Standard curve, not the camera's. So
    white balance and exposure on a JPEG are approximations. Nothing on screen says so.
12. Not built at all: contouring, dodge and burn, pits, glasses glare, brows and lashes, lips, face-shape sliders,
    wide-angle face stretch, flyaway hairs, background blur from depth, skin-based white balance (plan 23.9).

### 0.4 Decided by Bill, and what is still open

Decided 2026-11-18:
1. **Order:** Shape and Glasses first (section M).
2. **CSEC:** skipped for now (K.4 parked).
3. **Real portraits:** Bill has plenty. The first set is described in 0.5.
4. **"Light like this portrait":** Bill wants it, built so that it does not use the patented method. E.5 is now
   our own design. Read 0.6 before building it.

Still open. Ask once, in chat, in plain words (defaults if he is away):
1. **Unretouched raws** for the priority tracks: glasses with real glare (and a clean frame of the same sitter if
   there is one), a wide-angle group shot with faces near the edges, full-length people for body shaping, a
   background with straight lines. *Default: build on synthetic cases and the 0.5 set, and write down what could
   not be judged.*
2. **Public face sets as a measuring stick** (professionally retouched pairs; hand-marked wrinkle masks). They are
   non-commercial (CC BY-NC-SA). They would be used on Bill's PC to measure only, never shipped, never trained on.
   The tip jar is the part to weigh. *Default: do not download; the bench (block A) takes any folder Bill clears.*

### 0.5 Bill's first portrait set (looked at on 2026-11-18)

`D:\onedrive\Best Shots Folder\Portraits`: 16 JPEGs, Canon 5D Mark IV, 55 mm, f/3.5 to f/4, ISO 100 to 320, one
session against a mottled blue backdrop, one hard light from camera right. **Bill has already retouched them**
(saved from Photoshop CS6). So:

- **Not a "before"** for blemishes, lines or under-eyes: those are already gone. Do not tune the finders on them.
- **The same light on sixteen faces.** The measured light (E.4, E.5) and the skin white-balance estimate (G.1)
  must come out the same for all sixteen, whoever sits in front of the lens. A real-world truth test that needs
  no ground truth.
- **A wide range of complexions**, most of them deep skin tones. The pigment split (C), shine (D) and skin white
  balance (G) must hold across the whole range, not only on light skin (see N).
- **Shine:** several still carry strong highlights on forehead and cheek. Ask Bill whether he left them on purpose
  before using them as shine tests.
- **Glasses:** one sitter wears glasses, and there looks to be a second frame of the same sitter without them: a
  test for "From another frame" (I.3).
- Also: one profile (the face finder's weak case, B.8), long hair across a face and natural hair against the
  backdrop (hair guard, B.1), a tattoo on a chest (must be left alone), knitted patterns (warps show in patterns,
  H.3), and pairs of frames of the same sitter (recipes across a shoot, H.4).
- **Bill's finished look** is the measure for "subtle at 100": how much texture and shine he keeps.
- **Not in the set:** wide-angle shots, groups, full-length figures, straight-line backgrounds, candle light.
- These are private photos of people. Never copy them into the repo, the tests, the docs or an error report.

### 0.6 Patents and "clean room" (facts for Bill's decision; nobody here is a lawyer)

- **Clean-room work protects against copying code** (copyright). The Studio already works that way: methods are
  built from papers, never from research-only code. It does **not** protect against a patent: a patent covers the
  method, however it was arrived at.
- **Free software is not exempt in the US.** The statute reaches whoever "makes" or "uses" a patented invention,
  not only whoever sells it.
- **The honest ways round a patent:** do something its claims do not describe (a design-around); build from work
  old enough that any patent has run out (US patents last 20 years from filing); or ask the owner.
- **The one patent read so far, in part:** US 9,576,351 B1, "Style transfer for headshot portraits", Adobe, filed
  2015-11-19, shown as active, expected to run to 2035-11-19. Its first claim, as summarised from Google Patents:
  an input image and **two different reference images**; the input split into two levels of detail, each with an
  "energy"; the first level changed using the first reference's stored energy, the second using the second
  reference's; the levels added back. Claims after the fifth were not read. Whether the 2014 paper's own
  single-reference method has a separate patent is not known.
- **E.5 is designed to share none of those steps** (no splitting of the photo into detail levels, no energy
  matching, one reference). Whether that is enough is a lawyer's call and Bill's decision.
- Other blocks built from recent papers with Adobe, Google or Microsoft authors (H.2, E.2, D.2, F.3, J) have **not**
  been patent-searched at all. Say so in `THIRD_PARTY.md` when they are built.

---

## A. The bench first (everything else is judged on it)

`tools/portrait_bench.py`, with `Portrait Bench.bat`. Like the noise harness: truth where truth can be made.

1. **A synthetic face with known parts.** Extend the test face generator: a skin patch built from a known melanin
   map, a known haemoglobin map, known shading from a height field under a known light, a known shine lobe with the
   light's colour, pores, hairs (thin dark lines of known position), wrinkles, round pits and bumps in the height
   field, freckles, a mole, two lights of known colours with a known mixture map. Several complexions.
2. **Guards every tool must pass** (reported per tool, per amount):
   - *Pores kept:* energy of the finest layer inside skin after / before (target at least 0.98).
   - *Nothing moved:* for every tool that is not a warp, zero displacement of the mesh points and of hair lines.
   - *Hair kept:* contrast of the known hair lines after / before.
   - *Stayed on skin:* change outside the skin mask (eyes, lips, brows, hair, background) below a set level.
   - *Truth error:* against the known pigment maps, shine layer, light colour, mixture map, pit depth.
3. **A folder mode** for real portraits: before / after sheets and a "what changed" picture per tool into
   `outputs/portrait_bench/`, so Bill can judge twenty faces in one sitting.
4. **Prove or clear the three expected faults first** (0.3 items 4, 5, 6) with tests that fail today if the fault is
   real. Write the result in `docs/RESULTS.md` either way.

## B. Faults and blockers in what ships (B.5, B.7, B.8 go with the Shape track; order in section M)

- **B.1 Hair guard.** Hairs are thin, dark and long like wrinkles; what tells them apart: they are darker than a
  groove, colour-neutral or hair-coloured instead of skin-coloured, often run past the skin zone, and come in
  dense fields (stubble, moustache, brows). Build a hair-likeness map from `line_layers()` at the finest scale plus
  colour, add the hair segmenter where present and a beard / moustache prior from the mesh (upper lip, chin, jaw),
  and refine the skin map against the photo with the guided filter so hair-like structure drops out of it
  (FabSoften's "guided feathering"). Lines, Even skin, Smooth skin and every new skin tool multiply by it.
- **B.2 Freckles.** Many similar shallow dark spots are freckles, not blemishes. `find_blemishes()` counts them
  and leaves them, and says so ("38 freckles left alone"). A new control **Freckles: Keep / Soften** (soften uses
  the pigment layer, C.4; nothing is cloned).
- **B.3 Iris mask.** Lashes: run the line finder inside the iris disc and take dark thin lines out of the mask.
  Turned eyes: fit an ellipse in the edge search (axis ratio from the mesh's head turn as the first guess). Add
  "Place the iris by hand" (two clicks: centre and edge).
- **B.4 Faces across a shoot.** `process_files()` and the command line find faces by themselves whenever the edit
  holds an active Portrait slider and no faces. Report how many photos had none.
- **B.5 Per-person settings.** A face entry may carry its own amounts (`"amounts": {...}`) over the shared ones.
  On screen: a row of face thumbnails; click one to set that person, "Everyone" to set all.
- **B.6 Blemishes off the face.** The same finder on Person > Skin (shoulders, back, chest, arms), with the size
  taken from the body points instead of the face width.
- **B.7 Sculpting leaves the room alone.** Weight every warp by the Person mask the way Double chin is weighted by
  skin, so a wall or door frame behind a waist stays straight; where background is uncovered, stretch it by at
  most a few percent and report it, with "Fill behind" (existing Fill) as the option. Then the to-do items:
  shapes that snap to the body points, handles, both sides at once.
- **B.8 Small faces.** Run the detector on tiles when the photo is large and the first pass finds faces under
  about 60 px (group shots). Profiles with one eye visible: say plainly that they are not handled yet.

## C. The skin model (the centre of this plan)

New file `studio/engine/local/skinmodel.py`. No GUI imports.

- **C.1 The maths, in plain words.** Pigments absorb light, and absorption multiplies. Take the log of the
  white-balanced, scene-linear skin colour and the multiplying turns into adding: log colour = a melanin amount
  along one colour direction + a haemoglobin amount along another + shading (the same in all three channels) + a
  small remainder. Solve those three numbers per pixel. That is Tsumura, Haneishi and Miyake's 1999 separation.
- **C.2 The two colour directions.** Do not trust a per-photo guess alone. Start from fixed directions (shared
  defaults, violet), refine them on this face's own skin pixels by independent component analysis (a small FastICA
  in NumPy, no new package), and refuse any answer that drifts more than a set angle from the defaults (fall back,
  amber). When the camera has a Studio profile, derive the starting directions through its colour matrix (cyan).
- **C.3 Coloured light.** A candle skews the directions. Always solve on a neutralised copy (the light taken out
  with the best available estimate, block G), then apply the resulting gains to the photo as Bill balanced it.
  Report confidence; below it, the pigment tools grey out with the reason.
- **C.4 What it returns:** melanin map, haemoglobin map, shading map, remainder, confidence, on the face work copy.
  Cached per face like the gain map. Body skin uses the nearest face's directions.
- **C.5 Tools built on it** (all in `portrait`, all 0 by default, all through the gain map):
  - **Redness** (-100..100): evens the haemoglobin map at blotch sizes; below zero calms overall flush.
  - **Pigment** (-100..100): evens the melanin map (sun spots, patchy tan); the overall level moves only on its
    own **Tan** control, so nobody's complexion changes by accident.
  - **Even skin** gains a size control and works per pigment (the 2011 Ojima study: unevenness by pigment and by
    blotch size together). The old Even skin stays as method 1 for old edits.
  - **Freckles: Soften** (B.2) and **Fade** per blemish: fit the spot's own soft bump in the two pigment maps
    (a few Gaussians, as in the CGFR paper), scale it 0..100%, leave texture alone. Heal stays as the other method.
- **C.6 Views** (Bill: the flashy things are what people see): a melanin view in browns, a haemoglobin view in
  reds, the shading view in greys, switchable on the photo. They are also the "what changed" views.
- **C.7 Tests:** recover the known maps on the synthetic face for several complexions and three light colours;
  the candle case must either pass or grey out, never produce confident nonsense.

## D. Shine

In `portrait`: **Shine** (-100..100). Below zero takes shine off; above zero adds a healthy glow. Two methods.

- **D.1 "Skin model" (default).** A pixel is skin colour (C) plus a shine amount times the light's colour. With
  the pigment model saying what the skin under it should look like from its neighbours, solve the shine amount per
  pixel (Li, Lin, Zhou, Ikeuchi, CVPR 2017). Remove a share of it as a gain. Partly clipped pixels: use the
  channels that are not clipped. Fully clipped areas hold nothing to recover: lower them to the surrounding skin's
  level only, never paint texture, and say "blown here" in the status line (the highlight rebuild and Fill exist
  for that). *Best for:* oily forehead, nose, cheek hot spots in any light colour. *Weak at:* large blown patches.
- **D.2 "By detail" (band-sifting).** Split skin brightness by size, by strength and by sign; shine is the bright,
  strong, fine part. Scale it (Boyadzhiev, Bala, Paris, Adelson, 2015). *Best for:* a quick matte or dewy look,
  body skin. *Weak at:* it cannot tell shine from any other small bright detail (a rim of light, wet lips), so
  keep it gentle.
- The mesh gives a T-zone weight (forehead, nose, chin) as a starting mask the person can change.
- Tests: the known shine layer on the synthetic face, two light colours, with and without clipping.

## E. The light map: dodge and burn, and contouring

One brightness-only layer per photo, in stops. Everything in this block writes into it, so Bill can paint over
what the automatic tools did. New files `studio/engine/local/lightmap.py`, `studio/app/lightmap_panel.py`.

- **E.1 The map.** `edit.light_map = {"strokes": [...], "even": 0..100, "features": 0..100, "contours":
  0..100, "preset": "...", "amount": 0..100}`. Strokes are stored as points like mask brush strokes (so they are
  exact at any size and stay on the subject through crops); the automatic layers are computed at render. Applied
  as `2 ** map` on scene-linear light, capped at about one and a half stops either way, smooth by construction.
  **Show map** draws it in grey over the photo. **Export the map** writes it as its own 16-bit file for anyone who
  wants it as a Soft Light layer in Photoshop.
- **E.2 The brush (manual dodge and burn).** Tools **Dodge** and **Burn** (X swaps them), size, flow, feather.
  **Stay inside edges:** a stroke's value is spread to the region it sits on and stops where brightness or colour
  changes sharply, using the existing guided-filter machinery on a small copy (the idea of Lischinski, Farbman,
  Uyttendaele and Szeliski, 2006: a few strokes and a slider instead of careful masking). Each stroke keeps its
  own strength slider afterwards, so a stroke can be eased back next week without repainting it.
- **E.3 Even (automatic).** The retoucher's first pass: flatten uneven light on skin at middle sizes. Work on the
  **shading** part from block C, so a freckle or a flush is never mistaken for a shadow. Generalises what Under
  eyes and Smile lines do today to the whole face, neck and hands.
- **E.4 Sculpt (automatic contouring; our own design).** A contour template drawn once on the standard face mesh:
  darker in the hollow under each cheekbone, the temples, the sides of the nose, under the jaw and along the
  hairline; lighter on the top of the cheekbones, the bridge of the nose, the brow bone, the centre of the
  forehead, the chin and the cupid's bow. Stored as one number per mesh point in
  `studio/engine/local/contour_template.json`, laid onto each face through the mesh triangles, blurred, and
  **scaled by the light already in the photo** (fit with `relight.fit_light()` on the face's own surfaces): it
  deepens the modelling the light began, and does nothing on the side the light does not reach. Two sliders, the
  split Evoto uses: **Features** (eyes, nose, lips) and **Contours** (bone structure). Presets:
  Natural, Soft, Angular. Needs: the mesh's depth kept in the face entry (`"z"`, with the standard face's depth as
  the fallback for old edits) and the mesh triangle list (from MediaPipe's face mesh, Apache-2.0; add to NOTICE).
- **E.5 Light signatures ("Light like this portrait"; our own design; read 0.6 first).** *Measure* a reference
  portrait's light as a short list of numbers: (1) the light fitted on its face surfaces with
  `relight.fit_light()`: where the key light comes from and how strong it is against the fill (spherical-harmonic
  lighting: Ramamoorthi and Hanrahan 2001; Basri and Jacobs 2003, already the basis of Relight); (2) how much
  darker or lighter each zone of the contour template (E.4) is than the face's average shading: the cheek hollow
  against the cheekbone, the side of the nose against its bridge, and so on, about twenty numbers; (3) how deep the
  shadows go and how soft the edge between light and shadow is. That list is a **light signature**: a small text
  file, `profiles/light_signatures/NAME.psblight`, share-alike like the lens signatures. *Apply* it by measuring
  the same numbers on the photo in hand and writing into the light map the smooth gains that move them toward the
  signature (the ratio of wanted shading to measured shading: ratio-image relighting, 2001 to 2003), by Amount.
  The result is an ordinary light map Bill can paint on. It works on shading only: it never copies skin, pores or
  colour from the reference, and it does not split the photo into detail levels. Said on screen: it moves light
  that is there; a flat-lit face will not become hard-lit beyond the cap. First signatures: Bill's own best-lit
  portraits. Test: the sixteen portraits of 0.5, shot under one light, must give sixteen near-identical signatures.
- **E.6 Body.** The brush works anywhere. Automatic body contouring is not in this build.
- Tests: strokes stop at a known edge; Even flattens known uneven shading and leaves known pigment alone; Sculpt
  on the synthetic face is symmetric under frontal light and one-sided under side light; guards from block A.
- **Acceptance:** Bill, who says he is bad at contouring, gets a face he would keep from two sliders, and can fix
  what he disagrees with using the brush.

## F. Scars and pitted skin

Found in the hunt: no retouching paper solves this. The useful fact is from dermatology, which measures pitted
scars with 3D scanners because in an ordinary photo they are only light and shadow. So: treat them as shading.

- **F.1 Find them.** Take the shading map (C), subtract the face's smooth shape (the clinical method subtracts a
  fitted surface and measures the valleys that remain), and look for round or irregular hollows from about 0.3%
  to 3% of the face width. Use the light direction (`fit_light`): a pit is dark on the side nearest the light and
  bright on the far side; a bump is the reverse; a pore is too small; a blemish shows in the pigment maps, not
  here. Our own addition, untested: the bench decides.
- **F.2 Soften them.** **Pits** (0..100): reduce both halves of each hollow's shading in the gain map, never the
  finest layer. Wide soft hollows use the fold layer already there. Tiny deep ones (near-black) may be offered as
  Heal spots (amber, deletable). **Raised scars** are the same tool with the sign reversed.
- **F.3 Second method: "By detail"** (band-sifting again: the dark, middle-strength part at pit size).
  *Best for:* an overall rough cheek. *Weak at:* it softens every small shadow, pits or not.
- **F.4 Say the limit on screen:** this softens pitted skin; under hard side light at 100 some shape remains,
  because the alternative is painting new skin.
- Tests: pits and bumps of known depth in the synthetic height field under three light directions; pores kept.

## G. White balance from skin, and mixed light

New file `studio/engine/color/skinwb.py`; the interface in a new panel file.

- **G.1 Skin as the reference (a fourth estimate in `autowb.estimates()`).** Skin tones sit close together in
  colour space, so skin's departure from where it should be gives the light's colour (Bianco and Schettini, 2012
  and 2014, tested on raw files). Their weak point is that people differ; remove the person first with the pigment
  split (Tanaka, Kakinuma, Kamijo, Takahashi, Tsumura, 2015): what is left after melanin, haemoglobin and shading
  is the light. Use cheek and forehead skin with shine (D) and deep shadow left out. Shown beside the other three
  estimates with its confidence (amber); it never applies itself, and "Keep the mood" applies as always.
- **G.2 Two lights.** Trigger: the faces in a photo disagree, the two sides of one face disagree, or the Grayness
  Index already reports two lights. Then offer **Mixed light**: two light swatches (found, or picked with two
  clicks), a per-pixel mixture map of how much of each light reaches each place (Hsu, Mertens, Paris, Avidan,
  Durand, 2008), solved on a small copy and carried to full size along edges. The correction is a blend of two
  ordinary white balances by that map, which on a raw is exact. Controls: Amount, and Balance (which light to
  neutralise more: a window can go neutral while the candle stays warm). The map can be shown.
- **G.3 Scribbles.** Three strokes the person can add when the automatic map is wrong: "this is neutral", "this is
  skin", "these two are the same colour" (Boyadzhiev, Bala, Paris, Durand, 2012). Faces are automatic "skin"
  strokes.
- **G.4 Skin only.** A small option: correct the cast on this person's skin alone (green bounce from grass).
- Tests: skin of several complexions under known lights (error in degrees, as for the other estimators);
  two-light scenes with a known mixture map. Real: Bill's candle and window portraits.

## H. Shape: faces, wide-angle faces, bodies (PRIORITY TRACK)

Bill: "getting it to auto-detect or user select and having it work is difficult and will take us time to figure
out." So this block is built first, in small steps, each shown to Bill. It has its own sheet-maker,
`Shape Bench.bat`: before / after pairs with the found outlines drawn, into `outputs/shape_bench/`. Everything is a
warp on the existing machinery (capped, exactly inverted, pixels sampled once). New files
`studio/engine/local/shape.py`, `studio/app/shape_panel.py`.

- **H.0 Three rules for every shape tool.**
  1. *Find it, or be shown it.* Each tool first tries to find its part by itself. If it is unsure it says so and
     asks for one click. The person can always correct it by dragging handles. Cyan when found, amber when guessed.
  2. *The room stays still.* B.7's guard is part of every warp.
  3. *A recipe, not a painting.* What was done is stored relative to the face or body ("waist 6% narrower"), so it
     can be applied to the next frame of the same person (H.4).
- **H.1 Face shape sliders.** From the mesh, so they need no clicks: Face width, Cheeks, Jaw width, Chin length,
  Forehead height, Nose width, Nose length, Eye size, Lip fullness, Smile lift (the corners, a little), and
  **Even up** (bring left and right closer: one lower brow, one smaller eye). Built like Jawline today: push and
  scale primitives placed on mesh points and sized by the face. Caps of a few percent of the face. Irises stay
  round (Eye size scales, never stretches). Hair and ears follow smoothly past the face outline. On a turned head
  each side is scaled by how much of it faces the camera (from the mesh's depth). Per person (B.5). None of them is
  part of Younger.
- **H.2 Stretched faces in wide-angle shots** (Develop > Lens: "Fix stretched faces").
  - *Why it happens:* ordinary perspective stretches anything round near the edge of a wide frame.
  - *Find:* field of view from the focal length and sensor size; faces from the finder. The stretch at each face
    follows from geometry, so the tool can say "3 faces stretched by 9 to 22%". Suggested when the lens is wider
    than about 28 mm equivalent and a face sits in the outer third.
  - *Be shown:* click a face to include it or leave it out. One Amount slider.
  - *Method* (the known recipe: Shih, Lai and Liang, 2019; Tan et al., arXiv 2104.12464; built here as a mesh
    optimisation, no network): lay a mesh over the photo; inside each face, with its hair and neck from the Person
    mask, aim for the stereographic view, which keeps round things round; elsewhere keep straight-line
    perspective; keep the lines `lens/lines.py` found straight; change smoothly between; solve; resample once,
    together with the lens geometry stage.
  - Contactless lenses give no focal length: ask once and remember it for that lens.
- **H.3 Bodies: find the part, or be shown it.** Today the person draws an oval. New:
  - *Find:* from the 33 body points (with their depth and how visible each is), the Person mask and its parts: a
    centre line and a silhouette edge on each side for torso, waist (the narrowest place between ribs and hips),
    hips, each upper arm, forearm, thigh and calf, neck, shoulders. Facing or side-on is read from the points'
    depth; a tool that only makes sense one way (Tummy side-on, Waist facing) says so.
  - *Be shown:* moving the pointer lights up the part under it with its name; one click selects it. Where nothing
    was found, a click and drag along the limb defines it. Handles on the outline. "Both sides" mirrors.
  - *Warp along the outline, not a blob:* Slim moves the silhouette edge toward the centre line along the edge's
    own direction, fading to nothing at the centre line and past the ends of the part, so a limb narrows without
    shifting and the joints stay where they are. Lift and Size as today. New: Height and Legs (a gentle stretch
    below the hips with the feet kept on their floor line), Shoulders, Neck.
  - *Guards:* B.7; a warning where a warp crosses a pattern (stripes, knit, lettering: found as strong regular
    texture); hands and faces are never moved by a body tool.
  - The ovals stay, as the manual fallback and for old edits.
- **H.3 status (session 19, 2026-11-23):** built in `engine/local/bodies.py` + `portrait.py` (`outline`, `stretch`) + the
  Body sculpting panel (Find bodies, Pick a part, Both sides, Longer legs, drag along a limb). Differs from the text
  above: facing or side-on is read from the points' 2D layout (shoulder width against torso length), because the
  stored points carry no depth; Chest and Seat place the existing lift ovals (amber) rather than a new lift. Not yet:
  Height, Shoulders as a "broader" preset (Slim -100 does it), the pattern warning, handles on the outline, B.7 for the
  ovals. Finished only when Bill says the finding and the selecting work on his photos.
- **H.4 Recipes across a shoot.** Because shape changes are stored relative to the mesh and the body points,
  "Apply to the other frames of this person" works. The Studio suggests frames in the same folder that look like
  the same sitter (mesh proportions, clothing colour, time); the person confirms. With B.4 this makes shape work
  usable on a whole session.
- **H.5 Tests.** Synthetic figures with known measurements: a waist of known width narrows by the asked percent,
  joints do not move, straight lines behind stay straight within a pixel, every warp inverts exactly. Synthetic
  wide-angle scenes with spheres for heads: a sphere must come out round. Then Bill's eye on the Shape Bench
  sheets, early. **This block is finished when Bill says the finding and the selecting work.**

## I. Glasses (PRIORITY), then brows and lashes

Bill: "glasses glare is the absolute bane of photographers everywhere." New file
`studio/engine/local/glasses.py`, panel in its own file.

The frame to keep honest: glare is light added on top of the eye. Where some of the eye still shows through, the
added light can be taken off. Where the glare is blown white, nothing lies under it, and rule 1 forbids painting
an eye. Then the only true sources are another frame of the sitter, or the sitter's other eye.

- **I.1 Find it.** *Glasses:* frame edges round the eye regions the mesh gives (thin strong edges, mirrored about
  the nose). *Glare:* patches inside the lens area that are brighter than the skin beside them, colourless or
  tinted green, purple or blue (lens coatings), with sharp borders that do not follow the face. Each patch is
  classed as a **veil** (detail shows through) or **blown** (clipped), with what it covers (iris, lid, cheek,
  frame). Said in plain words: "Left lens: a veil over the iris (can be lifted). Right lens: a blown patch over
  half the iris (needs another frame)." Also reported: the frame's shadow on the cheek.
- **I.2 Method "Lift the veil"** (see-through reflections). A reflection adds light, which lowers contrast inside
  the patch. Estimate the patch's own colour and strength at its border, where the same skin carries on outside
  it; subtract it; remove the border edges that belong to the patch and not to the face; put the area back
  together smoothly with the solver Heal already uses (`retouch.py`). On a frontal face the other eye guides it:
  an edge present round one eye and absent round the mirrored other is reflection (the symmetry idea of Sandhan
  and Choi, CVPR 2017). *Best for:* window and softbox veils, coating tints. *Weak at:* blown patches; heads
  turned far.
- **I.3 Method "From another frame"** (what retouchers do by hand). **Find a clean frame** looks in the same
  folder for frames of the same sitter, close in time, where that lens area is clean or cleaner: glare moves when
  the head tips, and photographers often take one frame with the glasses tilted or off. The eye area is lined up
  through the mesh and laid in as a Composite spot with a seamless edge, matched in colour and brightness. The
  person sees which frame was used, side by side, and can pick another. *Best for:* any glare, when a second frame
  exists. *Weak at:* a changed gaze or expression between the frames.
- **I.4 Method "Other eye"** (frontal faces, last resort): the sitter's own other eye, mirrored, with the
  catchlight put back on the light's side. Marked amber. Off by default.
- **I.5 Glare off the eye** (on the frame, a cheek, the lens edge): ordinary Heal / Fill spots, made for review.
- **I.6 A shooting tip in the panel:** "Take one extra frame with the glasses tipped down a little. The Studio can
  use it."
- **I.7 Brows and lashes: Define** (darken, fill gaps) from the mesh's brow outlines and B.3's lash mask.
- Looked at and not used: the recent networks for eyeglass reflections (2024 to 2026) pair removal with
  inpainting, which paints the eye.
- Tests: a synthetic eye behind a synthetic lens with a known added reflection (veil strength recovered; a blown
  area reported, never filled in); the glasses pair in 0.5; then Bill's real glare shots.

## J. Background blur from depth (Finish)

Depth from the depth model, from Dual Pixel when the file has it, and the Person mask for clean edges (the recipe
phones use: Wadhwa et al., 2018). Blur in scene-linear light so bright points bloom the way a lens makes them,
layered so near things stay in front of far ones. Controls: amount (as an f-number), focus point by click.
Labelled "simulated" (amber). Hook for later: the blur's shape from a lens signature. Off by default.

## K. JPEGs and scans

- **K.1 Say it.** White balance and exposure on a JPEG or scan are marked "estimated" (amber) with one line why.
- **K.2 Studio JPEGs that reopen like raws.** On export the Studio knows exactly how it got from scene light to
  the JPEG. Store a compact description of the way back inside the file (well under 100 KB; the idea of Nguyen
  and Brown, CVPR 2016). Opening a Studio JPEG later then restores near-raw latitude. Test: export, reopen,
  compare with the raw render.
- **K.3 Other people's JPEGs.** A camera's processing has too many stages for one curve per channel to undo, so
  only offer an estimate: a tone-curve family chosen from the EXIF make, model and picture style. Experimental.
- **K.4 Exposure repair with CSEC: parked.** Bill, 2026-11-18: "Skip CSEC for now." Not in this build.

## L. In the Studio

- **Portrait** becomes five short panels, each in its own file under `studio/app/`: **Skin** (Blemishes, Fade,
  Freckles, Even, Redness, Pigment, Tan, Shine, Pits, Lines, Under eyes, Smile lines), **Light** (Dodge, Burn,
  Even, Sculpt, Show map), **Eyes** (with Glasses), **Shape** (faces, wide-angle faces, bodies), **People** (the
  thumbnails of B.5).
  Simple level shows one slider per job; Full shows methods with their "best for / weak at" lines.
- Every automatic tool has **Show what changed** and its own view (pigment maps, light map, mixture map).
- Command line (`docs/CLI.md`, tests in `tests/test_cli_tools.py`): `portrait` (apply sliders, faces found by
  itself), `find --what pigment | shine | pits | lights`, `light-map` (export). The MCP server follows by itself.
- Looks carry the new sliders; faces, strokes and sculpting stay with their own photo, as now.

## M. Order of work (Bill's priorities, 2026-11-18)

1. **A** (bench, kept lean), the three prove-or-clear tests, and the Shape Bench.
2. **Shape track:** B.7 (the room stays still), B.5 (per person), B.8 (small faces), H.1 (face sliders), H.3
   (bodies found or shown), H.2 (wide-angle faces), H.4 (recipes across a shoot). Show Bill after each.
3. **Glasses:** I.1 to I.6.
4. **Light map:** E.1, E.2 (the dodge and burn brush), E.4 (Sculpt, the automatic contouring).
5. **Faults in what ships:** B.1 to B.4, B.6.
6. **C** (skin model), then **D** (shine), then **E.3** (Even).
7. **E.5** (light signatures).
8. **F** (pits).
9. **G** (skin white balance, then mixed light).
10. **L** tidy-up, I.7, **J**, **K.1**, **K.2**.
11. Only if wanted later: K.3. Parked: K.4.

"Sculpting" in Bill's message is read here as the shape tools (faces, bodies, wide-angle). If he also meant
contouring (Sculpt, E.4), it is already next in line at step 4.

Steps 1 to 3 are Bill's stated priorities. Steps 4 to 9 are everything else he asked for by name.

## N. Risks, said plainly

- **Candle light and the skin model.** The pigment split was validated on flat, neutral clinical light. Bill's
  signature work is the opposite. C.3 is the mitigation; the bench and his candle raws decide. If it fails there,
  the pigment tools grey out in that light and the brightness-only tools (light map, pits) still work.
- **Sculpt can look painted** on a flat-lit face or a face it reads wrongly. That is why it follows the existing
  light, is capped, and lands in a map Bill can paint over.
- **Pits are only softened.** Anything stronger means inventing skin, which rule 1 forbids.
- **The mixture map can be wrong** where one material sits under both lights with no neutral nearby. Scribbles
  (G.3) are the answer, and the map is always shown.
- **Patents and licences** were checked only as far as 0.6 and section O say. Free software is not exempt, and
  clean-room work does not answer a patent. A lawyer's look before E.5 and H.2 go out to Bill's group is the
  careful path; it is Bill's decision.
- **Complexions.** Much of the skin-pigment work in the papers was measured on light skin. Bill's first set is
  mostly deep skin tones, where melanin dominates, redness is harder to see and shine stands out more. C, D and G
  must be shown to work across the range on the 0.5 set before they are switched on. Where the split is
  unreliable on a face, those tools grey out there instead of guessing. Face and iris finding get the same check.
- **Finding body parts is the hard part** (Bill said so). Expect several rounds with his eye. The fallback at
  every step is the hand-drawn oval that works today.
- **Glasses:** a blown patch cannot be undone from one frame. The Studio says so instead of inventing an eye.
- **Nothing here has seen an unretouched face yet.** Bill's first set (0.5) is retouched. Ask for the raws of
  0.4 early.

## O. Sources (what was read, and how far)

Read in full text (in `D:\Analyst_Toolkit\Retouching Related Papers`): CGFR, physics-based blemish fading
(`2406.13227v1.pdf`); FabSoften (`Velusamy_FabSoften...pdf`); wrinkle finder plus inpainting
(`2211.01930v1.pdf`); AutoRetouch (`Shafaei_AutoRetouch...pdf`); CSEC (`2405.17725v2.pdf`); Iris-SAM
(`2402.06497v2.pdf`); Afifi's thesis (`2107.13117v1.pdf`: abstract, contents, skin and face passages only).

Read as abstracts or summaries only (2026-11-18); read the paper before building from it:
- Tsumura, Haneishi, Miyake, "Independent-component analysis of skin color image", JOSA A 16(9), 1999.
- Ojima et al., unevenness of skin colour by melanin and haemoglobin components, SPIE 7897, 2011.
- Li, Lin, Zhou, Ikeuchi, "Specular Highlight Removal in Facial Images", CVPR 2017.
- Boyadzhiev, Bala, Paris, Adelson, "Band-Sifting Decomposition for Image-Based Material Editing", ACM TOG 2015.
- Lischinski, Farbman, Uyttendaele, Szeliski, "Interactive Local Adjustment of Tonal Values", SIGGRAPH 2006.
- Shih, Paris, Barnes, Freeman, Durand, "Style Transfer for Headshot Portraits", SIGGRAPH 2014 (**not** built;
  see 0.6). US 9,576,351 B1 (Adobe), read in part on Google Patents.
- Shu, Hadap, Shechtman, Sunkavalli, Paris, Samaras, "Portrait Lighting Transfer Using a Mass Transport Approach",
  ACM TOG 2017.
- Pegu, "Lightweight Additive Blend Maps for Texture-Preserving Face Retouching", AAAI 2026 (the map idea; no
  weights taken).
- Petit et al., 3D skin imaging of atrophic acne scarring, Skin Research and Technology, 2018.
- Bianco, Schettini, "Color Constancy Using Faces", CVPR 2012; "Adaptive Color Constancy Using Faces", TPAMI 2014.
- Tanaka, Kakinuma, Kamijo, Takahashi, Tsumura, illuminant estimation by pigment separation, SPIE 9394, 2015.
- Hsu, Mertens, Paris, Avidan, Durand, "Light Mixture Estimation for Spatially Varying White Balance", 2008.
- Boyadzhiev, Bala, Paris, Durand, "User-guided White Balance for Mixed Lighting Conditions", 2012.
- Afifi, Brubaker, Brown, mixed-illuminant white balance, arXiv 2109.08750 (idea only; licence not confirmed).
- Tan et al., wide-angle portrait correction, arXiv 2104.12464 (recipe only; no network).
- Sandhan, Choi, "Anti-Glare", CVPR 2017. Wadhwa et al., synthetic depth of field, 2018.
- Nguyen, Brown, raw reconstruction from a self-contained JPEG, CVPR 2016.

From memory, not checked that day: Shih, Lai, Liang, "Distortion-Free Wide-Angle Portraits on Camera Phones",
2019; ratio-image relighting (Shashua and Riklin-Raviv 2001; Wen, Liu and Huang 2003).

Looked at and ruled out: GAN and diffusion retouchers and shine removers (they redraw the face); Alibaba's
body-reshaping model (academic and non-commercial use only); Afifi's white-balance and exposure code
(research only); networks for wide-angle faces and glasses that use generative priors or inpainting.

Not searched yet: flyaway hairs, facial-hair segmentation, lips, detectors for small or profile faces, and
patents for anything except the one in 0.6.
