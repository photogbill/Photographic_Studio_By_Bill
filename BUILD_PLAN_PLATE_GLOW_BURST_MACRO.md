# Build plan: Card to finished stacks, Best of burst, Macro planner, Plate Reader, Glow guard (and the variable IR filter)

Written 2026-11-17, at the end of session 13, for a fresh session. Bill picked four of the ideas offered at the end of
session 13: "I would say 1-4, but make it a build plan." The four, in his order: **(1) Plate Reader** (multi-light
capture: relight an object, see its surface, tell real colour from tarnish), **(2) Glow guard** (keep a character
lens's glow safe from the haze and contrast tools), **(3) Best of burst**, **(4) Macro stack planner**. Block A (the
variable infrared filter) came from his answer about filters in the same message and is small; do it first.
**Block F (card to finished stacks) came an hour later and is now the main job** (0.5): Bill hand-holds every macro
shot, focuses by hand, and wants to dump a whole card of mixed stacks and get finished stacks back.

Pattern as before (`BUILD_PLAN_TABS.md`, `BUILD_PLAN_PANO_IR.md`): reuse what exists, synthetic-truth tests for every
algorithm, then the pages. If the session runs out, stop at the end of a block (section 6) and say exactly what is and
is not built.

---

## 0. Read first

### 0.1 Rules (all of `BUILD_PLAN.md` section 0 and `CLAUDE.md` apply; these are the ones that bite here)
1. **Do not commit or push.** Bill pushes with `push.bat`. One plain line in `docs/NEXT_UPDATE.md` per part built.
2. **Nothing written to C:\ or the Windows profile.** Per-PC data (glow profiles, focus-increment calibrations, filter
   white balances) under `paths.get("CALIBRATION_LOCAL")`, like the hot-spot, dust-map and glow profiles.
3. **Names:** single-photo work is an ordinary edit (sidecar), exported as `NAME_Studio.ext`. Multi-frame results:
   `FIRST_to_LAST_RTI_Studio.tif` (albedo, linear), `NAME_RTI-relit_Studio.jpg`, `NAME_RTI-normals_Studio.png`
   (`stack/frames.result_name`, `free_target`). Best of burst **never deletes**: it rates and flags only.
4. **Every method the person can choose has METHOD_INFO** (best for / weak at). Colours: **cyan measured** (light
   directions from the sphere, a measured glow profile, a calibrated focus increment, eyes found and measured),
   **amber estimated** (glow estimated from the photo, subject guessed without a face, planner numbers from
   formulas), **violet shared** (shipped starter values).
5. **Errors report themselves** (`common.report_problem`, `log.save_error_report`).
6. **No OpenCV, no new packages** without asking Bill (`tests/test_imgops.py` enforces it). NumPy/SciPy only.
7. **Tests** on synthetic data with known truth, writing only under `outputs/`; run the whole suite before saying it
   works (about 20 minutes on the 2-core test machine).
8. **Bill does not use the command line.** Questions in chat, plain words, with a recommendation (0.4).

### 0.2 What exists that you will use (do not rebuild)
| Need | Where | Notes |
|---|---|---|
| Lens glare spread function (measured) | `raw/glow.py`: `Profile`, `kernel_image`, `measure_profile`, `pick_profile`; `pipeline.glow_report`, `measure_glow_profile`, `glow_profiles` | Measured once per lens from a bracket of a small light in the dark (cyan). Used today to recover blown lights (DESAT idea). **The Glow guard's cyan tier.** |
| Character lenses | `lens/library.py`: `LensEntry.character` | Laack, Heliar, Petzval: "Recover lens sharpness" already stays off for them. Glow guard defaults on for them. |
| The stages that eat glow | `pipeline._post`, in order: veil removal (`detail/scales.apply_veil`), Dehaze (`detail/finish.dehaze`), Auto HDR (`color/hdr.py`), Detail by scale (`detail/scales`), Texture, masks (local dehaze, clarity, texture) | All on scene-linear light before the tone curve: one place to wrap (B.3). |
| Flare field machinery | `local/flare.py` (membrane background, normalised convolution), `pipeline.flare_map` (memo per raw settings, proxy size) | Same memo pattern for the glow layer. |
| Faces, eyes | `local/faces.py` (BlazeFace, face mesh 468 points, iris), `local/eyes.py` | Eye sharpness and eyes open/closed for Best of burst. Human faces only. |
| People, depth, objects | `local/people.py` (person mask), `local/depth.py` (Depth Anything V2 Small), `local/sam.py` | Subject when there is no face. |
| Noise model | `calib/noise.py`, `image.noise_model` | Sharpness minus what noise alone gives (as `stack/focus.py` does), so a noisy frame never wins on its noise. |
| Sharpness energy | `stack/focus.py` (scale-normalised Laplacian energy, Lindeberg 1998, noise-corrected) | Reuse for burst ranking and the stack gap report. |
| Focus Stack | `stack/focus.py` (depth index, confidence, breathing), `app/focusstack.py` | The planner and gap report go on this page. |
| Alignment | `stack/align.py: register` (translation .. homography), `phase_shift` | Plate Reader frames are on a tripod: translation is enough. |
| Stack pages | `app/stackui.py: StackPage, ResultView, MethodPicker, honest_card, JobBar` | The Plate Reader is a `StackPage`. |
| Library index | `io/libindex.py` (sqlite; capture time, camera, lens, rating), `app/library.py`, `app/review.py` | Bursts are groups in the index; Review shows a burst ranked. |
| Plate Lab (single scan) | `local/stains.py` (stains and corrosion unmixed, "Show what's lost") | The Plate Reader's albedo opens in Develop, where the Plate Lab tools then work. |
| Infrared filter guess | `color/infrared.py`: `FILTERS`, `FAMILY`, `guess_filter`, `colour_meter`, filter white balance save/load | Block A extends it. |

### 0.3 Facts checked on 2026-11-17 (do not re-derive)
* **Bill's infrared filters** (his words, 2026-11-17): most of the XSi infrared frames used **one variable filter, a
  FOTGA 67 mm "Six-in-One Adjustable" IR filter, 530-750 nm** (rotate to change the cut-off); he also has an
  **800 nm** filter and "a lot of lens filters"; the 850 frames are the converted 450D at 850. He remembers 720 raws as
  dark purple. Session 13's guesses: 2944 group 590 (not sure), 2994 group 720 (not sure), 5185 group 850-like
  (no colour at all; likely the 800 nm filter or the variable at its far end).
* **How these filters work** (savazzi.net/photography/variable-nir-filter.html, a review of this exact Fotga filter):
  a stack of **two polarisers with a deep-red filter**; rotating the front polariser moves the nominal pass edge along
  a 530-750 scale that covers only about 80 degrees of the ring. Variable ND filters, which are built the same way,
  show a **"rolling X" / cross artifact** past the end stop and with wide-angle lenses (provideocoalition.com,
  kolarivision.com). One reviewer measured this filter's real range as **closer to 640-800 nm** than 530-750
  (medium.com/live-view "FOTGA variable IR filter: A review"). So: the cut-off is soft and continuous, the colour can
  vary across the frame, and the scale markings are not wavelengths. Polarisers also change skies and reflections.
* **Bill cannot photograph the Hill plate** (NMAH refused access, even when he offered to photograph it himself). The
  Plate Reader is for objects he can put under his own camera: daguerreotypes, tintypes, ambrotypes, coins,
  engravings, paper, paintings, and his own future Hillotypes. The Hill plate stays a single-scan job (Plate Lab).
  Boudreau's colorimetry of Hillotype recreations (PROGRESS decision 89) says the real Hillotype colour does **not**
  change with viewing or lighting angle, while tarnish interference colours do: a multi-light capture can tell them
  apart. That is the Plate Reader's "Colour that moves" view (E.5).
* Camera facts: **Canon EOS R5**, 45 MP, pixel pitch **4.39 um** (8192 x 5464 on 36 x 24 mm); **450D / XSi** 12 MP,
  5.2 um. Bill's macro lens: **Voigtlander 90 mm** (manual focus: no in-camera focus bracketing; a rail or the ring).

### 0.4 Ask Bill once, at the start, in chat (one message; defaults if he is away)
1. **Plate Reader kit:** "Do you have a shiny black ball (a ball bearing, a black marble or a glass Christmas ornament)
   and a flash or LED you can move by hand? Anything old and shiny you'd test it on first (a coin, a tintype)?"
   Default: build and test on synthetic objects; the capture guide on the page says what to use.
2. **Glow:** "Have you measured the Laack's glow profile (Library > Tools > Measure lens glow: a small light in the dark,
   a short bracket)? It makes the Glow guard exact." Default: estimate from the photo (amber).
3. **Bursts:** "What do you shoot in bursts most: birds, people, events?" Default: people first (faces and eyes are
   measured), anything else by the sharpest subject (amber).
4. **Macro:** "For the 90 mm: a focusing rail, or turning the ring? On the R5, do you use focus bracketing?" Default:
   rail steps plus an R5 focus-bracketing suggestion that is calibrated by one ruler shot (D.2).
5. **Filter (block A):** "If you shoot a grey card or white paper at each mark of the variable filter, the Studio can
   learn each mark's white balance." Default: the estimate.

### 0.5 Bill's answers (2026-11-17, 03:28) - they change the order
* **Hand-held, manual focus, almost every macro he has shot**, relying on Adobe to align and stack. He uses focus
  stacking for portraits too. Low light, people moving slowly: he focuses by hand and shoots bursts while **rocking
  the focus back and forth wide open**, then stacks ("high risk, but I'm good at it"). Canon's in-camera focus
  bracketing needs autofocus and he never knows how many shots to take.
* **The request:** "Can I dump a whole SD card of different stacks, and a focus stacking importer handles the rest?"
  Find the stacks, cull the frames that would hurt (focus wrong, subject moved, etc.), stack them all. That is block F.
* **No shiny ball** for the Plate Reader (E gets a no-ball mode and a shopping line: a steel ball bearing or a black
  glass marble).
* **The Laack is in storage**, not shot in years, glow not measured: the Glow guard moves to last.
* **Bursts:** birds in flight yes, birds on branches no, events sometimes; his rocking-focus bursts are stacks (F).
* Sample work he shared: a hand-held corneal-reflection self-portrait (50 mm Zeiss macro), hand-held ant macros, and
  two hoverfly-on-blue-eyed-grass stacks (`stk11`, `stk16`); in `stk11` the right wing and one petal tip look doubled
  (the fly moved between frames): exactly what F.4's "subject moved" and F.5's motion-safe merge are for.

---

## A. Variable infrared filter (small; first)

A.1 `color/infrared.py`: `FILTERS["variable"] = "Variable IR (530-750 ring: two polarisers + deep red)"` with
METHOD_INFO (best: one filter for many looks; weak: soft cut-off, colour can change across the frame, the marks are not
wavelengths). `guess_filter` gains a **continuous estimate**: the nearest point on the polyline through the FAMILY
table in (log R/G, log B/G), reported as "about 690 nm equivalent" (amber), plus the nearest fixed family. Tests:
synthetic dye-model frames at 600-820 nm give a monotone estimate within 25 nm.

A.2 **Measured mark white balance:** the filter white-balance store (already per filter key) gets a position label
(530, 590, 650, 680, 720, 750, 800 or "my mark"), so a grey card shot at each mark is reused (cyan).

A.3 **Uneven filter colour across the frame:** fit a low-order surface (plane + radial term) to log(R/B) over
foliage/sky pixels (robust, clipped excluded); a spread above 0.08 shows an amber note ("this filter's colour changes
across the frame: a wide lens or a ring past its stop") and offers **Even out filter colour** (a smooth per-channel
gain field, off by default, saved in the edit). Detect the X pattern (two dark diagonal bands through the centre:
correlate the darkness with |x| = |y| bands) and say so plainly; it cannot be undone, only lightened.
Tests: synthetic radial colour change and a synthetic X are found; a uniform frame is not flagged.

A.4 **Panorama:** if per-frame colour gains differ more than 10% in R/B between frames, say "the filter ring may
have moved between frames" (the gains already even it out).

---

## B. Glow guard (Bill's idea 2; build last: the Laack is in storage, 0.5)

B.1 **What it is.** A character lens adds light around bright things: the candle glow of the Laack, a Petzval's halo.
To Dehaze, Veil removal, Detail by scale, Auto HDR and Clarity that light looks like haze to remove or soft contrast
to sharpen, so they eat the look Bill shoots the lens for. The Glow guard takes the lens's light out before those
tools run and puts it back after, so they work on the scene and leave the glow as shot. Develop > Lens:
**Protect lens glow** (on by default for lenses marked character), **Glow kept** 0-150% (100% = as shot; above 100%
deepens it), a chip saying which tier made the glow layer.

B.2 **The glow layer G** (scene-linear, added light), memoised per raw settings like `flare_map`:
* sources S = light above a knee (0.6 of clip), with blown lights at the brightness `raw/glow.py` recovered for them;
* **cyan:** G = S convolved with the measured glare spread function of this lens (`glow.kernel_image`, nearest
  aperture via `pick_profile`);
* **amber:** no profile: fit a radial kernel to the halos around isolated bright sources in this photo (sum of two
  exponentials and a power-law tail, by the lower-envelope fit `glow.envelope_fit` already uses: the scene can add
  light around a candle, never take it away), then convolve;
* G is capped at 90% of the pixel (never more than what is there).
Method texts: "Measured profile" (best: exact for this lens and aperture; weak: needs the one-time bracket) and "From
this photo" (best: no setup; weak: needs a few isolated lights; a lamp against a bright wall is not isolated).

B.3 **Pipeline:** in `_post`, before `apply_veil`: `out -= g * G` (g = the exposure gain already applied; floor at 5%
of the pixel); after the masks block, before the tone curve: `out += kept * g * G`. Add `glow_guard` to the Edit and
`stage_versions["glow_guard"] = 1`; it is not a raw setting (not in `raw_key`), but G's memo key is the raw key.
Region renders subtract exactly what the whole frame does (G is built whole-frame at proxy size, as the flare field).

B.4 **Tests:** synthetic candles (bright discs) convolved with a known kernel over a textured, hazed scene:
(1) G recovered within 10% of the true added light (both tiers); (2) Dehaze at 60 with the guard leaves the glow within
5% of the original around the lights while the haze elsewhere is removed as without the guard; (3) guard off renders
bit-identically to today; (4) Glow kept 0% removes the glow (a test that the layer is right); (5) a sidecar round trip.
Real check: `test_raws\candlelit_r5_exposures` (flames clipped, adapted lens) and the 850 candlelit portraits.

---

## C. Best of burst (Bill's idea 3: birds in flight and events; a burst with the focus moving is a stack, block F)

C.1 **Groups:** `io/bursts.py`: frames of one camera and lens with capture times (SubSecTimeOriginal when present)
less than 1.0 s apart (setting), confirmed by thumbnail similarity (phase correlation of 256 px greys: peak > 0.3), at
least 3 frames. Stored in the library index; Library shows a **Burst 12** chip on the first frame and can collapse a
burst to its best frame.

C.2 **Score per frame** (`library/burst_rank.py`, background, cached in the index):
* subject: faces (`faces.find`) -> eyes (iris boxes); else the person mask; else the depth model's nearest compact
  region; else the frame's sharpest 20% area (amber in all three);
* **sharpness** in the subject: noise-corrected scale-normalised Laplacian energy at 2-3 scales on the raw (half size,
  then a full-size crop of the subject), as in `stack/focus.py`;
* **eyes open:** eye aspect ratio from the face mesh eyelid points (Soukupova and Cech, CVWW 2016); closed below 0.18
  (setting); half-closed flagged;
* **why soft:** structure-tensor anisotropy of the gradients: whole frame directional = camera shake, subject only =
  subject moved, background sharper than the subject = focus missed (behind or in front, from the depth model);
* exposure: clipped share of the subject.
Rank: sharpness first, then eyes open, then expression is **not** judged (no smile model: say so).

C.3 **Page:** Library right-click "Find bursts" and Tools menu; Review opens a burst ranked left to right with the
score, the reason chips and the subject box; keys: **K** keep the best (rated 5 and picked), **R** reject the rest
(marked rejected, never deleted, undoable); "Show sharp areas" overlay.

C.4 **Tests:** synthetic bursts with known blur (Gaussian sigma 0-3 px, motion PSFs) ranked exactly; noise added to the
sharpest frame does not make it lose to a blurrier clean one by noise alone; eye aspect ratio on the stored face-mesh
sample (open) and a synthetic closing; grouping by time and similarity. Real: `test_raws\burst_spider_handheld`
(12 frames at 7 fps): report the ranking for Bill to judge. Honest limits on the page: no animal or bird eye detector
yet (birds are ranked by the sharpest subject, amber); expression is not judged.

---

## D. Macro stack planner (Bill's idea 4; small: Canon focus bracketing with AF lenses; the gap report is now in F)

D.1 **Planner** (Focus Stack page, "Plan a stack" group; engine `stack/plan.py`, pure arithmetic): inputs camera
(from a photo, or a list: R5 4.39 um, XSi 5.2 um), lens focal length, magnification m (or "the frame covers ___ mm
across": m = sensor width / that), f-number N, depth to cover (mm), overlap (default 30%), pupil magnification P
(default 1). Outputs: effective f-number Ne = N (1 + m / P); depth of field per frame DOF = 2 Ne c (1 + m / P) / m^2
style macro formula with c = 2 pixel pitches (setting: for a print size), checked against the general formula at low
m; step = DOF x (1 - overlap); frames = ceil(depth / step) + 1; **diffraction**: Airy diameter 2.44 x 0.55 um x Ne
against 2 pixels: "softening from f/__ effective" (amber, it is a formula). Output modes with METHOD_INFO: **Rail**
(mm per step) and **R5 focus bracketing** (number of shots, focus increment).

D.2 **Calibrating the R5's focus increment** (Canon does not publish what an increment is in mm): shoot a ruler at
45 degrees with focus bracketing at a chosen increment; the Studio finds the sharp band in each frame (focus energy
along the ruler) and stores mm per increment for that lens and aperture (cyan, per PC). The planner then suggests the
increment. Manual lenses (the Voigtlander): rail only.

D.3 **Gap report** after a stack (`stack/focus.py` result): where the texture of the merged result is high but no
frame was sharp there (best frame energy far below what neighbours reach, low confidence) and the depth index jumps
by more than one frame: amber overlay "nothing was sharp here", with "a frame is missing between 7 and 8" in the
report. Tests: the synthetic focus stack with one frame removed: the gap found at the right depth band and nowhere
else; planner formulas against hand values (m = 1, N = 8, R5: Ne 16, DOF about 0.28 mm).

---

## E. Plate Reader: multi-light capture (Bill's idea 1; the Plate Lab's multi-capture part)

E.1 **What it is.** A new sidebar tab after Panorama: **Plate Reader**. The object stays still under a fixed camera
while a light is moved around it, 24-48 frames, with a shiny black ball in the frame. From how each point brightens
as the light moves, the Studio learns its surface direction and its true colour. Then: relight it from any angle by
dragging a light ball, see scratches, tool marks and relief that one photo hides, get the colour with all shading and
glare removed, and see which colour **moves** with the light (tarnish, interference films) and which stays (the
image's own colour). Museums do this as Reflectance Transformation Imaging (RTI).

E.2 **Capture guide** (the page's honest card and a printable checklist): tripod, manual exposure and focus, raw; the
ball beside the object, same height; the light at a constant distance about 4x the object's width; 3-4 rings of
elevation 15-65 degrees, 8-12 positions each; a dark room; nothing else moves. Optional extras, each a layer: a
crossed-polariser frame (glare removed), infrared and UV frames (Bill's converted camera and filters).

E.2b **No ball (Bill has none):** a **no-ball mode** estimates the light directions from the object itself
(uncalibrated photometric stereo: Lambertian factorisation with the integrability constraint, Yuille and Snow 1997;
the generalized bas-relief ambiguity, Belhumeur, Kriegman, Yuille 1999, resolved by assuming the light circles at
a roughly constant elevation, as the capture guide asks). Amber, and the page says a ball is better: a steel ball
bearing (hardware store) or a black glass marble beside the object.

E.3 **Engine** `engine/stack/rti.py`:
1. Align (translation, `align.register`), develop all frames linear with one fixed white balance.
2. **Light directions from the ball:** circle found (Hough on the mean frame, or dragged by hand); the highlight in
   each frame (sub-pixel centroid of the brightest blob inside the circle); normal n at the highlight, light
   L = 2 (n.v) n - v with v = (0, 0, 1) (Mudge et al., highlight RTI). Cyan.
3. **Fits per pixel** (METHOD_INFO each): **PTM** (Malzbender, Gelb, Wolters, SIGGRAPH 2001: biquadratic in the light's
   x, y; best: fast, smooth, the classic; weak: soft on shiny surfaces) and **HSH** (hemispherical harmonics, order 2-3;
   best: shiny metal and glossy plates; weak: needs more frames). Robust: per pixel the darkest 10% (cast shadows) and
   the brightest 10% (glints) are left out of the fit.
4. **Photometric stereo** (Woodham 1980) on the robust frames: normal map and albedo (shading-free colour); relative
   height by integrating the normals (Frankot and Chellappa 1988), shown as a relief view.
5. **Views:** Relight (drag the light), Specular enhancement and Normal unsharp masking (the standard RTI enhancements
   that make tool marks and faint relief pop), Albedo, Relief, **Colour that moves**: per pixel, how much the hue
   changes across light directions after shading is removed (amber map): interference and tarnish colours move,
   process colour and pigment do not (the Boudreau test in 0.3).
6. **Outputs:** albedo as a linear intermediate (opens in Develop; the Plate Lab's stain tools then work on it),
   a relit JPEG/TIFF at the chosen light, a 16-bit normal map PNG; optionally a `.ptm` file for RTIViewer once its
   format is checked against the published HP Labs PTM format description (cite it in RESEARCH.md before writing it).

E.4 **Tests:** synthetic objects with known truth: a height field (engraved letters, a coin-like relief) rendered
Lambertian + Phong highlights + cast shadows under 36 known lights with a rendered ball: light directions within
1 degree, median normal error under 3 degrees, albedo within 2%, PTM relighting error under 3% at a held-out light;
a thin-film layer whose hue depends on the angle is flagged by Colour that moves, a pigment patch is not.

E.5 **Honest limits** on the page: needs the object in front of your camera (a scan or a single photo is not
enough: the Hill plate cannot be done this way); everything must stay still; very shiny mirror surfaces
(daguerreotypes) need the glints kept out, and a polariser helps; the relief is relative, not in millimetres.

---

## F. Card to finished stacks (Bill, 2026-11-17: "dump a whole SD card ... and a focus stacking importer handles the rest")

F.1 **What it is.** Point it at a card, a folder or a Library selection. It finds every focus stack, puts each in
focus order, leaves out the frames that would hurt the result (and says why), stacks every group, and shows a queue:
each finished stack beside its frames. Nothing is deleted, every decision can be undone with one click, and frames
that belong to no stack are left alone. It lives in the Focus Stack tab as a **Whole card** queue; Library > Import
from card offers "Find focus stacks" when the copy finishes.

F.2 **Finding the stacks** (`stack/sorter.py: find_stacks(paths)`):
* candidates: consecutive frames (capture time with sub-seconds; file number breaks ties) from one body, the same
  lens and focal length, less than 2 s apart (setting), exposure within 1 stop;
* the same scene: 512 px greys, high-passed, phase correlation peak > 0.25 with the neighbour and a shift under 25%
  of the frame (hand-held drift is normal);
* **the focus-sweep signature:** a sharpness map per frame at 256 px (noise-corrected Laplacian energy, as
  `stack/focus.py`); in a focus stack the sharpest region moves through depth (the sharpest-frame map spans 3 or more
  frames over textured areas); in an ordinary burst it stays put (that run goes to Best of burst); a single shot is
  left alone;
* Canon's in-camera focus bracketing: look for a focus-bracketing flag and shot number in the CR3 maker notes (check
  on a real R5 bracket before relying on it); when present the group is certain (cyan), otherwise inferred (amber);
* a long run is split where the scene changes (correlation drops) or a gap passes 2 s.
Tests: a synthetic card of three stacks (5, 9 and 14 frames, one rocking back and forth), one ordinary burst and
single shots, interleaved by time: grouped exactly.

F.3 **Order:** by the depth each frame is sharpest at (the magnification and breathing order `focus.analyse` already
measures), so a rocking sweep (Bill's wide-open low-light portraits) sorts itself; several frames at one depth: the
best is kept, the rest are "same depth as frame k, less sharp".

F.4 **Culling** (after alignment; `focus.analyse` already measures residuals, gains and the moved share per frame).
Each frame gets a status and a plain reason:
* **Adds nothing:** it is the sharpest frame for under 0.5% of the textured area and a neighbour covers its band
  better (a duplicate depth, or soft everywhere);
* **Shaken:** directional blur in its own sharp band (structure-tensor anisotropy against its neighbours);
* **Subject moved:** after local alignment, the residual motion or brightness change where this frame would
  contribute is above the ghost threshold: left out, or **kept only where it is safe** (setting);
* **Camera moved too far:** overlap with the reference under 70% or a scale jump (a hand-held lurch would crop the
  whole result);
* **Exposure or flash misfire:** more than 1 stop from its neighbours;
* **Not this scene:** the subject changed.
Red = left out, amber = used with care; the frame stays visible and one click puts it back. Strictness: gentle,
normal, strict. Tests: synthetic stacks with planted bad frames (soft everywhere, motion-blurred, the subject shifted
8 px, +1.5 stops, a big lurch, a duplicate depth): each found with the right reason, good frames kept, and the stack
closer to the truth with culling than without.

F.5 **Merging:** `focus.merge` (Auto) with Line up locally on (the hand-held default). New **motion-safe merge** for
portraits and anything alive: where the frames still disagree after alignment, the result takes **one** frame there
(the sharpest at that spot, else the reference) instead of a mix, the ghost-free rule HDR deghosting uses. With faces
found: the eyes come from the frame where they are sharpest, and skin may come from fewer frames (no ghosted pores).
**Gap report** (was D.3): textured places no frame got sharp (best energy far below what neighbours reach, low
confidence, the depth index jumping more than one frame) get an amber overlay and a line: "nothing was sharp between
frames 7 and 8". Test: the synthetic stack with one frame removed: the gap is found at the right depth band and
nowhere else.

F.6 **The queue page:** found stacks as rows (frame strip, N frames, N left out with their reasons, the result's
thumbnail when done); **Stack all** in the background with Cancel and a progress line; each result written beside
its frames as `FIRST_to_LAST_Stack_Studio.tif` (opens in Develop like every stack). End report, for example: "11
stacks found in 312 photos; 147 frames used, 23 left out (9 added nothing, 6 subject moved, 5 shaken, 3 camera moved
too far); 2 runs were bursts, not stacks (sent to Best of burst)". Settings: gap seconds, strictness, motion-safe.

F.7 **Real tests and honest limits:** `test_raws\focus_fly_handheld` (9 x R5, hand-held) culled and stacked; Bill's
own card next. On the page: a stack with focus gaps keeps its gaps (named, not invented); a subject that moved in
every frame can be made ghost-free, not sharp everywhere; a hand-held sweep also shifts the viewpoint, so near depth
edges alignment can only partly undo it.

---

## 6. Order of work (revised 2026-11-17 after Bill's answers, 0.5)
| # | Block | Done when |
|---|---|---|
| 0 | Message to Bill: only what 0.5 did not answer (an R5 focus-bracketing set and a hand-held card to test with). | Noted in PROGRESS. |
| 1 | A: variable filter estimate, mark white balance, uneven-colour and X checks, panorama ring note, tests. | `test_infrared.py` new tests green. |
| 2 | **F: card to finished stacks**: find, sort, cull with reasons, motion-safe merge, gap report, queue page, Import-from-card hand-off, tests. | F.2-F.5 synthetic tests green; the hand-held fly stack culled and stacked; a synthetic card sorted end to end. |
| 3 | C: bursts and ranking (birds in flight, events), Review view, tests. | Synthetic ranking exact; the spider burst ranked for Bill. |
| 4 | D: planner (Canon focus-bracketing shots and increment for AF lenses; rail steps), increment calibration, tests. | Planner hand values match. |
| 5 | E: Plate Reader (ball and no-ball modes, PTM/HSH, normals, Colour that moves) + tests, then the tab. | E.4 tests green; the tab builds and exports. |
| 6 | B: Glow guard, when the Laack comes out of storage (it also works from the photo for any glowing lens). | B.4 tests green; guard off is bit-identical. |
| 7 | Docs: NEXT_UPDATE lines, Help cards, README rows, PROGRESS, RESULTS, RESEARCH rows (PTM, HSH, Woodham, Frankot-Chellappa, Mudge H-RTI, Yuille-Snow, Belhumeur-Kriegman-Yuille, Soukupova-Cech, glare spread, HDR deghosting), THIRD_PARTY methods. Full test suite. | All green; Bill told what to try. |

Why this order: F is what Bill asked for most directly and matches how he actually shoots (hand-held, manual focus,
many frames per subject); A is an hour; C serves birds in flight and events; D is small and only helps with AF
lenses; E needs a capture Bill has not made yet; B waits for the Laack.
