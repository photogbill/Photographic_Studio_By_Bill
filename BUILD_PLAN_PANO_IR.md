# Build plan: Infrared / Full Spectrum tab and Panorama tab

Written 2026-11-16, at the end of session 12, for a fresh session. Bill asked for both tabs ("a tab specifically for
editing full spectrum / infrared photos, because those can be a challenge, with the color channels and white balance";
"we can use them to test panoramic stitching"). `BUILD_PLAN.md` section 25 is the short version; **this file is the
one to build from.** It follows the pattern of `BUILD_PLAN_TABS.md`, which built the Dual Pixel Lab, Focus Stack and
Super-Res Stack in one session: shared parts first, synthetic-truth tests for every algorithm, then the pages.

Build **Infrared first** (smaller, mostly colour, Bill shoots it now, and it edits the raw itself), then **Panorama**
(new geometry: features, bundle adjustment, seams). If the session runs out, stop at the end of a block (section 4)
and say exactly what is and is not built.

---

## 0. Read first

### 0.1 Rules (all of `BUILD_PLAN.md` section 0 and `CLAUDE.md` apply; these are the ones that bite here)
1. **Do not commit or push.** Bill pushes with `push.bat`. Write a plain line in `docs/NEXT_UPDATE.md` for every part
   built (`- New: ...`, `- Fixed: ...`), written for photographers.
2. **Nothing written to C:\ or the Windows profile.** Scratch goes through `stack/frames.Scratch` (beside the app,
   `paths.temp_dir()`); per-PC data (filter white balances, hot-spot maps) under `paths.get("CALIBRATION_LOCAL")` like the
   lens-glow and dust-map profiles.
3. **Names:** single-photo IR work is an ordinary edit in the photo's sidecar, exported from Develop as
   `NAME_Studio.ext`. Multi-frame results: `FIRST_to_LAST_IRfusion_Studio.tif`, `FIRST_to_LAST_Pano_Studio.tif`
   (`stack/frames.result_name`). `free_target` always; never a photo's own name.
4. **Every method the person can choose has `METHOD_INFO`** (one line best for, one line weak at): filter profiles,
   presets, fusion modes, projections, seam methods, blend methods. Colour coding: **cyan measured** (a filter white
   balance shot by the person, rotations solved from matches), **amber estimated** (filter guessed, focal length
   estimated, hot spot from one photo), **violet shared** (shipped presets).
5. **Errors report themselves** (`common.report_problem` / `log.save_error_report`); `Cancelled` and `StackProblem` are
   expected, not faults (already in `log._EXPECTED`).
6. **No OpenCV.** `THIRD_PARTY.md` says it is not used or installed, and the installers do not install it. Session 12
   found five modules importing `cv2` anyway (broken on every installed copy) and replaced them with
   `studio/engine/imgops.py` (`resize_linear`, `resize_cubic`, `warp_affine`, `gaussian_blur`, matching OpenCV to 1e-5).
   `tests/test_imgops.py` fails if any module imports `cv2` or a package `requirements.txt` does not install. So:
   features, RANSAC, seams and projection are written here in NumPy/SciPy. No new packages without asking Bill.
7. **Tests** on synthetic data with known truth, writing only under `outputs/`; run the whole suite before saying it
   works. Offscreen page tests like `tests/test_stack_pages.py`, including the window-size checks.
8. **Bill does not use the command line.** Questions in chat, plain words, with a recommendation (section 0.4).

### 0.2 What exists that you will use (do not rebuild)
| Need | Where | Notes |
|---|---|---|
| White balance as raw multipliers | `pipeline/edit.py: WhiteBalance(mode="multipliers", multipliers=[R,G,B])`; `pipeline.wb_multipliers` | Any positive values work (normalised so the smallest is 1). Applied to the mosaic **after denoise, before demosaic** (`pipeline.py` "white_balance" stage), so IR's big blue gain does not fool the denoiser. |
| Click-to-neutral | `pipeline.pick_white` -> `color/autowb.patch_multipliers` | Works in camera space: right for IR foliage picking as it is. |
| Auto white balance | `color/autowb.auto_multipliers` (grey edge, grayness index) | Untested on IR; see 1.4. |
| Camera matrix stage | `pipeline.py`, after demosaic: `cam.camera_to_working(raw.cam_xyz)` or a DCP (`edit.profile`) | **The IR hook goes here** (1.3). |
| Edit cache key | `Edit.raw_key()` lists the fields of stages 1-11 | Add `infrared` to it, or the preview will not re-render. Add `stage_versions["infrared"] = 1`. |
| Temperature / tint display | `pipeline.wb_temp_tint` (`cam.temp_tint_from_multipliers`), Develop's Temperature slider 2000-12000 K | Meaningless for IR (far off the daylight locus). In IR mode show the multipliers instead (1.6). |
| Lens falloff measurement | `lens/falloff.estimate`, `gain_map`, `model_falloff` | Base for the hot-spot model (1.5). |
| Colour grading (split tone) | Develop > Colour grading wheels | The "Candle glow" preset sets highlights there; do not write a new toner. |
| Alignment | `stack/align.py`: `register(kind=... "homography")`, `fit` (IRLS), `Transform`, `warp`, `flow`, `phase_shift`, `noise_sigma`, `analysis_copy`, `common_box` | `register` searches about a quarter tile per level: fine for fusion (frames nearly aligned), **not** for panorama overlaps (2.2). `phase_shift` windows with a Hann window, which kills the edge overlaps a panorama has. |
| Frames | `stack/frames.py`: `probe`, `check`, `stack_edit`, `scene_rgb(path, edit, half, region)`, `Scratch`, `result_name`, `default_out_dir` | `scene_rgb` develops stages 1-12 (lens geometry included when the edit asks), camera-crop frame, not upright. |
| Linear result file | `io/intermediate.write_intermediate(path, rgb_scene, meta, info)` | Takes any ndarray, a `Scratch` memmap included, and writes 256-px tiles: a panorama canvas can be streamed. Opens in Develop as `scene_linear`. |
| Pages | `app/stackui.py`: `StackPage` (tray, result view, job bar, 360-px panel, actions), `ResultView` (B toggles reference, Z 100%), `JobBar` (cancel), `MethodPicker`, `honest_card`, `CoverageDiagram`, `scene_to_qimage` | Panorama is a `StackPage`. Infrared is a one-photo page built from the same parts (1.6). |
| Library send-to | `library.py: send_to = Signal(str, list)`; `main.py: send_to_tab(where, files)` with rows dualpixel=2, focus=3, superres=4 | Add "Edit in Infrared tab" and "Send N to Panorama"; renumber the rows when the pages are inserted. |
| Pyramid blending | inside `stack/focus.merge` (5-tap kernel, `_up2`, `_lap_noise_factors`) | Factor out `stack/pyramid.py` (build, collapse, blend by masks) with the focus tests kept green, then use it for panorama seams. |

### 0.3 Facts checked on 2026-11-16 (do not re-derive)
**Bill's infrared and full-spectrum camera is one converted Canon EOS 450D (Rebel XSi)**, 12 MP CR2, CFA RGGB
(`raw_pattern [[0,1],[3,2]]`), LibRaw has its colour matrix. Measured on the real files in `test_raws\` (raw, black
subtracted, 8x8-binned, the brighter half of the frame below clipping):

| Files | R/G | B/G | log-correlation R-G, B-G | spread of log(R/G), log(B/G) |
|---|---|---|---|---|
| `infrared_850\IMG_3156, 3204`, `infrared_850_candlelit\IMG_3208` (850 nm) | 0.97 | 0.99 | 0.999-1.000 | 0.015, 0.016 |
| `ir_visible_fusion\IMG_7469, 7470, 7472` | 0.96-0.97 | 0.99 | 0.96-0.996 | 0.04-0.05 |
| `ir_visible_fusion\IMG_7471` (a window: daylight through glass) | 0.87 | 0.98 | 0.63-0.65 | 0.70, 0.62 |
| Visible-light reference, 5D Mark IV (`raws_5d4_pair\2I2A3249`) | 0.53 | 0.56 | 0.99 | 0.11, 0.14 |
| UV black light, 6D (`blacklight\IMG_5088`) | 0.30 | 2.27 | 0.95-0.97 | 0.37, 0.52 |

* **At 850 nm the three CFA dyes pass the same light: the raw is monochrome to within 1.5%.** That is the signature
  the filter guess uses for "850 nm and deeper": R/G and B/G within 5% of 1 and the chroma spread under 0.03.
  590 / 665 / 720 nm and full-spectrum frames from the same 450D are **not** in `test_raws` yet (Bill's
  `20210909 full spec` and `20210911_full_spec` folders on the archive drive have them): the thresholds for those
  families come from the synthetic model (1.4) until real frames arrive.
* The 450D files carry a custom in-camera white balance, as-shot multipliers [1.349, 1, 2.391] (850 frames) and
  [1.81-2.20, 1, 1.41-1.67] (fusion folder); LibRaw's daylight multipliers [2.489, 1, 1.217] are the visible-light
  ones and useless for IR.
* **`ir_visible_fusion` is not a matched pair.** Its four frames are 850-nm-looking frames of different indoor scenes
  (blinds, a chandelier, a window, a wall). Fusion needs a visible frame and an IR frame of the same scene; ask Bill (0.4).
* The colour of a candle in an 850 nm frame is not in the raw (it is monochrome): any warm candle in IR is added colour,
  and the page must say so (amber).
* WB with normalised multipliers keeps clipping consistent: the strongest raw channel has multiplier 1, so all channels
  of a neutral clip together. The cost of IR white balance is **noise**: at 590 nm blue may need 8-15x gain. The denoiser
  runs before white balance with a per-channel noise model, so it sees the true noise; suggest Clean when a channel gain
  passes 4x.
* **What the last 1.5% is (measured 2026-11-16, ask nothing more about it):** Bill asked whether the small channel
  differences at 850 nm hold anything to play with. On IMG_3156, 3204, 3208 (black from the optical-black margin, the
  one hidden dimension G / sqrt(R B), guided-filtered with the luminance as guide; **the two green photosites G1 / G2,
  which see identical light, as the noise control**):
  - raw: 3.3-3.6x the control, but it **rises with brightness** (+0.012 at G = 450 DN to +0.024 at 2000 DN on IMG_3204):
    a level-dependent channel response or flare, not material colour;
  - with that brightness trend removed: 2.0-3.0x the control, and what is left **follows edges and fine texture** (the
    R and B photosites sit one pixel away from the greens, so a ratio of them is partly a spatial derivative);
  - no material-specific colour (skin vs cloth vs wood) shows convincingly; the candle flames are clipped.
  **Conclusion: at 850 nm on this camera the raw is monochrome. The 1.5% is a sensor-level gain difference to remove,
  not colour to amplify.** That removal is the useful tool (1.5b: develop without demosaicing). Real IR colour needs a
  filter where the dyes still differ (550-720 nm) or two exposures through different filters (1.7).
* **Panorama test set** (`test_raws\panorama_iraq_6d_135mm`, from Bill): 7 portrait frames IMG_1565-1571, all full size
  (3648 x 5472), Canon 6D, **Samyang/Rokinon 135 mm f/2 (manual: focal length and aperture recorded as 0)**, Iraq
  2016-11-21, all seven within 6 seconds at the same exposure (1/125 s, ISO 100), sunset under smoke, processed in DxO
  OpticsPro 11.2 (JPEG; the raws are not to hand, and are not needed). `references\` holds three finished stitches to
  compare against: `Sunset_Under_Smoke_Bill_final.jpg` (Bill's own, 13392 x 5060), `MS_ICE_stitch_test.jpg` (Microsoft
  Image Composite Editor, 13617 x 4998) and its edited version. **True focal length** for the focal-estimation test:
  135 mm / 6.54 um pixel pitch = about 20,600 px; the lens's real focal length may differ by 1-2%. Cross-check: ICE's
  13617 px width at that focal length is 37.9 degrees, which matches 7 portrait frames of 10.2 degrees with the
  measured overlaps. A good acceptance test: register our result to ICE's and Bill's (they agree with each other) and
  report the layout difference in pixels.
  Measured overlaps (high-passed log luminance, 1/4 size, NCC 0.70-0.86): 61, 49, 60, 82, 54, 45% of a frame between
  neighbours, with 40-84 px of vertical drift at 1717 wide (hand-held tilt). **Hard case on purpose:** a smooth sky
  gradient and a thin skyline band carry almost all the detail. A plain NCC on unfiltered images got two of the six
  overlaps wrong (fooled by the sky gradient): match on high-passed images and features on the skyline, and let the
  rotation model (3 angles + one focal length) carry the geometry where the band is thin.

### 0.4 Ask Bill once, at the start, in chat (one message; use these defaults if he is away)
1. Already answered on 2026-11-16: **Bill has no visible / IR pair and no full-spectrum camera now** (he plans to get one
   again; the XSi files are all there is). He sent the Iraq panorama at full size with three reference stitches (0.3);
   the raws are not needed. Still worth one line at the start: "Any other panoramas, or XSi frames at 590 or 720 nm,
   you'd like me to test with? Drop them in `test_raws\_incoming`." Default if away: build and test on synthetic truth,
   the 850 frames, the black-light frames (strongly different channels, for 1.5c) and the Iraq panorama.
2. Default decisions (do not ask, just do, and say so): the IR tab edits the raw itself (settings in the sidecar, so
   every Develop tool works afterwards); 850 nm opens in black and white; the swap presets are linear-light channel
   mixes; panorama crops to a rectangle by default.

---

## 1. Infrared / Full Spectrum tab

### 1.1 What it is
A workspace in the sidebar, after Super-Res Stack: **Infrared**. It opens one photo (or a shoot) from the Library,
shows it large, and gives the controls an IR photographer needs **before** ordinary editing: which filter, white
balance far outside the camera's range, channel swap / mixer, false-colour presets, hot-spot correction. Everything
it sets lives in the photo's edit (`edit.infrared` + `edit.white_balance`), so **Open in Develop** continues on the raw
with every tool, and Paste settings carries a shoot's IR look to the rest. Fusion (visible + IR, several frames) is
the one part that writes a new file.

Page text (top card, plain words): *"Infrared and full-spectrum raws break normal colour: the camera's white balance
cannot go far enough and its colour matrix was made for visible light. Pick the filter, click foliage, choose a look,
then edit in Develop as usual."* What to shoot: *"A frame with a grey card or white paper in sun through each filter
lets the Studio measure that filter once (cyan). Foliage and sky in the frame help the presets."*

### 1.2 Files
| File | What |
|---|---|
| `studio/engine/color/infrared.py` | Filters, guess, multipliers, mixer matrices, presets, measured filter white balances, `camera_to_working` hook. |
| `studio/engine/lens/hotspot.py` | Hot-spot model, measure from one flat frame or several photos, gain map. |
| `studio/engine/stack/fusion.py` | Visible + IR fusion (align across spectra, modes, NDVI). |
| `studio/app/infrared.py` | `InfraredPage`. |
| `tests/test_infrared.py`, `tests/test_hotspot.py`, `tests/test_fusion.py` | Synthetic truth + the real 850 frames. |

### 1.3 Pipeline hook (do this first, with its test, before any UI)
`Edit.infrared: dict | None = None`:
```python
{"on": True, "filter": "850", "guess": {"key": "850", "confidence": 0.93},   # guess kept for the amber chip
 "profile": "identity",            # identity | visible (the camera matrix, for full spectrum) | measured:<name>
 "mixer": [1,0,0, 0,1,0, 0,0,1],   # 3x3, rows = output R, G, B from white-balanced camera R, G, B (linear)
 "preset": "mono_best",            # the preset the mixer came from ("" = edited by hand)
 "mono": True, "hotspot": None | {"model": {...}, "amount": 1.0}}
```
In `pipeline.py`'s camera-matrix stage: when `infrared` is on, `M = infrared.camera_to_working(edit.infrared, raw)`:
* `identity`: white-balanced camera RGB is taken as linear ProPhoto as it is (no visible colour science pretended),
  then the mixer: `M = P @ Mix` where `P` maps "camera-as-RGB" to ProPhoto by treating it as linear sRGB primaries
  (`cam.camera_to_working` of an identity-like sRGB camera; one fixed matrix, so swaps look like Photoshop's on a
  linear file). Write the exact matrix in the module text and test that white stays white.
* `visible`: the camera matrix, then the mixer (full spectrum with an IR-cut filter on the lens behaves like visible).
* `mono`: after the mixer, R = G = B = the mixer's first row (luminance only).
Add `infrared` to `Edit.raw_key()`; `stage_versions["infrared"] = 1`. Edits without the field render exactly as today
(test: a stack of existing sidecars renders bit-identically before and after).

### 1.4 `color/infrared.py`
```python
FILTERS = {  # key -> label, METHOD_INFO-style best/weak, default treatment
  "full":  "Full spectrum (no filter)",  # pink-red raw; WB on a neutral; profile "visible" if an IR-cut filter was on
  "550":   "530-550 nm",                 # strongest colour; swap red/blue; WB on foliage
  "590":   "590 nm",                     # classic false colour: gold foliage, blue sky after the swap
  "665":   "650-665 nm",                 # less colour, cream foliage; swap, low saturation
  "720":   "720 nm",                     # faint colour, mostly luminance; B&W or a gentle swap
  "850":   "850 nm and deeper",          # monochrome raw (0.3); B&W by default
  "unknown": "Not sure",
}
def guess_filter(image) -> dict                 # {"key", "confidence", "rg", "bg", "spread"}: numbers of 0.3
def foliage_multipliers(image, box=None) -> np.ndarray   # WB that makes foliage neutral (see below)
def neutral_multipliers(image, u, v) -> np.ndarray       # = pipeline.pick_white, kept here for the page
def mixer(preset: str, strength: float = 1.0) -> np.ndarray   # 3x3
PRESETS = {"none", "swap_rb", "blue_sky_590", "goldie", "aerochrome_like", "mono_best", "mono_red", "candle_glow"}
PRESET_INFO = {...}                              # best for / weak at, one line each
def camera_to_working(ir: dict, raw) -> np.ndarray
def filter_wb_path(name) / save_filter_wb(name, mult, meta) / load_filter_wbs()   # measured per filter (cyan)
```
* **Guess** from the statistics in 0.3 on a half-size raw: `850` when R/G and B/G are within 5% of 1 and the spread is
  under 0.03; otherwise nearest family on the synthetic model's R/G, B/G table, confidence from the distance; anything
  far from every family is `unknown`. Always shown amber with its numbers ("raw R/G 0.97, B/G 0.99: looks like 850 nm").
* **Foliage white balance:** IR photographers neutralise foliage (it is the brightest common thing in IR). Auto foliage:
  pixels in the top 40% of luminance whose local variance is moderate (not sky, not specular), median R/G and B/G of
  those -> multipliers. Amber, with a "click foliage" picker for a cyan-quality answer on that photo. Test on the
  synthetic scene (foliage within 1%).
* **Measured filter WB:** "Measure this filter" on a frame with a grey card or white paper in sun: the clicked patch's
  multipliers saved by filter name on this PC (`CALIBRATION_LOCAL/infrared/<name>.json`, camera model inside). Offered
  first for that camera + filter, cyan.
* **Presets** (each is just a mixer + an optional saturation and a WB rule; the person can still move the matrix):
  `swap_rb` red<->blue; `blue_sky_590` swap plus a little green into blue; `goldie` (foliage gold, sky blue: swap,
  then reduce blue saturation); `aerochrome_like` (IR->red, red->green, green->blue; a colour-infrared film look, for
  full spectrum or 590 with foliage); `mono_best` (the channel with the most detail by noise-weighted gradient
  energy); `mono_red`; `candle_glow` (mono_best plus Colour grading highlights warm: **added colour**, amber note).
  Row sums shown under the matrix so a mix that changes brightness is visible.
* **Synthetic model** (`tests/test_infrared.py` builds it; keep the curves in the test, not in the engine): a camera with
  Gaussian R, G, B dye curves that all become transparent past 800 nm (as measured, 0.3), filters as long-pass edges at
  550/590/665/720/850 nm, reflectances: foliage (low visible with a green bump, red edge at 700 nm, high NIR plateau),
  sky (high blue, low NIR), skin, grey card. Render the camera raw for each filter -> the guess table and the tests.

### 1.5b Monochrome raw for 850 nm and deeper (`color/infrared.py: mono_mosaic`) - the useful answer to the 1.5%
When the guess says 850 (or the person ticks **Develop as black and white raw**), skip demosaicing: every photosite is a
luminance sample once the four CFA positions are brought to the same gain.
* Gains: per CFA position, a smooth local ratio map to the greens (Gaussian about 16 plane pixels, clipped and very dark
  pixels left out, like the measurement in 0.3), so the brightness-dependent part is taken out too; then
  `mono = mosaic / gain_map`. Hot and dead pixels must be fixed first (a demosaic hides them; this does not): the
  existing hot-pixel / dust code, or a 3x3 median test against the neighbours of the same and other colours.
* Hooks into the pipeline in place of the demosaic stage (`edit.infrared["mono_raw"] = True`), output R = G = B; denoise
  runs before it as now. METHOD_INFO: best for sharp 850 nm daylight subjects (foliage, branches, architecture): full
  sensor resolution, no colour noise, no demosaic patterns; weak at: nothing gained when the lens or the light limits
  sharpness (Bill's candle frames at f/1.4-f/2 look the same either way: checked on IMG_3204 and IMG_7469), and wrong
  for any filter that still shows colour (the page offers it only for 850 or when the colour meter says "none").
* **"Is there any colour in this raw?" meter** (from 0.3, always shown in the filter card): the hidden dimension over
  the two-green control, with the brightness trend removed and strong edges left out of the measurement. Under 4x
  (the 850 frames measured 2.0-3.0x with edges in): "No usable colour: this raw is black and white"
  (cyan, measured). It also guards the guess: a "850" guess with real colour (a 720 frame) is caught.
* Colour for an 850 photo is a creative choice and labelled as added colour (amber "added", never cyan): duotone and
  split-tone from luminance (Colour grading), and **palettes by brightness** (a thermal-camera look such as
  "ironbow"/"inferno", a sepia platinum print, a cyanotype blue). `candle_glow` = warm highlights by luminance.
* Tests: a synthetic monochrome scene sampled through a CFA with gains 0.97 / 1 / 1 / 0.99 plus a 1% brightness-
  dependent term and noise: the mono path beats RCD demosaic + luminance in PSNR against the truth and leaves no grid
  (FFT peak at the Nyquist corner below the noise); hot pixels removed; the colour meter says "none" on that scene and
  "colour" on a 720-like scene from the 1.4 model.

### 1.5c Black and white from the channels: selective contrast (Bill's idea, 2026-11-16)
Bill: the three channels are three handles on the picture, not only colour: *"desaturate the entire image and then use
the channels to make subtle contrast changes to different parts of the image. You can adjust exposure by channel,
saturation by channel, etc."* This is how IR and UV photographers work in black and white, and it is the main tool of
this tab for every filter that leaves the channels different (550-720 nm, full spectrum, UV-pass, and visible light).
* **Channel tone, then mix** (Infrared tab, a "Black and white" group): per raw channel (after white balance, before
  the mix) **exposure** (stops), **contrast**, a small **curve**, and **saturation of that channel's contribution**; then
  the **mono mix** (weights per channel, negatives allowed, sum shown, "keep brightness" normalises). Live B&W preview,
  B toggles the plain desaturated version. Presets as IR "filters for B&W": *Dark sky, bright foliage* (590/720),
  *Soft skin* (the channel with the least texture on skin), *Most separation* (weights that maximise noise-weighted
  local contrast, solved, amber), *Even* (plain average).
* **Channel masks in Develop** (new mask type **Channel**, works on any photo): a mask from one channel, or from a
  channel **difference or ratio** (e.g. R/B), with range and feather, so a B&W photo gets local exposure or contrast where
  a channel says "foliage" or "sky". In a 590-720 nm frame the red and blue photosites see different mixes of visible and
  IR light, so their ratio separates IR-bright foliage from sky and skin (the single-frame cousin of the NDVI in 1.7).
  The mask preview shows what the ratio sees (cyan: measured from the raw).
* **850 nm, measured (Bill: "it's all fun and games until I adjust one channel down 200%, the other up 100%"):** the
  three channels hold nearly the same picture, but extreme mixes do change it, and the tab must show the price. On
  IMG_3208, channels evened out first, AHD demosaic:
  - R and B **move together** (their deviations from G correlate 0.8), so opposite weights on R and B cancel:
    (-1, 1, 2) changed tones by at most 0.3%, noise x1.3.
  - The live direction is **green against red + blue**: (-1, 3, -1) tones within 0.9%, noise x1.7; (-2, 5, -2) 1.6%,
    x2.7; (-4, 9, -4) up to 3% darker highlights, noise **x4.8**, and **the candle flames go darker** than everything
    else (a hot flame has a different near-IR slope from what it lights: the one real spectral difference at 850).
  So at 850 every 1% of tonal change costs roughly a doubling of noise. The tab allows it (it is the person's picture)
  and shows the cost live (the noise meter below); it also points to what moves tones without that cost on an 850 photo:
  the Light panel's tonal regions, Detail by scale (1.5d), and masks by brightness or object.
* **Remember the channel settings; adjust in black and white** (Bill: "it's hard to adjust in colour or false colour
  and go to black and white"):
  - The B&W channel settings (per-channel exposure, contrast, curve, saturation, and the mix weights) live in their own
    part of the edit, `edit.infrared["bw"]`, separate from the colour / false-colour settings. Turning B&W off and on
    loses nothing; each side keeps its own values.
  - Adjusting happens **in the black-and-white view**: the moment a B&W control is touched, the preview shows the B&W
    result (never the colour image), and B toggles against the plain desaturated version.
  - Three small **channel thumbnails** above the sliders show R, G and B each as its own greyscale picture (like
    Photoshop's channels palette), so the person sees what each handle holds before moving it.
  - Ranges: mix weights -300% to +300% per channel, per-channel exposure +-3 stops. A **noise meter** under the mix:
    how much the mix multiplies the noise, **measured on the mixed preview** (flat-area noise estimate, `align.noise_sigma`
    style) against the plain average, cyan up to x1.5, amber above x2, with the words "this mix multiplies the noise by
    4.8". Not the textbook sqrt(sum w^2): demosaicing makes the channels' noise correlated, so that formula says x18 for
    (-4, 9, -4) where the measurement is x4.8.
  - **Recipes:** "Save as B&W recipe" stores the channel settings by name on this PC (beside Looks, never on C:\), listed
    per filter; the **last recipe used for that camera and filter** is offered when the next photo opens; Apply to the
    shoot and Paste settings carry it to other photos.
* UV: UV-pass frames from a full-spectrum camera put the light mostly in the blue and red photosites; the same tools
  apply (often blue as the base channel, red for separation). UV-lit visible fluorescence (Bill's black-light set) is
  visible light and uses the ordinary B&W mixer or these tools on its strongly different channels.
* Tests: B&W settings survive B&W off / on and a sidecar round trip, and a saved recipe re-applies exactly; the noise
  meter matches the measured noise of a mix on a synthetic noisy frame within 10%; the Detail by scale panel in this tab
  and in Develop change the same edit; on the synthetic 590 scene (1.4) the *Dark sky* preset darkens sky against
  foliage by at least 1 stop while the result stays neutral (R = G = B); per-channel exposure on a scene with identical channels equals one global exposure of
  their mean (the 850 case, proved); a Channel mask from R/B selects the foliage of the synthetic 590 scene with at least
  90% agreement; on the black-light frames the mono mix changes local contrast measurably between presets.

### 1.5d Detail by scale in the Infrared tab (Bill: "include the microcontrast tool that is shades of shadow")
The tab includes the **Detail by scale** group from Develop (23.11: Amount, Balance tonal-crisp, presets, the meter, the
six methods in Full, "Where it works"), so tones can be shaped while the photo is in black and white. It is **the same
widget class bound to the same edit fields** (`edit.microcontrast` / the detail settings), not a copy: what is set here
shows in Develop and the reverse, and one test checks the two panels stay in step. **Judge in black and white** is on by
default in this tab. On an 850 photo this is the main tool for tonal separation (it costs no extra noise, unlike an
extreme channel mix), and its text says so.

### 1.5 Hot spot (`lens/hotspot.py`)
Many lenses show a bright centre in IR (internal reflections at that wavelength). Model: a radial Gaussian bump on top
of the falloff model, `g(r) = 1 + a * exp(-(r / s)^2)` in normalised radius, plus an optional off-centre point.
* Measure from **one flat frame** (sky, wall: the person says so) by fitting the bump to the radial profile; or
  **learn from several photos** of one lens + filter (median of the normalised frames, like Learn dust map) in the
  Library's Tools menu. Saved per lens + filter on this PC.
* Correction is **off by default** (like vignetting); amount 0-100%. Amber from one photo, cyan when learnt from 5+.
* Test: a bump of 0.4 stop with s = 0.35 added to a flat field and to a textured synthetic scene: measured within 5%
  (flat) and 15% (textured, with the honest note); nothing found on a frame without one.

### 1.6 The page (`app/infrared.py`)
Layout: top bar (title, chips: filter guess amber / measured cyan, channel gains, "noise: Clean suggested" when a gain
passes 4x), the photo large (`stackui.ResultView`; B before/after = camera as-shot vs this; Z 100%), a filmstrip of the
shoot under it (Library selection or the folder), the 360-px panel on the right, actions at the bottom.
Panel, top to bottom:
1. **Filter** picker with the guess pre-selected and its sentence; "Measure this filter..." (click a grey card).
2. **White balance**: Auto foliage / Click foliage / Click a neutral / Measured for this filter; the three multipliers
   as numbers with fine sliders (log scale 0.05-20x). Temperature is not shown (meaningless here); Develop shows
   "IR white balance: set in Infrared" in place of its Temperature and Tint while `infrared.on`.
3. **Look**: preset tiles (small live thumbnails of the current photo, rendered at 256 px in the background), each with
   its best-for / weak-at text; strength.
3b. **Black and white** (1.5c): channel thumbnails, per-channel exposure / contrast / curve / saturation, mix weights,
   noise meter, B&W recipes (save, last used for this camera and filter).
3c. **Detail by scale** (1.5d), Judge in black and white on.
4. **Channel mixer** (Full): 3x3 grid of spin boxes, row sums, Monochrome tick, profile (identity / visible / measured).
5. **Hot spot**: Measure / Learn (Library), amount, Show.
6. **Fusion** (folded): "Add the visible frame..." -> mode picker (1.7), Build.
Actions: **Apply to the shoot** (paste the IR part of the edit to the strip's photos), **Open in Develop**, **Reset**.
Keys: B, Z, Esc, Ctrl+Return (Open in Develop). Library: right-click "Edit in Infrared tab"; Tools menu too.

### 1.7 Fusion (`stack/fusion.py`) - **build last, synthetic tests only for now**
Bill has no visible / IR pair and no full-spectrum camera today (0.4), so this is built and tested on synthetic pairs
and waits for real ones. The page says how to shoot a pair once he has a full-spectrum camera again: tripod, manual
focus set for the IR frame, one frame with an IR-cut (visible) filter and one with the 720 or 850 filter; or two IR
frames, **720 + 850**, whose difference is real spectral information too (the dyes differ at 720 and not at 850).
Inputs: one visible frame and one IR frame of the same scene (tripod or close). Alignment across spectra: register on
**gradient-magnitude images** (foliage is dark in visible and bright in IR, so raw brightness does not match, edges do):
`align.register(grad(vis), grad(ir), kind="homography")`, then `align.flow` on the same pair for small parallax.
Modes (METHOD_INFO each):
* **IR detail, visible colour**: OKLab L from IR (blend amount), a/b from visible. Best for: haze cutting, glowing
  foliage with true sky colour. Weak at: things IR sees through (some fabrics) change shape of tone.
* **Colour infrared (CIR)**: R = IR, G = visible red, B = visible green (remote-sensing standard false colour; the
  Aerochrome look from real data, not a guess).
* **NDVI map**: (IR - red) / (IR + red), the vegetation index satellites use, shown with a colour ramp and a legend
  (cyan: measured from the two frames). Weak at: needs both frames white-balanced on the same grey card for absolute
  values; otherwise relative (amber).
Output: `FIRST_to_LAST_IRfusion_Studio.tif` (intermediate). NDVI also as a 16-bit greyscale PNG for other programs.
Tests: a synthetic visible/IR pair rendered from the 1.4 model with a known homography between them: alignment within
0.2 px; CIR channels equal the inputs; NDVI of the foliage and sky regions equals the model's values within 0.02.

### 1.8 Infrared done means
The tab opens an 850 frame from `test_raws` in black and white with a sensible tone, the guess says 850 (amber) with its
numbers, Auto foliage and Click foliage work, presets render live, Open in Develop shows the same picture with every
tool working, Paste carries it to the shoot; hot spot measured on a synthetic bump; fusion builds from a synthetic pair;
all tests green; the page states its limits; NEXT_UPDATE lines written.

---

## 2. Panorama tab

### 2.1 What it is
A `StackPage` (MIN_FRAMES 2): frames in from the Library ("Send N to Panorama") or by drop; **Find layout** (features,
pairs, bundle adjustment) shows the frames' outlines on a small sphere/plane preview, each outline coloured by how well
it is tied in (cyan solved from many matches, amber few, red not placed); **Preview** stitches at 1/8 size; **Build**
writes the full panorama. Page texts: what it does, what to shoot (*"Overlap each frame by a third, turn the camera
rather than walk, manual exposure and focus, and keep moving things out of the overlaps."*), honest limits (2.8).

### 2.2 Files
| File | What |
|---|---|
| `studio/engine/stack/features.py` | Corners, descriptors, matching, RANSAC homography (our own; no OpenCV). |
| `studio/engine/stack/bundle.py` | Rotations + focal length by bundle adjustment; wave correction (level horizon). |
| `studio/engine/stack/project.py` | Projections (rectilinear, cylindrical, spherical/equirectangular, Panini, Mercator): forward and inverse maps. |
| `studio/engine/stack/seams.py` | Seam finding (dynamic-programming cut through each overlap), label maps. |
| `studio/engine/stack/pyramid.py` | Gaussian / Laplacian pyramids and mask blending (factored out of `focus.merge`). |
| `studio/engine/stack/pano.py` | `PanoParams`, `analyse`, `layout_view`, `preview`, `build`, `METHOD_INFO`, texts. |
| `studio/app/panorama.py` | `PanoramaPage`. |
| `tests/test_features.py`, `tests/test_bundle.py`, `tests/test_pano.py`, page tests in `tests/test_stack_pages.py` |

### 2.3 Features and pairs (`features.py`)
Panorama overlaps are 20-50% of a frame: far outside `register`'s search, so match features first, then refine.
```python
def corners(lum, n=2000) -> (N x 2)        # Harris on a 2000-px analysis copy (align.analysis_copy), sigma 1.5,
                                           # adaptive non-maximal suppression (Brown, Szeliski, Winder 2005, "MOPS")
def describe(lum, pts) -> (N x 64), angles # MOPS: 8x8 samples spaced 5 px from the blurred image at the corner's
                                           # gradient angle, bias/gain normalised (zero mean, unit variance)
def match(d1, d2, ratio=0.8) -> pairs      # scipy.spatial.cKDTree, Lowe's ratio test, mutual best
def ransac_homography(p1, p2, thr=2.0, iters=800, rng) -> (H, inliers)   # 4-point DLT, normalised coordinates
def pair_ok(n_inliers, n_matches) -> bool  # Brown & Lowe 2007: inliers > 8 + 0.3 * matches
```
Then refine each accepted pair with `align.fit(kind="homography")` on the inliers and, at full analysis resolution,
`align.register(..., init=T)` restricted to the overlap box (sub-pixel). Pairs considered: every pair when n <= 12,
otherwise each frame with its 6 neighbours in time plus any pair whose layout predicts an overlap after a first pass.
Frames that join nothing are left out and named on the page (red outline, "no overlap found").

### 2.4 Bundle adjustment (`bundle.py`)
Rotating-camera model: `x_j ~ K R_j R_i^T K^-1 x_i`, one focal length `f` for all frames (same lens and zoom; a per-frame
`f` option for zooms), principal point at the centre, lens distortion already removed by developing each frame with
its lens correction (`stack_edit` + `edit.lens` as Develop would).
* Initial `f`: EXIF focal length and the sensor width (450D 22.2 mm, R5 / 5D IV / 6D 36 mm; a small table, else EXIF
  focal-plane resolution). **Adapted vintage lenses record nothing**: estimate `f` from the pairwise homographies
  (Szeliski and Shum 1997) and show it amber with "type it if you know it" (focal length field on the page).
* Initial rotations: chain the pairwise homographies along a maximum spanning tree of the match graph (weights = inliers).
* Solve with `scipy.optimize.least_squares` (method "trf", loss "huber", sparse Jacobian pattern given), residuals in
  pixels of the analysis copy, parameters = axis-angle per frame (frame 0 fixed) + `f`. Report RMS residual per pair.
* **Wave correction / level** (Brown and Lowe): the up vector that is most perpendicular to every frame's x axis,
  then one global rotation so the horizon is level. A "Level" checkbox (on).
* Tests: frames rendered from a known spherical scene with known rotations (yaw steps 25 degrees, small pitch and roll
  noise) and `f`: rotations within 0.05 degrees, `f` within 0.5%, with and without the EXIF start.

### 2.5 Projection and warping (`project.py`)
```python
PROJECTIONS = {"rectilinear", "cylindrical", "spherical", "panini", "mercator"}   # METHOD_INFO each
def canvas(rotations, f, frame_shape, projection, scale) -> (w, h, extent)
def to_camera(u, v, R, f, projection) -> (x, y, valid)   # canvas -> frame pixel (inverse map, for warping)
```
* Rectilinear: straight lines stay straight, up to about 120 degrees wide. Cylindrical: wide horizontal sweeps.
  Spherical / equirectangular: 360 x 180 (writes Google's GPano XMP fields so viewers treat it as 360: `ProjectionType`,
  `FullPanoWidthPixels`, `CroppedAreaImageWidthPixels`, ...; Export must keep them when the source is a 360 result).
  Panini (vedutismo, architecture interiors) and Mercator.
* Warping: each frame's footprint box only, `scipy.ndimage.map_coordinates` order 3 on `spline_filter`ed frames, in
  bands of 256 rows into a `Scratch` memmap per frame (a 20 x 45 MP spherical panorama is about 30000 x 15000 x 3
  float32 = 5.4 GB: never in memory at once).
* Exposure: one gain per frame in linear light from the overlaps (least squares with a prior pulling gains to 1,
  Brown and Lowe's gain compensation), then vignetting: use the lens falloff model when one exists (amber when only
  estimated); Develop's own vignette correction is off by default, so the panorama turns it on for its frames.

### 2.6 Seams and blending (`seams.py`, `pyramid.py`)
* Initial labels: each canvas pixel goes to the frame whose centre it is nearest (Voronoi on the sphere).
* **Seam:** along each boundary between two frames, a minimal-cost path through their overlap by dynamic programming
  (Efros and Freeman 2001 image quilting, extended along the whole boundary), cost = difference of the two frames in
  log luminance + gradient difference, so seams avoid moving people, water and leaves, and run along edges. (Graph cut
  needs a max-flow library we do not have; DP gives most of the benefit.) METHOD_INFO: "Smart seams" vs "Straight
  (Voronoi)".
* **Blend:** multi-band (Burt and Adelson 1983) on the label masks, levels from the overlap width, in tiles with a
  margin of 2^levels px so tiles join exactly (test: bit-identical to one-piece blending on a small canvas).
* **Ghost view:** where frames disagree after alignment (amber), so the person sees what was cut.
* **Crop:** the largest axis-aligned rectangle inside the valid area (maximal rectangle in a binary matrix on a downsized
  mask, then refined at full size); or "Keep the edge" (transparent border: needs an alpha channel in the intermediate,
  a small extension of `io/intermediate.py` with ExtraSamples; test that Develop still opens it).

### 2.7 `pano.py` and the page
`analyse(paths, params, progress, cancel)` -> `PanoAnalysis` (frames, pairs with inliers, rotations, f, residuals,
gains, canvas); `layout_view(an)` (outlines, colour by tie-in); `preview(an, params)` (1/8 size, < 10 s for 10 frames);
`build(an, params, out_path)` streaming; `result_tag` "Pano". Panel: projection picker (texts), Level, focal length
(shown, editable), seams (Smart / Straight), crop (Rectangle / Keep the edge), exposure match on, frames to leave out
(click an outline). Views: Result, Layout, Seams (labels coloured), Ghosts. Actions: Find layout, Preview, Build, Open
in Develop. Coverage-style honesty: show the pairs' RMS residuals; any pair over 2 px is listed with "parallax or
moving subject here".

### 2.8 Honest limits (state on the page, in these words or close)
Hand-held near subjects shift against the background between frames (parallax): no blend can fix that, and the seam
will show; use a tripod and turn around the lens. Moving water and people are cut along a seam, not merged. A lens with
no focal length in the file needs it typed or estimated. A full 360 needs overlap all the way round, top and bottom
included, or the poles stay empty.

### 2.9 Panorama done means
Synthetic sphere: layout, rotations and `f` within tolerance, gains matched within 1%, a straight line straight within
1 px in rectilinear, a moving object avoided by the seam, tiles bit-identical; a real set from Bill (if he sent one)
stitches with every pair under 2 px RMS or named; the page shows layout, preview and builds `..._Pano_Studio.tif` that
opens in Develop; all tests green; NEXT_UPDATE lines written.

---

## 3. Docs and wiring (both tabs)
* Sidebar order: Library, Develop, Dual Pixel Lab, Focus Stack, Super-Res Stack, **Infrared**, **Panorama**, Lenses,
  Cameras, Export, Help. Update `main.NAV`, the pages list and the `send_to_tab` rows.
* Help page: one card each (two sentences: what it does, what to shoot). README: two 🧪 rows in the feature table.
* `docs/PROGRESS.md`: a session section and the phase table rows; `docs/RESULTS.md`: the numbers measured;
  `docs/RESEARCH.md`: each method used, with citation status (session 12's rows were written from memory and are
  marked unchecked: check the ones you lean on, MOPS and Brown and Lowe first).
* `test_raws\README.txt`: sort whatever Bill dropped into `_incoming` and describe each set.

## 4. Order of work
| # | Block | Done when |
|---|---|---|
| 0 | Message to Bill (0.4). Sort `test_raws\_incoming` if he has dropped files. | Files described in `test_raws\README.txt`. |
| 1 | IR pipeline hook (1.3) + `infrared.py` guess, colour meter, multipliers, mixers, presets, mono raw (1.5b), channel tone, mono mix, noise meter and B&W recipes (1.5c) + synthetic model tests. | Existing sidecars render bit-identically; 850 frames guessed 850 with "no usable colour"; foliage neutral within 1%; mono path beats demosaic on the synthetic scene. |
| 2 | `InfraredPage` (B&W view while adjusting, channel thumbnails, Detail by scale shared with Develop) + Develop's IR white-balance chip + Channel mask type in Develop (1.5c) + Library "Edit in Infrared tab" + page tests. | An 850 frame from `test_raws` opens, presets live, Open in Develop matches. |
| 3 | Hot spot + fusion (synthetic pairs) + tests. | `test_hotspot.py`, `test_fusion.py` green. |
| 4 | `pyramid.py` factored out of focus (focus tests unchanged), `features.py` + tests. | Pairs found on the synthetic sphere; RANSAC rejects planted outliers. |
| 5 | `bundle.py`, `project.py` + tests. | Rotations / `f` within tolerance; projections round-trip within 0.01 px. |
| 6 | `seams.py`, `pano.py` (analyse, preview, streaming build) + tests; then Bill's Iraq set (0.3). | Synthetic panorama stitched, tiles bit-identical; the Iraq set stitches with every pair under 2 px RMS and an estimated focal length within 3% of 135 mm; compared side by side with Bill's own stitch. |
| 7 | `PanoramaPage` + Library send-to + page tests. | Builds from the page; opens in Develop. |
| 8 | Help cards, README rows, PROGRESS / RESULTS / RESEARCH, full suite. | Everything green; chat message to Bill: what to double-click and what to look at. |

Infrared goes first because Bill shoots it now and can judge it at once, and because it touches the pipeline (blocks 1-2
must not change any existing render). Panorama is pure addition on top of the stack foundation.
