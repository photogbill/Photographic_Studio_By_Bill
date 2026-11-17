# Build plan: Dual Pixel Lab, Focus Stack, Super-Res Stack

Written 2026-11-16 for the coding agent that builds it (Opus). Owner: Bill.
Bill's instruction: the three sidebar tabs that only show a description today get **built, all three, today**.
No "coming soon" labels.

This plan turns `BUILD_PLAN.md` sections 11, 12, 13, 23.5 and 23.7 into work you can do in one sitting, against the
code as it stands at version 0.1.5 plus the polish pass of 2026-11-16. Where this plan and those sections differ,
this plan wins for today; anything it leaves out stays as written there, for a later session.

---

## 0. Read first

### 0.1 Rules (all of `BUILD_PLAN.md` section 0 still applies; these are the ones that bite here)
1. **No new dependencies.** NumPy, SciPy, Pillow, tifffile, rawpy and PySide6 are enough for everything below. If you
   truly need one, it must be GPL-3.0 compatible and go into `requirements.txt`, `pyproject.toml` and `THIRD_PARTY.md`.
2. **Engine and interface stay apart** (2.2). Algorithms are pure functions under `studio/engine/`; no Qt imports
   there. Pages live in `studio/app/`, one file per tab.
3. **Stacking lives only in these tools.** The single-shot denoiser is not touched (0.4).
4. **More than one method is fine if the person can choose and understand** (23.1). Every method gets an entry in a
   `METHOD_INFO` dict (`name`, `does`, `best`, `weak`, `speed`), shown beside the picker, as
   `studio/engine/raw/denoise.py` and the demosaic picker in `studio/app/main.py` already do.
5. **Honesty in the interface.** Each tab states its real limits in plain words (the sentences are given below).
   If a method does not beat the simpler one on the tests, say so in `docs/RESULTS.md` and make the simpler one the default.
6. **Memory: an 8 GB laptop must not fall over.** A 45 MP frame is 540 MB as float32 RGB. Never hold more than about
   **2.5 GB** of arrays at once: stream frames one at a time, work in strips (`studio/engine/parallel.py`), keep raw
   mosaics as uint16, and put anything bigger in a memory-mapped file under `paths.temp_dir()` (create it with
   `mkdir(parents=True, exist_ok=True)`: `temp_dir()` does not), deleted when the job ends.
7. **Nothing on C:.** Results go beside the source frames (or the folder the person picks); scratch files go under
   the app's cache folder.
8. **Every file the Studio writes ends in `_Studio`** before its extension, and never replaces a file the Studio did
   not write: build names with `engine/io/export.py: export_name(source, spec, extra=...)` and pass the path through
   `free_target(path, source, taken)`. Names for these tabs are in 1.3.
9. **Errors report themselves.** In every `except` that catches a fault (not an unreadable file), call
   `studio.app.common.report_problem(what, exc)`. Long jobs run with `run_bg(..., keep=True)` and a
   `threading.Event` for Cancel, like `studio/app/process.py`.
10. **Write down what you built, as you build it:** one plain line per finished part in `docs/NEXT_UPDATE.md`
    (`- New: ...`). `push.bat` uses those lines as the comment. Also keep `docs/PROGRESS.md` and `docs/RESULTS.md` current.
11. **Do not push.** Bill pushes. Tell him in chat when a tab is ready for his eye.
12. **Tests first-class:** every algorithm gets a test on synthetic data with known truth (2.3 below gives the
    generators). The whole suite must stay green: 510 tests pass today (`python -m pytest`, about 7 minutes on two cores).
    Tests that pin the window (`tests/test_app.py`: every page fits 1280 x 600; `test_library.py`: 1366 wide) must
    still pass with the new pages.

### 0.2 What exists that you will use
| Need | Where |
|---|---|
| Open a raw, black and white levels, normalised mosaic (0..1, CFA kept) | `engine.open(path)`, `Image.levels()`, `engine/raw/linearize.py: normalise(raw, levels, box)` |
| Stages 1-12 to scene-linear ProPhoto RGB, whole frame or half size | `engine/pipeline/pipeline.py: linear_rgb(image, edit, region, half_size, cache=False)` (camera-crop frame, not yet turned upright) |
| Demosaic (RCD default, `half` for speed) | `engine/raw/demosaic.py: demosaic(mosaic, pattern, algorithm)` |
| Camera matrix, white balance | `engine/color/camera.py: camera_to_working(cam_xyz)`, `pipeline.wb_multipliers(image, edit)` |
| Noise model per frame (variance = a*signal + b) | `Image.noise_model()` |
| Turn upright | `pipeline.apply_orientation(img, raw.flip)` |
| Laplacian pyramids | `engine/detail/pyramid.py: gaussian, laplacian_from_gaussian, collapse` |
| Edge-aware smoothing | `engine/color/light.py: guided_filter(I, r, eps)` |
| Sub-pixel shift of two tiles by phase correlation | `engine/lens/auto.py: _phase_shift(a, b, iters)`: it already corrects the peak fit's pull toward whole pixels by re-measuring after a Fourier shift. Move it into the new `align.py` as `phase_shift` and import it back into `auto.py` (one copy) |
| Starlet (a trous) transform for moon sharpening | `engine/raw/denoise.py: starlet` |
| Strip-parallel work | `engine/parallel.py: strips(h, rows, job)` |
| Write TIFF / PNG / JPEG with profile and metadata | `engine/io/export.py: write(path, img_pp, spec, meta)` |
| Scans (TIFF, PNG, JPEG) opened as photos in Develop | `engine/io/scanfile.py`, `pipeline._scan_linear` |
| Thumbnails | `engine/io/thumbs.py: default_cache().get(path, size, mtime)` |
| Background jobs, settings, small widgets | `app/common.py: run_bg, Settings, label, chip, card, hbox`; `app/main.py: ValueSlider, Group` |
| A zoomable photo view (fit / 100%, drag) | `app/review.py: PhotoView` |
| Progress bar with Cancel in a page | pattern in `app/process.py: run_process` and the Library's `_build_progress` |
| Synthetic scenes and raws for tests | `tests/conftest.py: scene, mosaic_of, synthetic_raw, psnr` |
| The placeholder pages to replace | `app/main.py`: the three `InfoPage("Dual Pixel Lab" / "Focus Stack" / "Super-Res Stack", ...)` in `MainWindow.__init__`; `NAV` keeps the same nine entries |

### 0.3 Facts checked on 2026-11-16 (do not re-derive)
- `rawpy.imread(path, shot_select=1)` exists (rawpy 0.28 here, 0.27.1 on Bill's PC; LibRaw 0.22) and its own
  documentation names Dual Pixel RAW as the use. On a normal raw it raises `rawpy.LibRawRequestForNonexistentImageError`:
  that is the Dual Pixel test.
- LibRaw's CR3 reader treats `shot_select` as "which frame" for one-track files and "which track" for multi-track
  files. **Whether it hands back the second frame of an R5 Dual Pixel CR3 is not verified**: there is no Dual Pixel file
  in the project folder. This is step 1 of the day (3.1).
- Bill's measurements on a real R5 Dual Pixel file (`BUILD_PLAN.md` 11.1): frame 1 is A+B, frame 2 is one half (level
  ratio 2.007); black level 2048 on both; the two halves have strong opposite left-right shading (B/A from 2.7 to 0.31
  at f/11); in-focus detail agrees between halves (0.96), noise does not (0.02); stacking the two DNGs gains nothing.
- A float TIFF is already read as linear light by `scanfile.open_scan`; the pipeline then undoes the Standard curve
  (`scan_to_scene`) because it takes a scan to be display-referred. Stack results are scene-linear, so they need the
  flag in 1.2.
- Full-size export of an R5 frame peaks at 2.2 GB of memory; `linear_rgb` at half size takes a few seconds per frame.
- The citations in sections 11-14 of the master plan were written from memory (0.5 there). **Check each paper you
  lean on before you rely on a detail from it**, and note corrections in `docs/RESEARCH.md`: Fruchter & Hook 2002
  (Drizzle); Wronski et al. 2019 (handheld super-resolution); Hasinoff et al. 2016 (HDR+ merge); Burt & Adelson 1983
  (Laplacian pyramid fusion); Wadhwa et al. 2018 (dual-pixel depth); Punnappurath & Brown 2019 and Abuolaim & Brown 2020
  (dual-pixel reflection removal and defocus deblurring, both out of scope today).

### 0.4 Ask Bill once, at the start, in chat (one message, with these defaults if he is away)
1. **A Dual Pixel RAW file from the R5 and one from the 5D Mark IV**, put in the project folder. (Camera menu:
   Dual Pixel RAW > Enable; any subject with a bright light and some depth.) Without them the Dual Pixel tab is built
   and tested on synthetic data only, and the real-file check waits. His earlier eagle test was split with DPRSplit:
   the two DNGs from that work as input too (3.2).
2. **A focus bracket** (he has them: `D:\onedrive\Best Shots Folder\90mm Voigtlander`, needs folder access) and
   **a handheld burst** of anything detailed, 12-20 frames; a moon burst if he has one.
3. **Where results go:** default beside the source frames. (Recommended; say so and move on.)

---

## 1. Shared foundation (build this first; all three tabs stand on it)

### 1.1 Files
```
studio/engine/stack/
    __init__.py        (exists, empty)
    frames.py          probing a set of frames, the shared "stack edit", streaming them
    align.py           shifts, similarity / affine fits, warps, local flow
    focus.py           Focus Stack (section 4)
    superres.py        Super-Res Stack: noise-only, drizzle, moon (section 5)
    astro.py           night-sky registration and rejection (section 5.5)
studio/engine/dualpixel/
    __init__.py        (exists, empty)
    decode.py          the two frames, halves, shading (section 3)
    depth.py           disparity, confidence, noise-vs-detail map
    hdr.py             highlight recovery / Dual Pixel HDR
    views.py           split views, 3D outputs
studio/engine/io/intermediate.py     the linear intermediate file (1.2)
studio/app/stackui.py                shared widgets for the three pages (2.1)
studio/app/focusstack.py  studio/app/superres.py  studio/app/dualpixel.py     the three pages
tests/test_align.py  test_intermediate.py  test_focus_stack.py  test_superres.py  test_dualpixel.py  test_stack_pages.py
```
Add the public calls to the lazy table in `studio/engine/__init__.py` (`_EXPORTS`).

### 1.2 The linear intermediate (how a result gets back into Develop)
The master plan: a stack result "goes straight back into Develop as a 32-bit linear intermediate (plus its own
sidecar), so every raw tool except stages 1-10 still applies". Build exactly that on top of the scan path.

`engine/io/intermediate.py`
```python
MARK = "studio-intermediate-1"
def write_intermediate(path, rgb_scene, meta: dict, info: dict) -> str
    # rgb_scene: float32 HxWx3, scene-linear ProPhoto, 1.0 = sensor clip of a neutral, already upright.
    # 32-bit float TIFF, zlib-compressed, BigTIFF when over 4 GB (tifffile does this), linear ProPhoto ICC embedded
    # (engine/color/icc.py: add a linear-ProPhoto profile if there is none), Software tag as export.py writes it
    # (so is_own_export() is true), ImageDescription = JSON {"mark": MARK, "kind": "focus" | "superres" | "dualpixel",
    # "sources": [file names], "method": ..., "params": {...}, "scale": 1.0 | 1.5 | 2.0}.
    # Written to path + ".partial", then os.replace. Returns the path.
def read_info(path) -> dict | None       # the JSON above, or None for any other file
```
Changes elsewhere (small, each with a test):
- `scanfile.ScanImage`: new field `scene_linear: bool = False`; `open_scan` sets it when `read_info(path)` has the mark.
- `pipeline._scan_linear`: `rgb = disp if raw.scene_linear else scan_to_scene(disp)`.
- `scanfile.scan_preview` and `ScanImage.embedded_jpeg`: for a `scene_linear` file put the Standard curve
  (`engine/color/tone.py: standard_curve`) on before encoding, so the Library thumbnail looks like the photo, not flat and dark.
- Library card: a small chip on intermediates ("Focus stack", "Super-res 2x", "Dual Pixel HDR") from `read_info`.
- Test: a synthetic scene written as an intermediate and opened with `engine.open` renders within 0.5% of rendering
  the same scene from a synthetic raw (same default edit); values above 1.0 survive the round trip.

The result's own sidecar (`NAME.tif.psb.json`) is the ordinary one. Put a `"stack"` block in it (sources, method,
parameters, date) with `api.load_doc` / `api.write_doc`, so "Build again" can refill the page.

### 1.3 Names (rule 0.1.8)
| Result | Name |
|---|---|
| Focus stack | `<first stem>_to_<last stem>_Focus_Studio.tif` |
| Super-res | `<first stem>_to_<last stem>_SuperRes2x_Studio.tif` (`SuperRes15x`, `Stack` for noise-only, `Moon2x`, `Sky`) |
| Dual Pixel | `<stem>_DP-HDR_Studio.tif`, `<stem>_DP-A_Studio.tif`, `<stem>_DP-B_Studio.tif`, `<stem>_DP-Depth_Studio.png`, `<stem>_DP-3D_Studio.gif`, `<stem>_DP-Anaglyph_Studio.jpg` |
Default folder: beside the first source frame; if that folder cannot be written, `paths.get("OUTPUTS") / "stacks"`,
and say so. The page has a folder button and remembers the last choice (`Settings`).

### 1.4 `frames.py`
```python
@dataclass
class FrameInfo: path, name, width, height, camera, iso, exposure_s, fnumber, focal_mm, taken, flip, ok, problem
def probe(paths) -> list[FrameInfo]          # metadata only (engine/io/metadata.read_metadata, 512 KB head), sorted by capture time then name
def check(frames) -> list[str]               # plain-language warnings: different cameras or sizes (blocks the build);
                                             # different exposure / ISO / aperture (warns; focus stack and bursts expect the same)
def stack_edit(reference: Image, denoise: bool) -> Edit
    # ONE edit for every frame: white balance fixed to the reference frame's as-shot multipliers
    # (white_balance.mode = "multipliers"), lens.auto_ca on, lens.defringe on, sharpen off, highlight_mode "clip",
    # denoise on or off as asked. Same settings on every frame is what makes them comparable.
def scene_rgb(path, edit, half: bool) -> np.ndarray     # linear_rgb(engine.open(path), edit, None, half_size=half, cache=False); drop the Image at once
def mosaic(path) -> tuple[np.ndarray, RawInfo]          # normalised white-balanced CFA mosaic (float32 or uint16 + scale), cropped to the camera crop, plus pattern, cam_xyz, flip
```
Frames from different files are never all in memory as RGB. Unreadable frames (`UnsupportedRaw`) are listed with their
reason and left out; they do not stop the others.

### 1.5 `align.py` (the workhorse; get it right, with tests, before any merge)
```python
def luma(rgb) -> np.ndarray                                  # 0.25 R + 0.5 G + 0.25 B is enough for registration
def phase_shift(a, b, iters=3) -> (dy, dx, quality)          # moved from lens/auto.py (_phase_shift): windowed cross-power spectrum, parabola peak fit, refined by re-measuring; quality = the peak
def tile_shifts(ref, img, tile=128, step=96, max_shift=24) -> (centres Nx2, shifts Nx2, quality N)   # skip flat tiles (gradient energy below ~6x noise) and clipped ones
@dataclass
class Transform: matrix (3x3, maps reference pixel coords -> image coords), kind ("shift" | "similarity" | "affine" | "homography"), residual_px, tiles_used
def fit(centres, shifts, quality, kind) -> Transform         # weighted least squares, then 3 rounds of dropping tiles whose residual is over 3x the median (MAD)
def register(ref_luma, img_luma, kind="similarity", levels=3) -> Transform
    # coarse to fine: at 1/8 size one whole-frame phase_shift; then at 1/4, 1/2 and full analysis size:
    # warp by the current estimate, tile_shifts on the residual, fit, compose. Two passes at the last level.
def warp(img, transform, out_shape=None, order=3) -> np.ndarray      # scipy.ndimage.map_coordinates per channel, in strips (parallel.strips), mode="nearest"; order=1 for analysis copies
def flow(ref_luma, img_luma, tile=64, step=32, max_shift=6) -> (dy_field, dx_field, quality_field)
    # dense local displacement after global registration: tile_shifts on a fine grid, median filter 3x3 on the grid,
    # tiles with poor quality filled from neighbours, bilinear up to full size. For hand shake, rolling shutter and air wobble.
```
Registration always runs on **analysis copies**: half-size luminance (from `scene_rgb(..., half=True)`, or the mosaic
binned 2x2 for Super-Res), long edge capped near 2000 px by a further integer bin. Transforms are stored in
full-resolution pixel units (scale the matrix).

Tests (`tests/test_align.py`), on `conftest.scene` with added noise:
- a known shift of (3.37, -1.62) px (made by Fourier shift) is recovered within **0.05 px**;
- scale 1.012, rotation 0.25 degrees, shift (5, -3): recovered scale within **1e-4**, angle within **0.01 degrees**, corner error under **0.2 px**;
- a tile with a moving object does not pull the fit (one quarter of the tiles shifted by 4 px: global fit still within 0.1 px);
- `flow` recovers a smooth synthetic warp (sinusoid, 1.5 px amplitude) with mean error under 0.15 px;
- `warp` then inverse `warp` returns the image (PSNR over 40 dB inside the frame).

---

## 2. The three pages

### 2.1 Shared look and parts (`studio/app/stackui.py`)
All three pages have the same skeleton, so a person who learns one knows the others (and rule 0.3.12: each area its own tab):

```
+--------------------------------------------------------------------------------------------+
| Title   [chips: 12 frames . R5 . ISO 800]              [Add frames...] [Clear]  [Help ?]   |
+--------------------------------------------------------------+-----------------------------+
|                                                              |  Method (picker + Best for  |
|                    ResultView                                |  / Weak at text)            |
|     (PhotoView: fit / 100%, drag; B flips between the        |  Options (sliders)          |
|      result and the best single frame; Z zooms)              |  Honest limits (one card)   |
|                                                              |  [Preview]  [Build full]    |
+--------------------------------------------------------------+  [Open in Develop] [Folder] |
|  FrameTray: thumbnails in order, a tick on each,             |                             |
|  chips under each (sharpness rank, kept / dropped, offset)   |                             |
+--------------------------------------------------------------+-----------------------------+
|  Progress: text . bar . [Cancel]                                                           |
+--------------------------------------------------------------------------------------------+
```
- `FrameTray(QListWidget, icon mode)`: thumbnails from the Library's thumbnail cache; drag files or a folder onto the
  page to add; tick to include; a coloured chip per frame after analysis (cyan = measured and used, amber = used with
  a warning, red = dropped, with the reason as tooltip).
- `ResultView`: wraps `review.PhotoView`; `set_images(result_qimage, reference_qimage)`; **B** (hold or toggle) shows the
  best single frame in place of the result, at the same zoom and position: the only fair way to judge.
- `MethodPicker`: a combo plus a text block filled from `METHOD_INFO` (`does`, `Best for`, `Weak at`, `Speed`).
- `JobBar`: text, bar, Cancel; `start(fn(progress, cancel), on_done)`; uses `run_bg(..., keep=True)`.
- `empty state`: with no frames, the middle of the page says in three lines what the tab does, how to give it frames
  (Library: select, right-click, "Send to ..."; or drop files here), and what to shoot. Never a blank panel.
- Two-step flow everywhere: **Preview** works at half size and answers in seconds to a minute; **Build full size**
  writes the file. The person sees a result before committing minutes.
- Width: the right panel is the Develop panel's width; nothing in it may be wider than its space (the test in
  `tests/test_polish.py` for the Develop panel shows the check; add the same for these pages).
- Keys: B compare, Z 100%, Esc back to fit, Ctrl+Enter build.

### 2.2 Getting frames in
- Library right-click menu and Tools menu: **Send to Focus Stack**, **Send to Super-Res Stack** (selection of 2 or
  more), **Open in Dual Pixel Lab** (one photo). New signals on `LibraryPage` (`send_to = Signal(str, list)`), wired
  in `MainWindow` the way `open_photo` is; the nav switches to the tab.
- Each page: **Add frames...** (file dialog starting in the Library's folder) and drag-and-drop.
- After a build: **Open in Develop** opens the result (`MainWindow.open_in_develop(path)`), and the Library shows it
  if its folder is the open one (`library.reload(select=path)`).

### 2.3 Help, README, progress notes
- Help page: one card per tab (two sentences each: what it does, what to shoot).
- `README.md` table: the three rows go from planned to experimental preview, in the README's own words.
- Remove "Phase N" wording from text a tester reads (Lenses, Cameras and Export pages, and the new pages): say what
  works and what is planned, in plain words.
- `docs/PROGRESS.md`: phases 8, 10, 11 with what was built and what waits for Bill's files.

### 2.4 Tests for the pages (`tests/test_stack_pages.py`, offscreen like `tests/test_app.py`)
- each page builds, shows its empty state, fits 1280 x 600 with the window, and the right panel fits its width;
- frames dropped on the page appear in the tray; an unreadable file is listed with its reason, not raised;
- a tiny synthetic set (3-4 frames written as synthetic raws is not possible without a raw writer: monkeypatch
  `frames.scene_rgb` / `frames.mosaic` to serve synthetic arrays) runs Preview and Build end to end and writes a
  `_Studio.tif` that `engine.open` reads back;
- Cancel stops a build and leaves no `.partial` behind.

---

## 3. Dual Pixel Lab

### 3.1 First thing of the day: does the decoder give us frame 2? (15 minutes)
With Bill's two Dual Pixel files (0.4): `rawpy.imread(path, shot_select=1)` on each. Record in `docs/RESEARCH.md`:
size, black and white levels, mean level ratio to frame 0 (expect about 2), whether the CR3 works.
- Works for both: go on.
- CR2 works, CR3 does not: build on CR2, and accept **a pair of DPRSplit DNGs** as the R5 input (below). Write the
  finding down; our own CR3 second-track extractor (the CR3 box parser is in `raw_analyzer.py` and
  `engine/io/tiffmeta.py`) becomes its own task, not today's.
- No files from Bill yet: build everything on the synthetic generator (3.7) and leave the real-file check open in PROGRESS.

### 3.2 `dualpixel/decode.py`
```python
def is_dual_pixel(path) -> bool         # shot_select=1 opens without LibRawRequestForNonexistentImageError and is the same size; cheap (no decode)
@dataclass
class DualPixel:
    image: Image            # frame 0 as the Studio knows it (A+B): everything else in the app keeps working on it
    ab: np.ndarray          # normalised mosaic, black subtracted, 1.0 = clip (camera crop)
    a: np.ndarray           # the half frame, same scale as ab (so about ab/2), from frame 1 with ITS OWN black level
    b: np.ndarray           # ab - a, clipped at 0; invalid where ab is clipped (mask below)
    clipped_ab, clipped_a: np.ndarray (bool)
    pattern, wb, cam_xyz, flip
def open_dual(path) -> DualPixel                       # raises UnsupportedRaw("... was not shot in Dual Pixel RAW ...") with the camera-menu hint
def open_dual_pair(path_ab, path_half) -> DualPixel    # two DNGs from DPRSplit (frame 1 = A+B, frame 2 = one half), checked by size and level ratio
def shading(dp, sigma_frac=1/12) -> (gain_a, gain_b)   # per-half flat field, estimated from this photo
```
Shading (the requirement from 11.1): each half is darker toward one side, with a colour cast. Estimate per CFA colour:
low-pass `a` and `b` heavily (Gaussian, sigma = frame width / 12, on the 2x2-binned planes, clipped and near-black
photosites excluded by normalised convolution), fit `log(a_low / (ab_low / 2))` with a 2-D polynomial of order 3
(robust: two rounds, dropping the worst 10%), and return gains that bring each half to `ab / 2` on average. After
correction the halves must agree in brightness within 3% across the frame on a flat scene (test). A measured
per-lens flat field from the lens wizard replaces the estimate later; keep the function signature ready for it.

Sign convention: which half looks through which side of the lens is not assumed. Call them "half 1" (stored) and
"half 2" (derived); the depth map's near/far sign is fixed once on a real file (3.1) and written into RESEARCH.md,
and the page has a "Swap near / far" tick until then.

### 3.3 `dualpixel/depth.py`
```python
def halves_luma(dp, gains) -> (la, lb)          # shading-corrected, 2x2-binned luminance of each half (half resolution, CFA summed)
def disparity(la, lb, tile=48, step=16, max_shift=4.0) -> (disp, conf)
    # the split is along the sensor's long axis only: a 1-D search. Band-pass both (difference of Gaussians, sigma 1 and 4),
    # per tile: normalised cross-correlation at integer shifts -max..max along x, parabola through the peak for the
    # sub-pixel value. conf = (peak - second peak) * texture energy / noise energy, squashed to 0..1.
    # Flat or in-focus-ambiguous tiles get low confidence: say so, do not invent depth.
def refine(disp, conf, guide_luma, r=24, eps=1e-3) -> disp_full
    # confidence-weighted guided filter (guided_filter(disp*conf)/guided_filter(conf) with the A+B luminance as guide), up to full size
def detail_map(la, lb, disp, win=9) -> rho
    # local correlation coefficient of the high-passed halves after aligning by disp: near 0 = noise, near 1 = real detail
    # (Bill's file: 0.02 in sky, 0.96 on feathers). This is the "noise versus detail" map.
def focus_report(disp, conf) -> dict            # share of the frame in focus (|disp| < 0.15 px, confident), median front / back offset of the confident subject, in plain words
def air_steadiness(disp, conf) -> dict          # 11.2 item 3, experimental: the patchy (non-smooth) part of the offset map: one number + a map
```
`focus_report` drives the honest sentence from 11.1: "In focus: nothing to gain from re-aligning the halves" or
"Offset found (about 0.6 px behind the subject): micro-refocus could help" (the tool itself is later).

### 3.4 `dualpixel/hdr.py` (highlight recovery, 23.5: build as an option)
```python
def recover(dp, gains, disp=None, lo=0.80, hi=0.97) -> (rgb_scene, report)
```
- Demosaic A+B and half 1 separately (RCD; the half after shading correction), white-balance both, camera matrix both.
- Where the halves are offset (|disp| over 0.25 px) shift half 1 by half the local offset before use, so recovered
  highlights sit where the picture is.
- Per pixel weight `w = smoothstep(lo, hi, max channel of A+B before white balance / clip)`; result
  `(1 - w) * AB + w * 2 * half1`, only where half 1 is itself not clipped (`w = 0` where it is, with a 3 px feathered edge).
- The result is scene-linear with values above 1.0: write it as an intermediate (1.2); Develop's Highlights slider
  and tone curve then bring the recovered range into view. Nothing in the shadows changes.
- `report`: stops gained (log2 of the highest recovered value over clip, at most about 1.0), share of the frame recovered,
  share still clipped in both. Shown as a cyan chip (measured): "Recovered up to +0.9 stop in 3.1% of the frame".
- Honest label, fixed text on the page: **"About one extra stop in the highlights, nothing extra in the shadows:
  the normal picture already holds all the light."**

### 3.5 `dualpixel/views.py`
```python
VIEWS = ("A+B", "Half 1", "Half 2", "Difference", "Depth", "Confidence", "Noise vs detail")
def view(dp, gains, name, scale=0.25, maps=None) -> uint8 sRGB        # for the page: halves developed like the photo; difference amplified 8x around grey; depth as a colour ramp with low-confidence areas greyed
def export_half(dp, gains, which, path) -> str                        # a developed half as an intermediate (what DPRSplit does, built in)
def anaglyph(dp, gains, scale) -> uint8                               # red = half 1, cyan = half 2
def side_by_side(dp, gains, scale) -> uint8
def wigglegram(dp, gains, path, long_edge=1200, ms=120, boost=1.0)    # GIF flipping between the two views (Pillow); boost > 1 exaggerates by warping each half along x by +-boost*disp/2
def depth_png(disp_full, conf, path)                                  # 16-bit PNG, near = white, for Photoshop or Blender
```

### 3.6 The page (`studio/app/dualpixel.py`)
- Takes one photo ("Open in Dual Pixel Lab", or drop a file). A normal raw: the middle of the page says
  **"This photo was not shot in Dual Pixel RAW."** with how to turn it on (R5 and 5D Mark IV menu path) and what it
  costs (files twice the size, slower bursts). Offer "Use two DPRSplit files..." beside it.
- Left: `ResultView` with a view picker along the top (the seven views); **W** wiggles between half 1 and half 2 in place.
- Right panel groups: **Split** (export half 1 / half 2 / both), **Highlights** (Recover, strength, the report chip,
  the honest label, Build -> intermediate -> Open in Develop), **Depth** (map, confidence, focus report sentence,
  Swap near / far, Export depth PNG), **Noise vs detail** (the map and one line on what it means), **3D** (wigglegram,
  anaglyph, side by side, boost slider), **Lab** (air steadiness number and map, marked Experimental, 11.2).
- Chips: shading "estimated from this photo" (amber) until a measured flat field exists (cyan).
- Library badge "DP" on Dual Pixel files: only if `is_dual_pixel` proves cheap enough inside `libindex.load_entries`
  (measure it: it must not slow folder listing); otherwise leave the badge for later and say so in PROGRESS.

### 3.7 Tests (`tests/test_dualpixel.py`)
Synthetic split-aperture generator (put it in the test file, reuse `conftest.scene`):
a scene, a depth map (a ramp plus two blocks), per pixel offset `d = k * (depth - focus)`; half 1 = scene warped by
`+d/2` along x and blurred by `|d|`, half 2 = by `-d/2`; each multiplied by opposite left-right shading ramps
(1.6 to 0.6, with a mild colour cast); A+B = sum; mosaiced; independent shot + read noise on each half; black level
added; A+B clipped at the white level where the scene exceeds it.
- `b == ab - a` exactly on unclipped photosites; clipped ones are masked.
- `shading`: recovered ramps within **3%** of the truth; after correction the halves match within 3% on a flat patch.
- `disparity` + `refine`: correlation with the true offset map over **0.9** and mean error under **0.15 px** where
  texture exists; confidence under 0.3 on the flat block; in the in-focus band |disp| under 0.1 px.
- `detail_map`: under **0.2** on a flat noisy patch, over **0.8** on texture.
- `recover`: with the scene reaching 1.8x clip, recovered values within **5%** of truth wherever half 1 is unclipped;
  pixels below 0.8 of clip are bit-identical to the normal development; reported gain between 0.8 and 1.0 stop.
- a normal (single-frame) file: `is_dual_pixel` false; `open_dual` raises `UnsupportedRaw` naming the reason.
- views and 3D outputs: right sizes, written under `_Studio` names, GIF has two frames.
Real files (when Bill's arrive): both decode; level ratio 1.9-2.1; a half exported here matches the DPRSplit DNG of the
same file within 1% of mean level; the candle frame shows recovered flame detail (the Phase 8 acceptance).

### 3.8 Out of scope today (leave the hooks, list them in PROGRESS)
Micro-refocus, bokeh shift, ghost and flare control, defect detection from the halves, depth masks inside Develop's
Masks, feeding the noise-vs-detail map to the denoiser, synthetic aperture, defocus deblurring, reflection removal,
the long-range re-align experiment (the readout is in, the re-align is not), our own CR3 second-track extractor.

---

## 4. Focus Stack

### 4.1 Pipeline (`stack/focus.py`)
```python
METHOD_INFO = {
  "depth":   {"name": "Sharpest frame per place", "does": "Takes each place from the frame where it is sharpest, with soft joins.",
              "best": "simple subjects, clean colour, lowest noise", "weak": "hairs and bristles that cross in front of each other", "speed": "fast"},
  "pyramid": {"name": "Finest detail everywhere", "does": "Keeps the strongest detail at every size from any frame.",
              "best": "hairs, bristles, overlapping detail", "weak": "can raise noise and glare; colours slightly flatter", "speed": "medium"},
  "average": {"name": "Weighted blend", "does": "Blends frames by how sharp each is.",
              "best": "smooth subjects, lowest noise, no hard joins", "weak": "a little softer than the others", "speed": "fast"},
  "auto":    {"name": "Auto (default)", "does": "Finest detail where things overlap, sharpest frame elsewhere.",
              "best": "most stacks", "weak": "slowest", "speed": "slow"},
}
@dataclass
class FocusParams: method="auto", denoise_first=False, align="similarity", equalise=True, radius=1.0, smooth=1.0, halo_guard=True, order="auto"
def analyse(paths, params, progress, cancel) -> FocusAnalysis     # pass A, half size
def preview(analysis, params) -> np.ndarray                       # half-size result (scene-linear), from the half-size frames kept on disk as float16 memmap
def build(analysis, params, out_path, progress, cancel) -> str    # pass B, full size, one streaming pass; writes the intermediate
def source_patch(analysis, frame_index, box) -> np.ndarray        # one aligned source frame's pixels for the retouch brush
```

**Pass A (analysis, half size).** For each frame: `scene_rgb(path, stack_edit, half=True)`; keep its luminance and
store the half-size RGB as float16 in a memmap (20 frames of 45 MP at half size is about 1.3 GB on disk; fine).
1. **Order.** "auto": after registration, sort by the fitted magnification (focus breathing makes it change steadily
   with focus distance); if the magnifications do not form a clear trend (spread under 0.05%), keep capture order.
   The person can reverse or drag frames.
2. **Align.** Register each frame to its neighbour in the order (`register(..., kind=params.align)`; neighbours share
   the most focus, so they register best), compose to the middle frame as reference. `similarity` is the default
   (scale + rotation + shift: what breathing does); `affine` and `homography` are choices for a rail that tilts.
   Report residuals per frame; a frame over 1.5 px residual gets an amber chip ("did not line up well").
   The common area of all warped frames is the result's frame (crop the rest; say how much was cropped).
3. **Equalise** brightness (on by default): one gain per frame from the ratio of robust means of the low-passed
   luminance over the common area (flash or shutter flicker); clamp to +-0.3 stop and say so if a frame hit the clamp.
4. **Sharpness** per frame on the aligned luminance: `E = G_s * (Laplacian_1(L)^2 + Laplacian_2(L)^2)` (Laplacian of
   Gaussian at sigma 1 and 2 px times `radius`, smoothed with a Gaussian of 4 px times `radius`), minus the energy pure
   noise would give at that brightness (from the frame's noise model through the same filters; measure the factor on
   white noise once, as `pyramid.white_noise_factors` does), floored at 0. Noise must not look like focus.
5. **Depth map.** `idx = argmax_i E_i`; confidence `c = (E_max - E_second) / (E_max + eps)`. Smooth the index map
   edge-aware: confidence-weighted guided filter with the reference luminance as guide (radius 16 px times `smooth`),
   so low-confidence places take their depth from confident neighbours. Keep it as a float ("fractional depth").

**Merges** (preview at half size from the memmap; build at full size by streaming):
- `depth`: for each output pixel blend the two frames either side of the fractional depth
  (`w_i = max(0, 1 - |depth - i|)`). One streaming pass: for each frame, warp at full size, add `w_i * frame`.
- `average`: weights `E_i^2` (normalised, with a floor so flat areas average all frames and lose noise).
- `pyramid`: Laplacian pyramid of each aligned frame (per channel, 6-7 levels); at each level and pixel keep the
  coefficient from the frame with the largest **local energy** (the level's `|L|` smoothed over 5 x 5, luminance only,
  the same choice for all three channels, so no colour fringes; plain per-pixel maximum is what boosts noise);
  coarsest level: the `depth` result's coarsest level (keeps colour and brightness honest). Streaming: keep only the
  running best pyramid and its energy, one frame at a time.
- `auto`: `mix = smoothstep(0.55, 0.8, E_second / E_max)` blurred 8 px; result `= mix * pyramid + (1 - mix) * depth`.
- **Halo guard** (on by default, all methods): clamp every output pixel to the minimum and maximum that any aligned
  frame had within a 5 x 5 neighbourhood of that place (running min and max, updated per frame). Nothing can be
  brighter or darker than every source: this is what removes halos at depth edges and pyramid overshoot.
- **Denoise first** (off by default; the master plan suggests it, measure it): when on, `stack_edit(denoise=True)`.
  Record in RESULTS.md whether it helps on Bill's stacks; the weighted methods already average noise down.

**Pass B (build).** Full size, one frame in memory at a time: decode, warp (order 3), weight, accumulate; the depth
map and the weights come from pass A scaled up 2x (they are smooth). Memory: result + one frame + running min/max.
Then `apply_orientation`, `write_intermediate` (kind "focus"), sidecar `"stack"` block, done.

**Retouch from a source frame** (the master plan's fix-up; in scope if the day allows, after the three tabs work):
a brush on the result; the tray's selected frame is the source; painted areas take `source_patch` instead. Stored in
the `"stack"` block as strokes, applied at build. If there is no time, leave the engine call (`source_patch`) and the
note in PROGRESS; the person can still fix a ghost by building twice with different frames ticked.

### 4.2 The page (`studio/app/focusstack.py`)
- Tray in focus order with a "near -> far" arrow, Reverse button, a tick per frame; chips: line-up residual, brightness gain.
- Views (picker over the result): Result, Depth map (which frame each place came from, as a colour ramp, cyan = measured),
  Confidence, any single frame aligned (click it in the tray).
- Right panel: Method picker with the four texts; Align (Similarity / Affine / Perspective, with one line each);
  sliders Detail size (`radius`, 0.5-3), Smoothness of joins (`smooth`, 0-3); ticks Even out brightness,
  Halo guard, Clean each frame first; **Preview**, **Build full size**, **Open in Develop**.
- Honest limits card: **"Works on a subject that held still. Something that moved between frames (wind on a leaf)
  shows as a ghost there: untick the frames where it moved, or use the single frame that has it sharp."**
- What to shoot (empty state): "A focus bracket: same framing, same exposure, focus stepped from near to far.
  Tripod or rail. 5 to 100 frames."

### 4.3 Tests (`tests/test_focus_stack.py`)
Generator: an all-sharp scene (`conftest.scene`, 384 x 576) and a depth ramp with a step (a foreground block);
frame `i` = scene blurred per pixel by a Gaussian of sigma `1.2 * |depth - f_i|` (blend a small bank of pre-blurred
copies), scaled by `1 + 0.004 * i` about the centre (breathing), shifted by a known sub-pixel amount, with noise.
- alignment recovers each frame's scale within 2e-4 and shift within 0.1 px; "auto" order equals the true order;
- every method beats the best single frame by **at least 4 dB** PSNR against the all-sharp truth, and `depth` reaches
  within 1.5 dB of an oracle that picks the true sharpest frame per pixel;
- the depth map correlates with the true depth over **0.9** where there is texture;
- **halo test** at the foreground step: no output value more than 2% outside the range of the sources there; with
  Halo guard off the pyramid method fails this (shows the guard earns its place);
- **noise test**: on a flat grey patch the result's noise is not above the single frames' (`pyramid` allowed 1.2x; the others below 1.0x);
- a frame with a 3 px misalignment that registration cannot explain (a moved block) gets flagged, not silently used;
- memory: `build` on 8 frames of 2000 x 3000 synthetic stays under 1.5 GB (tracemalloc or resource peak);
- determinism: the same inputs and parameters give the same bytes.
Acceptance on Bill's files (master plan section 12): his 90 mm Voigtlander macro stacks merged with no visible halos
at 100%; put a blind comparison against his current stacking software in `docs/RESULTS.md` when he has judged it.

---

## 5. Super-Res Stack

### 5.1 Modes and methods (`stack/superres.py`)
```python
MODE_INFO = {
  "handheld": {"name": "Handheld burst", "does": "Lines up a hand-held burst and builds a larger, cleaner photo.",
               "best": "a subject that held still; 12-30 frames", "weak": "anything that moved falls back to one frame there"},
  "clean":    {"name": "Cleaner, same size", "does": "Averages the burst for less noise at the normal size.",
               "best": "low light without a tripod; 4 frames and up", "weak": "adds no detail"},
  "moon":     {"name": "Moon and planets", "does": "Keeps the sharpest frames, undoes the air's wobble, builds larger.",
               "best": "50 frames and up at a fast shutter speed", "weak": "needs many frames; thin cloud ruins it"},
  "sky":      {"name": "Night sky", "does": "Lines up on the stars and averages, removing planes and satellites.",
               "best": "10 frames and up on a tripod or tracker", "weak": "the ground blurs (Freeze ground comes later)"},
}
MERGE_INFO = {
  "drizzle": {"name": "Drizzle (from the raw data)", "does": "Drops every raw photosite onto a finer grid at its measured position: no demosaic.",
              "best": "real extra detail when the frames landed on different sub-pixel positions", "weak": "needs 12+ frames for 2x, 6+ for 1.5x; slower", "speed": "slow"},
  "average": {"name": "Robust average", "does": "Develops each frame, lines it up, averages what agrees.",
              "best": "cleaning noise; few frames; anything that moved", "weak": "no extra detail", "speed": "medium"},
}
@dataclass
class SuperParams: mode="handheld", merge="drizzle", scale=2.0, keep_best=1.0, pixfrac=0.7, reject=3.0, local=True, sharpen=0.0, crop=None
def analyse(paths, params, progress, cancel) -> BurstAnalysis     # rank, register, local flow, robustness (half size)
def preview(analysis, params, box=None) -> np.ndarray             # a 100% window of the result (default: the sharpest 768 px square), built the real way: honest preview
def build(analysis, params, out_path, progress, cancel) -> str
```
The mode sets defaults (merge, scale, keep_best, local flow on or off, rejection strength); the person can change them.

### 5.2 Analysis (all modes)
1. **Per frame, cheap:** `mosaic(path)`; analysis luminance = 2x2 bin of the white-balanced mosaic (R + 2G + B).
   Keep the mosaics as uint16 (90 MB each at 45 MP); when the set is over about 1.5 GB keep them in a memmap under the
   cache folder instead of RAM.
2. **Rank** by sharpness: variance of the Laplacian of the analysis luminance over the common textured area, divided
   by the noise variance. The reference is the sharpest frame. Drop frames below 60% of the median (blurred), and for
   `keep_best < 1` keep that share of the rest (moon default 0.25). Chips in the tray: rank and kept / dropped.
3. **Register** to the reference: `register(kind="similarity")` handheld and clean; shift only for the moon (the disc
   is the subject: mask to the bright disc, centre of brightness first); stars for the sky (5.5).
4. **Local flow** (`align.flow`, on for handheld and moon): residual hand shake, rolling shutter, air wobble.
   Moon: tiles 48 px on the half-size luminance, step 24 ("multi-point alignment").
5. **Robustness map** per frame: after warping the analysis luminance, `r = exp(-(d / (reject * sigma))^2)` where
   `d` is the 5 x 5-smoothed difference from the reference and `sigma` the noise at that brightness (noise model) plus
   a term for aliasing (local gradient x 0.5 px, so edges are not rejected for sub-pixel reasons). Moving things and
   bad tiles get weight near 0. The share rejected per frame goes on its chip.
6. **Sub-pixel coverage report** (honesty before the wait): the fractional parts of the global shifts. If they
   cluster (a tripod: all within 0.1 px), say **"These frames did not move between shots: stacking will clean noise
   but cannot add detail"** and switch the default to Cleaner, same size.

### 5.3 Merge: robust average (`merge="average"`, scale 1)
For each kept frame: `scene_rgb(path, stack_edit(denoise=False), half=False)`, warp by transform + flow (order 3),
weight by its robustness map (up-scaled) times its sharpness weight; accumulate `sum(w * frame)` and `sum(w)`;
where `sum(w)` is under 0.5 the reference frame alone fills in. One frame in memory at a time. Expected noise:
`1 / sqrt(N)` of a single frame where nothing moved (test).
Also offer rejection **"Sigma clip"** for the sky mode (5.5), which needs per-pixel statistics: two passes
(mean and variance, then a clipped mean), still streaming.

### 5.4 Merge: drizzle from the mosaics (`merge="drizzle"`, scale 1.5 or 2)
Verify the method against Fruchter & Hook 2002 first (0.3). Implementation that NumPy can carry:
- Output grid `scale` times the camera crop, three planes (R, G, B) of numerator and weight.
- For every kept frame and every CFA colour: the photosite centres `(y, x)` of that colour, mapped through the
  frame's transform and flow to output coordinates, scaled. Each photosite is a "drop" of width `pixfrac` input pixels
  (so `pixfrac * scale` output pixels). Deposit it with **bilinear weights to the 4 nearest output pixels**, scaled by
  the drop's overlap (for `pixfrac * scale <= 1` the 4-neighbour tent is the drop to good accuracy; for larger drops
  deposit to the 3 x 3 with the exact 1-D overlap lengths, separable), times the frame's robustness at that place.
  Use `np.bincount(flat_index, weights, minlength)` per band: it is the fast path (about 5-8 s per 45 MP frame).
- **Bands for memory:** process the output in horizontal bands of about 1500 rows; for each band take from every
  frame only the photosites that land in it (the transforms are near-identity, so that is a row range plus margin).
  Write each finished band into an output memmap. Peak memory: the mosaics (uint16) + one band's buffers.
  2x of 45 MP is a 180 MP result (2.2 GB as float32): fine on Bill's PC, heavy on a laptop, so the page offers
  **Whole frame at 1.5x**, **Whole frame at 2x** and **Crop at 2x** (drag a box on the preview), and says the size first.
- **Holes:** output pixels a colour never reached (weight under a floor) are filled by normalised convolution from
  their neighbours (3 x 3, then 5 x 5), per colour. With fewer than 6 frames for 1.5x or 12 for 2x, warn before
  building and suggest the lower scale; do not refuse.
- Then: camera matrix (`camera_to_working`) on the three planes -> scene-linear ProPhoto -> optional finish (5.6) ->
  `apply_orientation` -> `write_intermediate` (kind "superres", scale in the info).
- White balance: the mosaics are already multiplied by the reference frame's multipliers (`frames.mosaic`).
The kernel-regression merge (Wronski et al. 2019) and the HDR+ frequency merge (23.7) are the next step after
today; leave `MERGE_INFO` open for them.

### 5.5 Night sky (`stack/astro.py`) (build after handheld, clean and moon work)
```python
def stars(luma, max_n=400) -> (N x 3: y, x, flux)       # local maxima over 5 sigma of the background (median filter 31 px removed), sub-pixel centroid, drop elongated or saturated blobs
def match(stars_ref, stars_img) -> Transform            # start from phase_shift of the two star maps rendered as Gaussian dots, then nearest-neighbour pairs within 3 px, fit similarity (affine if 30+ pairs), iterate twice
def stack(paths, transforms, reject="sigma", k=3.0) -> (mean, rejected_map)   # streaming two-pass sigma clip; "winsorized" as the second choice (23.1)
```
The **rejection map** (what was thrown out: planes, satellites, hot pixels) is a view on the page, amber on black.
Frames are normalised to the same background level before rejection. Demosaic RCD (23.7). Out of scope today, listed
on the page as planned in plain words: Freeze ground, light-pollution removal, darks and flats from the camera
profile, star-sharpness weighting, drizzle for sky frames.

### 5.6 Finish
- Moon: starlet sharpening of the luminance (`denoise.starlet`, gains 1.6 / 1.3 / 1.1 on the three finest scales,
  slider 0-2, noise-floored); no deconvolution today.
- Handheld 2x: a light unsharp of the finest scale (slider, default 0.3): drizzle with `pixfrac` 0.7 is slightly soft by construction.
- Everything else (tone, colour, crop, export) happens in Develop on the intermediate.

### 5.7 The page (`studio/app/superres.py`)
- Mode picker first (four big choices with their `does` line), then the merge picker with Best for / Weak at.
- Tray: frames ranked, kept and dropped marked; chips: sharpness rank, offset in pixels (cyan, measured), share rejected.
- A **coverage diagram**: the frames' sub-pixel offsets as dots in a unit square (a glance shows whether there is
  anything to gain). Flashy and honest; Bill's rule about bright pretty things applies.
- Views: Result, Best single frame (B flips), Rejected (what was left out), Offsets.
- Output size line before building: "2x: 16,382 x 10,926 px, about 2.1 GB on disk, about 4 minutes."
- **Burst exposure advice** card (23.7): "For a burst, expose for the highlights (slightly under): averaging cleans
  the shadows. This is the opposite of the single-photo Canon rule (expose a little over), because here the shadows get
  many frames' worth of light."
- Honest limits card: **"The gain depends on a sharp lens and on real movement between frames: a hand-held burst has
  it, a locked tripod does not (that still cleans noise). 2x is a realistic ceiling. Where something moved, that
  place falls back to a single frame."**
- What to shoot (empty state), per mode in one line each.

### 5.8 Tests (`tests/test_superres.py`)
Generator: a high-resolution truth (a `conftest.scene` at 4x, plus a zone plate or fine stripes near the single-frame
limit); frame `k` = truth shifted by a known random sub-pixel amount (and a small rotation), box-averaged 4x down
(the photosite's area), mosaiced (`conftest.mosaic_of`), with shot + read noise.
- registration recovers every frame's shift within **0.05 input px**;
- **resolution gain:** drizzle 2x from 16 frames against the truth at 2x beats bicubic 2x of the best single developed
  frame by **at least 2 dB** PSNR, and resolves stripes at 0.7 cycles per input pixel that the single frame aliases
  (contrast at that frequency over 0.3 in the result, under 0.1 or false-patterned in the single frame);
- **tripod case** (all shifts 0): the coverage report says no detail to gain; the result's noise is under
  `1.3 / sqrt(N)` of a single frame;
- **noise:** robust average of 16 frames has noise under **0.3x** a single frame (ideal 0.25x) on a flat patch;
- **motion:** a block that moves in 4 of 16 frames leaves a ghost under **2%** of its contrast, and that area's noise
  is no worse than a single frame;
- **holes:** drizzle 2x with 4 frames still returns finite values everywhere (filled) and warns;
- **bands:** drizzle in bands of 200 rows equals drizzle in one piece (bit-identical);
- **moon:** `keep_best = 0.25` keeps the 25% sharpest of frames blurred by different amounts; local flow removes a
  synthetic per-frame wobble (residual under 0.2 px);
- **sky:** synthetic star fields with a drift, a plane trail in 2 of 12 frames: stars line up within 0.2 px, the trail
  is under 2% in the mean and bright in the rejected map;
- memory: 12 frames of 2000 x 3000 through drizzle 2x in bands stays under 2 GB.
Acceptance on Bill's files (master plan section 13): a hand-held 20-frame burst of a detailed static target and a
moon burst; at 2x the result shows detail the best single frame does not, at 100% side by side, with no ghosting.

---

## 6. Order of work for today
Each block ends with the suite green, a line in `docs/NEXT_UPDATE.md`, and something Bill can click. Do not start a
block until the one before it passes its tests. If the day runs out, stop at the end of a block and say exactly what
is and is not built: a working tab beats three half-working ones.

| # | Block | Done when |
|---|---|---|
| 0 | Message to Bill (0.4). Dual Pixel decode check as soon as his files arrive (3.1). | Findings in `docs/RESEARCH.md`. |
| 1 | `align.py` + tests. `frames.py`. | `tests/test_align.py` green. |
| 2 | `intermediate.py`, the `scene_linear` flag, thumbnail curve, Library chip + tests. | An intermediate opens in Develop looking like the developed scene. |
| 3 | `stackui.py`; the three pages as shells with empty states, frames in from the Library and by drop; page tests. | The three tabs show real pages (no build yet); window-size tests green. |
| 4 | **Focus Stack** engine (analyse, depth / average / pyramid / auto, halo guard, build) + tests; page wired (Preview, Build, Open in Develop). | `tests/test_focus_stack.py` green; a synthetic stack builds from the page. |
| 5 | **Super-Res Stack**: analysis, robust average, drizzle in bands, coverage report + tests; page wired. | `tests/test_superres.py` (handheld, clean) green. |
| 6 | Super-Res: moon mode, then night sky (5.5). | Their tests green. |
| 7 | **Dual Pixel Lab**: decode, shading, views and split export, depth + confidence + noise-vs-detail, highlight recovery, 3D outputs + tests; page wired. | `tests/test_dualpixel.py` green; real files checked if Bill sent them. |
| 8 | Help cards, README rows, "Phase N" wording out (2.3), `docs/PROGRESS.md`, `docs/RESULTS.md`. Full suite. | Everything green; chat message to Bill: what to double-click and what to look at. |
| 9 | If time remains: Focus Stack retouch-from-frame brush; Dual Pixel "DP" badge in the Library. | Each with its test. |

Focus Stack goes first among the three because Bill stacks macro himself and can judge it at once; the Dual Pixel
tab goes last because it leans on the same alignment code and on files that may not have arrived.

## 7. What "done today" means
- The three tabs are real pages: frames go in, a preview appears, a full-size result is written as
  `..._Studio.tif`, and **Open in Develop** shows it with every Develop tool working on it.
- Focus Stack: four methods with their texts, alignment with breathing, halo guard, depth-map view.
- Super-Res Stack: four modes; robust average and drizzle (1.5x, 2x, crop); coverage report; rejected view.
- Dual Pixel Lab: split views and export, highlight recovery with a measured gain, depth and confidence maps,
  noise-vs-detail map, focus report, wigglegram / anaglyph / side-by-side / depth PNG; a clear message on a normal raw.
- Every algorithm has a synthetic-truth test with the numbers above; the whole suite is green.
- Each tab states its limits in the words given here.
- `docs/NEXT_UPDATE.md` has a line per part built; `docs/PROGRESS.md` lists what waits for Bill's files and what was
  left for later (3.8, 5.5's planned items, the kernel and HDR+ merges, retouch brush if not reached).
- Nothing was pushed.
