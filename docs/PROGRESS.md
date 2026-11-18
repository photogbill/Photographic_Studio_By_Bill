# Build progress

Updated 2026-11-18 (session 14: drop-everything camera calibration; session 13: Infrared and Panorama tabs, `BUILD_PLAN_PANO_IR.md`, flare ghosts and Fill; then Card to Stacks on Bill's card). Plan: BUILD_PLAN.md v2.0.

## Session 14: drop-everything camera calibration (2026-11-18, chat session with Bill)

Bill, about to shoot his calibration frames: "I won't likely follow the wizard, but most people won't either. So it
should be able to handle a huge drop of raw photos at different ISO and temp. The key thing is the lens cap on."
So the step-by-step wizard of plan 7.2 was not built; this is what replaced it.

* **One frame at a time** (`engine/calib/frames.py`): each raw is opened, measured and let go (45 MP: about 2.5 s
  and under 1 GB on the 2-core build machine; up to 4 at once on a real PC). Kept per frame: a few numbers, a
  256-pixel centre crop (noise is told from fixed pattern by comparing two frames' crops) and its hot-pixel
  positions, in `calibration_local\frames` (about 100 KB a frame, private). A frame is known by its content, so a
  copy in another folder counts once and nothing is read twice. Drops add up across days.
* **Builder** (`engine/calib/builder.py`: `build_profiles`, `calibrate_drop`): any number of cameras in one pile,
  one profile each. Per ISO: black, read noise (pairs of short darks at the same temperature), banding, hot pixels
  present in 70% of frames. **Per camera temperature** (4 degree steps): read noise, banding, black
  (`IsoEntry.by_temp`). **Long darks**: level, noise and hot pixels by seconds and temperature, and
  `IsoEntry.thermal` = extra noise variance per second and the degrees that double it (fitted when the frames
  span 6 degrees or more). Flat pairs give gain from the crops; a pair whose light changed is left out. The
  profile's quality holds the coverage table and plain-word advice on what is missing. `build_profile(raws)` (the
  old call) now runs the same code.
* **Used by Clean:** `CameraProfile.noise_model(iso, range, temp_c, exposure_s)`; the pipeline passes the photo's
  camera temperature and shutter speed, so read noise comes from the nearest measured temperature (within 6
  degrees, pairs only) and a long exposure gets its measured extra noise.
* **Cameras page** (`app/cameras.py`, out of main.py): a drop zone (the whole page takes drops), Add a folder or a
  card, Add files, Stop; progress with the frame's kind, ISO, temperature and minutes left; what was left out and
  why; a card per camera with the ISO x temperature grid (cyan two or more, amber one), read noise by ISO with a
  line per temperature, the heat sentence, what would make it better, and Start this camera over.
  `Calibrate Camera.bat` / `studio-cli calibrate` do the same job in a console.
* **Not built:** the settings card for the user's exact camera and the printable checklist (7.2); a light-leak
  frame is noted, not left out; Part G (ETTR / ETTL) frames are left out as ordinary photos; the hot-pixel maps
  (`hot_pixels.json`, `hot_pixels_long.npz` by ISO, seconds and temperature) are saved but Clean does not use
  them yet; sharing a profile; a camera whose files record no temperature gets one profile for all temperatures.
* **Never run on real dark frames.** Tested on synthetic frames with known noise (tests/test_calibration_drop.py)
  and timed on a real R5 raw. Bill's first pile is the real test.

## Session 13b: Card to Stacks on Bill's own card (2026-11-17 evening, chat session with Bill)

Block F of `BUILD_PLAN_PLATE_GLOW_BURST_MACRO.md` (F.8 has the status), tested on Bill's real card copied to
`test_raws\DCIM\100CANON` (965 files: 511 CR3 from three days and 454 camera JPEGs beside their raws).

* **Card to Stacks** (`stack/sorter.py`, `app/cardstacks.py`): its own sidebar entry under Focus Stack. Point it at
  a card or folder (or Library right-click: Find focus stacks, or the import summary's button). It reads each raw's
  embedded preview, lines neighbours up, splits runs by pauses, lens and scene, tells focus stacks from bursts, and
  judges every frame of a stack: soft everywhere, adds nothing (naming the sharper frame at that depth), shaken,
  camera moved too far, exposure jump; after the engine lines a stack up, **subject moved** (the frame's own sharp
  part disagrees with its depth neighbours on both sides). Strictness gentle / normal / strict with best-for and
  weak-at text. Stack all runs one stack after another in the background; results are
  `FIRST_to_LAST_Focus_Studio.tif` beside the frames. Tick a frame to put it back. Nothing is deleted.
* **AI Servo bursts:** a run shot entirely in AI Servo AF is a burst (Canon CameraSettings FocusMode, now in
  `Metadata.focus_mode` and `FrameInfo.focus_mode`). Bill's dog at 400 mm (left on the card on purpose as a test)
  came out as nine "focus stacks" before this.
* **Show what's in focus** (Bill's request): in Card to Stacks, a stack shows cyan where some frame got it sharp
  and amber where there is detail but no frame got it sharp; a frame shows its own in-focus band. In Review, **F**
  lays cyan focus peaking over the photo (single photos, judged against the photo itself).
* **Import: Sort focus stacks** (Bill chose: copy everything, left-out frames into a separate folder): the frames a
  stack would leave out are copied and checked like the rest into `<day>\Left out` (second copy too); Open in
  Library opens the day itself.
* **Real card (heads and previews pulled on Bill's PC, sorted here):** 511 raws in about 2 minutes on the 2-core
  build machine: 11 focus stacks (144 frames used, 64 left out: 54 soft everywhere, 10 adds nothing), 16 bursts
  (all AI Servo), 2 single shots; every camera JPEG skipped because its raw is beside it. The raspberry stack
  (2459-2481) checked by eye: every red frame is past the berry or a softer repeat. Full-size stacks: see
  `docs/RESULTS.md`.
* **Not built yet:** the motion-safe merge, the gap report inside the result, Canon's focus-bracketing flag,
  handing bursts to Best of burst (block C), focus peaking in Develop.
* **Bill's question (R5 heat):** the R5 writes its temperature into every raw (28 to 36 °C in today's stacks).
  8K video is a good heater, not a calibration source (different readout and compression). The calibration
  builder already records dark current against shutter speed and temperature; using it in the denoiser (hot-pixel
  map and noise by camera temperature) is not built. Bill may shoot cold and hot dark frames for it.

## Session 13: Infrared tab, Panorama tab, flare ghosts, Fill (2026-11-17, chat session with Bill)

Built from `BUILD_PLAN_PANO_IR.md`, blocks 1 to 8, then three things Bill asked for during the session: automatic
lens-flare removal, a content-aware fill ("hopefully better" than Photoshop's: he chose LaMa, results over speed),
and panoramas whose exposure is so even nobody can tell they are stitched. Tests on the Linux test machine: see the
last line of this section. **Not yet run on Bill's PC.**

* **Infrared tab** (`color/infrared.py`, `lens/hotspot.py`, `stack/fusion.py`, `app/infrared.py`): filter guess with a
  colour meter (850 frames: "no usable colour", amber guess with its numbers); white balance by Auto foliage, Click
  foliage, Click neutral or a filter measured once (kept on this PC); looks (swap, blue sky 590, Goldie, colour-infrared
  film, B&W from the best channel); B&W from the channels (per-channel exposure, contrast, curve, separation, one
  -300%..+300% mix, keep brightness, live noise meter, B&W recipes per camera and filter); 850 nm mono raw path (no
  demosaic, per-CFA gains, hot pixels fixed); Detail by scale shared with Develop; channel mixer and profile; hot spot
  (measure or learn, off until on); colour by brightness (labelled added colour); visible + infrared fusion (detail,
  CIR, NDVI). Everything lives in the photo's edit: Open in Develop, Apply to the shoot. Develop gained **Channel
  masks** and an IR white-balance chip; the Library gained Edit in Infrared tab and Learn IR hot spot.
* **Panorama tab** (`stack/features.py`, `bundle.py`, `project.py`, `seams.py`, `pyramid.py`, `pano.py`,
  `app/panorama.py`): Harris + MOPS features, RANSAC, Brown-Lowe pair check; rotation bundle adjustment with one focal
  (estimated when the lens records none) and lens distortion k1; level; five projections plus Auto; per-channel
  exposure and vignetting solved from the overlaps; seams by dynamic programming with a clipping cost, a **flare
  guard** (keeps the frame without flare) and a centre pull; multiband blend in tiles that join exactly; largest
  rectangle crop; views Result / Layout / Seams / Ghosts; build at 100 / 50 / 25%, written as
  `FIRST_to_LAST_Pano_Studio.tif`, opens in Develop.
* **Flare ghosts** (`local/flare.py`, Develop > Lens): Find flare ghosts (the line through the centre), a ghost list to
  untick, Show (amber circles), Subtract the glow with an amount, and Rebuild ghosts with Fill.
* **Fill** (`local/fill.py`, Retouch > Fill): multi-scale PatchMatch with voting from the photo itself, guided by LaMa
  when it is installed (`Convert Models.bat` converts `lama_fp32.onnx` from huggingface.co/Carve/LaMa-ONNX; the
  NumPy runner gained DFT and dilated convolutions), grain matched; method picker Auto / LaMa / PatchMatch with their
  texts. Each spot fills the same way every time.
* **LaMa checked on the real model** (a byte-identical copy of Carve's file): the runner needed Einsum, and 8-bit
  weights painted cyan blobs in the hole, so LaMa is kept in 16 bits and Convert Models judges the hole on its own.
  Converted and placed in Bill's `models\inpaint` (lama.npz + NOTICE), so Fill uses LaMa on his PC without a download.
* **Panorama seams checked for lines** (Bill asked): seams now keep away from frame edges (a thin bright line on the
  5185 set came from a seam on a frame's edge) and all overlaps are levelled together (`pano.seam_levels`), which took
  out the step where a flare-veiled frame met a clean one. Iraq and 2994 had no lines before and have none now.
* **Numbers:** `docs/RESULTS.md` (session 13). Iraq set: 138 mm estimated for 135 mm, RMS 2.16 px.

**Bill's filters (answered 2026-11-17):** most XSi infrared frames used one FOTGA 530-750 variable filter (two
polarisers and a deep red filter; rotate to change), plus an 800 nm filter; next build plan block A handles it.
**Next:** `BUILD_PLAN_PLATE_GLOW_BURST_MACRO.md`: block F first (Bill: dump a whole card, find the focus stacks, cull
the frames that hurt, stack them all; he hand-holds and focuses by hand), then Best of burst, Macro planner, Plate
Reader, Glow guard (the Laack is in storage).
**Waits for Bill:** his eye on the Iraq stitch next to his own; Fill tried on his own flare photos (LaMa is in place);
the tabs on his PC.

**Not built from the plan:** preset tiles with live thumbnails (a list with texts instead); the per-channel curve
editor on the page (the engine has curves); GPano metadata for 360 panoramas; leaving a frame out by clicking its
outline (the tray ticks do it); adding a flare ghost by clicking.

Tests: **629 passed, 11 skipped** on the Linux test machine (21 minutes, 2 cores), with LaMa installed (its test runs)
and the overlap-levelling test.

## Session 12: Dual Pixel Lab, Focus Stack, Super-Res Stack (2026-11-16 evening, chat session with Bill)

Built from `BUILD_PLAN_TABS.md`, blocks 0 to 8 (block 9's retouch brush and DP badge not reached). The three tabs are
real pages: frames go in (Library right-click, Tools menu, or drop), a preview appears, a full-size result is written
as `FIRST_to_LAST_<tag>_Studio.tif` (Dual Pixel: `NAME_DP-<kind>_Studio.<ext>`), and **Open in Develop** opens it with
every Develop tool. Tests on the Linux test machine: see the last line of this section. **Not yet run on Bill's PC.**

* **Shared foundation.** `engine/stack/align.py` (phase correlation with Fourier refinement, Lucas-Kanade per tile on a
  B-spline, IRLS fits for shift / similarity / affine / homography, coarse to fine, local flow for hand-held frames,
  one warp), `stack/frames.py` (probe, checks, scene-linear frames, mosaics, scratch memmaps beside the app),
  `engine/io/intermediate.py` (32-bit float tiled TIFF, scene-linear ProPhoto with a linear ICC profile, an sRGB preview
  page for thumbnails, the edit history in its description). `scanfile.py` reads it back as `scene_linear`. Library chip
  "stack". `lens/auto.py` now uses the one `phase_shift`.
* **Focus Stack** (`stack/focus.py`, `app/focusstack.py`): Auto / Depth / Pyramid / Average with their texts; magnification
  order and breathing; Line up locally (on by default) for hand-held stacks; noise-aware sharpness; depth map and
  confidence views; halo guard; streaming full-size build. Engine has retouch-from-frame strokes; the brush on the page
  is not built.
* **Super-Res Stack** (`stack/superres.py`, `stack/astro.py`, `app/superres.py`): modes Hand-held, Tripod, Moon, Night sky;
  merges Drizzle (1.5x, 2x, or a crop at 2x), Average, Sigma clip, Winsorized; sharpness ranking with a shared noise
  yardstick; robustness maps (what moved is left out); lateral colour fixed inside the drizzle; coverage chart; rejected
  view. Night sky: stars found and matched, planes and satellites rejected.
* **Dual Pixel Lab** (`engine/dualpixel/*`, `app/dualpixel.py`): R5 Dual Pixel CR3 opened directly (track 5 read from a
  patched temporary copy; bit-identical to DPRSplit's second DNG), CR2 (5D Mark IV) through LibRaw's second frame,
  or a DPRSplit DNG pair. Shading measured per CFA position; half-views; disparity with confidence; detail map
  (the two halves' agreement); focus report; highlight recovery from the half (measured gain, about 1 stop);
  wigglegram, anaglyph, side by side, 16-bit depth PNG.
* **Also fixed:** colouring a part of a black and white photo did nothing without the DDColor model (`colourize.active`);
  `register` moved a featureless frame by the noise peak (now identity); **Composite, click-to-select (SAM), Depth haze
  and JPEG repair imported OpenCV, which the installers do not install** (THIRD_PARTY.md: not used), so they failed on
  every installed copy: replaced by `engine/imgops.py` (same results as OpenCV to 1e-5), and `tests/test_imgops.py` now
  fails on any import `requirements.txt` does not cover; `test_composite`'s grain test asked for more
  grain than the scene had over the subject (the grain match itself is right: subject 0.057 vs scene 0.056).
* **Test raws** (`test_raws\`, git-ignored, README inside): copied from Bill's drives so the external can go away.
  Dual Pixel eagle (+ DPRSplit pair), hand-held fly focus stack (9 x R5), hand-held spider burst (12 x 5D4), black
  light, IR/visible fusion, infrared 850 with 20 candlelit IR portraits, an R5 candlelit desk portrait at three
  exposures (7 frames, flames clipped), an underexposed R5 kitten, a 5D4 pair.

**Waits for Bill:** a Dual Pixel file with clipped highlights (the eagle has none, so recovery shows nothing on it);
a 5D Mark IV Dual Pixel CR2 (decoder written, never seen a real one); a tripod focus bracket of a still subject (the fly
moved, so a leaf/leg ghost remains even with local alignment); a tripod or hand-held burst of a distant detailed scene
for super-res; a moon burst; a night-sky series. His eye on all three tabs.

**Left for later (plan 3.8, 5.5, 7):** micro-refocus, bokeh shift, ghost/flare control, defect detection from the halves,
depth masks in Develop from Dual Pixel, feeding the detail map to the denoiser, synthetic aperture, defocus deblurring,
reflection removal, the long-range re-align; Freeze ground, light-pollution removal, darks and flats, star-sharpness
weighting, drizzle for sky frames; the kernel and HDR+ merges; the Focus Stack retouch brush on the page; a "DP" badge in
the Library. (Plan 3.8's "our own CR3 second-track extractor" was built after all: it was the only way to read frame 2.)

**Next:** Bill's eye on the three tabs; then the Infrared / Full Spectrum and Panorama tabs, in a new session, from
`BUILD_PLAN_PANO_IR.md` (Bill: build them in a separate discussion).

## Polish pass before the wider test round (2026-11-16, chat session with Bill)

Not a build phase: a bug hunt and four decisions from Bill. Tests: `tests/test_polish.py` (each test is a bug a tester
would have met). 510 tests green on the Linux test machine; **not yet run on Bill's PC, and the Windows installer and
push scripts could not be run at all there: Bill runs them once before pushing.**

* **Memory:** every background job (and the photo it held) stayed in memory until the Studio closed
  (`app/common.py: Job`); now a job is forgotten once its result is delivered. One photo in memory at a time.
* **Export names (Bill's rule):** every export is `NAME_Studio.ext` (`engine/io/export.py: export_name`), and a file
  the Studio did not write is never replaced (`free_target`, with the `taken` set for batches).
* **Error reports write themselves:** `error_reports/error_YYYY-MM-DD_HH-MM-SS.txt` (`log.py: save_error_report`,
  `common.report_problem`; a hard crash is reported at the next start, `log.watch_for_crashes`). Help has the folder button.
* **Installer:** the person picks the folder (the picker opens first); D: is only suggested when it is an internal
  drive with room; free space is checked before downloading.
* **Push comments write themselves:** `docs/NEXT_UPDATE.md` -> `push.bat` comment -> `CHANGELOG.md` and the GitHub
  release text (`tools/whats_new.py`). Rule 0.3.13: add a line there after any part is built. `CLAUDE.md` says the same.
* Also: one bad photo no longer ends an export; raws that cannot be opened say why in the viewer (X-Trans, sRAW,
  Foveon, damaged, a format LibRaw lacks); a sidecar is written only when something changed; measurements asked for
  together are made once (per-image locks in `pipeline.Image`); empty Library message; Develop panel width;
  Panasonic RW2 metadata; settings file swapped in whole; README brought up to date.
* Checked on other makers' raws (rawpy's public samples): Nikon D3S and D4, Canon 5D II, an old Kodak develop
  correctly. No Sony or X-Trans file was at hand.
* **Next: `BUILD_PLAN_TABS.md`**: build the Dual Pixel Lab, Focus Stack and Super-Res Stack tabs (Bill: "just build
  those tabs"). Done in session 12, above.

## Where things stand

| Phase | State | Notes |
|---|---|---|
| 0. Foundation | **Done, needs Bill's double-click** | Layout (2.2), `Install.bat`, `Run Tests.bat`, logging, `THIRD_PARTY.md`, true-ISO fix in the analyzer. Stack taken as recommended (Python + NumPy/SciPy + rawpy; PySide6 in Phase 4) since Bill was away. Acceptance: `Run Tests.bat` green **on Bill's PC** (green on the Linux test machine: 342 passed, 3 skipped). |
| 1. Decode + develop engine | **Done, awaiting Bill's eye** | Stages 1, 2, 10, 11, 15 (basic), 21. `studio-cli` / `Develop Raws.bat` turns CR2/CR3 into 16-bit Adobe RGB TIFF. Export now also writes lens, exposure and the true ISO (EXIF in JPEG, XMP in TIFF/PNG). |
| 2. Calibration engine | **Drop-everything tool built (session 14), awaiting Bill's frames** | Cameras page drop zone, `Calibrate Camera.bat` / `studio-cli calibrate`: any pile, any cameras, by ISO and camera temperature. Part G (exposure latitude) not yet. Acceptance needs Bill's Part A and B frames. |
| 3. Denoiser v1 | **Built, awaiting Bill's blind test** | Hybrid NLM + starlet default; MP-PCA and starlet candidates; blind-test tool (10 pairs ready). BM3D-class baseline still to write. Off by default until the blind test. |
| 4. Studio shell | **Built** | `Start Studio.bat`: Library, Develop (live render, wipe, loupe, histogram, panels, undo, sidecars, Make it good), Export, Cameras, Lenses, Help. |
| 4b. Library + card import (session 11) | **Built, awaiting Bill's card** | All four steps of 24.9. **Import from card** (Library, primary button): the Photos folder dialog (you choose it: drive buttons with free space, OneDrive warning, nothing created until you confirm), one-screen import dialog (cards = any drive with DCIM, refreshed every 2 s; raws only by default; what stays on the card always said; one folder per shooting day named by its date, a new day at 04:00, grouped by year; your own name per folder or one folder for all; space check; second copy), copied in the background with progress, Cancel and a finish summary; a quiet **Card found** notice. Engine `engine/io/ingest.py`: copy, check (BLAKE2b, read back), then name; never overwrites; import record `<Photos>\_studio\imports.jsonl`; `studio-cli import`. **Library speed**: disk thumbnail cache (1 GB cap, Clear cache), only what is on screen, ratings/details in the background with `cache\library_index.sqlite`. **Review** (Space): large view + filmstrip, keyboard rating with auto-advance, Z / click 100% (zoom kept from photo to photo), C compare with locked zoom, Camera preview (amber) or From the raw (cyan). **Process picks**: picks / selected / shown, own edit or Make it good, background with Cancel. Plus sort, filters (rating, camera, lens, ISO, date, not rated, name), size slider, subfolders, right-click menu, drop a folder, recent and pinned folders, Tools menu, ratings that cannot be saved are shown and can be kept on this PC. Acceptance: Bill goes card to exported picks without Explorer; a tester on another brand does it with no help. |
| 5. Lens | **Built, awaiting Bill's eye** | Lateral CA and purple-fringe removal (earlier), plus this session: **Fix lens** button with a colour-coded report; distortion from lensfun (violet) or measured from straight lines (amber, strict gate: never on organic subjects); lens falloff measured (lensfun or estimated) with correction OFF by default; stage 13 geometry in one resample (distortion, vertical/horizontal perspective, straighten with **Auto level**, crop with aspect presets drawn on the photo, constrain-to-image); **lens tagging** (Library: Tag lens for a whole shoot; Develop: lens picker; shared starter list + your own private list; written into exports). Acceptance: fringing on Bill's vintage-lens files fixed by one button with no colour damage. |
| 6. Full develop | **Built, awaiting Bill's eye** | Session 4: **Microcontrast** (stage 16) and **capture sharpening** (stage 12). Session 5: the full **Light** panel (highlights, shadows, whites, blacks in stops; **Smart lighting**, region by region with a guided filter; **Auto** that sets highlights / whites / blacks only and keeps low-key photos low-key, the named test; a point **tone curve** per channel), the **Colour** stage (OKLCh: saturation, vibrance with skin protection, **Pick white** eyedropper, **Auto** white balance (grey-edge), a **DxO-style colour wheel** with several ranges, **HSL** in 8 bands, **colour grading** wheels with balance and blending, **black & white** with a colour mixer, **.cube LUTs**), **Looks** (8 starter looks incl. a draft **"Bill's Candlelight"**, Save look), **Copy / Paste settings** (Develop, and Library for many photos), **virtual copies**, a labelled **History** with **snapshots** (kept in the sidecar), **export presets**. Session 6: **highlight rebuild** (clipped channels get the colour of the clipped area's edge), **Dehaze** (dark-channel prior), **Texture**, **film grain** (size, roughness, mono or colour), **film toe** and **highlight roll-off** with a drawn film curve, a **parametric curve** (four regions), **Auto colour** (vibrance from measured colourfulness), **B&W filter presets**, a **camera-profile picker** reading DNG camera profiles (.dcp), and **output sharpening** for screen / glossy / matte in Export. Acceptance: Bill's microcontrast blind test against DxO; "Bill's Candlelight" tuned by Bill. |
| 6b. Plan section 23 (session 10) | **Part built, awaiting Bill's eye** | **Detail by scale** (23.11) replaces the Microcontrast and Texture sliders: six methods, each with its own switch, amount and "best for / weak at" text (Tonal, Local contrast, Crisp, Tonal separation, Veil removal, Broad), stackable under one shared budget per scale; Simple level = Amount + Balance (starts on the Tonal side) + presets (Tonal, Crisp, Both, Soft glow); **Judge in black and white**; a live meter (amber when a guard limits the stack) and the balance number; diffraction note from f/8. **Denoise safeguard** (23.12, on by default, with a count). **Auto HDR** on one raw (23.6): Light / Medium / Heavy / Highlights only / Low-key / Shadows only, Amount 0-150%, three engines (exposure fusion, local Laplacian, Multi-Scale Retinex), noise-aware lift, off until turned on. **Auto white balance** (23.9): Grayness Index beside grey edge, both shown with their agreement, two light sources reported, **Keep the mood**. Method texts on the denoise picker (23.1). **Not built yet from section 23:** 23.2 neural runtime and 23.3 learned denoisers, 23.4 highlight methods (by area, detail transfer, grain match), 23.5 Dual Pixel HDR, 23.7 stacking, 23.8 PDAF line filter (needs a Sony raw that shows striping), 23.9 mixed light and skin estimate. **Session 11 built 23.8's demosaic picker** (see the 23.8 row). Acceptance: Bill's blind test of Detail by scale, in colour and in black and white. |
| 6c. Demosaic picker (session 11, plan 23.8) | **Built, awaiting Bill's eye** | Develop > Detail (Full): **RCD** (new default), Menon, **LMMSE**, **Dual** (RCD + LMMSE by local contrast), MHC, each with its best-for / weak-at text; **Even out the two greens** (green equilibration, off by default). RCD measured ahead of Menon on every crop, clean and noisy, and twice as fast; older edits keep Menon (their sidecars say so). Not built: the PDAF line filter. |
| 9d. Eyes (session 11) | **Built, awaiting real portraits** | Find faces now also finds each **iris** (MediaPipe Iris model, which Bill downloaded, run by tinynet; the edge then sets the size; without the model a classical fit). Portrait: **Eye colour** (Blue, Green, Hazel, Grey, Amber, Brown, Violet or Pick a colour; amount; both eyes or one), and in Full **iris brightness, iris clarity, iris colour strength, outer ring, catchlights** (all 0 by default, not part of Younger). **Fix red eye**: red pupils become Red eye spots, pupils the flash blew out white become **Pupil** spots (amber, removable). Retouch > **Pupil**: click a blown pupil or an animal's glowing eye. Checked on NASA's public-domain astronaut portrait (irises found on real eyes, model and fit agree) and drawn eyes (colour hits its hue to within 4 degrees, skin, pupil and catchlight untouched). Not built: an iris placed by hand. |
| 7 | Not started | Wizards. |
| 8. Dual Pixel Lab (session 12) | **Built, awaiting Bill's files and eye** | R5 Dual Pixel CR3 read directly; views, depth, detail map, focus report, highlight recovery, 3D outputs. Needs a DP file with clipped highlights and a 5D4 DP CR2. |
| 9. Local + retouch | **Built (masks session 6, retouch session 7), awaiting Bill's eye** | **Masks** in Develop: Brush (feather, flow, erase, **auto mask**), Gradient, Radial, **Object** (one click selects a leaf; Alt-click takes a part away), **Colour**, **Brightness** and now **Skin** (skin tones of any complexion); shapes combine (add, subtract, only-where), invert, amount; inside each mask: exposure, contrast, highlights, shadows, whites, blacks, temperature, tint, saturation, vibrance, texture, clarity, dehaze, sharpness, noise reduction and now **Smooth skin** (softens blotches and uneven colour, keeps pores and edges). **Retouch** (its own group): **Heal**, **Clone** and **Red eye** spots, clicked or brushed along a hair, source chosen automatically and draggable; **Find dust** (one photo); **Learn dust map** in the Library (the dust on your sensor, learnt from several photos, kept on this PC) and **Dust map** in Develop to heal it in any photo (also `--dust-map` for batch develop). New **100% view** (Z, mouse wheel or the 100% button; drag to move) where every tool works. Still to do: depth and focus range masks (need Dual Pixel, Phase 8), denoise strength and microcontrast per mask, editing a drawn gradient or circle by its handles (content-aware fill for large removals: built in session 13, Retouch > Fill). Acceptance: Bill selects one leaf in two clicks and adjusts it; dust removed automatically. |
| 9c. Person mask | **Built (session 9), awaiting a real portrait** | Masks > **Person**: everyone in the photo selected on this PC (MediaPipe person segmentation + BlazePose 33 body points), then a **Part**: whole person, background, skin, face, hair, clothes, arms, legs, torso; snapped to the photo's edges. Also in "Refine" (+ / - / x Person). |
| 9b. Portrait | **Built (session 9), awaiting a real portrait** | New **Portrait** group in Develop. **Find faces** (Google MediaPipe face detector and 468-point face mesh, run on this PC by the Studio's own NumPy code, no new install; a brightened copy is tried for candle-dark photos) or **Click the eyes** when a face is missed. **Remove blemishes** (one button: red spots and dark spots become Heal spots in Retouch, amber; moles and beauty marks are left alone). Sliders, all 0 by default: **Younger** (all of them at once), **Lines**, **Under eyes**, **Smile lines**, **Even skin**, **Jawline**, **Eyes**, **Teeth**. **Body sculpting**: Push / Shrink / Grow brushes and shape tools (**Breast lift & size**, **Butt improvement tool**, **Slim waist / tummy / arm / thigh**) with Lift, Size, Slim and width sliders; every warp capped. Acceptance: the person in a real portrait looks like themselves on a good day, judged by Bill and the sitter. |
| Install + updates | **Built (session 9), awaiting Bill's first run on Windows** | **Install Photographic Studio.bat**: one file for people without git or Python (downloads the Studio and a private Python into one folder; run again to update; never touches a git copy). **Release.bat** + GitHub Actions: GitHub builds the **MSI** (Briefcase + WiX) and publishes the release with both installers. **Help > Check for updates** (off until chosen; an installed copy asks once in the sidebar): automatic updates go in when the Studio next starts (.bat copies) or as it closes (MSI). Not yet run on Windows by me: Bill's first install and first release are the test. |
| 10. Focus Stack (session 12) | **Built, awaiting Bill's eye** | Four merges, breathing, local alignment for hand-held, halo guard, depth and confidence views. Retouch brush not on the page yet. |
| 11. Super-Res Stack (session 12) | **Built, awaiting Bill's eye** | Hand-held, tripod, moon, night sky; drizzle 1.5x / 2x / crop; robust merges; coverage chart; rejected view. |
| 12. Infrared / Full Spectrum (session 13) | **Built, awaiting Bill's eye** | Filter guess and colour meter, white balance beyond the camera, looks, B&W from the channels with a noise meter, 850 nm mono raw path, hot spot, colour by brightness, fusion, Channel masks in Develop. |
| 13. Panorama (session 13) | **Built, awaiting Bill's eye** | Layout with focal and distortion from the overlaps, five projections, exposure and vignetting evened out, seams with a flare guard, multiband blend. Iraq set: 138 mm for 135 mm, 2.16 px RMS. |
| 9e. Flare ghosts and Fill (session 13) | **Built, awaiting a real test** | Develop > Lens: find, subtract or rebuild flare ghosts. Retouch > Fill: PatchMatch from the photo, planned by LaMa once converted. |

## Waiting for you in the outputs folder (already generated)

* `outputs\blindtest\dxo_ready\index.html`: the Phase 3 blind test, 10 pairs (Studio vs your DxO exports, 100%).
* `outputs\develop\compare_20261114_080709\index.html`: all 17 sample raws, camera JPEG vs Studio, wipe slider.
* `outputs\develop\denoise_preview\index.html`: four high-ISO files denoised, with 100% crops beside Canon's own.
* `outputs\detail_preview\sample_raws\index.html`: microcontrast and Recover lens sharpness on three of the sample
  raws (fit-to-screen wipe and 100% crops). Run `Detail Preview.bat` for your Voigtlander folder: it puts your DxO
  export beside each crop.

## What Bill can double-click today

* `Install.bat`: builds the private environment inside the folder. Now also adds the optional lens database
  (lensfun) once; everything works without it.
* `Run Tests.bat`: every self-test, green/red summary.
* `Start Studio.bat`: the Studio. New this session in Develop: **Looks** at the top of the panel; the full **Light**
  panel (Auto, Highlights, Shadows, Whites, Blacks, Smart lighting) and, in Full, the **Tone curve**; **Colour** (Pick
  white, Auto, Saturation, Vibrance, Black & white with its mixer, LUT) and, in Full, the **Colour wheel**, **HSL** and
  **Colour grading** (these four start folded: click a header to open it); **History** with snapshots at the bottom;
  in the top bar the **version** menu (virtual copies), **Copy settings** and **Paste**. Library: **Paste settings**
  onto every selected photo. Export: presets (Archive, Print, Web, Social, and your own). Session 6 adds Dehaze and
  clipped-highlight rebuild (Light), parametric regions, Toe and Highlight roll-off with a drawn film curve (Tone
  curve), Auto colour, B&W filters and the camera-profile picker (Colour), Texture and film grain (Detail),
  output sharpening in Export, and the **Masks** group (Brush, Gradient, Radial, Object, Colour, Brightness).
  Session 7 adds the **Retouch** group (Heal, Clone, Red eye, Find dust, Dust map; Show spots), the **Skin** mask
  with **Smooth skin**, the **100%** view (Z or the mouse wheel; drag to look around), and in the Library **Learn dust
  map** (select 5-20 photos of different scenes from one camera; sky or a plain wall at f/11 or smaller is ideal).
  Session 8 adds **Light > Clipped highlights: measure from lens glow** (reads how bright and what colour a blown
  light was from the glow the lens spread around it) and, in the Library, **Measure lens glow** (once per lens: a
  bracket of one small light in a dark room; the Help button in the profile list explains how).
  Session 9 adds the **Portrait** group (below Retouch): Find faces or Click the eyes, Remove blemishes, Younger and
  the face sliders, and Body sculpting (brushes and shape tools). The face models are in `models\faces` (no install).
* `install-linux.sh` (new, for Linux PCs): `bash install-linux.sh` installs or updates the Studio in one folder;
  `start-studio.sh` (or the menu entry it offers) starts it.
* `Detail Preview.bat` (new): drag raws or a folder on it, or double-click for the Voigtlander folder.
* `Develop Raws.bat`, `Denoise Preview.bat`, `Run Benchmark.bat`, `Analyze Raws.bat`, `Calibrate Camera.bat`,
  `Blind Test vs DxO.bat`: as before. `studio develop` also takes `--microcontrast 0.6 --sharpen --fix-lens --rotate`,
  `--look "Warm film"`, `--smart-lighting 40`, `--highlights rebuild --dehaze 30 --texture 20 --grain 15
  --profile NAME.dcp --sharpen-for screen`, and now `--dust-map` (heal the camera's learnt dust in every photo).

**Session 11 adds** (Phase 4b, all in the Library of `Start Studio.bat`):
* **Import from card...** (top right): the first time it asks where your photos should be kept (pick a drive or a
  folder). Then: the card, what to copy (raws only by default), the folder names (the date, or type after it), and
  Import. Try it with a real card in the reader; a second import of the same card copies nothing.
* **Review** (or Space on a photo): Left / Right, 0-5 / P / X / U, Z or a click for 100%, C to compare two, Enter to
  develop, Esc or Space back. "Show" picks Camera preview or From the raw.
* **Process picks...**: export your picks in one go (or the selected photos, or all shown).
* The filter row (rating, camera, lens, ISO, date, Not rated yet, file name), the sort, the Size slider, Include
  subfolders (above the folder tree), Recent and pinned folders (right-click to pin), the **Photos** button, the
  right-click menu on photos, and **Tools** (Tag lens, Paste settings, Learn dust map, Measure lens glow, Photos
  folder, Keep this folder's ratings on this PC, Clear thumbnail cache).
* `Library Speed Test.bat` (new): drag a big folder from your archive drive onto it to measure how fast it opens.
* `studio-cli import F:\ --to E:\Photos [--name "Kelly candlelight"] [--jpegs] [--second F:\Backup] [--dry-run]`.

**Session 10 adds** (all in `Start Studio.bat`, Develop):
* **Detail** group: Preset, **Judge in black and white**, **Amount**, **Balance (tonal - crisp)**, the meter, the
  balance number; in Full, the six methods with their own switches and amounts, and "Where it works" (shadows /
  midtones / highlights). Try Amount 70 on a low-ISO file, then tick Judge in black and white.
* **Clean** group: **Protect stars, catchlights and sparks** (on) with its count; a text under the method picker.
* **Light** group: **Auto HDR** (off until ticked), its strength, amount and (Full) engine, and a report line.
* **Colour** group: **Auto uses** (grey edge / grayness index), **Auto: keep the mood**, and a report line after Auto.
* `Develop Raws.bat` / `studio develop` also take `--detail "tonal=70,crisp=30"` (or a preset name: `--detail Both`)
  and `--hdr medium` (`--hdr lowkey:laplacian --hdr-amount 80`).

## Decisions taken while Bill was away (all reversible)

1. Stack: Python + NumPy/SciPy + rawpy, as the plan recommends (open question 1). No Rust.
2. Default demosaic: Menon 2007 (measured best of the three implemented). RCD next.
3. Studio Standard look: contrast 1.35, mid grey 0.11, tuned against the camera JPEGs of the R5, 5D IV and 6D.
4. Gamut compression added (not in the plan) after the UV-lit violet produced colours outside the spectral locus.
5. ICC profiles are generated by our code and named "... compatible".
6. Denoise off by default; the isolated-outlier filter is off by default (it ate real texture: -2 dB).
   Default luminance strength 0.6 (1.0 wiped leaf texture on the R5 at ISO 4000 even though PSNR liked it).
7. `raw_analyzer.py` stays at the top of the folder; it imports the Studio metadata reader for the true ISO.
8. **Recover lens sharpness is off by default** (a look decision, and many of Bill's lenses are character lenses).
   Make it good turns it on at 40% unless the photo's lens is marked as a character lens.
9. **Microcontrast: 2-stop range by default.** 1 stop was too weak on real photos (+6%); see RESULTS.md.
10. **lensfun is optional**, installed once by `tools\ensure_env.bat`. Its numbers are frozen into the sidecar when
    Fix lens uses them, so an edit renders the same on a PC without lensfun or with a newer database.
11. Geometry numbers (straighten, perspective, crop) are stored as they look on screen (upright frame); the crop is
    fractions of the frame left after "constrain to image". No upsampling: straightening makes the photo smaller.
12. Order deviations from plan section 3: purple-fringe removal runs before geometry (detected better on pixels not
    yet resampled), and capture sharpening runs right after it (still before geometry and tone), so a 100% loupe
    can sharpen only the pixels it shows.
13. Lens list: a shared starter list in `profiles/lenses/manual_lenses.json` (CC BY-SA; Laack, Heliar, Apo-Lanthar,
    Zeiss 1898 Anastigmat marked as character lenses, plus common vintage glass); your own lenses go to
    `settings/my_lenses.json` (private, never committed).
14. The "before" in the before/after wipe now uses the same straighten / perspective / crop as the edit, so it lines up.
15. **Colour stage order (deviation from plan section 3):** HSL, colour wheel, saturation / vibrance, B&W, grading and
    LUTs run *after* the tone curve, on display-referred values in OKLCh. Hue and chroma changes then act on the colours
    you see and never alter lightness; doing them before the curve made the curve re-shift what the wheel had set.
16. **Highlights / shadows work in stops before the base curve** (not as local masks): global, so no halos, and
    provably monotone at +-100. The local part is the separate **Smart lighting** control (guided filter, radius 1.5%
    of the long edge), off by default.
17. **Auto (Light) never touches exposure or shadows.** It only sets highlights (pull back clipping), whites (only
    when the frame is not low-key) and blacks. Bill's rule: no automatic brightness changes unless he turns them on.
18. **Looks never carry brightness.** Exposure, mid grey and "match camera brightness" are stripped from every look,
    shared or saved, so applying a look cannot brighten or darken a photo. The lens tag is never taken from a look.
19. **Auto white balance neutralises candle warmth.** It is a grey-edge estimate of the light's colour; for candle-lit
    portraits As shot usually looks truer. Its tooltip says so. Picked / Auto white balance is stored as exact
    multipliers, shown as the nearest temperature and tint; moving either slider carries on from there as Custom.
20. **"Bill's Candlelight (draft)"** is my starting point, not Bill's look: highlights -30, blacks -10, vibrance +10,
    orange and red slightly desaturated, a faint warm tint in the highlights, microcontrast 50% (shadows-first). It
    never changes exposure. Bill tunes it and saves it under his own name (Save look).
21. History: moving the same slider again within two seconds updates one step ("Exposure +0.5") instead of adding
    fifty; the last 40 steps are kept in the sidecar per version. Snapshots and virtual copies live in the same
    sidecar, so nothing is ever written outside the photo's folder or the app folder.
22. The Full-only groups (Tone curve, Colour wheel, HSL, Colour grading) start folded; each group remembers whether
    you left it open (settings\app.json).
23. **Clipped highlights stay "plain clip" by default**; "rebuild colour" is one choice away in the Light group (Full).
    Rebuild only shows when highlights or exposure are pulled down (the default white point is below sensor clip).
24. **Camera profiles: the profile's own tone curve is not used.** A .dcp sets colour only; the Studio's tone curve
    and Light panel keep control of brightness. Profiles you add are copied into settings\camera_profiles (private:
    commercial profiles must never be shared) and offered only for photos from the camera they were made for.
25. Grain is generated from the pixel's position, so a 100% view matches the export exactly; fit-to-screen previews
    show the grain a downsized export would have. Judge grain and Texture at 100%, like sharpening.
26. Output sharpening is off unless chosen in Export (the Web, Social and Print presets turn it on for their medium).
27. **Masks are attached to the photo, not the screen**: shapes are stored in the camera's own frame (before rotation
    and crop), so straightening or re-cropping later leaves them on their subject. They are recomputed at render time,
    so they are exact at any size and a 100% view matches the export.
28. **Click an object is not AI**: it learns the colours around your click and of everything else (GrabCut-style
    colour models), scores every pixel, smooths the score along the photo's edges, and keeps the connected part you
    clicked. Brightness counts less than colour, so shading across a leaf does not split it. It works best on the
    object's typical colour (clicking a blown highlight finds other blown areas). Alt-click marks what is not part
    of it. Its result is stored as a small bitmap in the sidecar and snapped to the full-size edges when rendering.
29. Local adjustments order: exposure / white balance / texture / clarity / dehaze in scene light (before the tone
    curve), tone controls by blending the global curve with a per-mask curve, colour after the global colour stage,
    sharpness and noise last (before grain). Several masks overlapping apply in list order.
30. "Paste settings" offers Masks as a group (off by default, since masks are drawn for one photo).
31. **Retouch spots are attached to the photo** like masks (camera frame, before rotation and crop). A spot's source is
    always read from the photo *before* retouching, so the order of spots does not matter and a 100% view matches the
    export exactly. Heal works on log values (light is matched by multiplication, as in the Photoshop Healing Brush);
    the membrane is second order with a feathered edge rather than Georgiev's fourth order (simpler; the feather hides
    the seam). Clone copies exactly, for hard edges and patterns.
32. Order: retouch is applied to the scene-linear photo right after capture sharpening and before exposure (plan
    section 3 lists it at stage 18), so every later slider (exposure, curves, colour, masks) acts on the repaired photo.
33. **Find dust is conservative** (at least 5% darker, six times above the local texture, round, not in deep shadow) and
    draws what it found in amber for review: one photo cannot tell dust from a bird in the sky. The dependable route is
    the **dust map**: at least 3 different scenes; a blob is kept only if it sits in the same place on the sensor in at
    least 3 photos (and 35% of them); the same scene twice counts once. The map stays in calibration_local (never
    shared, plan rule 2) and, when used, its spots are written into the edit, so re-learning the map later never
    changes an edit you already made.
34. The dust map heals only the spots this photo shows: each is looked for near its place with a size window from the
    aperture (the shadow spreads roughly as f-number(map) / f-number(photo)); for lenses that record no aperture (most
    of Bill's vintage glass) the window is wider.
35. **Smooth skin is a mask adjustment**, not a global slider (the research: blind smoothing destroys skin texture; good
    retouching is region-aware). Pores and fine hair are never touched (only the middle layer is softened); inside any
    mask it eases off on colours far from skin (eyes, brows, lips). Off unless Bill sets an amount on a mask.
36. **100% view**: Z, the mouse wheel or the 100% button; drag to move around (right- or middle-drag while a tool is
    on). Masks, retouch spots and the eyedroppers all work in it. The before/after wipe is fit-to-screen only.
37. "Paste settings" offers **Retouch** as a group (off by default) so dust spots can be synced across a shoot; looks
    never carry spots.
38. **Measured highlights (DESAT for cameras, session 8).** A third clipped-highlight mode reads a blown light's true
    brightness and colour from the glow the lens spread around it, as NASA's DESAT does for the Sun's flares with the
    telescope's diffraction. With a measured lens glow profile (cyan) every channel is measured; without one, only a
    light blown in some channels but not all can be (amber), from its unclipped channel and the colour of its glow;
    anything else keeps the colour rebuild. Blown areas larger than 0.3% of the frame, or not standing well above
    their surroundings, are surfaces, not lights, and are left to the rebuild. The extra light is laid in with the
    shape of an unclipped channel where there is one, otherwise as a dome: the glow tells how much light and what
    colour, never the fine detail inside. Off unless Bill picks it; nothing changes at all where nothing clipped.
39. Glow profiles are per lens (and aperture), kept in settings\glow on this PC, picked automatically by the lens
    (the hand-picked lens tag first) and nearest aperture. "Copy settings > Light" now also carries the clipped-
    highlight mode and profile; looks never carry a profile.

40. **Portrait faces are found on this PC** with Google's MediaPipe models (Apache 2.0), shipped unmodified in
    models\faces and run by our own NumPy code (tinynet.py): no TensorFlow, OpenCV or MediaPipe install, nothing
    downloaded, photos never leave the PC. OpenCV would have run them too but adds ~40 MB for one feature.
41. Faces are stored in the photo's sidecar as 468 points in the camera frame, like masks and spots, so straightening
    and cropping later keep them on the face. Faces found automatically draw amber; faces from your eye clicks cyan.
42. **Every portrait slider starts at 0 and is subtle at 100** (lines filled at most fully, folds lifted only part
    way, a jaw warp of about 2% of the face). New features default off, as always.
43. **Lines fill only elongated grooves** (oriented morphology), so freckles, moles and pores stay: they are part of
    the person. Remove blemishes also skips probable moles and beauty marks.
44. Blemishes become ordinary Heal spots (amber, auto = "blemish"), so they can be checked and removed one by one in
    Retouch, and Paste settings > Retouch carries them like dust spots.
45. Face work is computed once per face and setting as a smooth gain map attached to the photo (on a copy of the
    face at most 640 px wide), then applied to every render: the 100% view and the export match exactly, and the
    full-resolution pores are untouched.
46. **Body sculpting is a warp applied after retouching, in stroke order**, each step inverted exactly (no smearing);
    spots placed on a sculpted photo land on what was clicked. Caps: push 30% of the brush, lift 14% of the shape,
    size about 10%, slim 12% of the width.
47. Copy settings / looks carry the portrait *sliders* only; faces and sculpting stay with their own photo.
48. **Person mask parts are worked out, not labelled by a model**: the person mask and 33 body points come from
    MediaPipe's person segmentation and BlazePose (Apache 2.0, from the official mediapipe 0.10.9 wheel); skin,
    face, hair, clothes, arms, legs and torso come from the body points, the face map (if Find faces was used) and
    the person's own skin colour. Google's multiclass model would label hair / skin / clothes directly but its
    server is blocked here; Bill downloaded it (models\people\selfie_multiclass_256x256.tflite), and when it is
    there hair, skin, face and clothes come from it directly (the worked-out parts stay as the fallback).

49. **Work in progress is shown where you are looking** (a friend's first test read slow renders as "the filters
    don't work"): a moving bar on the photo, a "Working... 2.4 s" pill, and an amber chip beside Controls; nothing for
    the first 0.15 s, so quick renders don't flicker. **One preview render at a time**: changes made while one runs
    wait, and only the newest goes next (they used to run side by side and slow each other down).
50. **Double chin** (a friend's request): a slider and a one-press button (sets 60) in Portrait, off by default and
    **not part of Younger** (it reshapes, so it is only ever asked for on its own). A warp draws the area under the
    jaw up towards it (nothing at or above the jaw line moves; at most 4.5% of the face width; never folds), weighted
    by where the skin is, so a collar, scarf or beard right under the chin stays put; plus the crease lifted part way
    and the jaw's own shadow deepened a little only where there already is one.
51. **Installer for people without git**: Bill chose the one-file .bat (downloads the Studio and python.org's
    embeddable Python into the chosen folder; D:\Photographic Studio by default when there is a D: drive) **and** an
    MSI built by GitHub ("the easy way": Release.bat sends a version tag, GitHub Actions builds and publishes).
52. **Updates are opt-in but offered**: an installed copy asks once (Update automatically / Just tell me / No thanks);
    a git copy is never asked and never changed. The only thing sent is a request to GitHub's public API for the
    latest release. Automatic updates are staged in `_update` and put in place by `Start Studio.bat` before the
    Studio starts (so no file is ever replaced while in use); MSI copies install the new MSI as the Studio closes.
53. **MSI copies keep the user's files in Documents\Photographic Studio** (settings, logs, calibrations, outputs):
    Windows Installer owns the program folder and may clear it on upgrade or uninstall. Every other copy keeps
    everything beside the app (rule 0.3.3). `STUDIO_DATA_DIR` overrides both.
54. The hair segmentation model (Apache 2.0, downloaded by Bill) drives the Person mask's Hair part on a 512-pixel
    crop of each head; without it the multiclass model's hair (or the worked-out hair) is used.
55. **Composition overlays for every method that has one** (Bill: "a set of overlays for every composition method in
    art"): rule of thirds, golden ratio (phi grid), golden spiral, golden triangles, diagonals (baroque and
    sinister), the 45-degree diagonal method, dynamic symmetry, harmonic armature, rabatment, centre, radial and a
    grid. O cycles (off is one of them; outside the crop tool the first O shows the overlay on the whole photo),
    Shift+O turns the spiral (4 ways) and the triangles (2); a combo and a Turn button do the same in the Crop group.
    Geometry is worked out in pixels where the frame's shape matters (a 45-degree line is 45 degrees).
56. **Suggested crops are rules, not a learnt model**: no crop model with a licence that fits was found (the public
    ones are trained on research-only datasets). The eyes (or the person, or what stands out and is sharp) go on the
    chosen method's strong points; heads stay whole with room above; space in front of a turned face; no edge on a
    joint or the neck; nobody half in the frame; bigger is better when all else is equal. Three different crops,
    each with a sentence saying why; the best one opens in the crop tool with its overlay.

57. **Every push is an update** (Bill: "otherwise I won't write them"): after a successful push, push.bat runs
    Release.bat /auto, which raises the version's last number by one (0.1.0 to 0.1.1), tags it and sends the tag,
    so GitHub builds the installers and copies with updates on pick it up. Pressing N within 10 seconds skips it for
    that push; Release.bat by hand is for bigger jumps (0.1.x to 0.2.0).

58. **Detail by scale keeps old edits exact.** "Local contrast" IS the first microcontrast engine: an edit that
    holds only that amount renders through the old code, bit for bit. The earlier Texture slider shows only on
    edits that used it. The new methods live in a new `detail` block of the sidecar, in percent, with an `off` list
    (a switched-off method keeps its amount).
59. **Tonal's remap is the smooth one, and noise protection sits on the coefficients.** Three stronger-looking
    remaps were tried and dropped because they rippled clean gradients (RESULTS.md). Tonal at 100% lifts small
    tonal differences about x2.5 (the monotone limit is x3.2); the first engine managed x1.3.
60. **Shared budget:** 0.8 x Local contrast + 0.446 x Tonal <= 1 per pyramid level; over that, both are scaled back
    together and the meter turns amber. Fine total is held between -0.5 and +2.0.
61. **A lightness that sees blue** (weights 0.25 / 0.60 / 0.15) drives every new method, Auto HDR included: ProPhoto
    luminance counts a violet flower as black. The first engine keeps true luminance (old edits unchanged).
62. **Veil, tonal separation, broad and Auto HDR are measured on a 1024 px copy of the whole frame** and sampled at
    each pixel's place on the sensor, so the 100% view and the export agree (measured: 0.0005) and a region render
    needs no extra context. They read the frame before Dehaze and Smart lighting; if those are on, the regions are
    measured slightly differently from what they then act on (harmless, stated here).
63. **Veil removal never treats a dark subject as haze**: the floor is the darkest 0.5% of the frame, capped at a
    quarter of mid grey; mid grey is held, so only the shadows deepen. Negative = add a haze (Soft glow preset).
64. **Tonal separation is a push away from the centre of each crowd of tones**, not histogram equalisation: flat
    areas keep their brightness and smooth gradients are untouched (the first version made tile-sized halos).
65. **Crisp leaves edges and their skirts alone** and also anything riding on a bigger structure two scales up (a
    flame's slope is not texture). It is for texture; edges belong to Recover lens sharpness. Make it good sharpens
    less when Crisp is already up.
66. **Order in stage 15-16:** exposure, Veil, Dehaze, Smart lighting, Auto HDR, then Tonal separation + Broad +
    Tonal + Local contrast + Crisp in one pass, then (old edits only) Texture. Same sliders, same picture.
67. **Denoise safeguard: 4.5 sigma, on for new edits, off for sidecars written before it existed** (denoise stage
    version 2), so those render as they did. It is applied per photosite in the variance-stabilised domain.
68. **Auto HDR never runs by itself, and looks never carry it** (it changes brightness: Bill's rule). Every engine's
    change is re-expressed as one value per level of lighting before it is applied; that is what keeps exposure
    fusion and Retinex from drifting into halos. Engines run at the strength's 100%; Amount scales the result.
69. **What "needs" means in Auto HDR:** shadows are lifted toward -3.5 EV (2nd percentile), highlights pulled under
    +2 EV (99.5th percentile); Light gives 35% / 80% of that, Medium 60% / 90%, Heavy all of it; a photo that
    already fits gets a small minimum (0.5 / 0.3 stops) so the button is never a no-op. Low-key lifts only tones
    between about -6.5 and -1.5 EV.
70. **The lift is limited by noise, not followed by more denoising**: the denoiser runs on the raw data long before
    this stage. The report says how much lift was held back and suggests turning Clean on.
71. **Auto white balance default: 60% of the way for a warm light (under 3500 K) at ISO 1600 and up, 90%
    otherwise.** Both estimates are always computed; under 3 degrees apart is "confident", over 7 is "check by
    eye". An estimate with a tint beyond +-60 is pulled back to the daylight / tungsten band. Moving "keep the
    mood" after Auto runs Auto again at that amount.

## Questions for Bill (one at a time, in chat)

1. (Answered 2026-11-14: no automatic brightness changes unless turned on.)
2. Phase 0 stack: Python + PySide6 as the plan recommends - confirm.
3. Phase 2: have the calibration frames (checklist Parts A-C) been shot?
4. Phase 6: are the raws from the Heliar / Sigma 135 / Canon 180L same-leaf test still around? (To measure what
   the Heliar does to tonal gradients and make microcontrast reproduce it.)
5. Phase 6: should Recover lens sharpness be on by default for new edits, or stay a Make it good / manual choice?
6. Phase 6: "Bill's Candlelight (draft)": does the starting point look like your candle portraits? (Tune it, then
   Save look under your own name.)
7. Portrait: one or two raws of a face (ideally a candle-lit one, and one in ordinary light) to tune the face finder,
   the blemish finder and the Lines / Under eyes strengths on a real face.

8. Detail by scale: is Amount 70 on the Tonal side the look you mean by microcontrast (judge it in black and
   white too)? If it is still too timid or already too much, the one number to change is `K_TONAL` in `scales.py`.
9. Auto HDR: which strength should "Make it good" never touch? (Today it touches none: Auto HDR is only ever on
   when you tick it.)

## Open items for the next session

* **Card to Stacks leftovers** (block F, F.8): motion-safe merge, gap report in the result, Canon focus-bracketing
  flag, bursts to Best of burst, focus peaking in Develop; an image-only test for action bursts from cameras that
  do not record the focus mode.
* **Hot sensor:** use the dark-current-by-temperature table in the denoiser once Bill has shot cold and hot dark
  frames (the R5 records its temperature in every raw).


* **Section 23, still to build** (in the plan's order): the demosaic picker (RCD, LMMSE, dual) and PDAF line filter
  (23.8: the Sony tester needs the PDAF filter); highlight recovery by area, detail transfer and grain match with a
  method picker (23.4); BM3D-class baseline, NL-Bayes, Zero-Shot Noise2Noise and the optional ONNX runtime with the
  learned candidates (23.2, 23.3); mixed-light white balance and the skin estimate (23.9); Dual Pixel HDR (23.5,
  with Phase 8); stacking (23.7, with Phases 10-11).
* **Detail by scale leftovers:** measure MP-PCA's residual noise (the starlet table stands in); per-mask Detail by
  scale; the focus-plane-only option (needs Dual Pixel depth); the Focus Stack hand-off; the blue-channel check
  after the lens fix; presets measured from real lenses (lens signatures); a "Recover diffraction loss" that knows
  the f-number's blur exactly instead of measuring it. (Fixed this session: a gradient running off the frame was
  treated as a crease and came out up to half a stop darker or brighter along the frame edge at Tonal 100%; the
  picture is now continued beyond the frame by point reflection. The first engine alone, for old edits, keeps its
  old border behaviour so those render as they did.)
* **Speed:** Tonal adds 1.1-1.5 s to a preview on the 2-core build machine (its reference tones run in parallel:
  Bill's 14 cores should make it a fraction of that); the denoiser still dominates any preview it is in (40 s here).
  A 45 MP export with Tonal + Crisp + Auto HDR peaked at 4.1 GB: check an 8 GB laptop.
* **Auto HDR:** Dual Pixel, bracket and burst sources; noise-matched denoise of lifted shadows; tune the six
  strengths on Bill's candle portraits (Low-key) and a backlit scene.

* Person mask: try on Bill's photos (candle-lit edges, long hair, hands); the MagicTouch click-to-select model for
  Object masks (the runner now reads its 8-bit weights; its click encoding still to be matched against MediaPipe);
  snap sculpt shapes to the body points (one click for waist / thigh / arm ovals).
* Composition: try the suggested crops on Bill's own photos (candle portraits, botanicals) and tune the weights;
  horizon placement for landscapes (thirds / golden lines on the horizon found by Auto level).
* Speed (the friend's test): Dehaze, Texture and Microcontrast are the slow sliders on big raws (a 45 MP preview goes
  from 0.7 s to 4.7 s with all three on, on the 2-core build machine).
* Installers: Bill's first run of Install Photographic Studio.bat and first Release.bat (I cannot run Windows .bat
  files or build an MSI here); then an app icon for the MSI and the desktop shortcut.
* Portrait: tune on Bill's real portraits (blemish thresholds, line and fold strengths, the candle-light face finder);
  editing a sculpt shape by handles (today: move by dragging, width and sliders; redraw to resize); the face outline
  shown after sculpting follows the unsculpted face; face tools on very small faces in group photos (the detector
  sees the whole frame at 192 px: tiling would find more); a "mirror" for paired shapes (both sides at once).

* Measured highlights next steps: Bill's first lens glow profile (the Laack, candle bracket) and a check on real candle
  portraits; then, if wanted, the star-filter route (Rouf et al. 2011: a cross-screen filter spreads the highlight into
  rays that can be read back tomographically, about 9 more stops with fine detail, at the cost of rays to remove).
* Phase 9 remainder: depth and focus range masks (with Dual Pixel), per-mask denoise and microcontrast, editing a
  drawn gradient or circle by its handles (today: delete and redraw), faster masks on 45 MP (two masks add about 16 s
  to an export on the test VM; a whole-frame Smooth skin about 40 s), content-aware fill (PatchMatch-style) for large
  removals, a noise-model-aware threshold for Find dust at extreme ISO (three false spots at ISO 32000), and an option
  to apply the dust map automatically on opening a photo.
* Phase 6 leftovers: a microcontrast blind test against DxO; the Heliar measurement once the Part E files are in;
  Bill's own Candlelight look; the Dual Pixel highlight mode (Phase 8); creative sharpening with masking (comes with
  masks in Phase 9).
* Speed: the colour stage costs about 7 s on a 45 MP export on the 2-core VM (OKLab both ways); Smart lighting 5.7 s.
  Both are strip-parallel, so Bill's 14 cores should be several times faster; port to compiled code if not.
* Capture sharpening: field-varying PSF (corners of vintage lenses are softer), the edge-preserving progressive
  RL candidate (section 22 change 8), and a deconvolution comparison beyond RL (section 21 change 6).
* Phase 5: auto vertical perspective ("upright") from converging verticals; use lensfun TCA as a cross-check for
  the measured lateral CA; a vintage-lens fringing test frame (branches against sky, checklist Part D).
* Phase 1: per-body baseline from calibration; RCD demosaic.
* Phase 3: BM3D-class baseline (our own implementation), auto strength from ISO, hot-pixel map.
* Speed: tone + output stages remain the slow part on a 45 MP frame; port hot kernels if Bill's PC is still slow.
* Nothing has been committed to git: the device shell cannot delete git's lock files, so `push.bat` should do the
  commit (it shows every file first).
* Session 10 was built and tested on the Linux build machine (342 tests pass, 3 skipped) and copied into the
  folder; `Run Tests.bat` on Bill's PC is still the acceptance check, and `push.bat` still does the commit.

64. **Builder's defaults of 24.2, as written** (Bill may flip any of them): the folder date is the day the photos were
    taken, a new day starts at 04:00 (a setting in the import dialog), folders are grouped by year (a tick). Import is
    a button in the Library, not a ninth sidebar section.
65. **The Library has two bars**: buttons on top (Import from card, Review, Process picks, Open folder, Tools) and a
    filter row under it (rating, camera, lens, ISO, date, Not rated yet, file name, sort, size). 24.7 C put filter,
    sort and size in the top bar; with the new filters it would not fit at 1366 x 768. "Include subfolders" sits above
    the folder tree. The Library's minimum width is 1050 px (the test checks it fits beside the 240 px sidebar).
66. **Found, not fixed (outside 4b):** the whole window cannot shrink below 1521 x 862 because the Develop panel needs
    1281 px and the Help page 840 px of height; that was already so before this session. The plan's 1366 x 768 aim
    (6.1) needs those two pages reworked (a scroll area for Help, a narrower Develop panel).
67. **Card found notice never keeps an archive drive spinning**: card readers (removable drives) are looked at every
    2 s, fixed disks only when a new drive letter appears (a CFexpress reader that mounts as a fixed disk). Cards
    already in when the Studio starts are not announced. The import dialog itself checks every drive every 2 s.
68. **Capture times come from the first 256 KB of each photo** (`read_capture_time`, the same EXIF field as
    `read_metadata(...).datetime`): the full reader reads 24 MB per file, far too much for a card.
69. **Checking a copy past the Windows file cache** is used only after it proves itself on a test file at the start
    of each import (so a fault in it can never fail good copies); otherwise the read-back after fsync, and the log
    says which. Matching the record allows 2 s between a file time and an EXIF time (FAT keeps file times to 2 s).
70. **Make it good moved into the engine** (`engine/presets/auto.py`, rule 2.2): Develop's button and Process picks
    run the same code. Process picks saves the Make it good edit into each photo's sidecar (so Develop shows what it
    did); photos with an edit of their own keep it. Process uses master edits (virtual copies: Export).
71. **Ratings are written in the background, one after another**, and the grid updates at once. The engine now
    takes a lock around every sidecar read-modify-write, so a rating saved by the Library and an edit saved by
    Develop at the same moment cannot undo each other.
72. **Read-only folders**: the first rating that cannot be saved shows a message (once per folder) offering "Keep
    them on this PC". That switches the folder to central sidecars (`settings\sidecars\<id>\`, private), and every
    read and write follows (Develop, Export, Process, the Library), not just ratings; sidecars already beside the photos
    are copied in once. Tools > "Keep this folder's ratings and edits on this PC" turns it on or off by hand.
73. **Review**: "Go to the next photo after rating" is on in Review (remembered), never in the grid. Compare shows
    the two selected photos if exactly two are selected, otherwise the current and the next. The Dual Pixel and
    exposure badges in the info strip wait for those detectors (Phases 7 and 8); the focus-check overlay stays in
    its own phase, as 24.7 A says.
74. **Thumbnails**: 320 px JPEGs in `cache\thumbs` (per 24.7 B); at the largest Size (400 px) they are upscaled a
    little. Up to 900 are kept in memory; the rest come back from the disk cache in a few milliseconds.
75. **Shared code left main.py** (24.8): `app/common.py` (jobs, settings, small widgets, the lens picker),
    `app/exporting.py` (Export dialog), `app/library.py`, `app/review.py`, `app/importer.py`, `app/process.py`.
    `main.py` re-exports the old names, so nothing that used them changed.
76. **RCD is the default demosaic for new edits** (plan 23.8), measured ahead of Menon everywhere (RESULTS). Our RCD
    uses Hamilton-Adams' additive green correction instead of RCD's ratio (the ratio measured worse on detail).
    Sidecars store the algorithm, so photos edited (or rated) before keep Menon exactly.
77. **LMMSE is offered, not switched on automatically at high ISO**: with noise added, RCD still scored higher on all
    three test files. Bill's eye on real high-ISO files (maze patterns, smoothness) may change that.
78. **Green equilibration** is a per-photo tick (off by default) applied to the mosaic right before demosaicing; it
    corrects only where the two greens' local ratio is within 8 % of 1, so real detail is left alone.
79. **The iris is found with the face** (plan 9.7 follow-up): MediaPipe Iris on each eye's crop gives where the iris
    is; a short search along the iris edge then sets its size (the model read radii about 8% small on drawn eyes; with
    the edge search they are within 4%). Without the model file, the classical fit does the whole job. Faces mapped
    before this session get their irises the first time an eye tool is used.
80. **Eye colour moves the iris to the colour's typical lightness relative to the eye whites**, not to an absolute
    value, so an eye in candle shadow stays in shadow; the change is capped (x0.75 to x1.35 in OKLab lightness), half
    of the iris's own colour variation is kept (the fibres), and the pupil and catchlights are left out. The gain
    map may move channels further inside the iris only (brown to blue needs about 10x more blue); everywhere else the
    old 0.3x-3x safety limit stands.
81. **Red eye, version 2**: a brown iris is red-dominant in linear light (red 3-4x the other two), so version 1 of the
    Red eye spot could grey the iris round a pupil. New Red eye spots (clicked or from Fix red eye) act only on
    strong red; spots in older edits keep version 1 and render as before. Fix red eye only calls a pupil red when it
    is strongly red and brighter than the iris around it.
82. **No "sharpen eyes" slider**: the face work is a smooth gain map (decision 45), which cannot carry fine
    sharpening. For crisper eyes, draw a Brush mask over them and raise its Sharpness.
83. **Face Landmarker v2 replaces the first face mesh when its file is present** (models/faces/
    face_landmarker_v2.tflite, taken unchanged from Google's face_landmarker.task). Measured on the astronaut and seven
    damaged copies (RESULTS): it moves about half as much under candle-dark colour, a 3-stop half shadow, noise and
    tilts of 20-35 degrees, which is the light this Studio is for; it is less steady than the first mesh on a blurred
    face (0.021 vs 0.013 of the eye spacing) and a face at 0.4x size (0.016 vs 0.012). It is 2.5x slower (0.17 s vs
    0.07 s per face), small next to the rest of Find faces. Its 10 iris points are the first guess for the eye tools,
    finished by the same edge search as the iris model's ("mesh" in the face's iris record). Without the file, or
    with STUDIO_FACE_MESH_V1 set, everything runs as before. Faces already saved keep their points; only new finds
    use the new mesh, and each face records which mesh made it ("mesh": 1 or 2).
84. **Linux: `install-linux.sh`, `start-studio.sh`, `tools/ensure_env.sh`** (for a friend of Bill's). The same design
    as the Windows installer: one file, asks for a folder (`~/Photographic-Studio`), downloads the newest release
    (or `main` if the release predates the Linux files), copies it in without deleting anything, writes
    `.install_kind` = "sh". Python: the PC's own 3.10+ in `.venv`; if it cannot make one (Debian/Ubuntu/Mint without
    python3-venv) or the packages will not install, a private python-build-standalone CPython 3.12 in `python/`, its
    file found in the release's SHA256SUMS and checked against it (no GitHub API, so no rate limit); if that fails
    too, it offers the system packages (asked, sudo). The window's system libraries (Qt's xcb/Wayland plugins,
    typically `libxcb-cursor0`) are found with ldd and mapped to apt/dnf/zypper/pacman packages; installing them
    is asked first, with one `apt-get update` retry. The launcher applies a staged in-app update before starting
    (Help > Check for updates now works on Linux: `restart_into_update` starts start-studio.sh), logs the last start
    to `logs/launcher.log`, and when started from the menu reports failures by desktop notification. Each script
    is read whole before it runs (`main "$@"; exit $?`) and updates write new files (`cp --remove-destination`), so a
    script can replace itself safely. Run end to end on Ubuntu 24.04 (glibc 2.39): fresh install, update over it
    (user files kept), git copy (Python only), the private-Python path against a stand-in for GitHub (good checksum
    used, wrong checksum refused), the missing `libxcb-cursor0` found and installed, the window opened on a real X
    server (Xvfb) from start-studio.sh, and an in-app update applied through start-studio.sh. Not tested: a real
    desktop session (Wayland, HiDPI) and the live GitHub downloads, which this sandbox cannot reach.
    Bazzite and the other Fedora Atomic systems (read-only system image; Bazzite's dnf refuses host changes): the
    Studio lives in the home folder, so it installs the same way. The installer recognises them (/run/ostree-booted),
    first tries the window (a Wayland session needs none of the X11 libraries; Qt falls back from Wayland to X11 by
    itself), and only if it cannot open offers `sudo rpm-ostree install <packages>` (default no, reboot needed).
    Tested with a stand-in rpm-ostree: window opens, nothing offered; X11 library missing, the layer is offered
    and not run without a yes.
85. **Zoom and pan rebuilt** (a friend: "the denoising was efficient, but zooming and dragging is very clunky").
    Measured cause: two states only (fit or a rendered 100% window); the wheel jumped straight to 100%, and every
    drag ended with the photo snapping back for the next window's render (2.2 s with Clean on, R5). Now: any zoom
    from fit to 400% (wheel around the pointer, stopping once at exactly 100%; Ctrl+= / Ctrl+- through Fit, 25,
    33, 50, 67, 100, 200, 300, 400%; Ctrl+0 / Ctrl+1; the toolbar button and its menu; touchpad pinch). Clicking
    the photo toggles Fit and 100% at that point (as in Lightroom); a drag moves it; middle or right drag, or
    Space + drag, move it with a tool in hand; Esc with nothing in hand goes back to fit; a navigator in the corner
    shows where you are and jumps where you click. The photo follows at once: the whole-photo preview already in
    memory is drawn enlarged (5-8 ms a frame), and a sharp render of just the part on screen plus 8% (app/zoom.py
    `tile_request`, engine `render_window`) is laid over it when ready; small nudges reuse it. Up to 60% the tile
    is cut from the half-size whole frame the preview already made (0.1-0.3 s instead of 3-7 s). The engine now
    keeps stage 1-12 results for the last whole frame and the last region separately (one slot made the tile and
    the preview evict each other: a tone slider while zoomed re-did the denoise twice). The 100% loupe inset is
    retired (the real zoom does its job). Develop's top bar is compacter and the file name shortens with "...",
    so the window fits a 1366-pixel screen (it needed 1,521).
86. **A friend's first Linux test (0.1.3), four fixes.** (1) *Opening a photo dropped the window out of full screen.*
    Cause: the window's minimum size grew past her screen (Develop needed 1,521 pixels across and more with a long
    file name; the Help page alone made the whole window at least 1,042 high), and a Linux window manager puts such a
    window back to normal size. Now 1,235 x 486 on every page and unchanged by opening a photo: Develop's top bar
    folds Copy, Paste, Before / After and the virtual copies into a ⋯ menu when narrow, the file name shortens with
    "...", the text pages scroll, the Library hint wraps and its filters are a little narrower. The window also
    opens as it was closed (place, size, maximized or full screen), and F11 toggles full screen. (2) *"The help
    section is a hot mess" and its "buttons don't work".* The cards were squeezed to slivers of clipped text (no
    scrolling), so they looked like broken buttons; now the page scrolls and every card shows its whole text in
    equal columns; the stale "100% view" card is now "Zoom and move". (3) *"The Studio stopped with an error" on
    closing.* Reproduced: closed during a Clean render, the process waited 37 s, then the render's worker pool refused
    work after Python had started shutting down and the error handler tried to show a message box with no
    application left: exit code 134, which the launcher reported. Now previews are dropped at once, export and
    import jobs are finished (the Studio asks before closing over them), no new work starts, and the process leaves
    without tearing Qt down once everything is saved: exit 0 within about 1.5 s. One Studio at a time (a lock
    beside the settings; "Open another window anyway" is offered), since a slow start invited a second click. (4)
    *Zooming and dragging took up to ten seconds*: 0.1.3 predates decision 85.
87. **Old photos, scans and film (Bill: "removing dust and scratches is a must", colour negatives "definitely",
    no hand colourization, no face restorers).** (1) *Scans open.* TIFF (8/16-bit, float), JPEG and PNG, grey or
    RGB, in the Library beside raws (a camera's JPEG beside its raw stays hidden) and in Develop. The embedded ICC
    profile is read field by field (matrix/TRC and grey; anything else, or none, is taken as sRGB and the
    info line says "assumed"), giving display-linear ProPhoto; then the default Studio Standard curve is inverted,
    so a default edit shows the scan exactly as the file (16-bit TIFF: mean 0.000%, 99.9th percentile 0.005%) and
    every slider works on it as on a raw. Clean stays raw-only (it is said in its panel). (2) *Negatives.* Black
    and white or colour, from a scan or a camera copy: density above the film base, each layer normalised by the
    frame's own density range (the orange mask and the layers' different contrasts go together), written as
    scene-linear light for the Studio's tone curve (the "paper"). Base and range are measured once on the whole
    frame; Pick the film base on clear film overrides; Negative contrast scales the range. White balance acts on
    the positive. (3) *Dust and scratches.* Top-hats both ways on log brightness, a threshold set by the local
    texture (a speck on a sky or cheek is found faint, the same contrast in foliage is left), hysteresis for whole
    shapes, specks or long thin marks only; the filled light is a harmonic membrane solved on the marked pixels
    only, with grain like the grain round it. Microsoft's scratch detector (Bringing Old Photos Back to Life, MIT),
    run in NumPy from 8-bit weights (33 MB, converted without PyTorch), judges which long marks are damage: on a
    clean photo it marks nothing, so the filter's texture false alarms on lines go. Specks are below its 256-pixel
    view and are judged by the filter alone. Faces' eyes, brows, lips and nostrils are never touched (catchlights).
    (4) *Colourize.* DDColor (Apache 2.0) from Qualcomm's ONNX export, converted to one NumPy file with 8-bit
    weights (53 MB, ships with the Studio: nothing to download) and run by a new small ONNX-graph runner
    (local/onnxnp.py: 19 layer types, matches ONNX Runtime to 0.0004, 1.1-1.5 s at 256 px on 2 cores). Only a, b
    are taken; each pixel keeps its L*; colour is put in scene-linear light through the inverse standard curve,
    brought up to full size edge-aware (guided filter), tapered near black and white. All four live in stages 1-12, are in
    the stage-1-12 cache key, and a zoomed region renders like the whole photo. CodeFormer (non-commercial licence)
    and GFPGAN (invents facial detail) are deliberately not used.
88. **Colour a part (Bill: Photoshop's colour points "imperfect to the point of annoyance, but in theory it could
    work very well").** Click a part of the photo (a coat, a car, the sky), choose its colour (the picker starts on
    the colour shown there): it spreads over that part and keeps the photo's light and shade. Where Photoshop
    struggles, and what is different here: (1) the spread follows brightness only, by geodesic distance with a
    squared-step cost, so a fold or a shadow (gradual) costs nothing however deep and the sharp edge of a collar
    or badge stops it, and the colour blotches an automatic colourizer leaves on a garment cannot cut it up; (2)
    the colour is laid as a material: the chosen colour's chromaticity at each pixel's own brightness (reflectance
    times shading), pulled to grey only as far as the gamut needs, so folds stay folds and the brightness does not
    change (to 0.0001 L* on the test); (3) you see where it goes: a white outline and a light tint of the part
    while the tool is in hand; Shift-click adds more of the part, Alt-click stops it at a place, Spread sets how
    strong an edge it may cross, Change colour and Remove in the list. Works with or without the automatic colour
    (the hints are laid over DDColor's colours). Honest limits: brightness-only spreading leaks where two parts are
    equally bright with a soft boundary (a pale cheek against a pale wall) and stops at strong edges inside a part
    (hair strands, deep creases): one Alt-click or Shift-click fixes each, and the outline shows where.
89. **The Plate Lab, first part: stains and corrosion unmixed (Bill: "go ahead and build the plate lab"; test plate:
    Levi L. Hill's Hillotype self-portrait, NMAH PG.003999.32, which Bill has worked on for 13 years).** A stain,
    foxing spot or corrosion bloom is a coloured layer over the picture, so it is measured by its colour instead
    of painted over (local/stains.py). On a small copy of the whole photo, in log space: the background of warm
    (log R/B), green-magenta and brightness is a median over 8% of the short side; the stains' colour direction is
    the weighted median of the strongest warm steps, searched only from -20 to +50 degrees (orange to
    yellow-brown); the brightness a stain causes per unit of its colour is one robust fit over all stain pixels
    (opaque cores left out of the fit). Each block is then cleaned with those numbers (so a 100% view equals the
    export): where a pixel steps from its background along the stain direction (within 40 degrees), the stain's
    brightness and tint are taken out; the picture under a thin stain stays. Real reds that lean magenta (lips)
    point another way and are left alone (tested). Evidence for Hillotype colour: Boudreau's colorimetry of his
    1980s recreation (Pioneers of Photography, 1987, ch. 18; Macbeth MS 1500+, D65, 10 degree observer), colour
    difference from the plate's background: yellow da -0.78 db +0.28, green -5.09 / -1.25, blue -2.31 / -2.48,
    red +0.54 / -2.98. All faint (1-5 units) and none toward orange (+a +b), where corrosion lies, so the stain
    cone leaves them alone. (A screenshot of one of his plates looked strongly pink-orange; that is the
    reproduction, not the plate.) Boudreau also reports the colour does not change with viewing or lighting
    angle, unlike tarnish interference colours: a two-angle capture could tell them apart. Not yet handled:
    hand-applied pigment (Hill used it on some plates) that happens to be orange would be taken for a stain;
    a shape and texture check (blooms are compact and crusty) is the planned fix. **Honesty:** bloom cores, where the picture may be gone (on a plate the corrosion
    eats the silver the image is made of), are kept in a map, shown amber by **Show what's lost**, and never
    filled with invented detail. Panel: Remove stains and corrosion, Stain strength, Sensitivity, Show what's
    lost, a chip with the share unmixed and the share possibly lost. Synthetic validation (known answer: sepia
    pictures with orange-brown blooms and opaque cores, add_stains): stain brightness error 0.10 -> 0.01 (log
    units) and tint error 0.14 -> 0.015 on four seeds; the fit's slope 0.67-0.89 against a true 0.72. Ordering in
    the scan path: stains, then dust and scratches, then colour. Still to come in the Plate Lab: multi-capture
    fusion (several lights / polarised / IR) and polarity for plates seen in their mirror state (the "show what
    changed" view came in decision 90).
90. **Cognitive Adjustments, batch 1 (Bill: "ready to build in neural filters that run locally?"; the name is his,
    "tongue in cheek at Adobe").** A new Develop group for tools built from research and run on this PC; each
    measures what is in the photo and invents nothing. (1) **Remove printing dots** (local/halftone.py): a
    halftone screen is a lattice of sharp peaks in the spectrum. On a full-resolution 1024 px crop of the middle
    (each channel and luminance), peaks more than 15 dB above the spectrum's median at their radius, between 1/25
    and 1/2 cycle per pixel, are kept only if they lie on one lattice (two fundamentals of equal frequency at a
    right angle, within 4 degrees, and their sums and multiples), so a fence or a fabric is never taken for a
    screen. Each block then has Gaussian notches (sigma 0.012 cycle/px, so the kernel fits the 48 px context
    and a 100% view matches the export) cut at those frequencies in log space; notches finer than a preview's
    pixels are skipped (already averaged). Tested: periods 3.2-6 px at 15-45 degrees found to 0.1 px and 2
    degrees, the dot texture cut by more than 65%; no screen found on clean photos or on Hill's plate. The chip
    reports the screen ("a screen every 5.0 px at 45 degrees"). (2) **Colourize**: stage version 2 shows the
    model a faded or badly exposed photo with its L* stretched (0.5/99.5 percentiles to 6/94; Adobe's advice for
    its colourizer is to fix brightness and contrast first; here it is automatic and only for the model's eyes),
    old sidecars keep version 1 and render as before; warm/cool and green/magenta sliders shift the suggested
    colour by up to 12 b*/a* units, tapered at black and white, the user's own coloured parts untouched. (3)
    **Show what the automatic tools changed**: the photo rendered with and without printing dots, stains, dust and
    scratches and colourize, the larger log difference per pixel as a heat map (clear = untouched, yellow to red
    = more), with the share changed in the status line. Next: colour match as editable settings (and Composite on
    it), JPEG repair inside the quantisation intervals, depth (Depth Anything V2 Small, Apache 2.0; Base and Large
    are non-commercial), click-to-select (SAM 2.1 Tiny first, any larger size dropped in is used), relighting.
91. **Depth from one photo (Cognitive Adjustments; Bill: "go ahead and build everything").** Depth Anything V2
    Small (Yang et al., NeurIPS 2024; the only Apache-2.0 size, Base/Large/Giant are non-commercial), the
    onnx-community export folded to 518 x 518 and converted to 8-bit weights (24 MB), run by our NumPy runner:
    2.9-3.3 s per photo on the build machine, 0.29% from ONNX Runtime on a real photo. local/depth.py. The photo is
    shown to it as displayed (standard tone curve, sRGB), squeezed to 518 x 518 as the model expects, ImageNet
    normalisation; its relative inverse depth is stretched back and normalised by the 1/99 percentiles (near = 1,
    far = 0). Measured once per photo and base settings (memo), from a <= 1024 px copy in the S frame, so masks
    and blocks sample one map (a 100% view equals the export, tested). Uses: (1) **Depth haze** (-100..100):
    Koschmieder's I = J t + A (1 - t), t = exp(-k far^2), the haze colour A the median of the brightest third of
    the farthest 15% of the photo, scaled to its 97th percentile; adding pushes the distance back, taking away
    inverts the same model with t >= 0.35 so distant noise is not blown up. Scene-linear, after colourize, in both
    the raw and scan paths; stage version "depth": 1; not in the "what the automatic tools changed" map (it is a
    choice, not an automatic repair). (2) **Masks > Depth**: a new shape type "depth" (the map as a bitmap in the
    S frame plus From/To/Softness, 0 = farthest, 100 = nearest), snapped to the photo's edges like the others,
    with + / - / x refine entries. (3) **Show depth map**: the map rendered through the mask path (so crop,
    rotation and geometry match the screen) and coloured red/yellow near to blue far. Honest limits in the
    tooltip: an order, not metres; glass, mirrors and water fool it (the espresso test photo's spoon reads as the
    saucer). Tests: haze model adds/clears by distance and clearing inverts adding; on the NASA astronaut photo
    the helmet in front is nearer than the wall by more than 0.4, haze changes the far corner 3x more than the
    helmet, a near-only depth mask selects the helmet and not the wall; panel round trip.
92. **Click to select with SAM 2.1, the size selectable (Bill: "i'll get both, and make it user selectable"; then
    "you could just use the tiny model and i'll test them both later").** local/sam.py. SAM 2.1 (Meta FAIR 2024,
    Apache 2.0 at every size), from the X-AnyLabeling ONNX exports Bill downloaded. The image encoder is folded
    to 1024 x 1024 (onnxsim; onnxslim, pure Python, as a fallback) and stored with 16-bit weights: 8-bit moved
    the features 3-6% and one outline to 0.82 IoU of the original, 16-bit is 0.1% and >= 0.998 IoU. The prompt
    decoder is kept unfolded so any number of clicks works (the runner now does shape arithmetic and takes named
    inputs; 0 in an input shape = any size); it adds SAM's own padding point, so no extra one is passed (an extra
    one changed the scores). As Meta's SAM2ImagePredictor: the photo as displayed squeezed to 1024 x 1024,
    ImageNet normalisation, clicks in that square, the best of the three outlines by the model's own score, the
    256 x 256 logits stretched to the photo (up to 2048 px) through a sigmoid for a soft edge. The look at the
    photo is kept per photo, size and base settings: Tiny 9-10 s here, then 0.15-0.3 s per click. **Masks >
    Object** and **Colour a part** use it (click, click again to add, Alt-click to take away); a shape records
    "model": "sam2.1-tiny"; the Spread slider hides for SAM-made colour parts (it has no meaning there). **Masks >
    Click-to-select**: SAM 2.1 Tiny (fast) / SAM 2.1 Large (finest edges) / Classic (no model, instant); a size
    that is not installed is listed greyed out with "not installed (Convert Models.bat)"; the choice is
    remembered. Why not convert Large here: the device bridge refuses files over 400 MB and the Large zip is 805
    MB, so **Convert Models.bat** (tools/convert_models.py) does it on Bill's PC: it finds model.onnx and the
    sam2.1_hiera_{tiny,large} zips in models\, unpacks into cache\convert_models (removed afterwards; TEMP and
    TMP point into cache\tmp, nothing on C:\), folds, converts, then runs each result on a made-up test picture
    and, with ONNX Runtime installed (it adds onnx, onnxsim and onnxruntime to the app's own Python once), says
    how far it is from the original; logs to logs\convert_models.log; finished models are skipped unless --force.
    Large is not shipped (it is too big for GitHub). **push.bat** now lets converted models (models\NAME\*.npz,
    which .gitignore lets in) through up to 95 MB (GitHub refuses 100 MB); before this, the colour and scratch
    models (53 and 33 MB) would have stopped the next push. Tests: runner named inputs and shape checks; on the
    astronaut photo one click outlines the face only (1.5-8% of the photo), a second click adds the suit, the mask
    renders where clicked, Colour a part outlines the suit, Classic still works; Convert Models finds exactly the
    three downloads among others; the panel lists Tiny, Large, Classic with uninstalled sizes greyed.
93. **Colour match from another photo, solved as settings (Cognitive Adjustments).** local/colourmatch.py. Photoshop's
    Color Transfer and Harmonization return a changed picture; here the answer is ordinary slider values. Nine
    statistics of the photo as displayed (OKLab): median L, L spread (5th-95th percentile), the mid-tones' mean a
    and b (the cast), colourfulness (75th percentile of each colour's distance from that cast, so a cast is not
    counted as saturation; median chroma was tried first and moved too little with Saturation), and the shadows'
    and highlights' a and b relative to the mid-tones. Nine settings: exposure, contrast (log), white balance as
    red and blue multipliers (mode "multipliers", temperature/tint filled in for the sliders), saturation, and the
    shadows and highlights grading wheels as 2-vectors. Solved on ~320 px renders of the real pipeline: a
    finite-difference Jacobian (one render per setting), then damped Newton steps with Broyden updates, a small
    pull toward no change (0.12 / limit per setting; 0.6 was too strong and under-matched), each step capped at
    half a setting's limit, the best kept: 14-20 renders, 0.6-1.1 s on the astronaut photo. Known-answer tests
    (the same photo developed with known settings as the reference): exposure -0.50 found as -0.499, multipliers
    [0.9, 1, 1.15] as [0.90, 1, 1.14], saturation -30 as -31; a split tone (shadows 180/40, highlights 40/40)
    found as 179/34 and 28/32; pixel colour error cut 7-100x. Panel: Match colours to another photo... (the
    reference's own sidecar is used), Match amount 0-150% (applied as the difference to the same sliders, so
    other changes stay), a chip with the change in words; Undo works as for any change.
94. **JPEG repair inside the quantisation intervals (Cognitive Adjustments; JPEG files only).** local/jpegfix.py.
    A JPEG keeps each 8 x 8 block's DCT coefficients only to within its table's step; decoders show the middle,
    hence blocks and ringing. The smoothest picture the file allows is found: Huber total variation (eps 2
    levels) lowered by gradient steps, every block projected back inside its intervals after each step (POCS;
    a hair inside, 0.48 of a step, so the file's own rounding is kept). In the file's YCbCr: brightness with
    the brightness table at full size, Cb and Cr at their stored size (sampling read from the file) with the
    colour table, then brought back; tables and sampling from Pillow (natural order, checked); output 16-bit.
    20 steps (40 smoothed fine texture at quality 60: PSNR 33.11 at 10 steps, 32.89 at 40). On the astronaut
    photo: quality 15 PSNR 28.3 -> 29.0 dB, blockiness (edge step over inner step) 1.81 -> 1.32; quality 60
    32.7 -> 33.1; the repaired brightness plane re-quantises to the file's coefficients (99.997-100%). The IJG
    quality is estimated from the brightness table for the chip. In the pipeline the repaired pixels replace
    the decoded ones (once per photo, kept with it) before anything else; stage version "jpeg": 1.
95. **Relight (Cognitive Adjustments; Bill liked idea 7, relighting).** local/relight.py. Surfaces from the
    depth map (decision 91): smoothed with a guided filter of itself (keeps steps), the scene's overall slope
    removed (a Gaussian of 25% of the short side subtracted; at 12% a face's own curvature was lost, without it
    the photo's floor-to-ceiling depth trend tilted every normal down so a light from above darkened
    everything), scaled so the steepest 10% tilt ~45 degrees, and steps in depth (outlines) left facing the
    camera - without that, the wall beside a head glowed. The light already on the photo: brightness ~ c0 + c .
    n fitted with centred normals (nz ~ 1 almost everywhere made c0 and c_z collinear until centred), Huber
    weights, trusted by fit quality (R^2/0.3, capped at 1). Gain = new Lambertian shading (ambient from
    Softness) over the measured one, clipped to 1/3..3, median-normalised (overall brightness kept), raised to
    Amount, smoothed; multiplied into scene-linear blocks before the depth haze. Controls: Relight (amount),
    Light from (-180..180, 0 right, 90 top), Light height, Light softness; the chip names both the new light and
    the measured one. Tests: on a synthetic sphere the measured light is found within 20 degrees and moving it
    brightens the side it moves to; on the astronaut, light from the left brightens the face's left half and
    from the right its right half, texture correlation > 0.97 (only light changes), a 100% view equals the
    whole render.
96. **Composite (Bill: "Let's add a composite feature to the studio, if only to ensure we are fully featured").**
    pipeline/composite.py, app/subject_picker.py. Add a subject from another photo...: a dialog shows that photo
    and click-to-select outlines the subject (Masks' Click-to-select model; Alt- or right-click removes); the
    outline is kept as a bitmap in the source's sensor frame (so it follows that photo's crop and rotation).
    Each layer: x, y (middle, fractions of this photo), size (height share), turn, mirror, edge softness, match
    grain, and a colour match solved between the two whole photos and applied to the SOURCE's settings, so the
    subject is re-developed under this photo's light as editable settings. Drawn last on the finished picture in
    every view (whole, zoomed, export), at that view's resolution: the subject is rendered at a quantised scale
    (renders and outlines cached), its alpha made solid inside (smoothstep 0.25-0.75; the soft SAM logits made
    suits translucent), warped by one affine; grain adds sqrt(dest^2 - subject^2) of measured noise (MAD of a
    fine Laplacian of log brightness over the flattest half). Never seen by masks or analysis views. Tests:
    placed where asked (centre within 6-8%, height within 8%), the rest untouched, a 100% view matches, mirror
    and turn change it, disabled = no change, masks unaffected, grain raised on a noisy scene, the matched
    source's statistics within 30% of the scene's residual.
97. **Auto white balance: Night (SGP-LRD), from RL-AWB (Bill found the code: Apache-2.0).** color/nightwb.py, a
    vectorised port of the authors' utils/Algorithm.py (copyright header and the licence kept; changes listed in
    the module): salient grey pixels (local contrast of log channels along (1,1,1), 7 x 7 mean), the two night
    filters (channel variance of log counts > 0.025; no channel further than 0.35 x the smallest mean log from
    its mean), brightness confidence (exponent from the skew), Minkowski mean of 3 x 3 local averages over local
    reflectance; on 14-bit counts of a ~750 px nearest-neighbour copy, as the reference. Checked against the
    reference code on the same input: 0.0001 degrees apart for three settings. The paper's learned agent (SAC,
    weights on Dropbox, tunes the grey share N 1-2% and the Minkowski order p 1-40) is replaced by a
    deterministic pick: all 40 settings of a 5 x 8 grid, the medoid answer used, its median angle to the others
    reported as the spread. Honest results on synthetic casts (median of 3-4 lights, our noise model, not real
    night raws): coffee (nothing grey) 2.0 degrees against grey edge 7.4 and grayness 6.1; at 1/5 the light 3.6
    against 56 and 4.6; astronaut (grey walls) 5.6 against 1.3 and 0.1 - so it is a third method, not a
    replacement. Auto WB now computes all three; the confidence is the widest disagreement between any two; the
    chip lists all three temperatures; the picker offers Night (SGP-LRD) with what it is good and weak at. Real
    night raws (Bill's) decide its worth; the paper's datasets (NCC, LEVI) were not reachable from here.
98. **Clean for scans and JPEGs, with raw shooting encouraged (Bill: "make sure Clean states clearly that it
    performs better on Raw files and that we encourage raw shooting").** local/scanclean.py. Clean was raw-only;
    the Multinex trial (decision 97's session) showed the gap. On a scan there is no sensor model, so the noise
    is measured in the picture: stored values into an orthonormal brightness / colour basis, each plane split
    into starlet bands; per band and per brightness bin (bins follow the picture's own brightness spread, so a
    dark photo gets its bins in the dark), the quietest tenth of block deviations, corrected by the same
    statistic for pure noise (texture can only add to a block), clipped blocks left out. Coarse bands are mostly
    picture even in quiet blocks (coffee measured 2.4/3.3/3.8 where noise implied 0.5/0.2/0.1), so they are
    capped by the finest band times white noise's band ratio times 2.5 (JPEG q75 noise: 2.2x at band 1); JPEG
    colour (half size, low-passed: noise flat over bands 0-2, measured) is recognised by its finest band being
    under 0.35 of brightness's and capped at the finest band's level instead. Removal: the raw starlet
    denoiser's two-pass empirical Wiener rule with that noise per pixel, brightness and colour strengths, keep
    grain, the 4.5-sigma safeguard. Results (grain-free references, known noise): +4 to +10 dB without JPEG, +0.5
    to +6.5 dB through JPEG q90; a clean picture is barely touched (> 38 dB). On the Multinex trial photos at
    equal brightness: Studio exposure + JPEG repair 24.6 dB / SSIM 0.54, + Clean 26.0 dB / 0.735, Multinex
    LOL-v2-real 23.9 dB / 0.738 - Clean ties its structure score and keeps the texture and colour it painted
    over. Some speckle survives in extreme pushes (a JPEG's own artifacts). In the pipeline: after JPEG repair,
    before everything else, once per setting (the noise measurement kept per photo); in strips with 160 px of
    context on every core (identical to one piece, tested): 24 MP in 18 s on 2 cores here. Panel: Clean now
    works on scans (row banding and the method picker stay raw-only), with an amber note: Clean does its best
    work on raw files, why, and "For the cleanest results, shoot raw"; the group's subtitle says the raw data is
    the best place to remove noise and to shoot raw; the noise chip shows the level measured in the picture; the
    old "Clean works on raws only" help line is gone; README updated the same way.
99. **The Studio from the command line, for programs (Bill, 2026-11-17: "Make a comprehensive command line
    interface so that forge and you can utilize it").** studio/cli_tools.py, added to the same `python -m studio`
    runner; `studio-cli.bat` for Windows callers; docs/CLI.md is the reference. Decision behind it: the Studio's
    technology is shared with the RPG Forge, but image generation stays in the Forge and the Studio stays a
    photographer's tool. The Forge is MIT and the Studio GPL-3.0, so the Forge runs the Studio as a separate
    program and never imports it. Commands: edit (any picture, any settings: the recipe is the Edit as JSON,
    only what differs from the defaults; --set, --recipe, --save-recipe), clean, relight, colourize, restore,
    jpeg-repair, descreen, negative, infrared, match and look-profile (colour match to a picture or to saved
    statistics), depth (map, view, normals, RGBA layers that stack back into the picture, optional filled
    background plate), cutout (RGBA or mask from person / click / depth / luminance / colour / skin / any shape),
    find (data), stack focus | superres | pano, dualpixel, schema (built from the code), looks, batch. Contract:
    --json prints one JSON object, exit codes 0 / 1 / 2, mistyped settings are errors with the nearest names,
    results follow the _Studio naming and never-overwrite rules (an RGBA PNG writer carries the Studio's mark),
    same picture and recipe give the same pixels. tests/test_cli_tools.py (12 tests). What was and was not run
    is listed at the end of docs/CLI.md. Asked for by the Forge side and not built (they need engine work):
    flatten light, make tileable, day to night.
100. **The Studio as an MCP server (Bill, 2026-11-17: "if an MCP server helps you and is easier for you, lets go
    ahead and add it ... Add details into the readme on how to enable an ai model to connect via mcp, so others
    can use it").** studio/mcp_server.py: stdio, no extra package. Tools are generated from the command line's
    own argparse parsers (one `studio_<command>` per command of decision 99, typed properties, `settings` as an
    object), so the two cannot drift; `studio_view` returns a picture to the model and a call with `preview`
    returns its preview. Dual-era: specification 2026-07-28 (stateless, `_meta` per request, `server/discover`,
    `resultType`) and the `initialize` handshake of 2025-11-25 and earlier. Progress lines go out as
    notifications. `Connect Studio to Claude.bat` (tools/connect_claude.py) adds the entry to Claude's
    claude_desktop_config.json, keeping everything else and a copy of the file as it was: the one write outside
    the Studio's folder, made only when the person runs it. `studio-mcp.bat` for other clients. README: how to
    connect, what a model can and cannot do, and that a picture a model looks at is sent to that model (the
    "never uploaded" line now says so). tests/test_mcp_server.py (8 tests). Not run yet: the Claude desktop app
    against it on Windows; the MSI copy cannot be connected (no separate Python).
