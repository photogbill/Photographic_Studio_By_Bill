# Photographic Studio By Bill — Master Build Plan

Version 2.0 · 2026-11-15 · Owner: Bill (photogbill) · Written for the coding agent that builds it

This is the single source of truth for building the software. Read all of it before writing code.
Build it phase by phase, in order. Do not start a phase until the one before it passes its acceptance tests.

---

## 0. Read this first (rules for the builder)

### 0.1 What exists today
- `raw_analyzer.py` + `Analyze Raws.bat`: measures noise, banding and hot pixels from CR2/CR3 and writes an HTML report. It works. Keep it, and reuse its code (CR3 box parser, noise-model fit, banding spectrum).
- `README.md`, `LICENSE` (GPL-3.0), `profiles/LICENSE.md` (CC BY-SA 4.0), `.gitignore` (allowlist), `push.bat`.
- A printable calibration checklist and a 4-screen GUI mockup exist as Claude artifacts (ask Bill for the links).
- **There is no denoiser, no develop engine and no GUI yet.** Everything below is to be built.
- The denoiser is built here from scratch. Bill's current denoiser is DxO (paid); it is the yardstick, and the point of this project is to give people who cannot buy it a raw denoiser of their own.

### 0.2 Who it is for
- Bill first: Canon EOS R5 (CR3), 5D Mark IV and 6D (CR2), a lot of adapted vintage lenses with no electronic contacts, candlelit portraits at high ISO, focus-stacked macro.
- Then his photography group (~150,000 people, every brand, many non-technical, many who cannot pay for software).
- The standard: **as easy as a phone app on the surface, fully featured underneath.** Bill dislikes darktable and RawTherapee. Do not copy their interface, their module-pipeline UI, or their naming. Every screen must be usable without reading a manual.

### 0.3 Non-negotiable rules
1. **Free forever, GPL-3.0 code, CC BY-SA 4.0 profiles.** Only use dependencies whose licences are GPL-3.0 compatible. Record each one in `THIRD_PARTY.md`.
2. **Photos never leave the PC.** No telemetry. No network calls except an explicit, user-triggered profile download/upload. Profile sharing is off by default and shows the exact file first. Never share: images, serial numbers, sensor fingerprint (PRNU), hot-pixel maps, dust maps, GPS.
3. **Nothing is written to C:\ or the Windows user profile (AppData, Documents, Temp) by default.** Settings, caches, logs, profiles and the Python environment all live inside the app folder (portable install). A user may move the cache; the default is beside the app.
4. **Non-destructive.** A raw file is never modified. Edits live in a sidecar (section 5).
5. **Comprehensive logging.** Rotating log files in `logs/`, one line per pipeline stage with timings, every exception with traceback, and a "Copy diagnostic report" button. Never log serial numbers or GPS.
6. **Bill does not use the command line.** Everything he runs is a `.bat` he can double-click: `Install.bat`, `Start Studio.bat`, `Run Tests.bat`, `Analyze Raws.bat`, `push.bat`. Each `.bat` must explain failures in plain English and keep the window open on error.
7. **The repo holds only software and shared profiles.** The `.gitignore` is an allowlist. Never commit a photo, raw, export, report or log. Test images are generated synthetically or fetched by a script into an ignored folder.
8. **Commits:** small and descriptive. Do not put a Claude-Session link in commit messages. A Co-Authored-By line is fine.
9. **Ask Bill questions in chat**, in plain language, with a recommendation. Do not make him open a file to answer. He delegates code-design decisions; he owns every decision about how a photo should look.
10. **Honesty over polish.** If a technique does not beat the baseline on Bill's own files, say so and drop it. Keep a `docs/RESULTS.md` with what was tried and what the measurements were.
11. **The look matters.** Bill's rule: the better a tool looks, the more it is used. Dark, cinematic, high-contrast UI; smooth previews; before/after wipes. Results that come from different methods are colour-coded (for example: measured from calibration = cyan, estimated from the image = amber, from the shared database = violet).
12. **Each area gets its own sidebar section** (section 6). Nothing may look like an email inbox.
13. **Write down what changed as you build it** (Bill, 2026-11-16). After any part is built or fixed, add one plain line for photographers to `docs/NEXT_UPDATE.md` (`- New: ...`, `- Fixed: ...`). `push.bat` sends those lines as the comment, `Release.bat` files them in `CHANGELOG.md` under the new version (`tools/whats_new.py`), and GitHub shows them on the release page that "What's new" opens. Bill does not write comments ("otherwise I won't write them"), and `push.bat` stays "just push".
14. **Errors report themselves** (Bill, 2026-11-16). An unexpected error writes `error_reports/error_YYYY-MM-DD_HH-MM-SS.txt` beside the app by itself (`studio/log.py: save_error_report`; in the app call `common.report_problem(what, exc)` inside the `except`). Testers send the file. Problems that are not faults of the Studio (an unreadable raw, a read-only drive, no internet) are said where they happen and write no report.

### 0.4 Decisions already made by Bill (do not reopen without asking)
| Topic | Decision |
|---|---|
| Name / repo | Photographic Studio By Bill · github.com/photogbill/Photographic_Studio_By_Bill · local folder `D:\Analyst_Toolkit\Photographic Denoiser` |
| Denoise | One single raw in, one clean result out. **No stacking in the denoiser.** (Stacking exists only as separate, clearly-named tools: Focus Stack and Super-Resolution Stack.) |
| Cameras | Canon CR2 and CR3 first. Then every brand. Users can calibrate their own body. |
| Lens correction | Fully automatic, including contactless vintage primes. Purple-fringe removal ON by default. **Vignette correction OFF by default** (Bill likes vignetting). Manual vintage zooms are out of scope. Build in lensfun for modern lenses. Lenstip-style review data may be used the way a photographer would; no DxOMark data. |
| Dual Pixel RAW | First-class support, used in as many useful ways as possible (section 11). |
| Masks | No Segment Anything (SAM). |
| Colour | Working space: linear ProPhoto RGB (ROMM primaries), 32-bit float, scene-referred. Default export: Adobe RGB. Other export spaces allowed. |
| Output | 16-bit TIFF default; PNG and JPEG; full-size flexible export. DNG only if unavoidable (Bill is not a fan). |
| Adjustments | Full set: exposure, contrast, microcontrast, DxO-style colour wheel, temperature/tint, auto colour, crop, and more (section 9). Inspiration: DxO PhotoLab and Exposure X7. |
| Microcontrast | Means "more shades of shadow", tonal separation, not edge sharpening. Bill says DxO gets it wrong. This is a headline feature. |
| Render like my lens | A shared, open library of real lens rendering signatures, built from Bill's lens collection and grown by users. |
| Calibration | A wizard: printable checklist → drop files in step by step → calibration runs. ISO 100 up to max and back down (cold pass, hot pass), plus long exposures. No artificial warm-up. Separate lens wizard. |
| Exposure latitude | Calibration also measures ETTR and ETTL behaviour per body (7.1): how much a frame can be pushed or pulled, and what shadow noise and highlight behaviour result. The wizard must be easy for non-technical users (7.2). |
| Integrations | Send to Photoshop, Send to PhotoCraft (github.com/storytold/photocraft), a Photoshop UXP panel, and a home inside ATK's Digital Forensics Workshop. |
| Export names | Every export is named after its photo with `_Studio` added: `IMG_0001.CR2` becomes `IMG_0001_Studio.tif` (`io/export.py: export_name`). Never the original's name; a file the Studio did not write is never replaced. (2026-11-16) |
| Installer | The person picks where the Studio installs: the folder window opens first. (2026-11-16) |
| Unbuilt tabs | Build them, do not label them "coming soon": `BUILD_PLAN_TABS.md` (Dual Pixel Lab, Focus Stack, Super-Res Stack). (2026-11-16) |
| Library and card import | The Library is where the photographer decides what to process, with no need for Windows Explorer. **Import from card** copies photos from a card into the photographer's own Photos folder (they choose it: internal or external drive), always into a subfolder they can name, the shoot date by default. Raws only by default. Section 24. |

### 0.5 How to work
- Keep `docs/PROGRESS.md` current: phase, what passed, what is open.
- Every algorithm gets: a pure function in the engine, a unit test on synthetic data with known ground truth, and an entry in the benchmark harness (section 17).
- Every phase ends with something Bill can double-click and see.
- Citations in sections 8, 11, 12, 13 and 14 were written from memory; several were confirmed in the section 21 literature review. **Verify each paper and each library capability before relying on it**, and note corrections in `docs/RESEARCH.md`.
- **Section 21 lists 14 changes from a 2024–2026 research review. Treat them as part of the sections they point to.**

---

## 1. Product shape

One application, eight workspaces, each a sidebar section:

| # | Workspace | What it does |
|---|---|---|
| 1 | **Library** | Browse folders, cull, rate, compare, filter. No import required, no catalogue lock-in. Optional **Import from card** (section 24). |
| 2 | **Develop** | The raw editor: denoise, lens fixes, tone, colour, microcontrast, masks, retouch, crop. |
| 3 | **Dual Pixel Lab** | Everything Dual Pixel RAW can do (section 11). |
| 4 | **Focus Stack** | Merge a focus bracket into one sharp image (section 12). |
| 5 | **Super-Res Stack** | Handheld burst → higher resolution and lower noise; moon and sky modes (section 13). |
| 6 | **Lenses** | Lens library, lens wizard, "render like my lens" signatures. |
| 7 | **Cameras** | Calibration wizard, camera profiles, the Raw Analyzer report. |
| 8 | **Export** | Export queue, presets, Send to Photoshop / PhotoCraft, print and soft-proof. |

Plus a **Help** section: built-in guided tours, the printable checklists, and "Copy diagnostic report".

Two interface levels, switchable at any time, same engine:
- **Simple:** one "Make it good" button (auto denoise + auto lens fix + auto tone + auto colour), then about eight sliders.
- **Full:** every control.

### 1.1 Positioning: what this is and is not

This project will not match Photoshop, Lightroom or DxO feature for feature, and must not try. Its place is:
1. **Free and private**, for people who cannot pay.
2. **Easy**, where the free alternatives are hard.
3. **First with techniques the big products do not ship**: per-camera calibrated raw denoising, automatic correction for contactless vintage lenses, a real Dual Pixel editor, lens rendering signatures, measured microcontrast, exposure-latitude advice per body, and methods taken from astronomy, MRI and signal processing.

When a choice has to be made between a common feature done adequately and an unusual one done well, build the unusual one first.

### 1.2 The Experimental Lab

Experimental techniques are a feature, not a risk to hide. They get a visible home so they can ship early without endangering the dependable tools.

- A **Lab** switch in settings turns on experimental tools. Each appears in its normal workspace with a flask badge and a one-line plain-English note: what it tries to do, where it came from ("from a Hubble technique", "from MRI"), and what can go wrong.
- **Three maturity levels**, shown in the UI: *Experimental* (may fail, may change) → *Beta* (works on Bill's cameras, needs wider testing) → *Stable* (passed the benchmark and blind tests; on by default where appropriate).
- Every Lab tool is **non-destructive and off by default**, has a before/after, and never changes how existing sidecars render.
- **Promotion rule:** a technique moves up only on evidence from the benchmark harness and blind tests (section 17). A technique that loses is removed and written up in `docs/RESULTS.md`; negative results are kept, since other people will want to know.
- **Feedback loop for the group:** a one-click "this helped / this hurt" on any Lab tool, saved locally; the user may choose to export a small anonymous report (tool, camera model, ISO, verdict; no image). Nothing is sent automatically.
- Sections 14 and 21 are the Lab's backlog. The first Lab tools are whichever denoiser candidates (8.1 b–e) are not yet the default, Dual Pixel items 12–13, blind lens correction (21, change 9), and HDR burst (21, change 11).
- **Generative methods** (ones that invent plausible detail) live only in the Lab, are never a default, and are always labelled "synthetic". Adopted 2026-11-14.

---

## 2. Architecture

### 2.1 Stack (recommended; confirm with Bill in one sentence before Phase 1)
- **Language:** Python 3.12 for the whole first version (the analyzer is already Python and the launcher pattern exists).
- **Raw decode:** LibRaw through `rawpy` for Bayer data and metadata; our own CR3/CR2 container code (already started in `raw_analyzer.py`) for anything LibRaw does not expose: Dual Pixel sub-frames, masked/optical-black borders, sensor temperature, Canon maker notes.
- **Numerics:** NumPy + Numba for the CPU reference path. GPU path through PyTorch (CUDA) as an optional accelerator, with every kernel also running on CPU. Target machine: 14-core CPU, 64 GB RAM, RTX 3080 Ti. Target for the group: any 8 GB laptop, no GPU required, just slower.
- **GUI:** PySide6 (Qt 6) with QML/Qt Quick for the flashy, GPU-drawn interface. The viewport is a GPU texture with shader-based preview adjustments.
- **Colour management:** LittleCMS (`lcms2`) for ICC; monitor profile honoured; soft-proofing.
- **Lens database:** lensfun (LGPL) for modern lenses; our own profile format for everything else.
- **Metadata:** ExifTool (bundled, portable) or `pyexiv2` for writing EXIF/XMP on export.
- **Packaging:** portable folder first (`Install.bat` builds a private environment inside the app folder, as the analyzer already does). Later: a single-download portable ZIP built by a release script, so group members install nothing.

If the Nov 11 plan document chose a different stack (the `.gitignore` already allows Rust and web files), ask Bill which one stands. Whatever the choice, keep rule 2.2.

### 2.2 Hard separation: engine vs interface
```
studio/
  engine/        # no GUI imports, ever. Usable from a script, the Photoshop bridge, ATK and tests.
    io/          # raw containers, decode, metadata, sidecars, export
    calib/       # camera noise model, dark/flat/defect maps, calibration wizard logic
    raw/         # raw-domain stages: black level, defects, banding, denoise, CA, demosaic
    lens/        # distortion, falloff, fringing, PSF, lensfun bridge, signatures
    color/       # white balance, camera matrices, working space, tone, grading
    local/       # masks, local adjustments, retouch
    detail/      # microcontrast, capture sharpening, output sharpening, grain
    dualpixel/   # everything in section 11
    stack/       # focus stack, super-resolution, alignment, rejection
    pipeline/    # stage graph, caching, tiling, ROI, proxy rendering
    presets/     # looks, sync, virtual copies
  app/           # PySide6/QML interface only; talks to the engine through one API
  bridge/        # Photoshop UXP panel, PhotoCraft hand-off, ATK plug-in entry point
  profiles/      # cameras/, lenses/, signatures/ (CC BY-SA, shared)
  tools/         # raw_analyzer, benchmark harness, synthetic test generators
  tests/
  docs/
```
The engine API is a small set of calls: `open(path) → Image`, `render(Image, Edit, region, scale) → array`, `export(Image, Edit, ExportSpec)`. A command-line runner (`studio-cli`) wraps the same calls for batch and for tests.

### 2.3 Pipeline engine requirements
- **Stage graph with caching:** each stage declares its inputs and parameters; changing a slider re-runs only the stages after it.
- **Region and scale aware:** the viewport asks for "this rectangle at this zoom". Fit-to-screen uses a proxy; 100% uses the visible tiles only.
- **Tiled:** a 45 MP frame in float32 RGB is about 540 MB per buffer. Process in overlapping tiles so 8 GB machines work and GPU memory never overflows.
- **Preview targets:** slider feedback under 50 ms on the proxy; settled 100% view under 1 s on Bill's PC for everything except denoise; denoise preview on a movable loupe first, full frame in the background.
- **Deterministic:** same raw + same sidecar + same version = bit-identical export. Each stage has a version number stored in the sidecar so old edits render the same after updates.

---

## 3. The develop pipeline, in order

Order matters. Each stage sits where the physics says it should. Stages 1–9 are on raw sensor data (before demosaic); that is where this tool wins.

| # | Stage | Domain | Notes |
|---|---|---|---|
| 1 | Decode | raw mosaic | Full sensor including masked borders. Dual Pixel sub-frames if present. |
| 2 | Linearise | raw | Black level per channel from optical-black area, per-row where the camera allows; white level; any linearisation table. |
| 3 | Defects | raw | Hot/dead/stuck pixels from the calibration map + single-frame detection (section 8.3). Dual Pixel: a defect hits one half only. |
| 4 | Dark-current / fixed pattern | raw | Subtract scaled calibration dark pattern (exposure time, temperature). Optional. |
| 5 | Banding / 1-f striping | raw | Row and column pattern removal using masked pixels and the measured banding spectrum (section 8.2). |
| 6 | Flat field | raw | PRNU (per-pixel gain, local to this PC only) and dust map. Optional. |
| 7 | **Raw denoise** | raw, variance-stabilised | The core (section 8). Needs the camera noise model. |
| 8 | Raw chromatic aberration | raw | Lateral CA corrected by resampling red and blue planes before demosaic. |
| 9 | Highlight reconstruction | raw | Clipped channels rebuilt from unclipped ones; Dual Pixel adds about one stop (section 11). |
| 10 | Demosaic | raw → RGB | Start with RCD or AMaZE-class quality; design the denoiser and demosaic together (joint path as research). X-Trans later. |
| 11 | White balance + camera → working space | linear RGB | Camera matrix / DCP-style dual-illuminant profile; into linear ProPhoto. |
| 12 | Capture sharpening | linear | Deconvolution with the lens PSF (section 10.5). |
| 13 | Geometry | linear | Distortion, falloff (off by default), perspective, rotate, crop. One resample for all of them. |
| 14 | Fringing | linear | Purple/green longitudinal fringe removal (on by default). |
| 15 | Exposure and tone | scene-referred | Exposure, highlights, shadows, whites, blacks, tone curve, tone mapper. |
| 16 | Microcontrast | scene-referred | Section 9.3. |
| 17 | Colour | scene-referred | HSL, colour wheel, grading, vibrance, B&W mixer, looks. |
| 18 | Local adjustments + retouch | scene-referred | Masks, heal/clone, dust removal. Masked versions of 15–17. |
| 19 | Lens signature | scene-referred | "Render like my lens" (section 10.6). |
| 20 | Grain, vignette (creative), borders | display-referred | |
| 21 | Output | display-referred | Resize, output sharpening, colour-space conversion, dither, metadata, write. |

---

## 4. File formats and camera support

- **Phase 1:** Canon CR2 (5D Mark IV, 6D and others LibRaw reads) and CR3 (R5), including C-RAW, and Dual Pixel variants of both.
- **Then:** everything LibRaw decodes (Nikon NEF, Sony ARW, Fujifilm RAF incl. X-Trans, Pentax PEF/DNG, Olympus/OM ORF, Panasonic RW2, Leica, DNG). A camera with no calibration profile still works using an image-estimated noise model (the analyzer already does this) or a shared profile.
- **Also open:** TIFF, PNG, JPEG, so the same editor handles non-raw files (stages 1–10 skipped, noise model estimated).
- **Support tiers shown in the UI:** Calibrated on this body (cyan) · Shared profile for this model (violet) · Estimated from this image (amber).
- **True ISO, not the EXIF number.** Canon bodies (confirmed on the 5D Mark IV; reported on the 1D X and 1D X Mark II) write ISO 65535 in standard EXIF for any ISO above 51200, because the field is a 16-bit number. The real ISO is in Canon's private maker notes. The decoder must read the true ISO from the maker notes and treat 65535 (also 32767 and negative values) as "invalid, look elsewhere". `raw_analyzer.py` currently reads the standard EXIF field and must be fixed in Phase 0; every profile lookup depends on it. Show "ISO 102400" in the UI, never "65535".
- **Expanded ISOs (L, H1, H2) are probably not real gain settings** but the nearest real ISO with the data scaled. On the 5D Mark IV, H1 = 51200 and H2 = 102400 sit above the top standard ISO of 32000. Calibration must determine the real gain behind each expanded setting; the noise model uses the real gain plus the scale factor. Bill shoots H1 and H2 regularly, so they are part of his calibration, not optional.
- **Unknown camera:** never crash. Decode with LibRaw defaults, estimate, and invite the user to run the calibration wizard.

---

## 5. Non-destructive editing

- **Sidecar:** `<name>.<ext>.psb.json` beside the raw (plain JSON, human-readable, versioned). Option to keep sidecars in a central folder instead for read-only drives (Bill's archive is on external drives).
- Contents: app version, per-stage version, every parameter, masks (vector where possible, small compressed rasters otherwise), crop, rating/label/keywords, history, snapshots, virtual copies.
- **History:** unlimited undo, visible list, click to go back, survives closing the app.
- **Snapshots** (named states) and **virtual copies** (several edits of one raw).
- **Copy / paste / sync settings** with a checklist of which groups to copy.
- **Presets ("Looks"):** a sidecar fragment in a single shareable `.psblook` file. Ship a starter set; "Bill's Candlelight" is the first. Drag onto a photo to apply. Looks are relative where it makes sense (do not overwrite exposure or white balance unless the look says so).
- Also read/write XMP rating, label and keywords so other software sees them.

---

## 6. Interface

### 6.1 Principles
- Dark theme, the mockup's palette. Big live image, controls on the right, filmstrip at the bottom, sidebar on the left.
- Every slider: double-click resets, hover shows a one-line plain-English tip and a tiny before/after animation, scroll wheel nudges, typed values accepted.
- **Before/after:** split wipe, side-by-side, and hold-to-compare key.
- **Loupe:** 100% magnifier that follows the mouse; denoise preview renders there first.
- Tools grouped by what the photographer wants, not by algorithm: *Clean* (noise, dust, hot pixels, banding) · *Lens* · *Light* · *Colour* · *Detail* · *Local* · *Retouch* · *Shape* (crop, straighten, perspective) · *Finish* (grain, vignette, border).
- An **Auto** button on every group, and one master "Make it good".
- Progress is always visible and cancellable. Long jobs go to a background queue.
- Keyboard shortcuts with an on-screen cheat sheet. Guided first-run tour.
- Histogram with raw-clipping indicators (shows what the sensor clipped, not what the display clipped), clipping overlays, colour picker readout.
- High-DPI and two-monitor support. Works at 1366×768.

### 6.2 Library
- Folder tree + thumbnail grid + filmstrip. Opens a folder instantly using embedded JPEG previews; real renders replace them in the background.
- Stars, colour labels, pick/reject, keywords; filter and sort by any of them and by camera, lens, ISO, date.
- **Cull mode:** full-screen, one key per rating, auto-advance, 2-up and 4-up compare with synchronised zoom, focus-check overlay.
- Auto-group brackets and bursts (same second, same settings, changing focus distance or handheld burst) and offer "Send to Focus Stack / Super-Res Stack".
- **Lens tagging for contactless lenses:** pick the lens once per shoot (or let the app suggest it from the measured rendering signature); it is written into the sidecar and into export metadata.
- Thumbnail and preview cache inside the app folder, size-capped, with a "clear cache" button.
- **Added 2026-11-15: see section 24** for what exists today, the Review view, speed on big folders, Process picks, card import and the Photos folder.

---

## 7. Camera calibration (Cameras workspace)

Implements the existing printable checklist (Parts A–C, plus optional Part G for exposure latitude, section 7.1). Ease-of-use requirements are in 7.2. The wizard is: pick camera → print checklist → drop each part's files into a drop zone → the wizard checks every frame immediately (missing ISO, light leak, wrong setting, C-RAW, lens-cap off) → build profile → show the report.

What a camera profile contains (`profiles/cameras/<make>_<model>.json`, shareable):
- Per ISO: gain (e-/DN), read noise, black level behaviour, clipping point, the Poisson-Gaussian noise curve.
- Row/column banding spectrum and its amplitude per ISO.
- Dark current vs exposure time, cold vs hot (from the two passes and the long exposures); sensor temperature mapping where the file records it.
- Noise correlation between neighbouring pixels (some cameras filter raws).
- Dual Pixel: A/B gain ratio, the A-frame clipping point, per-ISO noise of each half.
- Exposure latitude (7.1): ISO-invariance curve, push penalty per ISO, gain-step ISOs, highlight headroom and near-clip linearity.
- Colour: camera matrices; optional ColorChecker-based profile step.

What stays local, never shared (`calibration_local/<serial-hash>/`): hot/dead pixel map, PRNU map, dust map, master darks.

Rules: sorting is automatic from metadata (the user never renames or sorts files). A profile records how it was made and its quality score. A camera with a partial calibration still gets a usable profile.

### 7.1 Exposure latitude: ETTR and ETTL (checklist Part G, optional, once per body)

Bill's addition. Some sensors can be underexposed at a low ISO and brightened afterwards with almost no penalty ("ISO-invariant"); others pay heavily in shadow noise and banding and should have the ISO raised in camera instead. It differs by body, and often by ISO range within one body. **Working rule from Bill's field experience (years of low-light work on Canon bodies, taught to his students): every Canon body punishes underexposure severely and rewards slight overexposure, provided the highlights are not blown.** Use this as the default advice for any Canon without a profile. Calibration then measures how large the effect is on each body and ISO; the measurement refines the rule, it does not replace it. For other brands, measure before advising.

Two different things must be kept apart in the code and in the advice, because they are often confused:
- **Less light** (same ISO, faster shutter, then brightened): always worse on every sensor, because fewer photons were collected. At extreme ISO it is catastrophic: blotchy colour noise and colour shifts. Bill's reference pair shows this on the 5D Mark IV: a properly exposed ISO 102400 frame is usable; an underexposed one at the same ISO falls apart into colour blotches.
- **Same light, lower ISO setting, brightened afterwards**: this is the "ISO invariance" question, and it is what the matched pairs in the recipe below isolate.
The Exposure Advisor reports both, in plain words.

Why it belongs in calibration:
- **ETTL (expose to the left, then push):** a pushed frame shows the shadow-specific noise pattern (read noise, banding, fixed pattern, colour drift from black-level error) at full strength while the scene is otherwise normal. Dark frames show this noise alone; Part G shows how it behaves under real image content, which is what the denoiser faces.
- **ETTR (expose to the right, then pull):** shows the noise of an over-exposed frame brought back down, plus what happens near saturation: per-channel clipping points, non-linearity just below clipping, colour shifts in near-clipped highlights, and how much highlight headroom truly exists.

**Shooting recipe** (the wizard generates the exact settings card for the user's camera from one metered reading; the user types nothing):
One still scene with texture, deep shadow and a bright near-white area; tripod; steady light.
1. Reference: base ISO, metered at 0. 2 frames.
2. ETTL series: stay at base ISO, make the shutter faster by 1, 2, 3, 4 and 5 stops. 2 frames each.
3. Matched-ISO series: the same five shutter speeds, but raise the ISO by the same number of stops so the exposure is "correct" in camera. 2 frames each. (Each pair from steps 2 and 3 received identical light; only the ISO differs. That is the direct test.)
4. ETTR series: base ISO at +1, +2 and +3 stops. 2 frames each. Repeat at the user's usual low-light ISO.
5. Underexposure at high ISO: at the user's usual low-light ISO and at the highest ISO they really shoot (for Bill: 51200 and 102400), frames at −1 and −2 stops. 2 frames each. This is where Canon files break down and is the denoiser's hardest case.
About 42 frames, fifteen minutes.

**What the wizard computes and stores in the camera profile:**
- Input-referred read noise (electrons) versus ISO, from Part A, cross-checked against Part G pairs. This is the ISO-invariance curve.
- For every (ISO, push amount): the measured noise penalty in stops of pushing versus raising ISO in camera, and the banding/fixed-pattern amplitude after the push.
- The ISO at which the camera becomes effectively invariant, and any gain-step ISOs.
- Whether in-between ISOs (third-stops) are real gain changes or digital scaling.
- Highlight behaviour: true clipping point per channel, the linear range, usable ETTR headroom in stops per ISO.

**What the app does with it:**
- **Exposure Advisor card** in the Cameras workspace, in plain English and colour-coded, for example: "On this camera, from ISO 400 up you can underexpose up to 3 stops to protect highlights at no real cost. Below ISO 400, raise the ISO instead. Exposing to the right gains you about 1 stop of cleaner shadows at ISO 100." Printable.
- **Noise model follows the push.** Denoise defaults are driven by the actual signal level and the amount of brightening applied in Develop, not by the ISO label. An ISO 100 frame pushed 4 stops is treated as what it is. Banding removal and black-level care (section 21, change 1) strengthen automatically in pushed shadows.
- **Pulled ETTR frames** get lighter default denoise, and near-clip colour is corrected using the measured non-linearity before highlight reconstruction.
- **Library:** badge frames shot well left or right of normal exposure; "Normalise exposure" applies the brightening and the matching denoise settings in one click.
- **Shared database:** the latitude curves are model-level numbers and are shareable, so someone who never calibrates still gets the Exposure Advisor for their camera.

Named tests: **Bill's underexposed ISO 102400 candle portrait (5D Mark IV) is the torture test**: the result must have no colour blotches and no magenta/green drift, with the mask's green and the skin tone matching the properly exposed frame from the same session; a 5-stop push on each of Bill's bodies shows no visible banding after processing where the calibration says it can be removed; pushed shadows stay neutral; a +2 ETTR frame pulled down matches the reference frame's colour in the highlights.

### 7.2 The wizard must be easy (requirements)

- **Three levels, each a complete result:** *Quick* (Part A dark frames only, about 15 minutes; see section 21 change 2) → *Full* (Parts A–C) → *Complete* (adds Part G). The user can stop at any level and upgrade later; nothing is reshot.
- **One step per screen**, one picture of what to do, one sentence. No jargon: "Put the body cap on" not "acquire bias frames".
- **A settings card made for the user's exact camera** (its real ISO list, its menu names for the settings to turn off), viewable on a phone or printed.
- **Drop files in, in any order, in one pile.** The wizard sorts by metadata, ticks off what it has, and shows what is still missing as a short list.
- **Instant check on every dropped frame** with a plain reason and fix if it fails ("This one has a light leak: cover the eyepiece and reshoot ISO 3200").
- **Pause and resume** across days. Progress is saved.
- **No typing of numbers, ever.** Everything comes from the files.
- **A finish screen worth showing off:** the camera's noise curves, the Exposure Advisor card, a before/after of one of the user's own frames, and the profile's quality score.
- **Sharing is one clearly optional button** at the end, showing exactly what would be sent.
- Acceptance stays as Phase 7: a non-technical tester completes Quick calibration with no help, first try.

---

## 8. The denoiser (headline feature)

**Goal:** from one raw, at ISO 6400–51200 on the R5/5D IV/6D, match or beat Bill's old DxO on his own files with less detail loss. Judged blind by Bill (section 17).

### 8.1 Design
1. **Know the noise exactly.** Use the calibrated Poisson-Gaussian model (Foi et al. 2008) per channel, per ISO. Apply a variance-stabilising transform (generalised Anscombe, Mäkitalo & Foi) so noise becomes uniform, denoise, then use the exact unbiased inverse.
2. **Work on the mosaic**, before demosaic, on the four colour planes (R, G1, G2, B) or on a luminance/chroma transform of them. Demosaicing first turns clean per-pixel noise into coloured blotches that are far harder to remove.
3. **Remove structured noise first** (stages 3–6): defects, banding, fixed pattern. What is left is close to pure random noise, which is the easy kind.
4. **Core denoiser: a ladder of candidates, measured against each other.** Bill wants as many techniques as possible built and tried. Implement all of them in this order, keep every one that earns its place, and combine them where the combination measures better than any single method (for example: different methods per scale, per channel, or blended by the local signal-to-noise map):
   - a. Baseline: BM3D-class collaborative filtering on VST planes (the bar to beat).
   - b. Non-local patch PCA with a Marchenko-Pastur threshold (from MRI: Veraart et al. 2016, "MP-PCA"). Because the noise level is known exactly, the cut between signal and noise components is principled.
   - c. Multiscale: starlet / à-trous wavelet (astronomy: Starck & Murtagh) or Laplacian pyramid so low-frequency chroma blotches are handled at coarse scales and fine luminance detail is protected at fine scales.
   - d. Learned: a compact raw-domain network trained self-supervised on **this camera's own calibration frames plus synthetic noise from its measured model** (noise is synthesised from the profile onto clean low-ISO raws). No third-party image sets that carry unclear licences. Must run on CPU; GPU optional.
   - e. Dual Pixel assist when available (section 11.4).
5. **Detail-protection controls:** separate luminance and chroma strength; a "keep the grain" slider that blends back a controlled, fine, monochrome fraction of the residual (film-like, never blotchy); texture-aware strength from a local signal-to-noise map.
6. **Joint denoise + demosaic** as a research track once the above is solid.

### 8.2 Banding (Bill's cyclostationary idea)
- Banding is noise with a repeating row/column structure, which is exactly what cyclostationary analysis is for. The analyzer already measures the line spectrum from the masked border.
- Method: estimate the per-row and per-column offsets from the masked (optical-black) pixels and from a robust statistic of the image itself; model periodic components found in the calibration spectrum; subtract. This is the same problem NASA solves for JWST's 1/f striping (section 14).
- Must never create bands where there were none: gate by the measured amplitude from the profile.

### 8.3 Hot pixels and impulse noise
- Calibration map first. Then single-frame detection using Laplacian edge detection (van Dokkum 2001, "L.A.Cosmic", built to find cosmic-ray hits in single astronomy exposures): a real star or specular highlight is blurred by the lens; a hot pixel is sharper than the lens can physically produce.
- Use the lens PSF, when known, as the sharpness limit. Candle-flame highlights and catchlights must survive; this is a named test.

### 8.4 User controls (Simple level)
One slider: **Clean** (0–100, default from ISO). Full level adds: luminance, colour, detail protection, keep-grain, banding, hot pixels, method picker (for testing).

---

## 9. Develop tools

### 9.1 Light
Exposure · contrast · highlights · shadows · whites · blacks · tone curve (parametric and point, per channel) · a filmic tone mapper with a visible shoulder/toe · highlight reconstruction mode (clip / rebuild colour / Dual Pixel) · local "smart lighting" (lifts shadows without flattening) · dehaze · auto-tone that respects low-key images (a candlelit portrait must stay dark; auto must not turn night into day — named test).

### 9.2 Colour
Temperature/tint with eyedropper · auto white balance · **auto colour** · camera-profile picker · DxO-style colour wheel (pick a hue range on the wheel, shift hue/saturation/luminance, adjustable range and falloff, several ranges) · HSL panel · colour grading (shadows/mids/highlights wheels) · vibrance/saturation · skin-tone protect · B&W mixer with filters · split-tone · LUT/look loading · film looks (Exposure X7 territory) built as open, documented looks.

### 9.3 Microcontrast (do this properly)
Bill's definition: more distinguishable shades inside shadows and midtones, "more dynamic range within exposure". Not edge sharpening, not halo "clarity".
- Implement as tonal-separation enhancement at medium spatial scales on scene-referred luminance, using an edge-aware multiscale decomposition (guided filter or local Laplacian filters, Paris et al. 2011) so there are no halos.
- Separate strength for shadows, midtones and highlights; shadows are the default emphasis.
- Must be noise-aware: it uses the local SNR map from the denoiser so it never amplifies leftover noise.
- **Ground truth exists:** Bill's same-scene lens test (checklist Part E; Heliar chosen 19 of 20 times over modern lenses). Measure what the Heliar does to local tonal gradients versus the Sigma and Canon on the same subject, and make the control reproduce that difference. This also feeds the lens signatures (section 10.6).

### 9.4 Detail
Capture sharpening (section 10.5) · creative sharpening with masking · texture · output sharpening by size and medium · grain (monochrome or film-type, size, roughness, by tone).

### 9.5 Local adjustments (no SAM)
- Mask types: brush with edge-aware auto-mask ("stay inside the leaf") · linear gradient · radial · **luminance range** · **colour range** (click a colour) · **depth range** (Dual Pixel, section 11) · focus range (in-focus vs out-of-focus) · combine with add/subtract/intersect · feather and refine.
- "Click an object" without SAM: seeded region growing on colour + edges + depth (GrabCut-style graph cut), refined with a guided filter. Bill's test: select one whole leaf in two clicks.
- Every Light/Colour/Detail control is available per mask, including denoise strength and microcontrast.

### 9.6 Retouch
Heal and clone brushes · spot removal with auto source · **automatic dust removal** from the calibration dust map (scaled by aperture) · red-eye · simple skin smoothing by frequency separation (optional).

### 9.7 Shape
Crop with aspect presets and overlays · straighten (draw a line / auto horizon) · perspective/keystone (auto and guided) · rotate/flip · content-aware edge fill after rotation (optional, later).

### 9.8 Finish
Creative vignette · borders · watermark (text/logo, saved per export preset).

---

## 10. Lenses

### 10.1 Sources of correction data, in priority order
1. The user's own lens-wizard profile (cyan).
2. The shared community profile for that lens (violet).
3. lensfun entry for modern lenses with electronic contacts (violet).
4. **Estimated from the photo itself** (amber) — the fully automatic path for contactless vintage glass.

### 10.2 Automatic, profile-free correction (the vintage-lens path)
- **Lateral chromatic aberration:** estimate by registering the red and blue planes to green across radial zones (edges should line up between channels); fit a radial polynomial; correct on raw planes (stage 8). Needs no lens data at all.
- **Purple/green fringing (longitudinal):** detect fringe-coloured pixels next to high-contrast edges, limited to defocused/bright-edge regions; desaturate and re-hue toward the neighbourhood. On by default. Must not strip real purple (flowers, fabric) — named test.
- **Distortion:** from straight-line detection when the scene has lines; otherwise leave alone unless a profile exists. Never guess distortion on organic subjects.
- **Falloff:** estimated, but correction OFF by default.
- One button: **Fix lens**. The result panel shows what was fixed and where the data came from.

### 10.3 Lens wizard
Implements checklist Part D: falloff flats per aperture, straight-line target, candle (flare/glare), branches against sky (fringing), plus optional PSF target. Output: `profiles/lenses/<maker>_<name>.json` with distortion, falloff per aperture, lateral CA, fringing behaviour, field-varying PSF, flare/veiling-glare signature, and the rendering signature (10.6). Manual zooms out of scope.

### 10.4 Lens identity
For contactless lenses: user picks from their lens list; the app remembers per shoot and suggests a lens by comparing the image's measured aberration signature to stored profiles. Written to export metadata.

### 10.5 Capture sharpening by deconvolution
- Uses the lens PSF (measured, or estimated blind from edges in the image; for astro shots, from the stars).
- Richardson–Lucy deconvolution with damping/regularisation and noise awareness (the same family NASA used on Hubble images before its optics were corrected, section 14). Field-varying: different PSF per image zone.
- Default strength is conservative. Ringing is a failure; test on high-contrast edges and on the candle flame.
- Vintage-lens users often want the softness: the control is labelled "Recover lens sharpness" and is off for lenses with a signature marked "character lens" unless the user turns it on.

### 10.6 Render like my lens
- A **rendering signature** is what a lens does beyond geometry: local tonal contrast transfer by spatial scale and by tone (the microcontrast fingerprint), spherical-aberration glow, veiling glare, flare shape around bright sources (Bill's Laack candle flare), bokeh shape and edge character, falloff, colour cast, field curvature softness toward the corners.
- Measured in the lens wizard from the same-scene test and the candle and fringing frames, always relative to a neutral reference lens on the same scene.
- Applied at stage 19 as: tone/scale-dependent contrast transfer + glow + veiling glare + falloff + colour. Flare synthesis around detected bright sources is optional and labelled as synthetic.
- Honest limit to state in the UI: a signature can reproduce tonal and glow character. It cannot recreate true depth-of-field or true bokeh from a different lens. With Dual Pixel depth, a closer approximation is possible (section 11).
- Library lives in `profiles/signatures/`, CC BY-SA, grows as Bill adds lenses. First entries: Laack 13.5cm Anastigmat, Voigtländer Heliar, Apo-Lanthar, Zeiss Otus 55, Sigma 135 Art, Canon 180L Macro.

---

## 11. Dual Pixel Lab

Canon Dual Pixel RAW stores two images: frame 1 is the normal picture (both halves of every pixel summed, **A+B**); frame 2 is one half only (**A**). The other half is **B = (A+B) − A**. The two halves look through the left and right sides of the lens, so anything out of focus shifts slightly between them; anything in focus does not. Bodies: 5D Mark IV (CR2) and R5 (CR3), among others. Build a decoder that returns A+B, A and B as raw mosaics with correct black levels; verify whether LibRaw exposes frame 2 for both containers, and write our own extractor if not. Detect DPRAW files automatically and badge them in the Library.

Uses, in build order:

| # | Feature | How | Status |
|---|---|---|---|
| 1 | **Split and view** | Show A, B, A+B, and the A−B difference. Export any of them. (What DPRSplit does, built in.) | Build |
| 2 | **Highlight recovery, about +1 stop** | A alone holds half the light, so it clips one stop later. Where A+B is clipped and A is not, rebuild from 2×A, blended smoothly, with A's higher noise handled by the denoiser. Feeds stage 9. Candle flames are the showcase. | Build |
| 3 | **Depth / defocus map** | Per-region disparity between A and B (sub-pixel, sign tells front vs behind focus), refined edge-aware. Confidence map included. Low-texture and in-focus regions are ambiguous; say so. | Build |
| 4 | **Noise help for the single-shot denoiser** | A and B are two measurements of the same scene with independent noise. In focused areas their difference is pure noise: a direct, per-image noise measurement and a detail-vs-noise detector. Also enables self-supervised (noise2noise-style) training/tuning on the user's own file. Note honestly: A+B already contains all the light, so this improves *decisions*, not raw signal-to-noise. Stays within "one raw in". | Build + measure |
| 5 | **Depth masks** | "Subject", "background", "foreground" masks from the depth map for Local adjustments (blur, darken or desaturate the background; denoise it harder; sharpen only the focal plane). | Build |
| 6 | **Focus tools** | Focus-plane overlay; front/back-focus readout in the Library cull view (which frame of a burst is truly sharpest on the eye); lens focus-calibration helper. | Build |
| 7 | **Micro-refocus** | Shift the plane of sharpness slightly by re-weighting/shifting the halves by depth (Canon DPP's "Image Microadjustment"). The range is tiny; label it honestly. | Build |
| 8 | **Bokeh shift** | Shift the viewpoint of foreground/background blur (DPP's "Bokeh Shift"). | Build |
| 9 | **Ghost and flare control** | Flare and ghosts differ between halves; choose the half with less in affected zones (DPP's "Ghosting Reduction"). **Off by default**: Bill wants his candle flare. Also offer the opposite: emphasise it. | Build |
| 10 | **Defect detection** | A hot or dead sub-pixel affects only one half; compare halves to find defects with near-zero false alarms. Feeds stage 3. | Build |
| 11 | **Depth-aware lens signatures and synthetic aperture** | Use depth to vary blur by distance: stronger/softer background blur, closer "render like my lens". Label as synthetic. (cf. Wadhwa et al. 2018.) | Build after 3 |
| 12 | **Defocus deblurring** | The two halves have mirrored blur kernels, which makes deblurring out-of-focus areas far better posed than from one image (Abuolaim & Brown 2020). Rescue a slightly missed focus. | Research, then build |
| 13 | **Reflection removal** | A reflection in glass sits at a different depth than the subject, so it shifts differently between halves (Punnappurath & Brown 2019). Shooting through windows. | Research |
| 14 | **3D output** | Wigglegram GIF, red/cyan anaglyph, side-by-side stereo, depth map export (16-bit PNG) for Blender/Photoshop. Fun for the group. | Build |
| 15 | **Focus-stack assist** | Depth maps help order and blend focus brackets shot in DPRAW (section 12). | Later |
| 16 | **Lens measurement** | Per-half vignetting and aberration reveal pupil shape and decentering; add to the lens wizard report. | Later |

Calibration Part C frames are the test set. Every Dual Pixel feature must degrade gracefully on a normal raw (hidden or disabled with a one-line explanation).

### 11.1 First measurements on a real Dual Pixel file (Bill's eagle test, 2026-11-14)

File: EOS R5, RF 800mm f/11, ISO 1600, 1/400 s, split with DPRSplit into two DNGs. Measured on the raw data.

Confirmed:
- **DNG-1 is A+B, DNG-2 is A alone** (mean levels differ by 2.007×). B = DNG-1 − DNG-2 works. Black level 2048 on both.
- **The noise-versus-detail detector (item 4) works on a real file.** In clean sky the fine-scale fluctuations of A and B are uncorrelated (0.02): pure noise. On the eagle's feathers they agree at 0.96: real detail. This is a per-image, per-region measurement no single-image tool has.
- **The halves have strong, opposite shading.** At f/11 the B/A brightness ratio runs from 2.7 on the left third of the frame to 0.31 on the right third, with a colour cast. Any use of a single half, or any unequal mix of the halves, shows as a left-right brightness and colour gradient. **New requirement: flat-field each half (per lens and aperture, measured in the lens wizard or estimated from the image) before any Dual Pixel feature uses it.** Slow lenses are the worst case.

Negative result, to be stated honestly in the UI:
- **Stacking the two DPRSplit DNGs does not add resolution.** DNG-2 is already contained in DNG-1, and on an in-focus subject the halves show no measurable offset (under 0.1 pixel on the eagle), so there is nothing new to combine. The exposure-matched average of the two DNGs equals (3A + B)/4, which measured about 14% noisier in sky than the normal file and lower in detail-to-noise on the feathers. A straight average without matching brightness (what Bill did) equals (2A + B)/2: by calculation about 5% noisier than the normal file and 25% darker, with the same left-right shading cast. Either way nothing is gained.
- Visible differences between such a stack and other renderings come from the processing applied to each file (tone, sharpening, denoise), not from captured detail. The same raw data through DxO was clearly sharper from the CR3 than from the DNG, which shows how much the processing path matters. Lesson for this project: always compare methods on the same processing path.

Where Dual Pixel can truly sharpen a subject: when it is slightly out of focus, the halves are offset, and re-aligning them before summing removes a real blur (item 7). Build the offset map first and report "in focus: nothing to gain" or "offset found: gain available".

Where more resolution on a distant subject comes from instead: a handheld burst through Super-Res Stack (section 13), lucky-region selection for air shimmer, and deconvolution of diffraction (section 10.5). At f/11 on the R5 the diffraction blur is about 3.4 pixels wide and is known exactly from the f-number, so it is an unusually well-posed case; add **"diffraction recovery" as an automatic option whenever the aperture makes diffraction the limit.**

### 11.2 Lab experiment: Dual Pixel at extreme range (Bill's observation)

Bill's field conclusion from earlier tests: Dual Pixel RAW gave a visible resolution gain only when the subject was at the extreme range of the camera. The eagle file does not show it (halves aligned to under 0.1 pixel), and his original controlled files are not available, so this is **unconfirmed but physically plausible**, and worth a proper test.

Hypothesis: over a long air path, turbulence tilts the light differently across the lens opening. The two half-pixel images look through the left and right halves of the lens, so each is shifted by a slightly different, locally varying amount. The normal image (A+B) is then the sum of two slightly misregistered pictures, which is a real blur. Measuring the local A-to-B offset and re-aligning the halves before summing would remove that part of the blur. Astronomers measure air steadiness the same way, by comparing image motion through two sub-apertures (a differential image motion monitor). A short shutter speed freezes the effect, which is the normal case for long-lens work.

Build (Lab, Experimental):
1. Dense local offset map between A and B (after per-half shading correction). Separate the smooth, depth-related part (defocus) from the random, patchy part (air).
2. "Long-range re-align": locally warp the halves into register, then sum.
3. **Air steadiness readout** from any Dual Pixel frame: a number and a map. Useful on its own for choosing lucky frames in a burst (section 13), even if the sharpening gain turns out small.

Test protocol (needs a true control this time): tripod, distant detailed target across a long stretch of ground on a day with visible heat shimmer and again on a still, cool morning; for each, a burst in Dual Pixel RAW and a burst in normal raw, same settings, seconds apart; keep every raw and the file numbers. Compare from the raw data on the same processing path. Expected result if the hypothesis holds: offsets and gain on the shimmer day, none on the still day, none on near subjects.

---

## 12. Focus Stack

Separate workspace. In: a set of raws at different focus distances. Out: one all-sharp image that goes straight back into Develop as a 32-bit linear intermediate (plus its own sidecar), so every raw tool except stages 1–10 still applies.

1. **Detect and order** frames by focus distance (metadata when present, else by measured focus content).
2. **Raw prep per frame:** stages 1–10 with identical settings; single-shot denoise each frame first (cleaner focus decisions, cleaner result).
3. **Align:** focus breathing changes magnification, so fit scale + shift + rotation (+ slight perspective) per frame, coarse-to-fine, sub-pixel (ECC or phase correlation with a similarity model). Equalise brightness between frames.
4. **Merge methods** (user picks; default auto):
   - **Depth-map method:** per-pixel choice of the sharpest frame, smoothed edge-aware. Best colour and noise, clean for simple subjects.
   - **Pyramid method:** Laplacian-pyramid maximum-contrast fusion. Best for hairs, bristles, overlapping detail; can raise noise and glare, so apply noise-aware weighting.
   - **Auto:** pyramid where structure overlaps, depth-map elsewhere.
5. **Fix-ups:** halo suppression at depth edges; **retouch brush** to paint from any single source frame; ghost handling for subjects that moved (wind on a leaf).
6. **Extras:** slabbing for very large stacks; depth map export; stack preview while frames load; batch several stacks from one folder using the Library's auto-grouping.
7. **Acceptance:** Bill's 90mm Voigtländer macro stacks (`D:\onedrive\Best Shots Folder\90mm Voigtlander` — ask for access) merged with no visible halos at 100%, compared blind against his current stacking software.

---

## 13. Super-Res Stack (handheld bursts, the moon, the sky)

Separate workspace. This is the one place multi-frame averaging lives; it does not weaken the single-shot rule for the denoiser.

**Idea:** a handheld burst lands the image on slightly different sub-pixel positions in every frame. Each frame samples the scene on a different grid. Combine them on a finer grid and you recover real detail a single frame cannot hold, and noise falls by roughly the square root of the number of frames used.

Pipeline:
1. **Raw prep per frame** (stages 1–6; keep the mosaic).
2. **Rank frames** by sharpness; drop blurred ones.
3. **Align** to a reference: global shift + rotation, then local (tile-based or dense) sub-pixel alignment for hand shake, rolling effects and, for the moon, atmospheric wobble.
4. **Reject** what does not match (moving things, misaligned tiles) with a robustness map.
5. **Merge onto a finer grid** directly from the raw mosaics, so every output pixel gets true red, green and blue samples and **no demosaic is needed**:
   - **Drizzle** (Fruchter & Hook 2002, built for Hubble's undersampled dithered frames): each input pixel is shrunk and "dropped" onto the output grid at its measured position, with weights. Choose output scale 1.5× or 2×.
   - **Kernel-regression merge** (Wronski et al. 2019, "Handheld Multi-Frame Super-Resolution"): edge-adaptive kernels and per-pixel robustness; better on general scenes with motion.
   - Implement Drizzle first (simpler, well-defined), then the kernel method; keep the better per mode.
6. **Finish:** deconvolution with the combined PSF (section 10.5), then hand off to Develop as a linear intermediate.

Modes:
| Mode | Tuned for |
|---|---|
| **Handheld scene** | 8–30 frame burst of a static subject; strict motion rejection; 1.5–2× output. |
| **Moon / planets ("lucky imaging")** | Many frames (50–1000s, also from video); keep only the sharpest 10–30%; multi-point local alignment because air turbulence warps each frame differently; drizzle; wavelet (starlet) sharpening and deconvolution. Crop-to-subject so files stay small. |
| **Night sky** | Star-based registration; sigma-clipped averaging to remove planes, satellites and hot pixels; optional darks/flats from the camera profile; gradient (light-pollution) removal. |
| **Noise-only** | Same resolution out, just cleaner: tripod or handheld, for when the subject held still. |

Honest limits to show in the UI: gain depends on lens sharpness and real sub-pixel motion (a tripod with no movement gives noise reduction but no extra resolution); 2× is a realistic ceiling; moving subjects fall back to single-frame quality in the moving areas.

Acceptance: Bill shoots a handheld 20-frame burst of a detailed static target and a moon burst; result at 2× shows resolved detail absent from the best single frame, judged at 100% side by side, with no ghosting.

---

## 14. Research track: NASA, astronomy, MRI and signal-processing techniques

Bill asked for this investigation. Each row is a hypothesis to test on his files, not a promise. Record outcomes in `docs/RESEARCH.md`. Verify every citation first.

| Technique | Where it comes from | Use here | Plan |
|---|---|---|---|
| **Drizzle** (variable-pixel linear reconstruction) | Hubble Space Telescope, STScI; Fruchter & Hook 2002 | Super-Res Stack merge | Adopt (13) |
| **Richardson–Lucy deconvolution** | Used heavily on Hubble images before the 1993 optics fix | Capture sharpening with lens PSF | Adopt (10.5) |
| **Field-varying PSF modelling** | Hubble/JWST PSF models (Tiny Tim, WebbPSF) | Lens wizard stores PSF by image zone; deconvolve per zone | Adopt |
| **Bias / dark / flat calibration, bad-pixel masks, non-linearity correction** | Every space and ground telescope pipeline | The calibration wizard and stages 2–6 | Adopt (7) |
| **1/f striping removal with reference pixels** | JWST near-infrared detectors | Row/column banding from masked border pixels | Adopt (8.2) |
| **Cosmic-ray rejection by Laplacian edge detection** | van Dokkum 2001 (L.A.Cosmic) | Single-frame hot pixel and impulse removal | Adopt (8.3) |
| **Variance-stabilising transforms for photon noise** | Astronomy/microscopy statistics (Anscombe; Foi; Mäkitalo) | Core of the denoiser | Adopt (8.1) |
| **Starlet / à-trous wavelets, multiscale thresholding** | Starck & Murtagh, astronomical image processing | Multiscale denoise; moon sharpening; a candidate for microcontrast | Test |
| **Lucky imaging / shift-and-add** | Ground-based high-resolution astronomy | Moon/planet mode | Adopt (13) |
| **Sigma-clipped, noise-weighted stacking** | Standard astronomy co-adding | Night-sky and noise-only modes | Adopt (13) |
| **PSF from stars; blind deconvolution** | Astronomy | Auto-sharpen astro shots; measure a lens from a star field | Test |
| **Astrometric distortion solutions (plate solving)** | Astrometry | Optional lens distortion calibration from a photo of stars, extremely precise | Test, later |
| **Background / gradient modelling** | Astronomy sky subtraction | Light-pollution gradient removal; estimating falloff from flat-ish images | Test |
| **MP-PCA patch denoising (random-matrix threshold)** | MRI (Veraart et al. 2016) | Denoiser candidate b | Test (8.1) |
| **Non-local means with correct noise statistics** | MRI (Rician-noise NLM) | Baseline comparison | Test |
| **Cyclostationary analysis** | Bill's RF background | Banding detection and removal | Adopt (8.2) |
| **Noise2Noise / self-supervised denoising** | Microscopy, MRI, astronomy | Dual Pixel halves as a noise pair; camera-specific training | Test (8.1d, 11.4) |
| **Maximum-entropy / Bayesian reconstruction** | Radio and X-ray astronomy | Alternative deconvolution | Low priority |
| **Up-the-ramp sampling** | JWST/Hubble infrared detectors | Not applicable: needs repeated non-destructive sensor reads a camera does not give | Skip, documented |
| **Adaptive optics / wavefront sensing** | Telescopes | Hardware; not applicable. (Dual Pixel depth is the nearest cousin and is used in 11.) | Skip, documented |

Method for each test: implement on a branch → run the benchmark harness → blind A/B for Bill on three of his images → keep, tune or drop → write the result down.

---

## 15. Export and integrations

- **Export dialog + presets:** format (16-bit TIFF, 8/16-bit PNG, JPEG with quality; later JPEG XL / AVIF / HDR as options), colour space (Adobe RGB default; sRGB, ProPhoto, Display P3, custom ICC), size (long edge, megapixels, percentage, print size at dpi), output sharpening, metadata (all / copyright only / none; always strip serial and GPS unless the user opts in), watermark, naming template, destination. Presets such as "Facebook group", "Full-size TIFF", "Print 16×20".
- **Export queue** in the background, with a results list that opens the folder.
- **Soft-proof** with a printer/paper ICC and out-of-gamut warning. Simple print layout later.
- **Send to Photoshop:** export a 16-bit TIFF to a working folder beside the raw and open it in Photoshop. **Photoshop UXP panel** (Bill has the UXP Developer Tools): lists the Studio's current selection, "Open from Studio", "Re-develop this layer" (round-trips to the engine through `studio-cli` or a localhost-only bridge). Local only, no network beyond 127.0.0.1.
- **Send to PhotoCraft:** read PhotoCraft's actual input method from its repository before designing; default to file hand-off.
- **ATK Digital Forensics Workshop:** the engine is importable as a package; expose the Raw Analyzer and a read-only viewer there. No ATK code inside this repo.
- **`studio-cli`:** develop/export/stack from a `.bat` for batch jobs and tests.

---

## 16. Community and shared profiles

- Profile browser inside the app: search camera/lens, see quality score and contributor credit, one click to install. Source: the GitHub repo's `profiles/` folder (downloaded only when the user asks).
- Submit flow: the wizard produces the shareable file → shows it in full → the user chooses to save it for manual submission (pull request or an upload form later). Nothing is sent silently.
- Validation script run in CI on every submitted profile: schema, plausible ranges, no embedded identifiers.
- Merging: several users' profiles for one model combine into a model-level profile with per-ISO spread shown.

---

## 17. Quality, testing and benchmarks

- **Unit tests** on synthetic data with known truth: noise model recovery, VST round-trip, banding removal (inject known bands), defect detection, CA estimation (inject known shift), alignment accuracy (known sub-pixel shifts), drizzle resolution gain (synthetic undersampled target), focus-stack correctness (synthetic depth scene), Dual Pixel disparity (synthetic split-aperture render).
- **Golden-image regression tests:** a small set of renders compared against stored hashes/metrics so updates cannot silently change the look.
- **Benchmark harness (`tools/bench`)**: runs every denoiser candidate on (a) synthetic noise from each camera profile added to clean ISO-100 raws, reporting PSNR / SSIM / a perceptual metric / residual-structure score (is anything but noise being removed?), and (b) Bill's real high-ISO files with his DxO JPEGs alongside (`Voigtlander/`, `Zeiss/`, `high iso raw/` — local only). Produces a flashy HTML report with zoomable crops.
- **Blind test tool for Bill:** shows two unlabeled results side by side at 100%, he clicks the better one, the tool tallies. This is the final judge for denoise, microcontrast, stacking and signatures.
- **Named tests that must always pass:** candle flame survives hot-pixel removal · catchlights survive · purple flower survives defringe · low-key portrait stays low-key under Auto · no banding introduced on a clean frame · no halos from microcontrast at maximum · sidecar from version N renders identically in version N+1.
- **Named tests for card import (24.9, in `tests/test_ingest.py`):** the card is never written to (whole tree hashed before and after, once read-only) · a second import copies nothing · same name with different content keeps both · a `.partial` left by a crash is cleaned and copied again · a read error on one file does not stop the others · a card pulled mid-copy leaves no half-written file · too little space blocks the start · raws-only leaves the JPEG of a pair behind and says so · a card with no raws says so · several days make several folders and 02:00 belongs to the day before · unsafe folder names are made safe · a deleted photo is not imported again unless asked · the second copy is identical · no serial number in the log or the import record.
- **Performance budget** on Bill's PC, tracked per release: open-to-first-view under 1 s (embedded preview), slider response under 50 ms, 45 MP full denoise + export under 30 s on GPU and under 3 min on CPU, 8 GB RAM machine never crashes.
- **`Run Tests.bat`** runs the lot and prints a green/red summary.

---

## 18. Build phases

Each phase ends with a double-clickable deliverable and its acceptance test. Do them in order.

| Phase | Deliverable | Acceptance |
|---|---|---|
| **0. Foundation** | Repo layout (2.2), `Install.bat`, logging, test runner, `THIRD_PARTY.md`, commit the existing untracked work, fix true-ISO reading in the analyzer (section 4). Stack confirmed with Bill. | `Run Tests.bat` green on Bill's PC. |
| **1. Decode + develop engine** | Stages 1, 2, 10, 11, 15 (basic), 21. `studio-cli` turns a CR2/CR3 into a correct 16-bit Adobe RGB TIFF. | Colour and tone on R5, 5D IV and 6D files look right next to the camera JPEG; black/white levels verified against the analyzer. |
| **2. Calibration engine** | Camera profile builder from the checklist frames (headless), including Quick (dark-frame-only) profiles and Part G exposure latitude; profile format; analyzer upgraded to read/write profiles. | Bill's three bodies profiled; noise curves plotted; profile predicts measured noise on held-out frames within 5%. |
| **3. Denoiser v1** | Stages 3–7 with candidates a–c, benchmark harness, blind test tool. | Beats the BM3D baseline on the harness; Bill prefers it over his DxO output on at least half of ten blind pairs. If not, iterate here before moving on. |
| **4. Studio shell** | PySide6 app: Library (browse, rate, cull, compare), Develop viewport with loupe and before/after, sidecars, history, Clean + Light + basic Colour panels, Export. Simple/Full switch. | Bill edits and exports a real shoot end to end without help. |
| **4b. Library + card import** (added 2026-11-15, **do next**) | Section 24, in four steps (24.9): card import with the Photos folder and dated subfolders, Library speed, Review view, Process picks. | Bill goes from card to exported picks without opening Windows Explorer; the card is never changed; a 2,000-raw folder opens without stalling. |
| **5. Lens** | Automatic CA and defringe, lensfun bridge, distortion/falloff/perspective/crop (stages 8, 13, 14), lens tagging. | Fringing on Bill's vintage-lens files fixed by one button with no colour damage; named tests pass. |
| **6. Full develop** | Complete Light and Colour (9.1, 9.2), microcontrast (9.3), detail (9.4), presets/looks, copy/sync, virtual copies, batch export. | Microcontrast passes Bill's blind test against DxO's; "Bill's Candlelight" look ships. |
| **7. Wizards** | Calibration wizard (7.2 requirements, three levels) and lens wizard in the GUI, Exposure Advisor card (7.1), printable checklists, profile browser, share flow, capture sharpening from PSF (10.5). | A non-technical tester calibrates a camera from the printed checklist with no help. |
| **8. Dual Pixel Lab** | Section 11 items 1–10 and 14. | +1 stop highlight recovery demonstrated on the candle frames; depth mask selects subject on Part C frames; split/export matches DPRSplit. |
| **9. Local + retouch** | Masks (9.5) including depth and focus ranges, heal/clone/dust (9.6). | Bill selects one leaf in two clicks and adjusts it; dust removed automatically. |
| **10. Focus Stack** | Section 12. | Section 12 acceptance. |
| **11. Super-Res Stack** | Section 13: handheld and noise-only first, then moon, then night sky. | Section 13 acceptance. |
| **12. Signatures** | Render like my lens (10.6), first six signatures, Dual Pixel items 11–12. | Blind test: Heliar signature applied to the Sigma file is preferred over the plain Sigma file. |
| **13. Integrations** | Photoshop send + UXP panel, PhotoCraft, ATK hook, soft-proof/print. | Round trip to Photoshop and back works. |
| **14. Every brand + release** | Other makers' raws, X-Trans demosaic, model-level shared profiles, portable ZIP release, user guide, first public release to the group. | Ten volunteer testers on five brands install and export without help. |
| **15. Research backlog** | Section 14 "Test" rows, denoiser candidate d, joint denoise-demosaic, defocus deblur, reflection removal, HDR bracket merge, panorama stitch. | Each: keep or drop, written down. |

After Phase 4 the app is usable daily. After Phase 6 it replaces a paid raw editor for most work. Everything after is what makes it unlike anything else.

---

## 19. Risks and honest notes

- **Scope.** This is a multi-month build even with an AI doing the coding. The phase order is chosen so it is useful early and so the hardest promise (the denoiser) is proven in Phase 3 before a GUI is built around it.
- **Beating DxO** is the ceiling, not a given. The edge here is exact per-camera calibration and raw-domain processing; the plan keeps measuring instead of assuming.
- **CR3 internals are not officially documented.** Dual Pixel extraction and some metadata rely on community reverse-engineering; isolate that code and test it per camera model.
- **Python speed.** If the interactive budget cannot be met in Phase 4, port the hot kernels (not the app) to native code or GPU shaders. The engine/interface split makes this possible without a rewrite.
- **Vintage-lens auto-correction** can fix colour fringing and lateral CA reliably without a profile. Distortion without a profile is only safe on scenes with straight lines.
- **Lens signatures** reproduce tonal character and glow, not true optics. Say so in the UI.
- **Super-resolution** needs real sub-pixel motion and a lens that out-resolves the sensor; gains on soft vintage glass will be mostly noise reduction.
- **Support load.** With 150,000 potential users, the "Copy diagnostic report" button, the named tests and the profile validator are not optional extras.

---

## 20. Open questions for Bill (ask in chat, one at a time, when the phase needs it)

1. Phase 0: confirm the Python + PySide6 stack, or the stack from the Nov 11 plan if it differs.
2. Phase 2: have the calibration frames (checklist Parts A–D) been shot? Phase 2 cannot finish without Part A and B.
3. Phase 4: sidecars beside the raws, or in one central folder (your archive is on external drives)?
4. Phase 10: which focus-stacking software do you use now (the comparison yardstick), and may the builder read the 90mm Voigtländer folder?
5. Phase 11: do you shoot the moon as stills, video, or both?
6. Phase 14: where should group members download releases and submit profiles (GitHub only, or something simpler)?
7. Phase 4b: three builder's defaults in 24.2 stand unless Bill says otherwise: the folder date is the day the photos were taken, a new day starts at 04:00, and folders are grouped by year.

---

## 21. Recent research review (2024–2026), added 2026-11-14

A literature search across each area of the plan. Every item below was found and read at abstract level (not reproduced, not benchmarked). Before using any code or dataset, check its licence against rule 0.3.1 and record it in `THIRD_PARTY.md`. Before using any method, run it through the section 14 test loop.

**Rule (adopted, see 1.2):** many 2025–2026 papers use diffusion/generative models that *invent* plausible detail. This tool's promise is to recover what the sensor recorded. Recommendation: generative methods are never on by default, never used in the denoiser or the stacking tools' default paths, and are labelled "synthetic" wherever offered.

### 21.1 Changes this review makes to the plan

| # | Change | Affects |
|---|---|---|
| 1 | **Black level is a first-class accuracy target.** A July 2026 paper traces the colour casts that appear in very-low-light raw denoising mainly to small black-level errors. Stage 2 must estimate black level per frame, per channel (optical-black pixels + a residual estimate from the image), and the benchmark must report shadow neutrality. New named test: *deep shadows in a candlelit frame stay neutral after denoise.* | 3 (stage 2), 8, 17 |
| 2 | **Add a "Quick calibration" path: dark frames only.** Sony Research showed a noise model good enough to train a strong raw denoiser can be built from dark frames alone, skipping gain calibration. So: Part A of the checklist alone gives a usable profile (violet/cyan tier "quick"); Parts B–C upgrade it to "full". This cuts the barrier for group members sharply. | 7, 16 |
| 3 | **Iterate noise estimation.** Estimate noise → rough denoise → re-estimate from the residual → final denoise (coarse-to-fine, from YOND). Applies to the classical path and is the default for uncalibrated cameras. | 8.1 |
| 4 | **Check the variance-stabilising transform for bias at very low signal.** YOND reports that the standard inverse leaves an expectation bias that shows up as wrong colour, and corrects it from the image. Test our exact-unbiased inverse against this at ISO 25600+ and adopt the correction if it measures better. | 8.1 |
| 5 | **RawNIND as optional training/benchmark data** for the learned denoiser (8.1d) and for cameras nobody has calibrated: paired clean/noisy raws from many sensors, Bayer and X-Trans, openly published. Use only if its licence is compatible. Their result that a *linear-RGB* denoiser generalises across sensors better than a Bayer one is the fallback design for unknown cameras and non-raw files. | 8.1d, 4 |
| 6 | **Deconvolve to a target sharpness, not to infinity.** Astronomy's STARRED/Firedec line deconvolves to a chosen narrower PSF instead of trying to remove the blur completely, which avoids ringing. Adopt as the default behaviour of "Recover lens sharpness". Also test more than Richardson–Lucy: a 2024 JWST study compared five deconvolution methods and RL was not the winner. | 10.5, 14 |
| 7 | **Banding removal follows the JWST recipe:** mask the subject first, estimate the stripe pattern from what is left (median per row/column, per readout channel), optional FFT-based fit, and be explicit that a uniform background and striping cannot be told apart. JWST's own notes show it over-subtracts where real signal is not masked; our gate-by-calibrated-amplitude rule stays. | 8.2 |
| 8 | **Adopt the ECAS metric** (Edge Chromatic Aberration Score) from the first deep-learning purple-fringe paper as our objective fringing measure in the benchmark. | 10.2, 17 |
| 9 | **Blind lens correction for unknown lenses has a research base now** (OmniLens / OmniLens++): a universal model trained on a large simulated lens library, then adapted to one specific unknown lens from a few real photos. This is the vintage-lens problem exactly. Add as a candidate behind "Fix lens" and capture sharpening; keep it optional for character lenses. Their "adapt from a few photos" step maps onto our lens wizard. | 10.2, 10.3, 10.5 |
| 10 | **Focus Stack: add an uncertainty map and gap detection.** Recent fusion work treats pixels that are sharp in *no* frame, or that sit on a near/far boundary, separately (these cause halos), and notes that real stacks often have missing focus steps. Detect both, show them, and steer the retouch brush there. StackMFF is a candidate third merge method for large stacks. | 12 |
| 11 | **Super-Res Stack: add an "HDR burst" mode** (bracketed handheld burst → more resolution *and* more dynamic range in one merge, Lecouat et al.), and build from the published reference implementation of the Google handheld method (IPOL, "Implementing Handheld Burst Super-Resolution") instead of from the paper alone. | 13 |
| 12 | **Moon mode: pick the reference frame carefully and score "lucky" regions on two things**, geometric consistency and sharpness, not sharpness alone (Chan group, Purdue). Lucky *regions*, not whole frames. | 13 |
| 13 | **Dual Pixel: use the published physical model of the two half-aperture blur kernels** (they are mirror images that grow with defocus) for depth and deblurring, and use ray-traced synthetic dual-pixel data for unit tests. Explicit cross-correlation between the halves, which the plan already uses for depth, is also what a 2025 deblurring paper found worked. | 11, 17 |
| 14 | **Joint demosaic + denoise: use "mask-out" training** so the same network also repairs dead pixels (Tedla et al. 2025). | 8.1 (research track) |

### 21.2 Papers by area

**Raw denoising and noise modelling**
- Li, Jiang, Iso — *Noise Modeling in One Hour: Minimizing Preparation Efforts for Self-supervised Low-Light RAW Image Denoising*, CVPR 2025. Code: github.com/SonyResearch/raw_image_denoising. → changes 2.
- Feng et al. — *YOND: Practical Blind Raw Image Denoising Free from Camera-Specific Data Dependency*, arXiv 2506.03645 (2025). → changes 3, 4; design for the uncalibrated tier.
- Mohammadi et al. — *Why Low-Light Cameras Go Color Blind: Removing Color Bias in Raw Denoising*, arXiv 2607.11090 (July 2026). → change 1.
- Brummer, De Vleeschouwer — *Learning Joint Denoising, Demosaicing, and Compression from the Raw Natural Image Noise Dataset* (RawNIND), arXiv 2501.08924 (2025). → change 5.
- Liu et al. — *Blind2Sound: Self-Supervised Image Denoising without Residual Noise*, ICCV 2025. Learns from single raw noisy images under a Poisson-Gaussian model and adapts strength to the sensed noise level. → candidate for 8.1d.
- Li et al. — *Positive2Negative*, CVPR 2025. Self-supervised single-image denoising without masking/downsampling losses. → candidate for 8.1d.
- *SNIC: High-Fidelity Noise Modeling and Calibration*, arXiv 2512.15905. Shows manufacturer/DNG noise profiles are a poor match to real noise; supports doing our own calibration. → section 7 rationale.
- *Learning Physics-Informed Noise Models from Dark Frames*, arXiv 2310.09126; *Dark Noise Diffusion*, TPAMI 2025. Noise synthesis from dark frames, including fixed-pattern components. → 8.1d training data; low priority.
- AIM 2025 Challenge on Real-World RAW Image Denoising: multi-camera benchmark with dark frames and gains provided. → external benchmark for section 17 (verify availability).
- Tedla et al. — *Examining Joint Demosaicing and Denoising for Single-, Quad-, and Nona-Bayer Patterns*, arXiv 2504.07145 (2025); *JD3Net*, arXiv 2601.00703. → change 14.

**Dual Pixel**
- Swami — *Adjust Your Focus: Defocus Deblurring From Dual-Pixel Images Using Explicit Multi-Scale Cross-Correlation*, arXiv 2502.11002.
- *Revisiting Disparity from Dual-Pixel Images: Physics-Informed Lightweight Depth Estimation*, arXiv 2411.04714.
- *Simulating Dual-Pixel Images From Ray Tracing For Depth Estimation*, arXiv 2503.11213.
- Punnappurath et al. — *Modeling Defocus-Disparity in Dual-Pixel Sensors*, ICCP 2020; Abuolaim et al. — *Learning to Reduce Defocus Blur by Realistically Modeling Dual-Pixel Data* (2021); Pan et al. — *Weakly-supervised depth estimation and image deblurring via dual-pixel sensors*, TPAMI 2024.
- Confirmed from section 11: Abuolaim & Brown, ECCV 2020; Punnappurath & Brown, CVPR 2019 (reflection removal); Garg et al., ICCV 2019 (depth); Wadhwa et al. 2018 (synthetic depth of field).

**Burst super-resolution**
- Wronski et al. — *Handheld Multi-Frame Super-Resolution*, ACM TOG 38(4), 2019 (confirmed), and its open reference write-up *Implementing Handheld Burst Super-Resolution* (IPOL).
- Lecouat et al. — *High Dynamic Range and Super-Resolution from Raw Image Bursts*, ACM TOG 2022. → change 11.
- *Handheld Burst Super-Resolution Meets Multi-Exposure Satellite Imagery*, arXiv 2303.05879.
- Khan et al. — *MFSR-GAN*, arXiv 2502.20824 (2025): a synthetic data engine that preserves sensor noise and real handheld motion. Use the data-engine idea for tests; the GAN itself falls under the generative rule.
- Burst SR with one-step diffusion, arXiv 2507.13607: generative. Not for default paths.

**Focus stacking**
- Xie et al. — StackMFF series and *Generative Multi-Focus Image Fusion* (GMFF), arXiv 2512.21495. Code: github.com/Xinzhe99/StackMFF-Series. Deterministic stage is a candidate; the generative stage falls under the generative rule.
- Quan et al. — *Multi-Focus Image Fusion via Explicit Defocus Blur Modelling* (DMANet), AAAI 2025. → change 10.

**Lens correction**
- Jiang et al. — *OmniLens*, arXiv 2409.05809; *OmniLens++*, arXiv 2511.17126. Code: github.com/zju-jiangqi/OmniLens2. → change 9.
- Lu et al. — *DCA-LUT: Deep Chromatic Alignment with 5D LUT for Purple Fringing Removal*, arXiv 2511.12066 (2025). → change 8; method is a candidate.

**Astronomy / NASA**
- STScI JWST documentation — *1/f Noise* and *NIRCam 1/f Noise Removal Methods* (`clean_flicker_noise` step, jwst pipeline 1.16.0+). → change 7.
- Leist et al. — *Deconvolution of JWST/MIRI Images*, Astronomical Journal 167:96 (2024). → change 6.
- Millon et al. — *STARRED: wavelet-based two-channel deconvolution and PSF reconstruction*, arXiv 2402.08725. Also separates point sources (stars) from extended detail: useful for night-sky mode. → change 6.
- Akhaury et al. — multiband deconvolution with deep denoising, A&A 697, A136 (2025). Low priority.
- Chimitt, Mao, Hong, Chan — *Rethinking Atmospheric Turbulence Mitigation*, arXiv 1905.07498; Mao et al., arXiv 2009.00071. → change 12.
- Kim et al. — *AstroDiff*, arXiv 2506.02981: generative turbulence removal. Falls under the generative rule.

**Highlights**
- No new raw-domain method was found that beats channel-ratio reconstruction plus the Dual Pixel extra stop (11.2). Recent work is learned single-image HDR extension, which is generative.

### 21.3 Not found / still open
- No recent paper on **microcontrast as tonal separation** in Bill's sense. Section 9.3 (measure it from the same-scene lens test) remains original work.
- No published work on **Canon Dual Pixel RAW files specifically** for noise measurement from the A/B difference (11.4). Also original work.
- Nothing recent that changes the Drizzle recommendation.

---

## 22. Second research pass (2026-11-14): gaps the first pass did not cover

Same caveats as section 21: read at abstract/summary level, verify before use, check licences. Nine additions.

| # | Addition | Source | Affects |
|---|---|---|---|
| 1 | **Texture-preservation score using a "dead leaves" chart.** A printed chart of overlapping random discs is the camera industry's standard target for measuring how much fine texture a pipeline keeps (it became an ISO test). Add it to the benchmark, and offer it as an optional printable target in the calibration wizard. This gives the denoiser an objective "detail kept" number to go with "noise removed". | Cao et al. 2009; Achddou et al., *Hybrid Training of Denoising Networks to Improve the Texture Acutance of Digital Cameras*, arXiv 2404.07212 | 8, 17 |
| 2 | **Do not trust PSNR on fine texture.** A study built a test set of low-contrast, noise-like textures and found PSNR had near-zero correlation with what people preferred. Keep PSNR as a sanity check only; blind tests and the texture score decide. | Egiazarian et al., arXiv 1711.00693 (FLT database) | 17 |
| 3 | **Banding: correct each colour channel's rows separately, and use the masked edge pixels.** One sensor study cut read noise from 4.77 to 2.89 units just by subtracting each row's masked-pixel average. Row noise shows up as fast changes between neighbouring row averages, while real image content changes slowly from row to row; use that to tell them apart where masked pixels are too few. Model row noise as one random offset shared by a whole row. | arXiv 2312.13539; arXiv 2607.11090 (noise model) | 8.2 |
| 4 | **Hot pixels: subtract, do not just patch.** A hot pixel's dark response (an offset plus a slope that grows with exposure time) stays constant once it forms. Measure both in calibration and subtract the predicted excess, keeping the real signal underneath; fall back to interpolating from neighbours only for severe ones. Choose per pixel by severity, exposure time and ISO. Defects accumulate over a sensor's life, so the hot-pixel map must be refreshable, ideally updated quietly from ordinary photos. | Chapman, Thomas, Koren, Koren, *Improved image accuracy in hot pixel degraded digital cameras* | 3 (stage 3), 7, 8.3 |
| 5 | **Canon's own Dual Pixel guidance, to show in the Dual Pixel Lab.** Canon confirms the file holds A+B and A. Canon recommends at least 50mm, f/5.6 or wider, ISO 1600 or lower, and gives working distances (about 1–10 m at 50mm, 2–20 m at 100mm, 4–40 m at 200mm). Canon's developers also say the user must tell whether softness comes from focus (a Dual Pixel fix) or diffraction (a lens-optimiser fix). Our eagle measurement at f/11 agrees: no offset, diffraction-limited. The Lab should state this up front and steer f/8-and-slower files to diffraction recovery. Note this guidance concerns the focus-offset effect; the long-range turbulence hypothesis (11.2) is a different mechanism and still needs its own test. | Canon EOS 5D Mark IV Dual Pixel RAW pages and developer interview | 11, 11.1, 11.2 |
| 6 | **A second way to extract the halves:** published research on the 5D Mark IV converted Dual Pixel CR2 files with Adobe DNG Converter, then scaled and black-subtracted to get the two views. Use as a cross-check for our own extractor. | *Passive Snapshot Coded Aperture Dual-Pixel RGB-D Imaging*, arXiv 2402.18102 | 11 |
| 7 | **Auto white balance that works on a camera it has never seen**, by using that camera's own colour matrices to tell it where neutral light sources fall in its raw colours. Candidate for "auto colour" on every brand. | Kim et al., *CCMNet*, ICCV 2025, arXiv 2504.07959 | 9.2 |
| 8 | **Ringing-free deconvolution candidate:** an edge-preserving variant of Richardson–Lucy applied progressively across scales, designed specifically to suppress ringing. Test alongside plain RL for capture sharpening and diffraction recovery. | Yuan et al., *Progressive Inter-scale and Intra-scale Non-blind Image Deconvolution*, SIGGRAPH 2008 | 10.5 |
| 9 | **Veiling glare as two maps** (how much scene light gets through, and how much stray glow is added, varying across the frame and independent of distance). Use this decomposition when measuring vintage lenses for signatures: it is the "glow" part of a lens's character. The removal method itself uses a generative prior, so removal belongs in the Lab. | Qian et al., *Learning Latent Transmission and Glare Maps for Lens Veiling Glare Removal*, CVPR 2026, arXiv 2511.17353 | 10.6 |

Searched and found nothing that changes the plan: local tone mapping and microcontrast (recent work is learned tone-mapping networks; local Laplacian filtering remains the reference), demosaicing beyond section 21, highlight reconstruction.

**Considered and not applicable: time-folded optics** (Heshmat, Tancik, Satat, Raskar, *Photography optics in the time dimension*, Nature Photonics 2018). It shortens a lens by bouncing light between mirrors and picking out the right bounce by its arrival time, which needs an ultrafast time-resolving sensor and pulsed laser light. It is a hardware design; an ordinary camera file contains nothing it could be applied to. Documented here so it is not re-investigated.

**What it suggests for this project:** the software way to "mimic a longer lens" is already in the plan. Package it as one named preset in Super-Res Stack, **"Reach"**: handheld burst → sharpest-frame selection → 2× merge → diffraction recovery → crop. A 2× result is the practical equivalent of doubling the focal length for a static subject. State the limits in the UI (static subject, real hand movement between frames, lens sharp enough to benefit).

---

## 23. Third research pass and Bill's decisions (2026-11-15)

Same caveats as sections 21 and 22: verify every citation and every licence before use, record each in `THIRD_PARTY.md`, and run each method through the section 14 test loop. Everything here was decided by Bill in chat on 2026-11-15 unless marked "candidate".

### 23.1 New rule: more than one method is fine, if the user can choose and understand

Where several techniques do the same job and each wins on different photos, ship them all as a **Method** picker. Every option carries a short text beside it, in plain words: **Best for**, **Weak at**, **Speed**. One option is marked "Recommended" (chosen automatically from the photo where possible: ISO, lens, amount of clipping). Simple level shows only the recommended one; Full level shows the picker. The method used is written into the sidecar and colour-coded like every other result (rule 0.3.11). This applies to: denoise, demosaic, highlight recovery, HDR, stack merge, rejection, auto white balance.

### 23.2 Optional neural runtime (decided: yes)

- Add **ONNX Runtime** (MIT; verify) as an optional install, the way lensfunpy is optional. GPU through DirectML on Windows, CPU fallback. Off until the user turns it on; when off, nothing is loaded. No network use except a user-triggered model download that shows the file, its size, its licence and its source first.
- Models are never committed to the repo (rule 0.3.7). A `models/catalog.json` lists name, job, licence, source URL, SHA-256. Verify the hash after download.
- A model's **weights licence** is checked separately from its code licence (example: GraXpert code is GPL-3 but its models are CC BY-NC-SA, so they are **not** shipped or offered).
- Neural results are colour-coded as their own tier and labelled "learned". Generative models (ones that invent detail) stay in the Lab and are labelled "synthetic" (section 1.2 rule).

### 23.3 Denoiser candidates added to the ladder (8.1) and to the blind test (decided: all of them)

| Candidate | What it is | Licence (verify) | Notes |
|---|---|---|---|
| **RawNIND UtNet2 (Bayer)** | Raw-domain network, what darktable 5.6 ships | GPL-3.0 | Works on the mosaic. First learned baseline to beat. |
| **RawForge TreeNet** (light to heavy) + DeepSharpen | NAFNet-style networks trained mostly on RawNIND | MIT | Several sizes: offer by speed. |
| **SCUNet** (real-world blind) | RGB denoiser | Apache-2.0 | For JPEG/TIFF input and already-developed files, not for raws. |
| **MAXIM** (Google) | RGB denoiser | Apache-2.0 | Include for the comparison. Expected weak for us: trained on phone pictures that were already developed (SIDD), not raw data; very heavy on 45 MP; published as TensorFlow/JAX weights that must be converted. Keep only if it earns its place in the blind test. |
| **Zero-Shot Noise2Noise** (Mansour & Heckel, CVPR 2023) | A tiny two-layer network trained on the photo itself; no dataset, no noise model | method only, our code | Needs noise that is independent pixel to pixel: run on the mosaic planes after banding and hot-pixel removal. |
| **Learning to Recorrupt** (Monroy, Bacca, Tachella, arXiv 2603.25869, 2026) | Self-supervised training that does not need to know the noise distribution | method only | For cameras nobody has calibrated, and for the structured noise our model does not capture. |
| NL-Bayes (Lebrun, Buades, Morel 2013), dual-domain denoising (Knaus & Zwicker 2013) | Classical | methods only | Candidates beside the BM3D-class baseline (still to write). |

**"Do better than the published models" (candidate, the project's own edge).** Nobody else has a per-camera library of real dark frames, cold and hot. Use it:
1. **Camera-specific fine-tuning.** Take a general raw network (RawNIND or TreeNet architecture). Make training pairs from clean low-ISO raws: add photon noise from the calibrated gain, then add a **real dark frame sampled from this camera's own library** at the matching ISO, exposure time and temperature band (this carries the true read noise, banding and fixed pattern, which formulas miss; cf. Sony Research "Noise Modeling in One Hour", section 21). Fine-tune on the user's PC; result is a per-camera model stored with the profile. GPL-3 weights we train ourselves may be shared with the profile library.
2. **Temperature-matched everything.** Read the camera temperature from the maker notes where present (verify per brand); pick the dark library band (cold / warm / hot) for hot-pixel subtraction, banding and the training noise.
3. **Per-photo tuning.** Run Zero-Shot N2N or Learning to Recorrupt for a few seconds on the photo itself, starting from the camera model (test-time adaptation). On Dual Pixel files the A and B halves are a true noise pair in focused areas (11 row 4): use them as the training pair.
4. **k-sigma normalisation** (PMRID, Wang et al., ECCV 2020; verify) so one network serves every ISO: scale the raw by the calibrated gain and read noise before the network, invert after.
5. **Fusion by local signal-to-noise.** Blend candidates per region: learned model where it is confident, classical (MP-PCA / starlet) where the learned one smears low-contrast texture. Decide by the dead-leaves texture score and the blind test, never by PSNR alone (section 22).

### 23.4 Highlight recovery (stage 9)

Re-implement darktable's **ideas** in our own code (Bill: "their ideas, not their code entirely"; no source copied, cite the documentation):
- **By area** (darktable "segmentation based"): treat each clipped area as its own region, take colour from that region's own unclipped rim, join regions that belong to one object, rebuild fully clipped cores from the surrounding brightness gradients.
- **Detail transfer** (darktable "guided laplacians"): copy fine detail from channels that did not clip into the ones that did, and continue colour gradients inward, scale by scale. Aimed at lamps, speculars and candle flames.
- **Grain match:** add photon-like noise from the camera profile to rebuilt areas so they do not look smoother than their surroundings at high ISO. On by default at high ISO.
- **White level per ISO** measured in the calibration wizard: a wrong white point defeats every method.
- **Declipping, self-supervised** (Sechaud, Jacques, Abry, Tachella, TMLR 2026): a network learns to rebuild clipped values from clipped photos alone, by assuming pictures look statistically the same at any brightness. Train on the user's own raws (re-clip them lower to make examples). Lab first; promote if it beats the classical methods on candle and sun tests. Label "learned".
- **Already built, keep and combine:** "Rebuild colour" (`highlights.py`) and "Measured highlights from the lens glow" (`glow.py`, the NASA SDO/AIA solar method: the glow around a blown light tells how bright it really was). Use the glow measurement as the brightness target for the declipper and the detail-transfer method.
- **Dual Pixel +1 stop** (11 row 2) feeds all of the above.
- Method picker texts (23.1), for example: *Rebuild colour: fast, best for one blown channel, weak on big white-out areas. Detail transfer: best for lamps and flames, slow. Measured from glow: only method that knows the true brightness, needs a lens glow profile.*

### 23.5 Dual Pixel HDR (decided: build as an option)

"**Dual Pixel HDR**" in the Dual Pixel Lab and as a source for Auto HDR (23.6). A+B is the normal exposure; A alone received half the light, so it is a second exposure one stop darker taken at the same instant (no ghosting, ever).
- Merge: use A+B everywhere it is not clipped; fade to 2×A where A+B approaches clipping; noise-weighted blend in between.
- Required corrections first: per-half shading (A is darker toward one side of the frame; measure in Calibration Part C), and local A-to-B alignment in out-of-focus areas (the halves see from slightly different positions).
- Honest label in the UI: **about one extra stop in the highlights, nothing extra in the shadows** (A+B already holds all the light). Real gain depends on where the sensor itself saturates; measure per body in calibration and show the measured number, cyan tier.
- Only on files shot with Dual Pixel RAW turned on. Otherwise the option is hidden with a one-line explanation.

### 23.6 Auto HDR (decided: build; Bill's words: "I suck at balancing the dynamic range")

One button, then a strength choice. It never runs by itself (decision 0.4: no automatic brightness changes unless turned on).
- **Sources, picked automatically:** one raw; Dual Pixel raw (23.5); an exposure bracket; a burst (section 13 "HDR burst").
- **Engine A (default): exposure fusion** (Mertens, Kautz, Van Reeth 2007) of several virtual exposures made from the merged scene-referred data, as HDR+ does (Hasinoff et al. 2016). Cannot make halos by design. **Engine B: local Laplacian tone compression** (reuses `microcontrast.py`). Both offered under rule 23.1.
- **Auto part:** measure the scene's range from the raw histogram (robust percentiles of shadows and highlights), decide how many stops must be compressed, and set the lift and the pull-down separately. Shadows are denoised harder in proportion to the lift, using the calibrated signal-to-noise (noise-aware lift). Local contrast is kept by the microcontrast stage.
- **Strengths:**

| Option | What it does |
|---|---|
| **Light** | Holds the highlights, opens shadows a little. Looks like a careful single exposure. |
| **Medium** | Balanced; the default. |
| **Heavy** | Everything visible, flatter global contrast, strong local contrast. |
| **Highlights only** | Tames bright areas, leaves shadows exactly as shot. |
| **Low-key** | For candle and night work: protects flames and lamps, keeps blacks black, lifts only the lower mid-tones. Named test: a candlelit portrait stays a candlelit portrait. |
| **Shadows only** | Opens shadows, leaves highlights as shot (backlit subjects). |

  Each has an Amount slider (0–150%) and a short strengths / weaknesses text. Colour is handled so lifted shadows do not go grey and compressed highlights do not shift hue.
- **Bracket merge** in linear raw with noise-optimal weights (Granados et al. 2010; Hanji et al. 2020; verify), ghost rejection from the HDR+ robust merge.

### 23.7 Stacking

- **HDR+ align-and-merge** (Hasinoff et al. 2016; reference: Monod, Delon, Veit, IPOL 2021; article CC BY-NC-SA, check the code licence, re-implement from the paper if needed) as the default "Noise-only" merge: tile alignment, then a per-frequency merge that trusts a frame only where its difference from the reference is explained by the calibrated noise. Keep sigma-clipped mean as the other choice (23.1).
- **Burst exposure advice** in the UI: for a burst, expose for the highlights (slightly under); averaging cleans the shadows. State clearly that this is the opposite of Bill's single-frame Canon rule and why.
- **Pixel-shift merge** (Sony, Pentax) with motion masking.
- **Night sky:** rejection picker: percentile (up to 6 frames), sigma, winsorized sigma, linear-fit, MAD (for drizzled colour data); normalise frames before rejection; frame weighting by star sharpness; **rejection map** shown as a colour overlay (satellite and plane trails visible).
- **Freeze ground** (nightscapes on a fixed tripod, 4+ frames): find what moves with the stars and what stays still, align and stack the two separately, join along a soft mask (auto, with brush correction). Correct lens distortion **before** star alignment (wide lenses are where other stackers fail). No open-source tool does this well: a headline feature.
- **Background / light-pollution removal:** radial-basis-function surface fitted to sky samples (classical, no model), auto-sampled with the ground masked out.
- RCD demosaic by default for star fields.

### 23.8 Raw processing

- Demosaic picker: **RCD** (default), **LMMSE** (recommended automatically at high ISO), **dual demosaic** (detail method where there is detail, smooth method in flat areas), Markesteijn 3-pass for X-Trans. Our own implementations from the papers.
- **PDAF line filter** for Sony and Nikon Z striping (first outside tester shoots Sony), green equilibration, lateral CA correction on the mosaic before demosaic.

### 23.9 Auto white balance

- Add **Grayness Index** (Qian, Kämäräinen, Nikkanen, Matas, CVPR 2019): learning-free, finds the pixels most likely to be grey; can report more than one light source. Offered beside grey-edge (23.1); show both estimates and their agreement as a confidence.
- **Mixed light** (idea from Afifi, Brubaker, Brown, WACV 2022, no code used): render with two or three white balances and blend them with a smooth map built from the Grayness Index. For candle plus window light. Optional.
- **"Keep the mood" amount** on Auto: 100% = fully neutral, lower = partial correction toward the as-shot or a preferred rendering (research shows people prefer not-quite-neutral; Afifi et al., ICCV 2025). Default below 100% for warm low-light scenes so Auto no longer kills candlelight.
- **Skin-based estimate** (Bianco & Schettini 2014; verify) when faces are found, reusing the face and skin masks.
- Keep every estimate inside a sensible band around the daylight/tungsten curve unless the user overrides.

### 23.10 Where this goes in the phases

Phase 3 (denoiser): 23.2, 23.3 candidates into the blind test. Phase 6 follow-up: 23.4 classical parts, 23.6 Auto HDR on a single raw, 23.8, 23.9. Phase 8 (Dual Pixel Lab): 23.5. Stack phases: 23.7. Research backlog: declipping network, camera-specific fine-tuning, per-photo tuning. Phase 6 follow-up also takes 23.11 (Detail by scale) and the 23.12 safeguard and Retinex engine.

### 23.11 Microcontrast, revised: "Detail by scale" with a Balance control (decided 2026-11-15)

**Why.** Review of the built tool (`microcontrast.py`) on the detail-preview renders: safe (no halos, noise-aware, colour-safe) but too timid. At 80% it is not visible at screen size; measured gain is about +4 to +11% in shadows, and at high ISO the noise protection turns it off in the shadows while highlights get +17 to +27% (backwards). The strength cap (monotone limit, `A_MAX`) keeps it "subtle".

**What microcontrast means here (Bill's ruling).** Photographers disagree on the word (Hogan and others: pixel-level differences; Khong and others: richness of tonal transitions; several: clean shadows from low stray light). **Bill is on the tonal-transitions side**: more distinguishable shades, smooth and rich transitions, and **it shows when the photo is converted to black and white**. That is the default and the acceptance test. The other meanings are offered as controls, not argued about.

**Evidence that it is a balance, not an amount** (Bill's finished same-leaf JPEGs, Heliar vs Sigma 135 Art vs Canon 180 macro; weak evidence: focus stacks finished in Photoshop, light changed between lenses; not a deciding test, by Bill's instruction): the Sigma has the *most* contrast at every scale and the widest tonal span, yet the Heliar is the one people pick. What differs is the ratio of medium-scale tonal variation (about 4-8 px at 2600 px wide) to pixel-level contrast (1-2 px): Heliar 1.35, Canon 1.21, Sigma 1.06. Tonal variation concentrated at medium scales with relaxed pixel-level crunch reads as "modelled"; equal at both reads as "etched".

**Build: one panel, "Detail by scale".** It replaces the separate Texture, Clarity and Microcontrast sliders and owns their scales so they cannot fight each other; capture sharpening (10.5) stays a lens correction but the panel shows its effect on the Fine band.

*Simple level*
- **Amount** (0-100).
- **Balance**: left = **Tonal** (medium scales up, finest scale slightly relaxed), centre = both, right = **Crisp** (finest scale up). **Default sits on the Tonal side.**
- **Judge in black and white** toggle: shows the preview as a luminance-only image while adjusting (does not change the edit). This is how Bill sees microcontrast.

*Full level* (each slider can go negative; each has the rule 23.1 text: best for / weak at)
- **Fine** (about 1-4 px on a 6000 px frame): crispness. Negative = gentle softening, capped so it never looks blurred.
- **Medium** (about 8-90 px): shades and modelling. The existing local Laplacian engine.
- **Broad** (about 100 px and up): overall punch.
- **Veil**: shadow separation. Removes (or adds) a small uniform stray-light haze, measured per lens where a glow profile exists (10.6, section 22 row 9).
- Existing shadows / midtones / highlights weighting stays, shadows first.

*Presets* (rule 23.1): **Tonal**, **Crisp**, **Both** (sharp and tonal together: Bill's Zeiss Otus), **Soft glow**. Later: presets measured from real lenses in the signature library ("render like my lens", 10.6), stored as per-band gains plus veil.

**Methods are selectable and stackable (Bill, 2026-11-15: "every version gets a selection ... use the slider for one, and then a different one").** Every microcontrast method ships, each with its own on/off, its own Amount slider and its rule 23.1 text (best for / weak at). Any number can be on at once; the result is their combination. The Simple-level Balance and presets are just saved combinations of these sliders.

| Method | What it does | Best for | Weak at |
|---|---|---|---|
| **Tonal** (default; Bill's meaning) | Medium scales up, finest scale slightly relaxed | Modelling, shades of shadow, black and white | Can look soft on subjects that need bite |
| **Local contrast** (the engine built first) | Medium-scale differences made larger, edges left alone | General depth | Subtle; weak in noisy shadows |
| **Crisp** | Finest scale up | Fur, feathers, fabric, lenses that are a little soft | Raises grain; can look etched |
| **Tonal separation** | Spreads tones apart inside shadows and midtones region by region (no scale boost) | Flat, dull files; dark-on-dark subjects | Can look processed if pushed |
| **Veil removal** | Takes off the stray-light haze floor | Backlit and hazy scenes, uncoated lenses | Nothing to gain on clean modern lenses |
| **Broad** | Large shapes, overall punch | Flat light | Deepens shadows; watch clipping |

Rules for combining:
1. **Fixed order, whatever order the user moves the sliders:** Veil removal, then Tonal separation, then the scale methods (Broad, Medium-based, Fine-based) computed together in one pass from the same pyramid. Same sliders always give the same picture.
2. **One shared budget per scale.** Methods that touch the same scale add into a single gain for that scale, and the edge guard, the noise protection and the no-tone-reversal limit apply to the **total**, not to each method separately. Stacking three methods can never produce halos or amplify noise that one method alone could not.
3. **A live meter** in the panel shows the total change per band and the medium-to-fine ratio, colour-coded by which method contributed what, and turns amber when the total reaches a guard ("limited here to protect edges / noise").
4. The sidecar stores each method's amount, so a combination can be saved as a preset and shared.
5. Named tests: every pair of methods at 100% passes the no-halo and candle-flame tests; a stack of all methods at 100% never reverses tones.

This stackable pattern (own slider per method, shared guards on the total) extends rule 23.1 wherever combining is physically sensible: microcontrast, denoise (fusion in 23.3), highlight recovery (23.4). Where only one can apply at a time (demosaic, stack rejection), it stays a single pick.

*Engine changes*
- Replace the single scale bump (`level_weights`) with separate gains per band; negative gains allowed on Fine.
- Raise the usable strength: the effect at 60-80% must be plainly visible at screen size on a low-ISO file. Keep the no-halo and no-tone-reversal tests.
- Fix the high-ISO behaviour: the noise toe must use the noise left *after* the denoiser; shadows must receive at least as much as highlights when the denoiser is on. Named test: ISO 4000 R5 frame, denoiser on, shadows gain >= highlights gain.
- Fine band up uses the same noise protection as Medium.
- Report in the panel, colour-coded: change per band and the **medium-to-fine ratio** before and after (the balance number).

*Acceptance*: Bill's blind test, judged in black and white as well as colour. The controlled raw comparison (checklist Part E) is still wanted for the lens-signature library, but it no longer blocks this tool.

Sources read (definitions only, no code): Hogan, "What is Micro Contrast?" (dslrbodies.com); Khong, "Micro-contrast, the biggest optical luxury of the world" (2016); Griffin, "Micro-contrast: is it real?"; The Monochrome Collective, blog 328; Ice Cream Geometry, "Sharpness vs Microcontrast".

**Added from the Zeiss article (Lloyd Chambers, "Micro Contrast and the ZEISS 'Pop'", Lenspire, 2017; Bill supplied the text 2026-11-15; ideas only):**
- **Veil direction settled.** Stray light sets a grey floor that dark detail cannot go below, so dark tones merge. Removing that floor *is* shadow microcontrast (Chambers calls telling dark from slightly darker "penetrating power", which is Bill's "shades of shadow"). So **Veil defaults to removing haze**; adding haze is the "Soft glow" look only. Apply before any shadow lift (Auto HDR, 23.6), because lifting shadows that sit on a haze floor gives grey mush.
- **Bands are coupled.** Fine contrast cannot be high unless medium and broad contrast are too. The Balance control therefore never drives Fine up while Medium is negative; the "Both" preset raises all three.
- **Diffraction awareness.** Contrast at every scale starts to fall around f/8 and is plainly reduced by f/11 on high-resolution bodies. Read the aperture from EXIF: from f/8 the panel offers "Recover diffraction loss" (10.5) and says why; at f/16 and beyond it states honestly that the finest detail was never recorded.
- **Focus Stack hand-off (section 12).** Stacking keeps broad and medium contrast but loses some fine contrast. After a merge, measure the Fine band against the best source frame and offer to restore the difference (cyan tier: measured).
- **Blue light.** Lenses with weaker colour correction lose fine contrast in blue/violet light (faint violet halos). Check the blue channel's sharpness against green after the lens fix (10.2); candidate: guide blue-channel detail from green where they disagree.
- **Focus-plane only (with Dual Pixel depth, 11 row 5).** Sharp against blurred is what makes a subject stand out; option to apply the panel only to the in-focus plane.
- **Less sharpening.** When the panel has raised Fine, reduce the default capture and output sharpening so noise is not sharpened twice.

### 23.12 Additions from Bill's paper list (2026-11-15; arXiv numbers not yet verified, verify first)

- **Noise-aware shadow lift**: "A Retinex-based Image Enhancement Scheme with Noise Aware Shadow-up Function" (arXiv 1811.03280) is the published reference for the noise-aware lift in 23.6. Use its idea (the lift curve is limited where the signal is mostly noise) with our calibrated noise model.
- **Multi-Scale Retinex** (Jobson, Rahman, Woodell, NASA Langley, 1997; Retinex theory itself is Land's) as **Engine C** of Auto HDR (23.6), under rule 23.1. Picker text: *strong at pulling detail from very deep shadows; weak: can halo at hard edges and grey out colour.* Use the colour-restoration variant and our edge guard.
- **Denoise safeguard ("restore" step)**, idea from the solar Self2Self "train, denoise, restore" method (arXiv 2502.16807): after any denoiser, learned or classical, no pixel may move from its original value by more than a set multiple of its calibrated noise; beyond that, the original is kept (blended). Protects stars, catchlights and candle sparks from being smoothed away. On by default; shows a count of protected pixels.
- **Candidate**: "Learning to See Everything in Ultra-High Dynamic Range" (arXiv 2510.07741): one short raw exposure, highlights kept, shadows rebuilt by a denoising network. Same idea as exposing for highlights and letting our calibrated denoiser rebuild the shadows (Bill's ETTL calibration). Test their network against our pipeline; check licence.
- **Lab only (generative)**: Intrinsic Single-Image HDR (2409.13803); single-shot HDR via a video diffusion prior (2605.11628).
- **Not used, recorded so they are not re-investigated**: HDRUNet (2105.13084; for developed 8-bit images, not raw); the extreme-dark Mamba network (2508.03336) and latent-space Retinex transformer (2603.15131) (developed-image low-light networks that tend to invent texture); solar CNN denoising (1908.02815; same family as Zero-Shot N2N, nothing new); cloud-shadow removal for solar images (2407.13379) and NeRF relighting of satellite images (NASA NTRS 20230005357; needs many views of one scene).

---

## 24. Library upgrade and card import (Bill's decisions, 2026-11-15)

**Do this next, as Phase 4b (24.9).** Section 6.2 still stands. This section says what exists, what to build now and in what order, and adds card import and the Photos folder.

### 24.1 What Bill asked for, and what exists today

Bill's words: a photo browse tab "where someone can look at all of the raw images and make decisions on which to process, without having to utilize Windows Explorer. Just point it at the folder." And an "import from card option, where it copies all images from an SD card into a folder."

**Exists (do not rebuild):** `LibraryPage` in `studio/app/main.py`: folder tree, thumbnail grid from the embedded camera previews, stars 0–5, pick / reject / clear (P, X, U), a rating filter, double-click to Develop, Tag lens, Paste settings, Learn dust map, Measure lens glow. Ratings live in the sidecar. Exporting a selection exists.

**Missing (this section):**
1. **No big view.** Thumbnails are 240 px wide, so focus cannot be judged. Nobody can decide what to process from that.
2. **It will not survive a big folder.** Every folder open extracts every thumbnail from its raw again (nothing is cached on disk) and reads every sidecar before the window answers. Bill's archive is on external drives.
3. **No card import and no Photos folder.**
4. **No obvious step from "these are my picks" to "process them".**
5. No sorting, no filter by camera / lens / ISO / date, no thumbnail size control.
6. A rating that cannot be saved (read-only drive) only writes a log line. The photographer is not told.

Browsing in place stays: the Library opens any folder as it is, and nothing has to be imported first. Import is an extra, for getting photos off a card.

### 24.2 Decisions

**Bill's own words (do not reopen without asking):**
- The Library is where the photographer decides what to process. No Windows Explorer needed. "Just point it at the folder."
- **Import from card** copies the photos from a card into a folder.
- Imports always go into subfolders: "it should automatically have a sub folder arrangement, not just dump into the same folder."
- "Give the photographer the option of naming the folder or default it to that day's date."
- "Allow the photographer to set the default folder, so they can decide for external or internal drives."
- "Raws only by default."

**Builder's defaults (build as written; Bill has not ruled on these and may flip them, ask in chat):**
- "That day's date" is the day the photos were **taken**, not the day of the import. Several shooting days on one card make several folders.
- A new day starts at 04:00, so a night shoot stays in one folder.
- Folders are grouped by year.
- Import is a button in the Library, not a ninth sidebar section. It is one action that ends with the new folder open in the Library; rule 0.3.12 is about areas, and this is not one.

Everything else in this section was proposed to Bill in chat on 2026-11-15 and accepted as a whole ("put everything in a build plan").

### 24.3 The Photos folder (where imports go)

- One setting, `photos_dir`, in `settings/app.json`. **The photographer chooses it.** It is not set, and no folder is created, until they confirm it.
- **First import (or the first press of "Photos folder..."):** a small dialog, "Where should your photos be kept?", with:
  - a suggestion already filled in: `Photos` beside the app (`paths.PHOTOS = DATA_ROOT / "Photos"`, override `STUDIO_PHOTOS_DIR`). On a portable or git copy that is inside the app folder, which keeps rule 0.3.3. On an MSI install it follows the MSI exception already in `studio/paths.py`.
  - one quick-pick button per connected drive, with its name and free space ("E: My Passport · 1.2 TB free"). Pressing one proposes `<drive>:\Photos`.
  - "Choose a folder..." for anything else, and one line of plain text: "An external drive is fine. Raw files are large: about 20 to 60 MB each."
  - the free space of the chosen drive, always visible.
- Changing the Photos folder later never moves or copies existing photos. Say so in the dialog.
- **If the Photos folder is on a drive that is not connected,** say which drive, and offer "Try again" and "Use another folder for this import only". Never fall back to another drive silently.
- **OneDrive warning (verify):** if the chosen folder is inside a folder that OneDrive syncs (compare against the `OneDrive`, `OneDriveConsumer` and `OneDriveCommercial` environment variables), warn once that raw files will fill cloud storage quickly. Do not block.
- `paths.get()` creates a folder when it is called. Do not call it for `PHOTOS` before the photographer has confirmed.
- The Library gets a **Photos** shortcut at the top of the folder tree that jumps to this folder. On start the Library reopens the last folder, as today; if there is none and a Photos folder is set, it opens that.
- `.gitignore` already ignores `/Photos/` and every `*.psb.json` sidecar wherever it sits (done 2026-11-15; before that, a sidecar beside a sample raw in the project folder would have been committed by `push.bat`). Add a test that `git check-ignore` hides both; skip it when the folder is not a git copy.

### 24.4 Folder arrangement (never one big pile)

- **Layout:** `<Photos folder>\<year>\<folder name>\`, for example `E:\Photos\2026\2026-11-14 Kelly candlelight\`.
- **Folder name:** a text box that already holds the shoot date as `YYYY-MM-DD`. The photographer may leave it (the default), type after it, or replace it. Whatever is in the box is the folder name. Year-month-day order is fixed, so folders sort by time.
- **The date is the day the photos were taken** (default), read from each photo. A card emptied the next morning still lands under the right date.
- **A card with several shooting days** (default): one folder per day, each with its own name box. A tick, "Put everything in one folder", collapses them to one box, pre-filled with the first day's date.
- **Night shoots** (default): a "day" changes at 04:00, not at midnight, so a shoot that runs past midnight stays in one folder under the date it started. One setting, "A new day starts at", default 04:00. Bill shoots candlelight and night sky; a midnight split would cut his sessions in two.
- **Year level** (default): on. A setting, "Group folders by year", turns it off. The year is the year of the folder's date.
- **Capture time** comes from the engine's metadata (`read_metadata(...).datetime`), in camera local time, with no time-zone conversion. If a file has none, use its modified time (cameras set it to the capture time). If neither is usable, the folder is `Undated`.
- **Names are made safe for Windows:** strip `\ / : * ? " < > |` and control characters, trim trailing dots and spaces, refuse reserved names (CON, PRN, AUX, NUL, COM1–9, LPT1–9), cap at 80 characters. If the box ends up empty, use the date. Show the full path that will be used, live, under the box. Warn if a full file path would pass 240 characters.
- **A folder that already exists is added to,** never replaced and never duplicated as "(2)". That is how a second import from the same card works.
- File names stay exactly as the camera wrote them. No renaming in this version.

### 24.5 Import from card: one dialog

Opened by **Import from card...** in the Library (the primary button). One screen:

```text
 Import from card

 Card         [ EOS_DIGITAL (F:)  ·  412 photos  ·  38 new      v ]   [ Choose a folder instead... ]
 Copy         [x] Raw files     [ ] JPEGs     [ ] Videos
              214 JPEGs and 3 videos stay on the card.
 Into         E:\Photos   [ Change... ]            1.2 TB free  ·  this import needs 1.9 GB
 Folders      [x] 2026-11-14   26 new   ->   [ 2026-11-14 Kelly candlelight ]
              [x] 2026-11-15   12 new   ->   [ 2026-11-15                   ]
              [ ] Put everything in one folder
              Saved as:  E:\Photos\2026\2026-11-14 Kelly candlelight\
 Second copy  [ ] Also copy to a second drive   [ ...................... ]   [ Choose... ]

                                                         [ Cancel ]   [ Import 38 photos ]
```

- **Finding the card:** a "card" is any drive with a `DCIM` folder at its root, whatever Windows calls the drive type. Do not test for "removable": many CFexpress readers (the R5's card) show up as fixed disks. List every candidate with its label, letter, photo count and how many are new. Check again every 2 seconds while the dialog is open. "Choose a folder instead..." imports from any folder, with the same engine and the same rules.
- **Card found notice:** when a new card appears while the Studio is open, show a quiet card in the Library ("Card found: 412 photos, 38 new. Import..."). Never start copying without a press.
- **A camera plugged in by USB cable** usually has no drive letter (it speaks MTP). Not supported in this version. When no card is found, say "Put the card in a card reader."
- **Scan:** walk everything under `DCIM`. Fill the dialog at once from the directory listing (name, size, modified time), then confirm capture times from metadata in the background, one reader on the card at a time. Sort files into raws (`RAW_EXTENSIONS`), JPEG/HEIF, videos, other.
- **What to copy:** three ticks. Raw files on, JPEGs off, Videos off. The choice is remembered. With raws only, the JPEG of a RAW+JPEG pair stays on the card. Videos may sit outside `DCIM` (for example Sony's `PRIVATE\M4ROOT`; verify per brand) and are looked for only when Videos is ticked. "Other" files are never copied in this version.
- **Always say what is being left behind,** in the dialog and again in the finish summary: "214 JPEGs and 3 videos stay on the card." Someone who formats the card afterwards must have been told.
- **A card with no raw files:** do not import nothing in silence. Say "This card has no raw files, only JPEGs" and offer to tick JPEGs.
- **Days:** one row per shooting day, with a tick, the count of new files, and the folder-name box (24.4). No thumbnail picker in this version: the photographer imports, then chooses in the Library, which is what the Library is for.
- **Space check before starting:** needed bytes against free bytes on the destination (and on the second-copy drive), keeping 2 GB or 5% free, whichever is larger. If it does not fit, the Import button is off and the line turns red with the shortfall.
- **Second copy** ("Also copy to a second drive"): off by default, path remembered. Same folder arrangement, written and checked the same way. If the second copy fails, the first still counts; report it in amber.
- **While it runs:** file count, bytes, speed, time left, the current file name, Cancel. It runs in the background and the Studio stays usable. Keep the PC awake until it ends (verify: `SetThreadExecutionState`).
- **Finish summary:** copied; already imported (skipped); same-name clashes kept under a new name; failed, each with its reason; left on the card by choice. Then, only if every copied file was checked: "All photos are copied and checked. You can take the card out. The Studio wrote nothing to the card." Buttons: **Open in Library** (default) and View log.
- "Open in Library" shows the new folder (the first one, if several) with the filter on All photos.

### 24.6 Copy rules (never break these)

One lost shoot in a group of 150,000 is the story everyone hears. Each rule has a named test (24.9).

1. **The card is never changed.** Source files are opened read-only. The Studio never writes, renames, moves or deletes anything on the source, and never formats it. There is no "delete after import" option and none is to be added. The camera formats the card.
2. **Copy, check, then name.** Write to `<name>.partial` in the destination, flush and `fsync`, check it (rule 3), set the file's modified time to the source's, then rename it to the final name. A crash can only leave `.partial` files; the next import removes the ones it finds in the folders it writes to. (`*.partial` is already in `.gitignore`.)
3. **Every file is checked.** Hash the bytes while reading the card (one pass over the card; `hashlib.blake2b`, no new dependency), then read the written file back and compare the hash and the size. On a mismatch: remove the `.partial`, try the file once more, then list it as failed. Verify whether the read-back can bypass the Windows file cache without slowing the import badly; if it cannot, keep the read-back after `fsync`, and write in the log which kind of check was done.
4. **Never overwrite.** Same name and same content (size and hash) in the destination: skip, "already imported". Same name, different content: keep both. The new file becomes `<stem>_2<ext>` (then `_3`, and so on), is logged, and is listed in the summary. Camera counters wrap at 9999 and two bodies can share a prefix, so `IMG_0001.CR3` can be two different photographs.
5. **Only what is new.** Keep an import record at `<Photos folder>\_studio\imports.jsonl`: one line per imported file, with its original name, size, capture time, hash, where it went and when. A card file whose name, size and capture time match a line is "already imported", even if the photographer has since moved, renamed or deleted the copy. That way deleted rejects do not come back from a card that was not formatted. A tick, "Show photos imported before", overrides it. The record travels with the Photos folder. It is a convenience: if it is missing or unreadable, import still works by looking at the destination folder (rule 4). Never store a camera or card serial number in it (rule 0.3.5).
6. **One bad file does not stop the rest.** On a read error: retry twice, then mark the file failed and go on. If the card disappears: stop cleanly and report what was done. Starting again picks up where it left off (rules 4 and 5).
7. **One reader on the card.** Cards do not like parallel reads. Read in 4–8 MB blocks with one thread. Writing the main copy and the second copy may run in parallel from the same blocks.
8. **Everything is logged** (rule 0.3.5): `logs/import_<date>_<time>.log`, one line per file (source, destination, bytes, hash, seconds, result), then the summary. No serial numbers, no GPS.
9. **A dry run exists** (`plan_import`, 24.8). It decides every destination path and every skip or rename without touching a disk, and the dialog shows exactly what it returns.

### 24.7 Library: decide what to process, without Explorer

**A. Review view** (6.2's "Cull mode"; the button says **Review**, the wording is Bill's call). This is the main gap.
- **Space** opens the selected photo large, with the filmstrip below. **Esc** or Space returns to the grid at the same photo. **Left / Right** move. **0–5, P, X, U** work as in the grid. A tick, "Go to the next photo after rating", is on here (off in the grid). **Enter** opens the photo in Develop.
- **Z** or a click switches between fit and 100% at the pointer; drag to move around. The zoom and position stay when moving to the next photo, so the same eye can be compared across a burst.
- **C:** two photos side by side (the current one and the next, or the two selected), with zoom and position locked together. Four-up: later.
- **What is shown is a picker under rule 23.1,** with its plain text beside each option:
  - *Camera preview* (default): the JPEG the camera embedded in the raw. Best for: speed, it is instant. Weak at: it carries the camera's own sharpening and noise reduction, so it is not the raw. Chip: amber, "Camera preview".
  - *From the raw*: a quick render by the Studio's own engine, with the photo's saved edit or default settings. Best for: judging noise and real detail. Weak at: slower; the camera preview is shown until it is ready. Chip: cyan, "From the raw".
  - Verify, per brand, which embedded preview `rawpy` returns and how large it is, and take the largest the file holds. Where that is under about 2,000 px on the long side, 100% must switch to "From the raw" by itself and say so.
- **Info strip:** file name, date and time, camera, lens (or its tag), ISO, shutter, aperture, focal length; the Dual Pixel badge and the exposure badge (7.1) where they apply.
- Load the next two photos and the previous one in advance, so the arrow keys never wait.
- The focus-check overlay and the front/back-focus readout (6.2; section 11, item 6) stay in their own phases.

**B. Speed on big folders.**
- **Thumbnail cache on disk** (already called for in 6.2): `cache/thumbs/`, keyed on the full path, size and modified time; small JPEGs, about 320 px on the long side. Size-capped (default 1 GB, least recently used removed first), with a "Clear cache" button. It lives in the app's cache folder, never beside the photos, so read-only drives work.
- **Only what is on screen:** ask for thumbnails for the visible rows plus one screen ahead; drop the requests when the view scrolls on or the folder changes. Replace the `QListWidget` with a `QListView`, a list model and a delegate that paints the stars, flags and badges.
- **Nothing slow on the window's own thread:** sidecar ratings and metadata are read in the background. Keep them in a small index (`cache/library_index.sqlite`, Python's own `sqlite3`), keyed on path, size and modified time. The sidecar stays the truth; the index can be deleted at any time.
- Use few workers on a spinning drive (2) and more on an SSD. Measure.
- **Aims** (measure on a 2,000-raw folder and write the numbers in `docs/RESULTS.md`): the window answers within half a second whatever the folder size; a first screen of thumbnails within 2 s from an SSD and 5 s from a USB hard drive when cold, and within half a second when cached; scrolling never stalls.

**C. Choosing and sending on.**
- **Process picks...** (and "Process selected..."): one dialog. Which photos (picks / selected / everything shown), what to apply ("each photo's own edit" or "Make it good on photos that have no edit yet"), the export preset and the destination. It runs in the background with a progress bar and Cancel (6.1). This is the step from "decided" to "processed". If "Make it good" lives in the interface code today, move it into the engine first (rule 2.2).
- **Sort** by capture time (default), file name, rating, ISO. **Filter** by camera, lens, ISO range, date, "not rated yet", and a text box for file names, on top of today's rating filter.
- **Thumbnail size** slider (about 120 to 400 px).
- A count chip that tells the story: "412 photos · 38 picks · 12 rejects".
- **Include subfolders** tick. `collect(recursive=True)` skips folders named `outputs`, `logs`, `cache`, `reports` and the like wherever they are; that is right inside the app folder and wrong inside a photographer's own tree. For the Library, skip only `_studio` and hidden or system folders.
- Right-click menu: Open in Develop, Process, Tag lens, Paste settings, Show in Explorer, Copy path.
- Drop a folder on the window to open it. Keep a short list of recent and pinned folders above the tree.
- **Tidy the top bar:** it holds nine controls today. Left: the path. Middle: filter, sort, size. Right: **Import from card...** (primary), **Review**, **Process picks...**, Open folder..., and a **Tools** menu holding Tag lens, Paste settings, Learn dust map and Measure lens glow. It must still fit at 1366×768 (6.1).
- **A rating that cannot be saved must be seen:** show a message once per folder and offer the central sidecar folder (section 5).
- Colour labels and keywords (6.2): later.
- No deleting from the Library in this version.

### 24.8 Code layout

- `studio/engine/io/ingest.py`, no GUI imports (rule 2.2): `find_cards()`, `scan_source(path)`, `plan_import(files, options)` (pure, no disk writes: the dry run) and `run_import(plan, on_progress, cancel)`, which returns a report. The drive listing sits behind one small function, so tests can fake cards on any system (`STUDIO_CARD_ROOTS` for the Linux test machine).
- `studio-cli import <source> [--to DIR] [--name TEXT] [--jpegs] [--videos] [--second DIR] [--dry-run]` wraps the same calls, for tests.
- **New interface code goes in new files:** `studio/app/library.py` (move `LibraryPage` there first, with the tests green), `studio/app/review.py`, `studio/app/importer.py`. `studio/app/main.py` is about 5,900 lines; do not add these to it.
- `studio/paths.py`: `PHOTOS`, with its override.

### 24.9 Build order, tests and acceptance (Phase 4b)

| Step | Deliverable | Acceptance |
|---|---|---|
| **4b-1 Card import** | The copy engine first, with its tests; then the import dialog, the Photos folder dialog and the card-found notice. | Bill puts a card from a real shoot in the reader, presses Import from card, names the folder or leaves the date, and ends in the Library looking at the photos, without opening Explorer. No file on the card has changed. A second import of the same card copies nothing. |
| **4b-2 Library speed** | Thumbnail cache, on-screen-only loading, background ratings and metadata, the index. | The aims in 24.7 B, measured and recorded. |
| **4b-3 Review view** | 24.7 A. | Bill rates 200 frames with the keyboard alone and checks focus at 100% without leaving the Library. |
| **4b-4 Choose and send on** | 24.7 C. | Bill filters to his picks and presses Process picks; the exports appear. A tester on another brand does the whole path, card to exports, with no help. |

**Named tests that must always pass** (add them to the list in section 17): the source is never written to (hash the whole source tree before and after; run once with the source made read-only) · a second import copies nothing · same name with different content keeps both files · a `.partial` left by a crash is cleaned and the file copied again · a read error on one file does not stop the others · a card pulled mid-copy leaves no half-written final file · too little space blocks the start · raws-only leaves the JPEG of a pair behind and says so · a card with no raws says so · several days make several folders, and 02:00 belongs to the day before · unsafe characters in a folder name are made safe · a photo deleted after import is not imported again, and is when "Show photos imported before" is ticked · the second copy is identical to the first · no serial number appears in the log or in the import record.

### 24.10 Later, and out of scope

**Later:** a thumbnail picker inside the import dialog; renaming files on import; cameras over a USB cable (MTP); reading in-camera ratings into the Library (verify per brand); an Eject button; "Move rejects to a `_rejects` folder" (a move, never a delete); four-up compare.

**Out of scope:** any catalogue or database that the photos depend on (section 1: no catalogue lock-in; the index and the import record can both be thrown away), tethered shooting, phone imports.

## 25. Two more tabs: Panorama and Infrared / Full Spectrum (Bill's ideas, 2026-11-16)

**Build from `BUILD_PLAN_PANO_IR.md`** (the detailed plan, with the facts measured on Bill's files). This section is
the short version.

Bill's words: his folders "can test panoramic stitching", and "we can even make a tab specifically for editing full
spectrum / infrared photos. Because those can be a challenge, with the color channels and white balance, etc."
Written down after the three stack tabs were built (`BUILD_PLAN_TABS.md`), so both can stand on `stack/align.py`,
`stack/frames.py`, `io/intermediate.py` and `app/stackui.py`. Not started. Rules of section 0 and `BUILD_PLAN_TABS.md`
0.1 apply (names `..._Studio.tif`, nothing written to C:\, every method says what it is best for and weak at, colour
coding, a test on synthetic truth for each algorithm).

### 25.1 Infrared / Full Spectrum tab (build this one first: Bill shoots IR, and it is mostly colour work)

**Why a tab and not sliders in Develop:** an IR raw breaks the assumptions Develop makes. The camera's white balance
range stops near 2000 K, but IR needs the red channel pulled down far further than that; the camera matrix was made
for visible light, so it turns IR into nonsense colours; the photographer wants channel swaps that no visible-light
editor offers; and the filter decides everything, so the tab starts by asking which filter was on.

**Test files (in `test_raws\`, git-ignored):** `infrared_850` (2 frames) and `infrared_850_candlelit` (20 candlelit
portraits), 850 nm; `ir_visible_fusion` (visible + IR experiment, 2008). All of Bill's infrared and full-spectrum work
is on one converted **Canon EOS 450D (Rebel XSi)**, so the filter profiles can be built for that one sensor first;
his full-spectrum folders (`20210909 full spec`, `20210911_full_spec` on the archive drive) have many more examples. Bill also shoots
530 / 590 / 650 / 720 nm and full spectrum; ask him for one frame of each with a grey card or white wall in it.

**Filter profiles** (picker at the top; each says what it gives and what it is weak at):
| Filter | What the camera records | Default treatment |
|---|---|---|
| Full spectrum (no filter, or a hot-mirror off) | Visible + IR: everything pink-red | White balance on a neutral; offer "Remove IR" (subtract a scaled IR estimate) only if a matching 720+ frame exists |
| 530 / 550 nm | Strong colour, foliage gold to orange | Channel swap red/blue, white balance on foliage |
| 590 nm | Classic "false colour IR": gold foliage, blue sky after swap | Red/blue swap, white balance on foliage |
| 650 / 665 nm | Less colour, cream foliage | Swap, low saturation |
| 720 nm (Bill's very purple frames) | Faint colour, mostly luminance | Swap optional; offer B&W with the channel that has most detail |
| 850 nm and up | Almost pure luminance, same in all channels | Black and white by default; the colour that is there is noise and filter leak |
| Unknown | Estimate from the raw: the ratio of the channel means and how alike the channels are (850 nm makes them nearly equal) | Show the guess in amber |

**Controls (in the tab, before Develop):**
1. **White balance beyond the camera's range.** Channel multipliers in the raw space directly (not temperature/tint),
   with a picker for "click foliage" (foliage is the IR photographer's neutral) and "click a neutral". Range
   0.05x-20x per channel. Stored as multipliers in the sidecar so Develop shows the same thing.
2. **Channel mixer / swap** (3 x 3 matrix in linear camera RGB, after white balance): presets *Red/blue swap*,
   *Blue sky (590)*, *Goldie*, *Mono from the best channel*, and a full matrix in Full mode. Each row sums are shown so
   brightness is not changed by surprise.
3. **IR camera profile.** A replacement for the visible-light matrix: identity (no colour science pretended) or a
   profile measured from a frame of a printed colour chart under daylight through that filter (a "calibrate this
   filter" button, same pattern as the Cameras page). Measured = cyan, shared preset = violet, estimated = amber.
4. **False-colour presets** for the look Bill teaches: *Classic IR (590 swap)*, *Wood effect (850 B&W, glowing
   foliage)*, *Aerochrome-like* (IR to red, red to green, green to blue), *Candle IR* (keep the candle warm while the
   rest is IR). Presets are ordinary settings of 1-3, so the person can still move them.
5. **Hot-spot finder.** Many lenses show a bright centre in IR. Measure it the way lens falloff is measured
   (`lens/`), show it, and offer a correction (off by default, like vignetting).
6. **Fusion.** Visible + IR frames of the same scene (Bill's 2008 folder): align with `stack/align.py`, then put IR
   luminance under visible colour, or IR into one channel. Mode picker with texts.

Then **Open in Develop** with every tool; the IR settings stay in the sidecar so a second photo from the same shoot can
take them with Paste settings.

**Tests (synthetic truth):** a scene rendered through known filter curves (590, 720, 850) with known foliage and sky
reflectances: the filter guess gets the right family; white balance on foliage brings foliage to neutral within 1%;
the swap matrix maps the sky to blue; multipliers out to 20x do not clip in float; a hot spot added to a flat field is
measured within 5%. Real-file checks on the 850 nm candle frames: the default is B&W, the candle flame is not tinted
by the channel swap.

### 25.2 Panorama tab

**Reuses:** `stack/align.py` for the pairwise overlap (tiles, IRLS fits; homography for a rotating camera), `frames.py`
for probing and checks, the intermediate for the result.

**Pipeline:**
1. **Order and overlap.** Frames in time order; pairwise overlap found by phase correlation of strongly downsized
   copies at a few candidate offsets (a panorama overlaps 20-50%, outside `register`'s search range), then refined
   with `register` (homography) on the overlap region only.
2. **Global adjustment.** Bundle adjustment of focal length and per-frame rotation (a rotating camera: homographies
   are rotations under one focal length), Levenberg-Marquardt, robust loss. The EXIF focal length is the start; an
   adapted lens has none, so ask for it or estimate it (amber).
3. **Projection** picker with texts: rectilinear (up to about 120 degrees, straight lines stay straight), cylindrical
   (wide horizontal), spherical / equirectangular (360 x 180, for viewers), and Mercator / Panini for architecture.
4. **Exposure matching** in linear light: one gain per frame from the overlaps (least squares), then vignetting from
   the lens profile or measured.
5. **Seams.** Graph-cut seam through the overlap that avoids moving things (people, water, leaves) and high-contrast
   edges, then multi-band blending (Burt and Adelson) along it. A "ghost" view shows where frames disagree.
6. **Crop** to the largest rectangle, or keep the irregular edge with transparent borders (TIFF alpha).
7. **Build in bands** so a 20-frame, 45 MP panorama fits in memory (the streaming pattern of the stack tabs).

**Honest limits to state on the page:** parallax (handheld near subjects) leaves seams that no blend fixes; moving
water and people get cut, not merged; a lens with no focal length in EXIF needs it typed or estimated.

**Tests (synthetic truth):** a large known scene cut into overlapping views by known rotations and focal length:
recovered rotations within 0.05 degrees and focal length within 0.5%; a moving object in one overlap is avoided by the
seam; frames with different gains match within 1%; the result in rectilinear projection keeps a known straight line
straight within 1 px. Real files: Bill's folders with overlapping frames (ask him which shoots were panoramas).

### 25.3 Order

Infrared first (a smaller job, mostly colour, and Bill shoots it now); Panorama second (bundle adjustment and seams
are new code). Each tab: engine and tests, then the page on `stackui.py`, then a Help card, a README row and a line in
`docs/NEXT_UPDATE.md`.
