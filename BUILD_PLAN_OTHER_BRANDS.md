# Build plan: what other brands put in their raw files (Bill, 2026-11-21)

Bill, 16:54: "I know that some other brands have features that Canon does not. Thus I don't know what they are
precisely. BUT, if there are any features that would be seen within their raw files, I am all about adding them to
Studio." 17:02: "make a build plan for other brands. The build plan should include the research pass, because we'll
do that at a later date as well."

So this plan has two halves: **R, a research pass** (to run first, at a later date, with sources), and **B, the
build**, whose sections are written from what is known now and are **unverified until R confirms them**. Nothing in B
is built before R has checked it: every brand-specific claim below carries "(R)" where the research pass must confirm
the tag, its format and the models that write it.

---

## 0. Read this first

### 0.1 Rules (BUILD_PLAN.md section 0 applies; these matter most here)

* **No new required packages.** Maker notes are read by our own parser (as `metadata.py` does for Canon's ShotInfo);
  rawpy (LibRaw) stays the decoder; ExifTool is a reference for the research, never a dependency.
* **Truth labels.** Every value read from a file says where it came from (cyan: recorded by the camera and checked
  against a known scene; amber: recorded but coarse or not yet checked; violet: shared from another camera or a
  profile). A raw that is already a computational result (several exposures merged in camera: pixel shift merged,
  hand-held high resolution, multi-shot noise reduction, computational phone DNGs) is labelled as such, and the noise
  calibration never treats it as one exposure.
* **Sample raws** for tests come from raw.pixls.us (CC0) or Bill's students, kept in a local folder that is never
  committed (like `test_raws\`), and are used only by tests that skip when the file is absent. Every feature also gets
  a synthetic test with known truth.
* **Every method says what it is best for and weak at**; every new setting is in `schema`; every part built gets a line
  in `docs/NEXT_UPDATE.md`.

### 0.2 What exists to build on

* `engine/io/metadata.py`: EXIF plus Canon's maker notes (ISO, focus distance, focus mode, temperature).
* `engine/stack/superres.py` (Super-Res Stack: hand-held pixel shift by alignment), `engine/stack/sorter.py` (Card to
  Stacks: grouping by time, scene, focus), `engine/lens/` (lensfun profiles, auto corrections), the noise calibration
  (`engine/calib/`), Dual Pixel (`engine/dualpixel/`), the camera solver (`engine/scan/`: its `View.f_sigma`,
  `View.lens` and the scale step take what this plan reads).

---

## R. The research pass (first; at a later date)

### R.1 Questions, per brand

For **Sony, Nikon, Fujifilm, OM System / Olympus, Panasonic, Pentax / Ricoh, Leica, Hasselblad, Phase One, Sigma**, and
the computational phone formats (**Apple ProRAW, Google Pixel, Samsung Expert RAW**), and Canon for comparison:

1. **Focus distance**: is it recorded, where (tag), in what units and steps, how precise (a lookup table, a lens
   encoder count, metres), and for which lenses (native only? adapted?).
2. **Focus bracketing / focus shift**: does the camera record that a frame belongs to a bracket, its index, the number
   of frames and the step size (and in what unit)?
3. **Pixel shift / high-resolution multi-shot**: one file with several frames, separate files with a shared id, or an
   already-merged raw? The shift pattern, motion correction flags, and how to tell a merged raw from a single exposure.
4. **Phase-detection pixels**: are masked or split PDAF pixels in the raw data (positions, pattern)? Do they need
   correcting (stripes), and could they give sparse depth (as Dual Pixel does, on fewer pixels)?
5. **Embedded lens corrections**: distortion, chromatic aberration and vignetting parameters in the raw (format,
   model: polynomial or table, which focal lengths and distances), and how they relate to the in-camera JPEG.
6. **Orientation and level**: roll and pitch angles (electronic level), not only the 90-degree orientation tag.
7. **AF points and subject detection**: which AF point or area was used, face / eye / animal boxes.
8. **Exposure intent**: dynamic-range modes (underexposure plus a tone flag), bracket sequence numbers, the camera's
   look (film simulation, picture control, creative look) by name and settings.
9. **Sensor and calibration**: sensor temperature, shutter count, black level per channel, flat-field / lens cast
   calibration data (Phase One LCC), any per-camera noise or defect maps.
10. **Depth and mattes** (phones): embedded depth maps, semantic mattes (skin, sky, hair), and their resolution.

### R.2 Sources to check, in order

1. ExifTool's tag tables (per brand maker notes: names, offsets, formats, models) and its change log.
2. LibRaw / rawpy: what it decodes (multi-frame files, pixel shift, X-Trans, PDAF handling), its open issues.
3. The Adobe DNG specification (opcodes for lens corrections: WarpRectilinear, FixVignetteRadial, GainMap; ProfileName;
   the computational-DNG fields) and Apple's ProRAW documentation.
4. darktable, RawTherapee and ART source (maker-note readers, pixel-shift merging for Pentax and Sony, PDAF line
   correction for Sony and Nikon, embedded lens-correction readers).
5. The manufacturers' manuals for each feature (what the camera writes, on which models).
6. lensfun's database notes on embedded corrections.
7. Sample files: raw.pixls.us (CC0) for each feature; Bill's photography group for models not there.

### R.3 What the research produces

`docs/RAW_FEATURES.md`: one row per (brand, feature): verified yes / no, tag and format, models, how precise, a sample
file, whether rawpy exposes it, and our recommendation (build / skip / later). Each claim with its source. Then this
plan's B sections are corrected to match, the "(R)" marks removed where confirmed, and Bill is asked the questions in
section Q with the table's findings.

---

## B. The build (after R; each section unverified until then)

### B.1 A maker-note reader for each brand (foundation)

`engine/io/makernotes/` with one small parser per brand (TIFF-style IFDs inside the maker note; the offsets and byte
orders R records), reading only the fields this plan uses, into `Metadata` (new fields, all optional, with a `source`
per field). Test: every parser against the sample files' known values (ExifTool's output written into the test), and
a fuzz test that a corrupt maker note never stops a raw from opening.

### B.2 Focus distance, everywhere it is recorded

Several brands are thought to record the focus distance more finely than Canon's coarse ShotInfo value (R: Sony's
focus position, Nikon's lens data, OM System's distance in millimetres, Fujifilm, Panasonic). Uses: the scale of
surfaces and models (the Studio already turns a distance into magnification and the view edge to edge: C.6 c of
BUILD_PLAN_DP3D), the starting focal length of each 3D Scan view (focus breathing: `View.f_sigma` tightens from 0.35
to about 0.05 when the distance is known), and the self-distance check (a recorded distance cross-checks the stack's
own measurement). Colour: amber until a known scene checks a lens (then cyan in its profile).

### B.3 Focus brackets recorded by the camera

In-camera focus bracketing / focus shift (R: Nikon Z focus shift, OM System focus bracketing, Fujifilm focus
bracketing, Panasonic, Sony's newer bodies) may record the frame index, count and step. Uses: Card to Stacks groups a
bracket with certainty (cyan) instead of by time and scene (amber); the stack's frame order is known (near to far);
a recorded step size (when in a physical unit) gives the depth scale of a Focus Stack surface without a calibration.

### B.4 Pixel shift and high-resolution files

* **Separate frames with known shifts** (R: Sony's pixel shift sets, Fujifilm's, Nikon's, Pentax's four frames in one
  file): merged with the shifts known (no alignment search), giving full colour at every pixel (no demosaicing) and
  lower noise; motion between frames found by comparing the frames and filled from one frame (the Super-Res Stack's
  ghost handling), shown on a map.
* **Merged in camera** (R: OM System / Olympus and Panasonic high-resolution raws): labelled "merged in camera" so the
  noise calibration and Clean treat it correctly; no second merge.
Test: a synthetic scene sampled four times with known one-pixel shifts must merge to the truth within noise.

### B.5 Phase-detection pixels

If R confirms PDAF pixels in the raw data (R: Sony, Nikon Z, Fujifilm, others): first, correct them where they cause
stripes (a known pattern, filled from neighbours, before demosaicing); second, if their values are readable as
left / right pairs, a sparse version of the Dual Pixel depth (fewer pixels, the same physics: `engine/dualpixel/depth`
on the PDAF lattice), labelled amber, giving other brands' cameras some measured depth for focus stacks and 3D Scan.

### B.6 Lens corrections from the raw

Embedded distortion, chromatic aberration and vignetting parameters (R: Sony, Fujifilm, Panasonic, OM System, Nikon;
DNG opcodes from any camera) used by Develop > Lens as the first choice for that lens (cyan: the maker's own
calibration), lensfun second (violet: shared profile), the Studio's auto measurement third (amber). The 3D Scan solver
takes the distortion as fixed when it is known (no k1 to solve).

### B.7 The camera's level and AF point

* **Roll and pitch** (R: Nikon, OM System, Sony, Fujifilm; Canon on some bodies): the 3D Scan solver gets each view's
  gravity direction (an "up" prior in the bundle adjustment: hand-held views are rarely level, so the measured roll is
  better than assuming the average camera is level), Panorama levels the horizon from it, and Develop offers
  straightening by the level.
* **AF point / subject boxes** (R): Focus Stack and Develop know where the photographer focused (the subject mask
  starts there; Portrait finds the face the camera found).

### B.8 Exposure intent and the camera's look

Dynamic-range modes (R: Fujifilm DR200 / DR400, others) recorded with their underexposure, so the default develop
matches the brightness the photographer saw; bracket sequence numbers for HDR grouping in Card to Stacks; the camera's
look by name (R: Fujifilm film simulations, Nikon picture controls, Sony creative looks) offered as a starting look
in Develop, built from Bill's own reference pairs where possible (labelled "approximate" in amber: the makers' looks
are not published).

### B.9 Calibration data in the file

Sensor temperature (R: which brands), per-channel black levels, Phase One's lens cast calibration and any defect maps:
used by Clean's noise model (temperature) and Develop's flat-field.

### B.10 Computational phone raws

Apple ProRAW, Google, Samsung (R): labelled as computational (several frames merged, tone already applied in part);
embedded depth maps and semantic mattes (if R finds them) offered as amber masks in Develop and Portrait; their
noise calibration kept separate from single-exposure raws.

---

## T. Tests

* Synthetic with known truth for every algorithm (pixel shift merge, PDAF correction and sparse depth, embedded
  correction models against a known distortion grid, level prior in the solver on `tests/scan_synth.py` views with
  known roll).
* Sample-file tests (skip when the file is absent): each maker-note field against ExifTool's value for that file; each
  label (computational, merged in camera) on the right files.
* The Canon path unchanged (every existing test passes).

## J. Order

| # | What | Done when |
|---|---|---|
| R | The research pass (R.1-R.3) | `docs/RAW_FEATURES.md` with sources; this plan corrected; Bill answers section Q |
| OB1 | B.1 maker-note readers, B.2 focus distance | the fields read on every sample file; 3D Scan uses them |
| OB2 | B.3 focus brackets, B.7 level and AF point | Card to Stacks groups brackets with certainty; the solver uses the level |
| OB3 | B.6 lens corrections, B.8 exposure intent and looks | Develop > Lens prefers the maker's data |
| OB4 | B.4 pixel shift | a sample pixel-shift set merges with known shifts |
| OB5 | B.5 PDAF pixels, B.9, B.10 | stripes corrected; sparse depth where readable |

## Q. Questions for Bill (ask in chat after R, with the table's findings and a recommendation)

1. Which brands and models do you and your students shoot most? (The order of OB1-OB5 follows them.)
2. May the research pass download sample raws from raw.pixls.us (CC0) into a local folder that is never committed?
3. For cameras with pixel shift: do you want the merge in Super-Res Stack (beside the hand-held method) or as its own
   tab?
4. Phone raws: worth supporting for your group, or skip?

## K. Not in this plan

* Video files (some cameras' "post focus" and 4K / 6K photo modes store frames as video): a later plan if Bill wants
  them.
* Writing maker notes back into exports (exports keep the standard EXIF the Studio writes now).
