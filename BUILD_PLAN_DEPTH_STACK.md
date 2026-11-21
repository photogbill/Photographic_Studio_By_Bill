# BUILD PLAN: Depth Stack (focus-stacked Dual Pixel depth, and surfaces for games)

*Written by Claude Opus, 2026-11-21 04:20, from Bill's words in the RPG Forge chat. For one Studio session. Read
`BUILD_PLAN.md` section 0 (the rules) and `CLAUDE.md` first; everything there holds here.*

## What Bill said

- 2026-11-21 04:04: "Dual Pixel Raw could be useful if you are wanting textures?"
- 04:06: he shoots textures with the Zeiss 50mm f/2 Makro-Planar (1:2, essentially no distortion), wide open if
  that helps.
- 04:13: "I can also focus stack. So if you can focus stack dual pixel raws, or add that capability to Studio, we
  can do that as well." Handheld; no tripod.
- 04:17: "I want to do it now."

## Why

A photo texture needs a **height map** as well as colour (the game derives its normal map from it). Today the only
way to get one from a single photo is to guess height from brightness, and that guess is fooled by colour: a dark
knot in bark reads as a hole. The Studio can already **measure** depth in two independent ways, and they are
strong in different places:

| Measure | Where it is strong | Where it is weak |
|---|---|---|
| **Depth from focus** (Focus Stack already computes "which frame is sharpest where": `engine/stack/focus.py`, `depth()`) | Anywhere with texture, across the whole depth of the stack | Only as fine as the focus step between frames; sign-free inside one frame's depth of field |
| **Dual Pixel disparity** (`engine/dualpixel/depth.py`, `disparity()`, signed, sub-pixel) | Fine, signed offsets near each frame's focus plane; strongest wide open | Shallow: only trustworthy within a few pixels of offset; nothing on plain areas |

A focus stack of Dual Pixel raws gives **both at every point**, and they correct each other: the stack index says
roughly how far, and the frame's own signed disparity says exactly where between two frames. This is Bill's rule
from his RF work in a new place: measure two ways and trust the agreement. No published tool does this fusion as
far as we know, which is a reason to build it, not a reason to doubt it.

## What exists (read these before writing anything)

- `engine/stack/focus.py`: Pass A (half size), alignment for handheld stacks with focus breathing, sharpness with the
  noise energy removed, `depth()` (fractional frame index, edge-aware smoothed), confidence, the merges, the halo guard.
- `engine/stack/align.py`: similarity transforms, tile flow, `warp`.
- `engine/dualpixel/decode.py`: `is_dual_pixel`, `open_dual` (R5 checked), shading `GainField`s.
- `engine/dualpixel/depth.py`: `halves_luma`, `disparity` (1-D, band-passed, with confidence), `refine` (guided),
  `detail_map` (real detail versus noise from the two halves' correlation), `IN_FOCUS_PX`.
- `cli_tools.py`: `cmd_stack` (`stack focus|superres|pano`) and `cmd_dualpixel` (`--make report,depth,...`).
- `mcp_server.py` builds its tools from the CLI commands by itself: add nothing there by hand.

## What to build

**A. Dual Pixel frames in a focus stack.** `stack focus` accepts Dual Pixel CR3s exactly as it accepts ordinary
raws (the A+B image is the frame; nothing changes for photographers who never shoot Dual Pixel). When every frame
is Dual Pixel, Pass A also keeps each frame's half-size disparity and its confidence (`dualpixel.depth.disparity`
on `halves_luma`), warped by the same transform as the frame, on disk as float16 like the frames themselves.

**B. The fusion (the heart of it).** Per place:
1. From the focus measure: the fractional sharpest-frame index `k(x)` and its confidence.
2. From each frame `i` near `k(x)` (the two or three either side): its signed disparity `d_i(x)` and confidence.
   A point exactly at frame `i`'s focus plane has `d_i = 0`; one step of focus moves `d` by a slope `s` (pixels of
   disparity per frame of focus).
3. **Calibrate `s` from the stack itself:** wherever a place is confidently sharp between frames `i` and `i+1`, the
   difference `d_i - d_{i+1}` is one focus step. A robust median over the whole picture gives `s` (and its spread,
   which goes into the report). No lens data and no tripod needed.
4. Fused depth: `z(x) = Σ_i w_i(x) · (i - d_i(x)/s)`, with weights from each frame's disparity confidence and its
   nearness to `k(x)`, and `k(x)` itself as one more vote weighted by the focus confidence.
5. Where Dual Pixel has nothing to say (plain areas), `z = k(x)`. Where neither does, "do not know", filled only by
   the existing edge-aware smoothing and marked as filled.
6. **The agreement map:** `|k(x) - (i - d_i/s)|` per place, shown and saved. Large disagreement is a finding (a
   moving leaf, a reflection, a bad frame), never averaged away silently.

Without Dual Pixel frames, the same output is made from depth from focus alone (step 1 and 5). A single Dual Pixel
frame (no stack) still gives its own disparity depth, as now. All three cases write the same files.

**C. Height in real units, when it can be known.** Relative height (0..1 over the stack) always. Millimetres when
the focus step is known: from EXIF focus distance where the lens reports it, or from the magnification (`--ratio
1:2`) and an optional `--step-mm`. Say which, in the notes and the report.

**D. Surface outputs for games and 3D (`--make surface`).** From the fused result:
- `NAME_colour_Studio.png`: the all-in-focus merge (the existing auto merge), sRGB.
- `NAME_height_Studio.png`: 16-bit, white = near (the forge's convention), normalised over the confident range.
- `NAME_normal_Studio.png`: from the height, OpenGL convention (`--normals-y up|down` as `depth` already has).
- `NAME_confidence_Studio.png` and `NAME_agreement_Studio.png`.
- `NAME_surface_Studio.json`: the scale (relative or mm, and how it was known), `s` and its spread, the frames used,
  the share of the picture that is measured versus filled, and the method.
Optional `--tileable`: offset by half and heal the seams of all channels with one geometry (the RPG Forge's method,
`forge/night4/tileable.py`, is MIT: the idea may be reused; write the Studio's own code).

**E. Command line.** `stack focus FRAMES --make surface` (and `--make depth` for the depth alone), plus `dualpixel
FILE --make surface` for one frame. Every new setting in the schema with what it does. Results follow the contract
(`--json`, `NAME_Studio.ext`, nothing overwritten). Tests in `tests/test_cli_tools.py`.

**F. The Focus Stack tab.** A **Depth** view beside the merge: the fused depth with the colour coding of plan 23.1
(cyan measured by both, amber by one, violet filled), the agreement map as an overlay, and a "Save surface for games"
button that writes D. One line under the method saying what it is best for and weak at.

## Tests (synthetic, with known truth; nothing written outside `outputs/`)

- A synthetic height field (bumps, a ramp, a step edge), rendered as a focus stack with a known blur per frame and a
  known signed disparity per frame (noise added at R5 ISO 100 and 800 levels). The fused height beats depth-from-focus
  alone and disparity alone in RMS error, and `s` is recovered within 5 %.
- A plain area stays "do not know" (never invented); a moving patch shows up in the agreement map.
- Handheld: frames shifted, rotated by up to 1 degree and scaled up to 2 % (focus breathing) still fuse.
- A stack of ordinary raws gives the same files from focus alone; one Dual Pixel frame gives its own.
- The output file names and the JSON contract.

Then one real check if Bill has a Dual Pixel stack on disk (ask him; do not search his drives).

## Done when

`stack focus` on a handheld stack of R5 Dual Pixel raws writes colour, height, normal, confidence and agreement, with
the scale said honestly; the same command on ordinary raws gives the focus-only version; the tab shows the Depth view;
tests pass (`Run Tests.bat`); `docs/CLI.md`, `docs/PROGRESS.md` and `docs/NEXT_UPDATE.md` say what changed. Bill
pushes; nothing is pushed by the session.

## Not in this build

Photometric stereo from several light directions (a later idea from the same chat; it needs no Dual Pixel and could be
its own small build). Full 3D reconstruction from many angles (that is RealityScan's job in the forge). Training any
network.

## For the RPG Forge (the reason it was asked for)

The forge's intake (`D:\RPG_Forge\forge\scans.py`) will call `studio-cli.bat stack focus <folder> --make surface
--json` when a folder of Bill's photos is a stack, and `dualpixel <file> --make surface` for a single Dual Pixel
frame, then put the five files on its Libraries shelf as "William's own". Keep the file names and the JSON exactly as
in D so the forge can rely on them.

## Status (session 16, 2026-11-21)

Built: A to F. Where it differs from the words above, and why:

* **B.3, the step.** Each pair of neighbouring frames gets its own step (a hand does not step evenly; the synthetic
  hand-held stacks vary by 39 %); s is the median of the pairs' steps, its spread is reported, and each frame's
  position goes in the JSON. The sign convention used: s = d_(i+1) - d_i, so z = i - d_i / s holds as written.
* **B.4, the weights.** Inverse variance, both variances measured from the stack itself (Dual Pixel's from the step
  calibration, focus's from how far it strays from Dual Pixel where both are sure), so on a rail focus leads and by
  hand Dual Pixel leads, by themselves.
* **B.5, filling.** Not by the edge-aware (guided) smoothing: it copies the picture's brightness into what it
  smooths, which is the "dark knot reads as a hole" fault this build exists to avoid (seen as texture stripes in the
  height and as bokeh blobs in Bill's fly stack). Holes are filled from the measured places alone (push-pull,
  Gortler et al. 1996) and still marked as filled, confidence 0.
* **B.6, the agreement map** is how far every vote at a place strays from the answer: focus against Dual Pixel as
  written, and also the frames' Dual Pixel votes against each other (a leaf that moved gives focus two peaks, focus
  then rightly says it does not know, and only the frames' disagreement shows it).
* **C, millimetres**: from `--step-mm` (and `--ratio` with the camera's pixel size for the lateral scale). The EXIF
  focus distance is not read yet (the Studio does not read Canon's focus distance tags; at macro distances they are
  too coarse to give a step anyway).
* **Which way is near** cannot come from the halves alone (see `docs/RESEARCH.md`). Until the R5 is checked on a known
  scene, a stack takes its first frame as near and says so; Flip near / far (tab) and `--near` (command line).

Not checked yet: a real hand-held Dual Pixel stack, and the R5's near/far reading. Bill is shooting both (a texture
stack with the Zeiss 50 mm, and one Dual Pixel frame of a tape measure lying at an angle), into
`test_raws\dual_pixel_stack\`. Then: run `stack focus <that folder> --make surface`, look at the Depth view, set
`NEAR_SIGN["Canon EOS R5"]` from the tape measure, and check the window size (16) on real halves.

