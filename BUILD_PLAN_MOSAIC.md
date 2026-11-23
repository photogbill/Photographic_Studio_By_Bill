# BUILD PLAN: Flat Mosaic (hand-held surfaces, stitched, with Dual Pixel height)

*Written by Claude Fable, 2026-11-23 15:15, in Bill's chat. For a Studio build session. The rules in `BUILD_PLAN.md`
section 0 and the standing rules in `CLAUDE.md` apply to everything here.*

## 0. Why, and what Bill said

The game's road texture came from one of Bill's R5 surface photos. That photo covers about 1.6 m, so in the game
the cracks repeat every 1.6 m, and the street reads as a tiled pattern. Also, the Studio didn't know the photo's
scale, so the 1.6 m figure was a guess.

- ✦ Bill 2026-11-23 14:45: "Would it help if I took more cracked pavement examples, but in a way that they could be
  panoramically stitched? That way, there's far less repetition?"
- ✦ Bill 14:47: "stitching will have to use Studio, just like it will have to use the dual pixel raw features within it."
- ✦ Bill 15:08: "I want a flat mosaic mode, and I also want it to be able to handle images shot handheld, and
  therefore differences in distance."

### 0.1 What the Studio has today, and why it isn't enough
- `engine/stack/pano.py` is **rotation-only**: the camera turns on one spot (`bundle.py`: rotations plus one focal
  length). Its own honest text says parallax from a moving camera cannot be blended away.
- Walking along a road with the camera pointing down is a moving camera. The rotation model fails on it.
- **A flat surface is the exception.** Every view of a plane maps onto every other view through a single
  perspective warp (a plane-induced homography), whatever the camera did: moved, tilted, rolled, rose or dropped.
  So a hand-held mosaic of a flat surface can be stitched *exactly*. Height changes between frames are just a
  change of scale inside that warp.
- Parts already built that this plan reuses:
  - `features.py`: corners, descriptors, `match_pair`, `ransac_homography`, `refine_lk`
  - `bundle.py`: `spanning_tree`, `undistort` / `distort`, `rodrigues`
  - `scan/ba.py`: sparse least squares
  - `scan/intake.py` and `scan/model.py`: `focus_distance_m`, and scale from focus distance in amber
  - `scan/coach.py`: shooting advice
  - `seams.py`: `refine`, `ghosts`, `largest_rectangle`
  - `pyramid.py`: blending
  - `align.flow`: local nudges
  - `depthfuse.py`: Dual Pixel depth
  - `surface.py`: colour, height, normal, confidence and agreement, tileable

### 0.2 Bill's decisions this plan must keep
- Hand-held is the normal case. Bill hand-holds 95% of the time and "shoots unpredictably" (DP3D 0.x). No
  tripod, rail or pole may be required.
- **No coins for scale.** Scale comes from the lens's recorded focus distance plus its field of view (amber). A
  tape or ruler in a frame is optional, and makes the scale cyan when used. With no scale at all, the result is
  relative and says so. Per-lens distance calibration is ruled out ("I have over 100 lenses").
- Raws only by default, Dual Pixel RAW used where present.
- Exports are named `NAME_Studio.ext`; nothing the Studio didn't write is ever replaced.
- Long jobs go through the queue (`jobqueue.py`), so they can't be cut off.
- Every method says what it is best for and weak at; results are colour-coded cyan measured / amber estimated /
  violet shared.
- **Vignetting:** Bill likes vignetting in photos, and correction is normally optional and off. For a *texture*,
  though, vignetting becomes a repeating dark blotch. So Mosaic turns it on by default, says why on the page, and
  lets him turn it off.

## 1. The model: one plane, a camera pose for every frame

### 1.1 Geometry (`engine/stack/mosaic.py`, new)
- The surface is the plane z = 0 in metres (or relative units until the scale is known).
- Each frame i has a pose (R_i, t_i) over that plane, plus the lens's intrinsics: focal length in pixels and the
  centre, with k1 distortion from `bundle.py`, solved jointly when the lens doesn't report it.
- The frame-to-plane map is `H_i = K [r1 r2 t]`. This covers tilt, roll, yaw, sideways movement **and height**.
  Each frame's camera height comes out of the solve, so differences in distance are part of the model, not an
  error.
- **Two methods, each with its own text:**
  - **Plane poses** (default): 6 numbers per frame plus the shared lens. Physically constrained, drifts least, and
    reports each frame's height and tilt.
    - Best for: a lens with a known or reported focal length, any hand-held set.
    - Weak at: a zoom whose focal length changed mid-set; a set with no frame looking roughly down.
  - **Free warps:** 8 numbers per frame, with no lens assumed.
    - Best for: a contactless vintage lens with an unknown focal length, scans, phone pictures.
    - Weak at: long strips, which drift more; it gives no camera heights.

### 1.2 Solve
1. Find features in every frame, on lens-corrected luminance. Match candidate pairs, using neighbours in time first
   and then every pair whose footprints could overlap after a first guess, so serpentine rows and return passes
   link up.
2. Fit a RANSAC homography per pair (`ransac_homography`) and polish it with `refine_lk`.
3. Build a spanning tree from the best links, chain warps to a reference frame (the most central, most
   straight-down one), and turn them into first-guess poses (a homography decomposition, keeping the candidate with
   the plane in front of the camera).
4. Run a **global adjustment** over every inlier match of every pair at once (`scan/ba.py`, robust loss). Loop
   closures between rows remove drift, so drift is never left to chaining.
5. Gauge: the reference frame fixes the plane's axes. The plane's x axis follows the main direction of travel, so
   a road strip comes out horizontal.
6. **Scale:**
   - (a) a ruler or tape marked in one frame (two clicks and a length): cyan;
   - (b) otherwise, focus distance plus field of view, from every frame that records it, with the median taken and
     the spread reported: amber;
   - (c) otherwise relative, and the JSON says so plainly.
   The output's resolution (mm per pixel) and the surface's size in metres are written only when (a) or (b) holds.

### 1.3 Nearly flat, not perfectly flat
Real ground has a crown, a slope, kerbs and potholes, and a plane alone leaves misfit there.

- **Surface shape option:**
  - **Plane:** for floors, walls and short patches.
  - **Gentle curve:** a low-order surface, plane plus quadratic, for crowned roads and long lawns. Solved inside
    the same adjustment.
- **Relief parallax is real and must be handled.** Example: a camera at 1.5 m, a 3 cm kerb lip 0.6 m from the
  frame's centre moves about 1.2 cm between views. At 0.15 mm per pixel that is about 80 px. A 5 mm crack edge at
  the same spot still moves about 13 px. Three tools, in order:
  1. **Dual Pixel orthorectification** (section 3): each pixel lifted by its own measured height and dropped
     straight down. This is the real fix, used when the frames are Dual Pixel RAWs.
  2. **Seams through flat ground:** the seam cost (`seams.pair_costs`) gains a relief term, so cuts avoid edges
     with height (kerbs, crack walls) and run through flat asphalt.
  3. **Local nudge** (`align.flow`) inside overlaps, for what is left. It is said honestly on the page: it hides
     small parallax, it does not measure it.
- A **misfit map** (violet where the plane didn't explain the overlap) is saved and shown, so Bill sees where
  relief or movement was.

## 2. Making the picture

### 2.1 Resolution, when every frame is at a different distance
- Every frame's ground resolution is computed from its pose: closer frames carry finer detail.
- **Output resolution presets**, each with its own text:
  - **Game texture, 1 mm/px** (default for Mosaic: a 10 m strip is 10,000 px long)
  - **Fine, 0.5 mm/px**
  - **Full detail:** the median frame's own resolution. This can be huge; the page shows the size before it runs.
  - **Custom**
- Warping uses proper anti-aliasing when a frame is shrunk, so close frames don't shimmer.

### 2.2 Which frame wins where
Blend weights favour, in order:
1. **finer ground resolution** (the closer frame)
2. **local sharpness** (hand-held tilt puts part of a frame out of focus)
3. **distance from the frame's centre** (less distortion, vignetting and relief parallax)
4. **less glare**

Then:
- Seams are cut on those weights (`seams.refine`), and the result is blended across the seams with the pyramid.
- **Exposure:** a gain per frame is solved from the overlaps. An even-out across the strip is optional (on by
  default), so a cloud passing during the walk doesn't leave a bright band.
- **Vignetting:** estimated from the overlaps themselves (amber), or taken from the lens's profile where one exists.
- **Things that moved or shouldn't be there** (`seams.ghosts`): blowing leaves, Bill's feet or shadow at a frame's
  edge, a passing car. Each spot is taken from one frame only, and the ghost map is saved.

### 2.3 Size and memory
A 10 m strip at Full detail can pass 70,000 px. So:
- warping and blending run in tiles and stream to disk (16-bit TIFF, PNG for the game sizes);
- a preview at about 2,000 px comes first, so the page shows the result before the full run;
- the run goes into the queue whenever the output passes about 100 MP.

## 3. Dual Pixel height for the whole strip

- Each frame's Dual Pixel depth comes from `depthfuse.py`'s single-frame path (no focus stack needed: Bill shoots
  this at a normal aperture, straight down).
- Depth is put into metres using the frame's pose and scale, then expressed as **height above the fitted plane (or
  curve)**. That removes the camera's tilt and the road's crown from the height map.
- **Orthorectification:** each frame's pixels are lifted by their own height and projected straight down onto the
  output grid before blending. This removes relief parallax at kerbs, potholes and crack walls. It replaces the
  local nudge wherever height is known (cyan); the nudge is kept only where it isn't (amber).
- The height mosaic is blended with the **same seams and weights** as the colour, so colour, height and normal
  stay in register.
- Where two frames' heights disagree by more than the agreement threshold, it is shown in `agreement`, as in
  `surface.py` today.
- Frames without Dual Pixel data still stitch. Their area of the height map is marked "not measured" (confidence
  0), and the page says how much of the strip has measured height.

## 4. Out of the Studio

- **Files**, all through `surface.py`'s writer, so the RPG Forge's intake (`scans.py`: `studio_intake`,
  `shelve_studio`) takes them with no change:
  - `NAME_colour_Studio.png/.tif`
  - `NAME_height_Studio.png`
  - `NAME_normal_Studio.png` (OpenGL or DirectX)
  - `NAME_confidence_Studio.png`
  - `NAME_agreement_Studio.png`
  - new: `NAME_misfit_Studio.png`, `NAME_coverage_Studio.png`
  - `NAME_surface_Studio.json`, which adds: kind "mosaic"; frames used and left out (and why); mm per pixel and
    size in metres with how the scale was known; each frame's height and tilt; drift at loop closures; the share
    of height measured; the method choices.
  - **The JSON's `repeat_m`** is the measured repeat size, so the game sets its tiling from a measurement instead
    of a guess.
- **Tileable** (optional, on for the "Game texture" preset): `surface.py`'s half-offset healing, applied to every
  map with one set of weights.
- **Crop:** the largest clean rectangle (`seams.largest_rectangle`) by default, or the whole covered area with
  transparency.
- **Pieces for decals** (optional, small): in **Pick pieces**, Bill draws boxes round features in the finished
  mosaic (a pothole, a patch, a manhole, a stain). Each is cut out with all its maps and a transparent edge, as
  `NAME_piece_01_Studio...`. This feeds the game's decals.

## 5. The page, the coach, the command line

### 5.1 Where it lives
- In the **Panorama** tab, a **Kind** switch: *Turn on the spot (panorama)* / *Move over a flat surface
  (mosaic)*.
- **Auto** (default) tries both fits on the matches and picks the one that explains them. If the camera clearly
  moved, it says so in one line: "These were shot while moving over a surface; using Mosaic."
- A shortcut to the same page from the **3D Scan** section, since the results are game surfaces.
- Panel: method (Plane poses / Free warps), surface shape (Plane / Gentle curve), output resolution, scale (from
  focus distance / mark a ruler / relative), vignetting, exposure even-out, tileable, crop, Dual Pixel height
  on/off. Every choice has its best-for / weak-at text.

### 5.2 The coach (`scan/coach.py`, extended for mosaics)
- After the solve: a **coverage map seen from above** (each frame's footprint, coloured by its ground
  resolution), gaps in red, weak links in amber.
- Plain-words advice, such as: "Frames 14–15 overlap 22%; aim for 60%." · "Frame 9 was shot at 2.3 m, the rest
  near 1.5 m; it is used but its detail is coarser." · "This strip has no second row or return pass, so drift
  can't be checked (estimated 0.4% over 9 m)." · "Your shadow is in frames 3–6; it was removed from the result."
- **Shooting card**, on screen and printable, from the first answer to Bill:
  - point straight down, at roughly the same height
  - 35–50 mm lens
  - about 60% overlap
  - a second row or a walk back along the strip, for drift
  - overcast light or full shade
  - polarizer if there's glare
  - Dual Pixel RAW on
  - optional ruler in the first frame

### 5.3 Command line, MCP, queue
- `stack mosaic` in `cli_tools.py` with every panel option. The command is documented in `docs/CLI.md` and
  tested in `tests/test_cli_tools.py`.
- The MCP tool appears by itself (`studio_stack` / the generated tools). Do not add it by hand.
- Queue-able like every long job.
- The RPG Forge reaches it through its `studio.run`, so Claude can stitch Bill's next pavement set end to end:
  intake, mosaic, surfaces, shelve.

## 6. Tests (synthetic, with known truth; `tests/test_mosaic.py`)

1. A textured plane of known size, rendered from **hand-held poses**: camera height 1.0–2.0 m, tilt up to 15°,
   any roll and yaw, two serpentine rows, k1 distortion, a different exposure per frame. Pass when:
   - registration error is under 0.5 px RMS
   - poses come back within 1 cm and 0.5°
   - scale is within 2% when given exact focus distances, and within the stated spread when the focus distances
     are noisy
2. A 10 m single row against the same strip with a return pass: drift reported for the single row, and removed
   with the return pass.
3. Relief: a plane with known bumps and a 3 cm kerb, plus synthetic Dual Pixel depth. Orthorectified seams show
   no doubling (edge offset under 1 px), the height mosaic matches the truth within the agreement threshold, and
   the misfit map lights the kerb when Dual Pixel height is off.
4. Gentle curve: a crowned road with a 2% crown. The plane fit leaves misfit and the curve fit removes it.
5. A moving blob and a shadow at a frame's edge: both absent from the result and present in the ghost map.
6. Mixed distances: one frame at twice the height. It is used where it alone covers the ground; elsewhere the
   closer frames win (checked by the weight maps).
7. A rotation-only set goes to Panorama, and a moving set goes to Mosaic (the Auto switch).
8. Free warps on a set with no focal length: it stitches and reports no heights.
9. Output: the JSON's fields, `repeat_m`, `NAME_Studio` naming, nothing written outside `outputs/` in tests, and
   the forge's intake accepting the files (a fixture copy of `scans.py`'s reader checks).

## 7. Order

| Step | What | Done when |
|---|---|---|
| M0 | `mosaic.py` solve: pairs, tree, plane poses, global adjustment, scale; free warps | tests 1, 2, 7, 8 pass |
| M1 | Colour: resolution presets, weights, seams, gain, vignetting, ghosts, tiled streaming, crop | tests 5, 6 pass; preview and full run on a synthetic 10 m strip |
| M2 | Gentle curve, misfit map, relief-aware seams, local nudge | test 4 passes; test 3 without Dual Pixel shows the kerb in misfit |
| M3 | Dual Pixel height: per-frame depth, metres, height above the surface, orthorectification, height mosaic | test 3 passes |
| M4 | Surfaces out: `surface.py` writer, tileable, `repeat_m`, pieces for decals | test 9 passes |
| M5 | Page: Kind switch and Auto, the panel's texts, coverage map, coach, shooting card, 3D Scan shortcut | page opens on a synthetic set; coach lines appear |
| M6 | `stack mosaic` CLI, `docs/CLI.md`, queue, MCP check, forge `studio.run` | CLI test passes; queued run finishes |
| M7 | Bill's real set (when he has shot it): first real solve; read which way "near" is on the R5 (DEPTH_STACK status) | the coach's report and the result sent to Bill |

After each part: one line in `docs/NEXT_UPDATE.md` (`- New: ...`), `docs/PROGRESS.md` kept current, tests run
before anything is called working. Do not push; Bill pushes.

## 8. For Bill (ask in chat, with a recommendation; none of these block M0–M3)
1. **Default output resolution for game textures:** 1 mm/px recommended (sharp from the game camera, a 10 m strip
   = 10,000 px). Finer is available per run.
2. **Where it lives:** the Panorama tab's Kind switch, plus a shortcut from 3D Scan (recommended), or its own tab.
3. **Pieces for decals:** cut out by hand in Pick pieces (recommended for now), or also suggested automatically
   later.

## Status (2026-11-23, Claude Fable, the same day; ✦ Bill 15:11: "Go ahead and build M0-M6")

**Built: M0 to M6.** Tests on synthetic hand-held sets with known truth (`tests/mosaic_synth.py`): `test_mosaic.py`
(10), `test_mosaic_build.py` (7), `test_mosaic_io.py` (7), `test_mosaic_page.py` (2), `test_cli_tools.py` (+1).

What the tests measured:
- Two rows of 12 views at 1.0 to 1.7 m, tilted up to 10°, k1 0.04: every camera within 1 cm and 0.5°, the scale from
  focus distances within 2 %, k1 back within 0.015, the camera heights within 2 cm.
- A view at twice the height of its neighbours joins, and the nearer views win where they overlap it.
- Bumps 40 mm high: the relief's size from parallax within 3 % (fit 83 %); the peak comes out 35 mm.
- Orthorectified, the leftover misfit at the 99th percentile falls from 1.9 px to 1.1 px.
- Vignetting is measured from the overlaps (−0.29 against the set's −0.30).
- A red blob seen by one view, and a shadow at one frame's edge, are left out by majority.

Decided while building (◆ = Claude's choice; change any of them):
- ◆ **Heights per view in the view's own picture.** Each view's offset and ramp live in its own picture
  coordinates, not on the ground. A flat ground seen by a tilted camera gives a Dual Pixel offset that is a plane
  in the picture, so ramps taken off on the ground left leftovers that read as relief.
- ◆ **Raw depth, scales matched symmetrically.** The views' raw depth (Dual Pixel offset or focus steps) is used,
  not each frame's normalised depth. The scales between views come from a symmetric fit (the smallest generalised
  eigenvector over every overlap), then the offsets and ramps from least squares. A plain least-squares fit of one
  noisy depth against another shrank or collapsed the scales.
- ◆ **Relief size measured in two passes.** The relief's size comes from raw tile shifts: phase correlation, then
  Lucas-Kanade, with no smoothing across tiles, so narrow bumps keep their full parallax. The heights are then
  matched a second time on the lifted (orthorectified) views.
- ◆ **Plane warps rectified.** Plane warps (no lens) are rectified on the assumption that views look straight down
  on average. Otherwise the reference view's tilt keystones the whole layout.
- ◆ **Majority de-ghosting.** Where three or more views overlap and disagree, the view nearest the median wins.
  Smart seams alone only move borders, so a thing inside one view's patch stayed.
- ◆ **Auto.** Focus stacks, Focus Stack results and Dual Pixel views always make a mosaic. Otherwise Auto checks
  whether neighbouring frames join by a pure turn (the singular values of K⁻¹HK). It is not fooled by a wrong focal
  length (tested at 25 % off).
- ◆ **Section 8 defaults taken:** 1 mm per pixel for games; the Kind switch in the Panorama tab; pieces typed as
  boxes.
- ✦ **Focus stacking at each spot** (Bill 15:14). Stacks come in through `scan.views` (Card to Stacks' grouping,
  the Focus Stack merge, depthfuse's fused depth), with no new code.
- **Faster matching:** `features.match` now uses exact brute-force neighbours (the same matches as the k-d tree,
  about 3× faster), for the panorama too.

Not built from the plan:
- The 3D Scan shortcut to the page (5.1).
- Drawing piece boxes on the preview (4: they are typed as `x0,y0,x1,y1;...` on the page and the command line).
- A printable shooting card (it is a dialog and is in the JSON).

Not checked yet:
- Bill's real sets (M7).
- Which way is near on the R5. The mosaic's parallax will say it, and turns the relief the right way up when it is
  the other way round (tested).
- Memory on a large set: every view keeps its full-size colour until the build.
- The whole test suite on Windows.

**Next: M7**, the first real solve on Bill's pavement set. Read the coach's lines, the misfit and coverage maps, and
whether the relief's size was measured.
