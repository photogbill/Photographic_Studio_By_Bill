# BUILD PLAN: Dual Pixel 3D (macro, surfaces, foliage, our own camera solver, models for Blender and Godot)

*Written by Claude Opus, 2026-11-21 15:00, from Bill's words in the Studio chat, after a survey of his new Dual Pixel
folder (section 0.3). Read `BUILD_PLAN.md` section 0 (the rules) and `CLAUDE.md` first; everything there holds here.
This plan builds on `BUILD_PLAN_DEPTH_STACK.md` (built, session 16): read its "Status" section and
`engine/stack/depthfuse.py`, `engine/stack/surface.py` before writing anything.*

---

## 0. Read this first

### 0.1 What Bill said (2026-11-21, afternoon)

- 14:27 "How can we optimize the dual pixel processing and what else can we squeeze out of this session? I took a lot
  of images."
- 14:30 "What other image processing gains, be it blender textures or macro photography advantages, can we glean from
  dual pixel raw. My goal here is to see if we can improve the tool before testing."
- 14:32 "I also took 360 shots of some items, can we use that to build 3d models of any item i did that to with dual
  pixel raw?" 14:34 "But we also get depth maps via dual pixel raw and we get them from 360 degrees."
- 14:36 "We need to forget the turntable, I shoot handheld and outdoors. I also shot wide open with a 50mm macro lens
  (F2), and shot focus stacking frames."
- 14:41 "I think we should build our own camera position solver, because I tend to shoot unpredictably." Things to
  consider: "Macro Photography, Photogrammetry, 360 degree depth maps, Blender, and that i'm going to add some aspects
  of these photos, be it depth maps, leaves, etc, into a godot 4 video game via blender. My tool for building the game,
  rpg forge, can utilize studio. So if you can put together a comprehensive build plan that leaves nothing out for
  Studio, i'll start building."
- Earlier the same day (04:36, 14:27): he will shoot a tape-measure frame for near/far, but "I think I have enough
  photos overall that you shouldn't need it" (true: section A.1 reads it from scenes whose near end is obvious).

Decided by Bill in this chat: **our own camera position solver** (not RealityScan's), hand-held, outdoors, unordered
shooting. Everything else in this plan is a code-design decision (Bill delegates those) or is asked in section L.

### 0.2 What exists (read before writing)

| Piece | Where | State |
|---|---|---|
| Dual Pixel decode (CR3 fifth track, CR2 shot_select, DPRSplit pair), shading GainField (fast path) | `engine/dualpixel/decode.py` | built, checked on R5 |
| Disparity (1-D NCC + LK refine, confidence), detail map, near/far table `NEAR_SIGN` (empty) | `engine/dualpixel/depth.py` | built |
| Focus Stack pass A, merges, halo guard; Dual Pixel offsets per frame when every frame is DP | `engine/stack/focus.py` | built |
| Fusion focus + DP, per-pair steps, agreement, push-pull fill | `engine/stack/depthfuse.py` | built, synthetic + 2 real files |
| Surfaces: colour, 16-bit height, normal, confidence, agreement, JSON, tileable | `engine/stack/surface.py` | built |
| Features (Harris + MOPS, k-d matching, RANSAC homography), rotation-only bundle adjustment | `engine/stack/features.py`, `bundle.py` | built for the panorama |
| Card to Stacks (finds stacks on a card, judges frames from the embedded JPEG) | `engine/stack/sorter.py`, `app/cardstacks.py` | built |
| Cut-outs (classic, SAM 2.1 optional), depth masks (Depth Anything V2, estimated) | `engine/local/masks.py`, `cli_tools.cmd_cutout` | built |
| Command line + MCP built from it | `studio/cli_tools.py`, `studio/mcp_server.py` | built |
| RPG Forge intake of photos: develops raws with `studio-cli.bat edit`, then **guesses** height/normal/AO from the photo's detail | `D:\RPG_Forge\forge\scans.py` | forge side; this plan gives it measured maps instead (G.5) |
| Forge Blender runner (profiles, `--out` folder) and graphics-card lock | `forge/blenderrun.py`, `forge/threed/cardlock.py` | forge side; the Studio respects the lock (A.6) |

Nothing in the Studio yet: camera positions for a camera that moves, meshes, glTF, foliage cards, AO, de-lit colour,
batch processing of a folder of stacks, a cache of analyses.

### 0.3 Bill's test data (surveyed 2026-11-21 15:00; nothing processed yet)

`D:\dual pixel raw test` (connected to the session as its own folder; 594 CR3, 43 GB). Read from the files' own
boxes (no decode). Two shoots, split here into stacks by a gap of more than 4 s between frames (Card to Stacks will do
its own, better split; use it):

**Shoot 1, 2026-11-17 17:47-17:57**: 212 frames, 90 mm, f/3.5-5.6, ISO 400-12800 (late light), only 57 Dual Pixel.
14 stacks: #1 2345-2346 (2), #2 2347-2369 (23), #3 2370-2395 (26), #4 2396-2417 (22), #5 2418-2441 (24), #6 2442
(1), #7 2443-2458 (16), #8 2459-2481 (23, portrait), #9 2482-2499 (18) are ordinary raws; **#10 2500 (1), #11
2501-2513 (13), #12 2514-2523 (10), #13 2524-2537 (14), #14 2538-2556 (19) are Dual Pixel**. Subjects: plants and
leaves against soft backgrounds, a berry in Bill's palm (#8), blackberry leaves on litter (#11, #12), dawn redwood
(#13, #14). Use: high-ISO macro, the focus-only path, mixed shoots, Clean before stacking.

**Shoot 2, 2026-11-21 15:09-15:26** (camera clock): 382 frames, **all Dual Pixel**, 50 mm Makro-Planar, **f/2,
ISO 100**, 1/320-1/100 s, 56 stacks of 4-15 frames (209 frames portrait). The ideal test bed:

| Use | Stacks (first-last file, frames) |
|---|---|
| **Near / far** (ground receding at an angle: the bottom of the frame is nearer) | #38 3238-3252 (15, leaf litter to the trees), #14 3096-3102 (7, blocks on gravel), #25 3160-3167 (8), #9 3058-3064 (7) |
| **Textures** | #1 3006-3012 and #2 3013-3018 (painted board), #3 3019-3027 (clover and leaves), #13 3091-3095 (concrete block), #15 3103-3110 (gravel), #17 3119-3124 and #18 3125-3129 (bark), #39 3253-3260 (brick), #40 3261-3266, #41 3267-3276, #42 3277-3282 (cracked asphalt), #47 3320-3328 (roof-tile ball), #48 3329-3342 (rock slab) |
| **Objects from several angles** (the solver's test sets) | **log**: #28-#34 3183-3221 (7 stacks around it); **angel statue on a stump**: #4-#7 3028-3052 (4); **gnome**: #52-#56 3359-3387 (5); concrete blocks #13-#14; rocks #43 3283-3287, #49 3343-3347, #50 3348-3353; turtles #51 3354-3358 |
| **Foliage (leaves for the game)** | #11 3074-3081 (fig leaf), #20 3134-3139, #21 3140-3144, #22 3145-3150 (bramble leaves on bokeh), #35 3222-3228, #36 3229-3233, #37 3234-3237 |
| **Limits** (big scenes a 50 mm macro was not made for) | #16 3111-3118 (pallets), #46 3302-3319 (a whole tree), #10 3065-3073, #19 3130-3133, #24 3152-3159 (fences) |
| One frame alone | #23 3151 |

Honest reading of the object sets: **4 to 7 stacks around each object, roughly 50-90 degrees apart**, hand-held,
outdoors. That is far sparser than photogrammetry expects (10-20 degree steps, 60 % overlap). It is exactly why our
solver uses each view's measured depth (E.3): with depth, two views 60 degrees apart can still be placed; without it,
mostly they cannot. Expect the backs and bottoms of objects to be missing from these sets; the Coach (H.4) tells Bill
how to shoot the next ones.

Thumbnails and the survey script's JSON are in `outputs\_session_depth_stack\prvw\` (git-ignored) if the builder wants
them; do not copy photos into the repository.

### 0.4 Rules that matter most here

Everything in `BUILD_PLAN.md` 0.3 and `CLAUDE.md`. In particular for this build:

1. **Measured, estimated, shared: say which, everywhere.** Colours: cyan measured, amber estimated, violet filled in /
   shared, red disagreement. Every map, every camera, every mesh region carries how it was made, in the JSON and on
   screen. Nothing estimated is ever passed off as measured (a de-lit colour, a guessed roughness, a filled hole).
2. **Measure two ways and trust the agreement** (Bill's rule from his RF work, already the heart of the Depth Stack).
   This build adds two more pairs: Dual Pixel depth against multi-view stereo depth (E.6), and our camera positions
   against RealityScan's when RealityScan happens to be installed (E.9, optional cross-check, never required).
3. **No new required packages.** NumPy, SciPy, Pillow, rawpy, tifffile, PySide6 only. Marching cubes, quadric
   decimation, glTF, the solver: our own code. An optional package (for speed only) goes in `lab_packages\`, is never
   needed to start, and needs a GPL-compatible licence recorded in `THIRD_PARTY.md`.
4. **Bill does not use the command line**; the Forge and Claude do. Everything Bill runs has a page or a `.bat`.
5. **Each area its own sidebar section** (Bill, 2026-11-11). The 3D work gets its own section (H.1), not more tabs
   inside Focus Stack.
6. **Errors report themselves; comprehensive logging; nothing on C:\ by default; `NAME_Studio.ext`; never replace a
   file the Studio did not write; `docs/NEXT_UPDATE.md` line for every part.**
7. Tests on synthetic data with known truth for every algorithm (section I), then the real checks named in section I.6.

---

## A. Foundations (first: everything else stands on these)

### A.1 Which way is near, read from Bill's scenes

The halves cannot tell near from far on their own (see `docs/RESEARCH.md`, session 16), so read it once from scenes
whose near end is obvious, and lock it only when independent scenes agree.

1. Stacks #38, #14, #25, #9 of shoot 2 (ground receding: the bottom of the frame is nearer), plus any single frame
   from them. For each frame: the Dual Pixel offset (`depth.disparity`, confident places only) regressed against the
   image row, in the upright frame. A receding ground plane gives a monotonic offset-versus-row; its sign against
   "bottom = near" gives the sign of the offset in front of focus (careful: the offsets live in the sensor frame,
   before turning upright; `orient` 8 is a portrait frame, so "bottom of the upright frame" is a sensor column).
2. Every frame of every one of those stacks votes. Lock `NEAR_SIGN["Canon EOS R5"]` only when at least three stacks
   vote the same way with no stack against (a stack whose fit is weak abstains). Write the votes in `docs/RESULTS.md`
   and `docs/RESEARCH.md` (then: "checked on Bill's scenes 2026-11-21").
3. A tool for anyone's camera: `dualpixel FILES --make near-check` (command) and a button in Dual Pixel Lab, "Teach
   the Studio which way is near", which runs the same vote on frames the photographer picks (any receding ground,
   path, table) and saves the answer to the camera's profile (`profiles/`, shared-able, violet when it came from the
   shared database, cyan when measured on your own frames). `NEAR_SIGN` becomes a fallback table read after the
   profile.
4. When Bill's tape-measure frame arrives, it is one more vote, not a requirement.
5. Once the R5 is known: a stack's near end is measured from its Dual Pixel offsets whatever order the frames were
   shot in (already coded in `surface.decide_near`); the "assumed: first frame" wording disappears for the R5.

Done when: the R5's sign is locked from at least three of Bill's scenes, the profile mechanism works for any camera,
`decide_near` reports "measured" on shoot 2's stacks.

### A.2 Read the camera's focus distance (for scale)

Canon writes a focus distance in its maker notes for many lenses (FocusDistanceUpper / Lower, Canon FileInfo or
CameraInfo, depending on the body; the builder verifies on Bill's R5 files which tag and which units; ExifTool's Canon
tables are the reference to check against). The Makro-Planar ZE has a chip (its focal length and aperture are
recorded) but its lens name is blank in these files, so check whether it reports a distance at all.

- If it does: `metadata.read_metadata` gains `focus_distance_m` (and the upper/lower bounds), shown amber
  ("recorded by the lens, coarse"). Used by C.6 and E.7 as a starting scale, never as a measurement of height.
- If it does not (likely for adapted or older-chip lenses): say so in the scale notes; scale comes from C.6 b/c.

### A.3 Speed: Dual Pixel frames

Measured in session 16 on the 2-core build machine: shading fit was 39 s (now under 1 s); the offset search (17 s at
48/16, more at 16/4) and the double decode remain. Shoot 2 alone is 382 Dual Pixel frames, so this comes first.

1. **Decode once.** `frames.dual_halves` opens the photo again although pass A already decoded it for `scene_rgb`.
   New `frames.develop_and_halves(path, edit)`: one `api.open`, the second frame read from the same bytes, the
   half-size developed frame and the two halves' luminance returned together. Pass A calls it when the stack is Dual
   Pixel.
2. **Second frame from memory.** `decode._cr3_second_frame` copies the whole CR3 (70-120 MB) to the temp folder to
   patch three tables. Instead: read the file into a `bytearray` once, patch the tables there, and hand LibRaw a
   file object (`rawpy.imread(io.BytesIO(buf))`; verify rawpy accepts a file object in the version installed; if not,
   keep the temp copy and say why in the code). The same bytes serve the A+B decode.
3. **Shading once per stack.** Same lens, same aperture: fit `GainField`s on the reference frame (or pooled from three
   frames) and reuse them for every frame; refit only if a frame's level ratio drifts (focus breathing moves the exit
   pupil a little; measure on shoot 2 how much the gain changes across a stack and write it down). Then cache per
   (camera, lens, focal length, aperture) in the camera profile (cyan, measured), so later stacks skip the fit.
4. **Search only where the answer can be.** After the first two frames of a stack the step s is known roughly; for
   frame i the expected offset at a place is `s * (q_i - z)` from the focus estimate. Search +-2 px around it instead
   of +-8, and compute the correlation sums with integral images sampled on the grid (the grid step is 4: no need to
   filter every pixel). Keep the full search as the fallback where the prediction is unsure (low focus quality).
5. **Frames in parallel.** Bill's PC: 14 cores, 64 GB (`/topics/hardware`). A process pool for the Dual Pixel step:
   workers = min(cores // 3, free RAM // 6 GB), each worker one frame (decode + halves + offsets), results written to
   the stack's scratch memmaps. Progress and Cancel keep working (the existing `_tick` / `cancel` pattern). The engine
   already has `engine/parallel.py`: use it if it fits, extend it if not.
6. **The graphics card, later and optional.** The offset search is ideal for the RTX 3080 Ti, but no GPU package is
   installed. If added (CuPy, MIT, in `lab_packages\`), it must take the RPG Forge's card lock
   (`D:\RPG_Forge\forge\threed\cardlock.py`; same protocol, from the Studio side) when the Forge might be rendering,
   and fall back to the processor silently. Not in the first pass.
7. **Benchmark.** `tools/dp_bench.py` + `DP Benchmark.bat`: seconds per Dual Pixel frame, per step, before and after,
   on one stack of shoot 2; numbers into `docs/RESULTS.md`. Target: 4x faster per frame on Bill's PC; report what it is.

### A.4 A cache of analyses

Shoot 2 has 56 stacks; Bill will make surfaces, then foliage, then models from the same stacks. Pass A (half-size
frames, energy, Dual Pixel offsets) is the expensive part and does not change with merge or surface settings.

- `analysis_cache/` beside the app (`paths.py`), one folder per stack, keyed by the frames (path, size, mtime) and
  `FocusParams.analysis_key()`. Stores the memmaps (`aligned`, `energy`, `dp_disp`, `dp_conf`) and the small fields
  (transforms, flows, gains, box, notes) as `.npz` + JSON.
- `focus.analyse(..., cache=True)` reuses a hit; the Focus Stack page and the commands say "from the cache" in the
  log. Size cap (default 60 GB, a setting; oldest first), and a "Clear the analysis cache" button in Settings.
- Never stored in the repository; never on C:\.

### A.5 Folders of stacks, in one go

- `engine/stack/batch.py`: given a folder (or a card), find the stacks (Card to Stacks' `sorter`; its frame judging
  applies), group stacks that look at the same object (E.2 gives the grouping once the solver runs; before that, by
  time and by the match graph of their reference frames), and run a list of jobs over them (surface, foliage, model)
  with a queue that **resumes** after a crash or a Cancel (a `batch.json` in the output folder saying what is done).
- Command `stacks FOLDER` (data: the stacks found, kinds, frames left out with reasons, Dual Pixel or not, groups) and
  `stacks FOLDER --run surface|depth|foliage|model` (each stack or group through the command it names; one JSON result
  with one entry per stack).
- In the app: Card to Stacks gains "Make surfaces for every stack" and the new 3D section's Sets page (H.2).

### A.6 Mixed shoots

Shoot 1 has stacks where only some frames are Dual Pixel. Today pass A uses Dual Pixel only when every frame has it.
Change: use the Dual Pixel offsets of the frames that have them (the fusion already copes with frames whose
confidence is 0); say "Dual Pixel on 9 of 14 frames" in the notes and the JSON. Test with a synthetic stack where every
other frame has no halves.

---

## B. Macro photography gains (Focus Stack)

All of these use what pass A already measures. Each is a choice the photographer can see, with best-for and weak-at.

### B.1 Merge from the measured depth ("Measured depth" method)

For Dual Pixel stacks: pick each place's frame from the fused depth (`depthfuse.fuse`), not from contrast. Contrast
picks a blurred bright edge as "sharp" and makes halos at depth edges; the Dual Pixel offset is near zero only where a
frame really is in focus. New merge method `measured` in `focus.METHOD_INFO`: weights = a tent around the fused depth
in frame positions (`res.positions`), the pyramid method's detail picking kept for fine overlapping detail (hairs) only
where the measured depth says two frames are both in focus. Default for Dual Pixel stacks; ordinary stacks keep `auto`.
Test: synthetic DP stack with a bright foreground edge over a busy background: halo energy at the edge lower than
`auto`, PSNR not worse elsewhere.

### B.2 No ghosts where something moved

Where the agreement map is red (votes stray more than half a step), take that place from the single frame where it is
sharpest instead of blending frames (blending a leaf that moved makes a double leaf). Soft-edged (the agreement map
smoothed), on by default for every method, a tick "Take moving parts from one frame". Test: the moving-patch stack:
ghost energy in the patch near zero, no seam visible (gradient energy at the patch border within 1.2x of elsewhere).

### B.3 The gap map and the focus coach

The stack knows each frame's real plane of focus (`res.positions`) and, from the blur per step, each frame's depth of
field in steps. A place whose depth is farther than half a depth of field from every frame's plane was sharp in no
frame: a gap. Map it (amber overlay in the Depth view, a share in the JSON), and say it plainly: "Between frames 6 and
7 the focus jumped 2.3 steps: places at that depth are a little soft. Rock a little slower there next time." Without
Dual Pixel, positions are frame numbers and the coach says so. Test: synthetic stack with one frame removed: the gap
found at the right depth.

### B.4 Sharpen the gaps (known-blur deconvolution)

In a gap, the blur of the nearest frame is known: its radius from the place's distance to that frame's plane (the
disc radius per step is measured from the Dual Pixel offset per step: on the R5 the two half-disc centroids are 0.849 R
apart, so R per step = s / 0.849 in full-size px; verify this ratio on Bill's frames by fitting the blur of an edge
against its offset and write the measured value in `docs/RESULTS.md`). Non-blind, spatially varying deconvolution of
that frame in gap places only (Richardson-Lucy or Wiener in overlapping tiles with the disc PSF; the detail guard from
Clean against ringing; capped at a blur radius of 3 px: beyond that say "too far out of focus to recover").
Background: Xin et al., ICCV 2021 (defocus map estimation and deblurring from a single dual-pixel image), verify before
relying. Amber in the result's "how made" map. Off by default, a tick "Sharpen places no frame had in focus".

### B.5 Highlights: the extra stop, in a stack

A Dual Pixel frame holds about a stop more highlight in its halves (`dualpixel/hdr.py`). For wet leaves and insect
eyes: at full-size build, each frame's clipped places recovered from its halves before merging. Costs a second decode
per frame at build: only when the stack has clipped places (count at pass A). A tick "Recover highlights from the
halves" (on when clipping is found). Test: synthetic DP stack with a specular spot clipped in A+B but not in a half.

### B.6 Depth of field after the fact

The all-in-focus merge plus the measured depth: render a chosen plane of focus and aperture ("Keep sharp from here to
here, soften the rest", with a slider for how soft), so a stack whose background became too busy can get its bokeh
back. Layered, occlusion-aware gather blur by depth (near layers drawn over far ones, colour bleeding guarded at depth
edges by the agreement and confidence maps), disc kernel to match the lens's own bokeh shape; filled-in places blur
like their neighbours. In Focus Stack as a finishing panel and in Develop for any stack result (the depth travels in
the result's sidecar: B.7). Cyan where the depth was measured, amber where filled.

### B.7 The depth travels with the result

`build()` writes the fused depth, its confidence and labels into the intermediate's sidecar folder
(`NAME_Focus_Studio.depth.npz`, half size, float16) and the sidecar JSON points at it. Develop then offers:

- **Measured depth masks** (the mask system's depth shape gains a "measured" source beside Depth Anything; cyan): select
  by distance, haze by distance, darken the background. One Dual Pixel photo can also save its depth for this.
- **Relight** with measured normals instead of estimated ones when the photo has them.
- **Depth of field** (B.6).

### B.8 Card to Stacks with Dual Pixel

When frames are Dual Pixel, "adds nothing" and "soft everywhere" can use the measured in-focus area (|offset| near 0,
confident) per frame instead of contrast from the embedded JPEG. This needs the halves (slower than the JPEG pass), so
only on request ("Judge with the halves") or when the stack is going to be analysed anyway.

---

## C. Surfaces for games (textures)

Extends `surface.py`; every new map is a `--maps` choice and goes in the JSON under the same contract (version bump to
`"studio_surface": 2`, every version-1 key kept).

### C.1 Ambient occlusion and cavity

From the height in real or relative units: horizon-based AO (8 directions, radius in pixels a setting with a default
of 1/32 of the long edge), cavity (negative curvature, small radius), curvature (signed). Tileable-aware (wrap at the
edges when tileable). Measured-from-height = cyan in spirit, but say in the JSON "computed from the measured height".

### C.2 De-lit colour (albedo, estimated)

The colour photograph holds the light. The height is measured independently of brightness, so the shading can be
estimated from the shape: fit one directional light plus ambient so that `luminance ~ albedo * (ambient + k max(0,
n.l))` with a smooth albedo prior, robustly (shadows and speculars out), then `albedo = colour / shading` smoothed at
the shading's own scale only (never the texture's fine detail). Written as `NAME_albedo_Studio.png` (amber, estimated)
beside the colour, never replacing it; the JSON gives the light direction found and the share of the picture where the
fit is trusted. Test: synthetic height + known albedo + known light: albedo error well under the shaded colour's.

### C.3 Fine relief (estimated) kept apart from the measured height

The measured height is coarse (windows about 32 px). A texture artist often wants fine relief too. Offer it honestly:
`NAME_detail_Studio.png`, a high-passed relief estimated from the de-lit albedo's shading residual (amber), as its own
file, never mixed into the measured height. The JSON says how to combine (and Blender's material from F.3 adds it as a
second bump, strength 0 by default).

### C.4 Roughness and specular: not measured

Nothing in these photos measures shine. Do not invent it silently: `roughness` is either not written (the default) or,
when asked (`--maps roughness`), an amber estimate from the albedo's local variance with the JSON saying so. Note for a
later build: a stack shot twice with the light moved would give it (out of scope; see K).

### C.5 Better tiling

The session-16 tileable blends copies (ghosting in the blend band). Add a min-error seam: in each blend band, a
dynamic-programming cut through the place where the copies differ least (image quilting, Efros and Freeman 2001, as an
idea), feathered over a few pixels; one cut for every map (colour, height, normal from the tiled height, AO from the
tiled height). Keep the blend as the second method (best for even, random textures; the cut for structured ones such
as brick). Test: a synthetic brick pattern: no mortar line crossing a ghost.

### C.6 Scale in real units

Height in mm and pixel size in mm are what a game needs to size a texture and set its displacement.

a. `--step-mm` (rail) and `--ratio` (magnification), as now.
b. **A known object in the frame**: the photographer clicks two points a known distance apart (a coin's diameter, a
   ruler) on the reference frame, or the Studio finds a coin (circle fit on the all-in-focus frame, size chosen from a
   short list: US quarter 24.26 mm, penny 19.05 mm, 1 euro 23.25 mm ...). Gives mm per pixel at that depth; with the
   magnification known from it, the Dual Pixel offset per step converts to mm per step through the lens formula (C.6 d).
c. **The focus distance from the file** (A.2), amber, as a starting value.
d. **Dual Pixel physics, calibrated once**: at a given lens and aperture, offset per unit of defocus is a constant of
   the sensor and lens (Garg et al. 2019). One stack with a known object (b) or a rail (a) measures it; saved in the
   camera profile per (lens, focal length, aperture); afterwards every stack and every single Dual Pixel photo at that
   setting gets mm heights (cyan, "calibrated on <date> from <how>").
The JSON says which of a-d gave the scale. Ask Bill (L.1) to put a coin in one frame of each new set.

### C.7 Game-ready sizes and conventions

- Sizes: power-of-two outputs on request (`--size 2048|4096`), square crop from the measured area's best part, mip-safe
  (no single-pixel detail at the edges of alpha).
- Normal maps: OpenGL (Godot and Blender: green up) by default, as now.
- **Godot heightmap orientation**: Godot 4's StandardMaterial3D heightmap (parallax) has its own convention and a "flip
  texture" option; verify on Godot 4 with the Forge's Godot connection which way white goes and write it into the JSON
  as `godot.heightmap_flip` (true / false) so the Forge sets it without guessing. Do not change the height file itself
  (white = near stays the contract).
- **ORM** (occlusion, roughness, metallic in R, G, B; Godot's ORM material and glTF's convention): `NAME_orm_Studio.png`
  when asked, roughness from C.4 if asked else a constant the JSON names, metallic 0.
- sRGB for colour and albedo; linear for everything else (stated per file in the JSON).

### C.8 Decals and scatter

From a texture stack: pieces that stand out from the surface (a leaf on asphalt, #41) cut out with alpha from the
measured height and colour (C.9's mask method), packed as decals (D.3's atlas) for scattering in the game.

### C.9 Subject masks from measured depth

Outdoors there is no plain backdrop, so cutting out by colour fails. The fused depth separates subject from background:
the connected measured region nearest the camera that the photographer clicks (or the largest), grown along the
colour edges for the outline (the classic cut-out engine for the edge, the depth for what is in and out). Places where
nothing was measured (the bokeh) are background by definition. `cutout --depth-measured` (command) and a "Cut out by
measured depth" button on the Depth view. Used by D (leaves), E (masks for the solver and the carving), C.8 (decals).

---

## D. Foliage for the game (leaves, twigs, small plants)

Bill's words: "leaves, etc, into a godot 4 video game via blender".

### D.1 Leaf cards

From a stack (or one Dual Pixel photo) of a leaf or sprig against a soft background (shoot 2 #11, #20-22, #35-37;
shoot 1 #11-#12):

- Alpha from C.9 (measured depth + colour edge), edges anti-aliased, colour bled into the transparent margin (so
  mipmaps do not draw a dark halo), premultiplication stated in the JSON (straight alpha).
- Colour (the merge), albedo (C.2), normal from the measured relief (a leaf's cup and fold), a **thickness /
  translucency** map (amber: from the leaf's luminance under back light where the frame shows light through it, else
  a constant), and the card's real size in mm when scale is known (C.6).
- The leaf's 3-D bend as a low mesh (a grid over the alpha with the measured height, 200-800 triangles) as well as the
  flat card: games use both.

### D.2 Several leaves from one frame

When a frame holds several leaves, each separate measured region becomes its own card (the photographer can tick which
ones to keep on a page of thumbnails).

### D.3 Atlas

Many cards into one texture sheet (power of two), packed by size with padding (shelf packing is enough), every map
packed with the same layout; a JSON with each card's rectangle in UV, its size in mm, its pivot (the stem end, from the
lowest point of the alpha, or clicked).

### D.4 Hand-off

Cards and atlases go out as PNGs + JSON and as glTF (F.1): one glb per card or atlas, a quad (or the bent mesh) per
card, alpha-scissor material (glTF `alphaMode: MASK`, cutoff 0.5), double-sided. Godot notes in the JSON (alpha
scissor, back lighting from the thickness map, wind via vertex colour: the stem end painted black, the tip white, so the
Forge's foliage shader can sway it).

Command: `foliage FRAMES_OR_FOLDER --make cards,atlas,glb [--size 2048] [--pick N]`. Page: in the new 3D section
(H.5).

---

## E. 3-D models from hand-held Dual Pixel stacks: our own camera position solver

Bill: "I think we should build our own camera position solver, because I tend to shoot unpredictably." The design
target is his real shooting: hand-held, outdoors, unordered, few wide views (4-7 per object in shoot 2), every view a
focus stack at f/2 with Dual Pixel. The edge over ordinary photogrammetry: **every view brings its own measured depth**
(the fused depth of its stack) and a measured subject mask (C.9). Classical structure from motion throws that away and
needs many narrow views; ours uses it to place wide views and to fill what triangulation cannot.

### E.1 The view

One view = one stack (or one Dual Pixel photo): its reference frame's geometry (the stack's box frame, upright), the
all-in-focus merge at the size the solver works at (long edge 2048 for matching, full size for texturing), the fused
depth `t` (steps, with confidence and labels), the subject mask, and the camera data (focal length from EXIF, sensor
pixel pitch from `surface.PIXEL_MM`, focus distance if A.2 found one). Views come from the analysis cache (A.4).

### E.2 Features and matching, for unordered sets

- **Detection**: multi-scale (difference of Gaussians over 4 octaves) on the all-in-focus image, inside the subject
  mask (plus a margin) by default so the moving background does not vote; an option to use the whole frame for sets
  where the background is static (a bench in a garden).
- **Description**: oriented and scale-normalised (the panorama's MOPS is upright and single-scale: orbiting an object
  rolls and rescales it). A gradient-histogram descriptor of the SIFT kind (Lowe 2004; the patent expired in 2020;
  our own NumPy code), 128 values, root-normalised (RootSIFT, Arandjelovic and Zisserman 2012).
- **Matching**: every pair of views (shoot 2's largest object set is 7 views; a few hundred views is still fine with
  k-d trees), ratio test, mutual best, then geometric verification (E.3). The match graph's connected components are
  the objects: views of the log fall together, textures stay alone. A view placed in no component is listed with a
  reason ("no shared detail with any other view").
- **Depth-assisted matching** (ours): each matched point has a measured depth in both views; after a first pose
  estimate, matches whose depths cannot agree under any consistent scale are dropped, and new matches are sought along
  the predicted position (guided matching), which matters at 60-90 degree baselines.

### E.3 Two views

Calibrated camera (K from EXIF; focal length refined later). For each verified pair:

- Essential matrix by the normalised 8-point method on calibrated coordinates with RANSAC (5-point Nistér is better
  for few points; implement 8-point first, 5-point later if tests demand it), and a homography in parallel; choose by
  which explains the matches better (planar textures and walls give homographies: the GRIC idea, Torr 1998).
- **With depth** (ours): each view's fused depth is an affine function of inverse distance, unknown per view (offset =
  the plane of focus, scale = the step), over the shallow range of a macro view nearly affine in distance. So matched
  points give 3-D points in each view up to a per-view (a, b); a 3-point similarity RANSAC (Umeyama 1991) on those,
  with a and b solved jointly, gives the relative pose even when the 8-point method is unstable (wide angle, few
  matches). Both are tried; the pair keeps the one with more inliers and says which ("pose from matches", "pose from
  depth", "both agree": cyan when both agree within a threshold, amber when one alone).

### E.4 Many views

Incremental: start from the strongest pair (most inliers with a real baseline), triangulate, then add the view with the
most matches to already-triangulated points by PnP (a 6-point DLT then refined; EPnP optional) with RANSAC, or by the
depth similarity of E.3 when 2-D/3-D matches are few; triangulate new points; repeat; bundle-adjust every few views.
Order of shooting is never assumed.

### E.5 Bundle adjustment with Dual Pixel depth

`scipy.optimize.least_squares` (trf, Huber), sparse Jacobian pattern (as `bundle.py` does). Parameters: each view's
pose (6), each view's focal length (a tight prior around EXIF: focus breathing changes it by up to 2 %), one radial
distortion k1 per lens (shared), each view's depth affine (a_i, b_i), the 3-D points. Residuals: reprojection in
pixels, and **depth residuals**: a point's depth in view i against that view's fused depth at its pixel, weighted by the
fused confidence (and by the agreement map: red places weigh nothing). The depth terms hold wide, few-view sets
together; the reprojection terms hold the fine geometry. Report per view: reprojection RMS, depth RMS, points seen.
Outliers (reprojection > 3 px or depth > 3 sigma) removed in rounds.

### E.6 Dense depth, two ways

For each view, two depth maps in the shared frame:

1. The **Dual Pixel fused depth**, turned into distance by the view's solved (a_i, b_i).
2. **Multi-view stereo**: plane sweep against its two or three best neighbour views (normalised cross-correlation in
   7 x 7 windows on the all-in-focus images, depth hypotheses around the Dual Pixel depth +- its uncertainty, which
   cuts the sweep tenfold), winner-take-all with a left-right check.

Fused per pixel by their measured noise (the Depth Stack's pattern), with **an agreement map per view**: places where
Dual Pixel and stereo disagree are red and do not enter the mesh. Stereo covers detail Dual Pixel's windows blur;
Dual Pixel covers plain areas stereo cannot match only where it measured something, and nowhere else.

### E.7 Scale

The solved model is up to scale. Scale from (in order): a known object clicked or found in any view (C.6 b), the Dual
Pixel calibration of C.6 d (offset per mm at this lens and aperture: one number turns every view's (a_i) into mm), the
focus distance from the files (A.2, amber), else "relative" with a scale bar the photographer can set later. The JSON
says which, and the glTF is written in metres only when the scale is not relative (else in "model units" with a note).

### E.8 The mesh

- **Volumetric fusion**: a truncated signed distance volume over the object's bounding box (from the cameras and
  points), each view's fused depth integrated with its confidence as weight (Curless and Levoy 1996), resolution from
  the pixel footprint (a setting, default so the volume fits in 4 GB); views' subject masks carve away what every
  view sees as background (visual hull, Laurentini 1994), which closes the object where no view saw the surface (the
  back, the underside) with an honest, smooth hull marked violet ("filled, not measured").
- **Surface**: marching cubes (our own, the standard tables), largest component, small holes closed (violet), light
  Taubin smoothing that keeps volume.
- **Texture**: each face takes its colour from the view that sees it most squarely, sharpest and nearest (the stacks
  are all-in-focus, so sharpness is mostly the view angle); faces with the same view form charts; charts packed into an
  atlas (4096 max), colour levelled between charts (a gain per chart solved over the seams: outdoor light changes
  between stacks), seams feathered. A second texture records how each texel was made (cyan measured, violet filled).
- **Normal map and LODs**: the full mesh decimated by quadric edge collapse (Garland and Heckbert 1997, our own) to a
  game mesh (default 30 000 triangles; LOD1 7 500, LOD2 2 000), the full mesh's detail baked into a tangent-space normal
  map for each LOD. Godot 4 makes its own LODs on import as well: write LOD0 always, the others on request.
- **Collision**: a convex hull or a small set of convex pieces (approximate convex decomposition is out of scope; a
  single hull plus the decimated LOD2 as an optional trimesh collider).
- **Ground**: objects on the ground (the log, the gnome) usually include some ground; a "Cut at the ground" option
  fits a plane to the lowest measured region and removes everything below a small margin, with the cut edge marked.

### E.9 Cross-check with RealityScan (optional)

If RealityScan's command line is installed (the Forge's `tools.list` knows), the 3D section can run it on the same
all-in-focus images and masks and read back its cameras; the report gives the angle and position differences per view
(two methods agree: cyan; disagree: red). Never required; the Studio's solver is the product.

### E.10 What to show the photographer about quality

Per object: views placed / given, mean reprojection error, mean depth agreement, how much of the surface is measured
(cyan) versus filled (violet), the largest unseen region and from where to shoot it next (H.4).

---

## F. Out to Blender and Godot

### F.1 glTF 2.0 writer (our own)

`engine/io/gltf.py`: binary glTF (.glb) with meshes (positions, normals, UVs, tangents when a normal map is used,
computed MikkTSpace-style or left for the importer: verify Godot and Blender both accept no tangents and compute them;
write them if either does not), materials (PBR metallic-roughness: base colour sRGB, normal, ORM, alpha mode),
embedded PNG textures, LODs as separate meshes named `_LOD1`, `_LOD2`, units in metres, Y up (glTF's convention; Blender
and Godot convert), extras holding the Studio's "how made" summary. Validate by reading back our own files and against
the glTF 2.0 schema's required fields; check once by importing into Blender (the Forge's Blender runner) and Godot 4
(the Forge's Godot connection), results written in `docs/RESULTS.md`.

### F.2 Other files

OBJ + MTL (for any tool), PLY point clouds (each view's measured points with colour and a "how measured" field),
the camera positions as JSON (and in a simple format Blender can read through F.3's script).

### F.3 A Blender script, not an add-on

`studio/blender/build_material.py` (runs inside Blender, Python standard library + `bpy` only): given a surface JSON
or a model folder, builds the material (Principled BSDF; colour or albedo; normal map node in OpenGL mode;
displacement from the height with the scale from the JSON, or the detail map as a second bump at strength 0), a
displaced plane for a texture or the imported glb for a model, cameras placed where the solver put them (a check
anyone can see), and saves a .blend beside the files. The Forge runs it through `forge/blenderrun.py` (it passes
`--out`); the Studio's "Open in Blender" button runs Blender directly when Bill has set its path in Settings (never
guessed, never required). The script is GPL like the Studio.

### F.4 Godot

The Studio does not write Godot scenes (the Forge owns the game). It writes everything a Godot import needs and says
it in the JSON: `godot.import`: normal map = OpenGL (no flip), heightmap flip (C.7, verified), sRGB / linear per
texture, alpha scissor for foliage, suggested scale in metres, collision files. Add a short section to `docs/CLI.md`:
"Getting Studio files into Godot 4 through Blender or directly as glb".

---

## G. Command line, MCP and the RPG Forge

### G.1 New commands (each with a test in `tests/test_cli_tools.py`; the MCP server picks them up by itself)

| Command | Makes |
|---|---|
| `stacks FOLDER [--run surface,depth,foliage,model]` | the stacks found (data), or each stack / object through those jobs; resumable |
| `stack focus FRAMES --make surface --maps colour,albedo,height,detail,normal,ao,cavity,orm,confidence,agreement,gaps` | the surface files chosen; `--size`, `--tileable cut|blend`, `--scale coin|click:...` |
| `foliage INPUTS --make cards,atlas,glb` | leaf cards, atlases, glb |
| `cutout INPUTS --depth-measured [--click U,V]` | subject masks from measured depth |
| `model FOLDER_OR_STACKS --make cameras,points,mesh,glb,obj,blend [--triangles N] [--lods] [--ground-cut]` | the 3-D model and its report |
| `dualpixel FILES --make near-check` | the near/far vote (A.1) |
| `profile-dp FRAMES --lens ... --aperture ...` | the Dual Pixel scale calibration (C.6 d) |

Every command keeps the contract (`--json`, `NAME_Studio.ext` / `NAME_<kind>_Studio.ext`, the set moves together to
`-2` when a name is taken, nothing replaced that the Studio did not write, exit codes 0/1/2), and every new setting is
in `schema` with what it does.

### G.2 Long jobs

`model` and `stacks --run` take minutes to hours: progress lines on standard error as now, a `--resume` flag, and a
`*_job.json` in the output folder (state, what is done, what failed) so the Forge's queue can poll it.

### G.3 Contracts the Forge relies on

- `NAME_surface_Studio.json` version 2 (C), `NAME_foliage_Studio.json` (D.3), `NAME_model_Studio.json` (files, scale
  and how known, views placed, quality numbers, measured / filled shares, Godot import notes). Each has
  `"written_by"` (so the Studio replaces its own), a version number, and keeps every key of earlier versions.
- File names exactly as in this plan; a test per contract.

### G.4 Performance under the Forge

When called by the Forge, respect its card lock if the GPU path ever exists (A.3.6), run below normal priority when
`--background` is given, and cap workers with `--workers`.

### G.5 For the Forge (forge side, not built here; write it into `docs/CLI.md` for the Forge's builder)

`forge/scans.py` today guesses height, normal, roughness and AO from the photo's detail. Proposed: a texture folder of
Dual Pixel stacks calls `studio-cli.bat stacks <folder> --run surface --json` and shelves the measured maps (labelled
measured / estimated as the JSON says); a scan folder calls `model`; leaves call `foliage`. RealityScan stays the
Forge's fallback for sets the Studio cannot place.

---

## H. The app

### H.1 A new sidebar section: 3D Scan

Its own area (Bill's rule), with pages:

### H.2 Sets

Point it at a folder (or the Library's selection): stacks found (Card to Stacks' judging), grouped into objects and
textures automatically (E.2 match graph, refined after solving), each group a card with its thumbnails, Dual Pixel or
not, frames left out and why. Buttons per group: Surfaces, Foliage, Model. "Do them all" runs the batch (A.5) with a
progress panel that survives closing the page.

### H.3 Cameras and Model

A 3-D view of the solved cameras (frustums coloured by how each was placed: cyan both ways, amber one way, red not
placed) around the point cloud, then the mesh: a software turntable render (NumPy rasteriser, Lambert + texture; no
OpenGL dependency) the photographer can drag to turn, colour-coded overlay of measured / filled surface, the quality
numbers of E.10 in plain words, buttons: Export glb, Export for Blender, Open in Blender.

### H.4 Coach

For an object: a ring diagram of the angles covered and the gaps ("You have the front half from about 0 to 200 degrees;
the back and the underside were never seen. Shoot 6-8 more stacks around the back, and a few from lower down, each
overlapping its neighbour by about half"). For stacks: the focus gaps of B.3. For scale: "No coin or ruler was found:
the model is in relative units". Plain, short, actionable. Also printable as a one-page checklist ("How to shoot a 3D
scan with the Studio").

### H.5 Foliage page

Thumbnails of every leaf found in the chosen stacks, tick to keep, pivot click, atlas preview, export.

### H.6 Focus Stack and Develop additions

Depth view: gap overlay (B.3), the Measured depth merge (B.1), the moving-parts tick (B.2), Sharpen gaps (B.4), Recover
highlights (B.5), Depth of field panel (B.6), "Cut out by measured depth" (C.9), "Save surface for games" with the new
maps (C). Develop: measured depth masks, relight with measured normals, depth of field (B.7). Dual Pixel Lab: "Teach
the Studio which way is near" (A.1.3), "Calibrate Dual Pixel scale" (C.6 d).

---

## I. Tests

### I.1 Synthetic stacks (extend `tests/depth_synth.py`)

Speed changes must give identical results (A.3: decode once, from memory, shading once: max abs difference of the
offsets within 0.02 px of the slow path; the predicted search: within 0.05 px where the prediction is sure). Mixed DP
(A.6). B.1-B.6 as each section says. C.1-C.5 against analytic heights (a sphere cap, a sine field, bricks).

### I.2 Synthetic views for the solver

`tests/scan_synth.py`: a textured object (a bumpy log: a cylinder with a displaced, textured surface lying on a
textured ground plane) rendered by a small NumPy ray caster from N known camera poses, hand-held: unordered angles,
some rolled, distances varying 15 %, with each view's true depth turned into a "fused depth" with a random per-view
affine and noise like the Depth Stack's (and its confidence), and a subject mask. Cases:

- 16 views 22 degrees apart: rotation error < 0.5 degree, position error < 1 % of the object size.
- **5 views 72 degrees apart (shoot 2's case): placed with depth, and the test also shows plain structure from motion
  (the depth terms switched off) fails or is clearly worse.** This is the reason the solver exists.
- Unordered input and one stray view of something else: the stray is not placed and is named.
- A planar texture seen twice: the homography path chosen.
- Mesh: Chamfer distance to the true surface < 1 % of object size where measured; the unseen underside violet.
- Scale from a synthetic coin: within 2 %.

### I.3 Files

glTF read back (our reader in the test), structure checked against the spec's required fields; OBJ/PLY parse; the
contracts of G.3 (names, keys, versions); a second run replaces its own files.

### I.4 Pages

Sets, Cameras and Model, Coach, Foliage pages open and run on synthetic data (`tests/test_scan_pages.py`, as
`test_stack_pages.py` does).

### I.5 Speed and memory

The benchmark of A.3.7; a test that a 45 MP Dual Pixel stack's analysis stays under a memory cap (tracemalloc, as
`test_build_memory_stays_bounded`).

### I.6 Real checks on Bill's folder (after the synthetic tests pass; nothing written outside the Studio's outputs)

1. Near/far: A.1 on #38, #14, #25, #9.
2. Speed: one stack of shoot 2 before and after A.3.
3. Surfaces: board #1, bark #17, brick #39, asphalt #41, gravel #15; look at every map; tileable on brick and bark.
4. Foliage: #11, #20-22.
5. Models: the log (#28-#34), the angel (#4-#7), the gnome (#52-#56); report views placed, errors, measured share;
   import into Blender and Godot 4 once.
6. Mixed / high-ISO shoot 1: #11-#14 (DP) and #2-#5 (ordinary).
Write every number and every fault found (and fixed) into `docs/RESULTS.md`.

---

## J. Order of the build (milestones; each ends with something Bill can double-click)

| # | What | Done when |
|---|---|---|
| M0 | A.1 near/far from Bill's scenes, A.2 focus distance check | R5 sign locked from >= 3 scenes; focus distance read or its absence written down |
| M1 | A.3 speed, A.4 cache, A.5 batch + `stacks`, A.6 mixed shoots | 4x per DP frame (or the honest number), a re-run of a stack comes from the cache, `stacks FOLDER --run surface` does shoot 2's textures |
| M2 | B.1, B.2, B.3, B.7 (depth in the sidecar, measured depth masks in Develop) | Depth view shows gaps; Measured depth merge default for DP; masks in Develop |
| M3 | C.1-C.9 surfaces, scale (C.6 a-c), Godot conventions (C.7) | Bill's board, bark and brick give every map; Godot heightmap flip verified |
| M4 | F.1 glTF, F.2, F.3 Blender script, D foliage | a leaf card glb imports into Blender and Godot; Blender script builds a displaced bark plane |
| M5 | E.1-E.5 solver + H.2 Sets, H.3 Cameras | the log, angel and gnome sets placed (or each unplaced view named with its reason); synthetic 5-view test passes |
| M6 | E.6-E.8, E.10 dense, mesh, texture, LODs, H.3 Model, H.4 Coach, `model` | a textured log glb from shoot 2 in Blender and Godot; the coach's advice for the next shoot |
| M7 | B.4, B.5, B.6, C.6 d, E.9, B.8 | each tick works on a real stack |

M0-M4 is one long session's work; M5-M6 one or two more (the solver and the mesh are the largest pieces in this plan);
M7 after Bill has tested. `docs/PROGRESS.md` gets a line per milestone, `docs/NEXT_UPDATE.md` a line per part.

---

## K. Not in this build, and honest limits

- **Few views give partial models.** Shoot 2's objects have 4-7 views from one side and one height: the solver may
  place them, but the backs and undersides will be filled (violet) or missing. The Coach says how to shoot better sets.
- **Plain, shiny or moving surfaces** are hard for every measure: Dual Pixel and stereo both need texture; leaves in
  wind move between stacks (the agreement maps show it).
- **Outdoor light** changes between stacks; colour levelling hides seams but cannot remove a shadow that moved.
- **Roughness and specular** are not measured (C.4). Photometric stereo (the same stack lit from several sides) would
  measure normals and shine; it is its own later build.
- **No neural networks are trained** here (Depth Anything stays as the estimated, amber alternative it already is).
- **No Godot scenes**: the Forge owns the game; the Studio delivers files and import notes.
- **Not a RealityScan replacement on large, well-textured scenes** with hundreds of photos; the Studio's solver is built
  for Bill's few, wide, Dual Pixel views of small things.

---

## L. Questions for Bill (ask in chat, plain words, with the recommendation; none blocks M0-M4)

1. **Scale**: "Can you put a coin (a quarter) somewhere in one frame of each new object or texture set? It lets the
   Studio give real sizes for the game." Recommendation: yes; the Studio finds the coin by itself.
2. **Where the 3D work lives**: "A new sidebar section called 3D Scan, with Sets, Cameras and Model, Coach and Foliage
   pages?" Recommendation: yes (his rule: each area its own section).
3. **Blender path**: "Where is Blender installed, so Open in Blender can start it? (The Forge already knows; the Studio
   can read it from the Forge's settings if you prefer.)" Recommendation: read it from the Forge's config when present,
   else ask once in Settings.
4. **Next shoot**: "For the next object, would you shoot 12-16 stacks all the way round and a few from lower down?"
   Recommendation: yes, after M5, guided by the Coach's printout.

---

## Status (session 17, 2026-11-21 afternoon)

Built: **M0 to M4** (with the app pages for them), tested on synthetic data with known truth (`tests/test_dp3d.py`,
`tests/test_scan_pages.py`); real checks on Bill's frames for M0 and M1 (docs/RESULTS.md, session 17). Where it
differs from the words above, and why:

* **A.1** Built as written. R5 locked -1 from stacks #38, #25, #9 (#14 abstains: shot nearly straight down, so the
  orientation tag cannot say which edge is nearer). The end frames of each stack abstain (Bill rocks focus past both
  ends). `nearcheck.py`; the profile is `dpprofile.py` (numbers only: near sign, shading per lens setting, room for the
  C.6 d scale).
* **A.2** The Makro-Planar ZE records no focus distance on the R5 (checked). Read for lenses that do (ExifTool's
  ShotInfo scaling, unverified on such a lens); used as an amber starting scale through the thin-lens formula.
* **A.3** Decode once, second frame from memory, shading once (refit when the level ratio drifts > 2%; kept in the
  profile per lens setting), a rewritten offset search (3.8x; the first implementation kept as
  `depth.disparity_reference` for the tests), frames in parallel by **threads** (not processes: a process pool
  re-launches the program in the MSI install, whose Python is a stub). 2.4x per frame one at a time in the 2-core
  workspace; the parallel gain is Bill's PC's to measure (`DP Benchmark.bat`). A.3.4 (predicted search) not built:
  with grid-only window sums a shift costs the same wherever it is needed. A.3.6 (GPU) not built, as planned.
* **A.4** As written, plus the full-size colour merge kept per merge setting (it is the slow part of a surface). The
  cache folder is created at its final name and counts once `meta.json` is written (Windows cannot rename a folder
  whose files are open as memory maps).
* **A.5** `batch.py` and `stacks`. Grouping stacks by object is left to the solver (E.2): every stack is its own
  group until then (`StackInfo.group` is null). Output: a `Studio surfaces` folder beside the photos.
* **A.6** As written (neighbouring Dual Pixel frames paired across the ordinary ones; positions shared out).
* **B.1** As written; "Auto" resolves to "measured" on Dual Pixel stacks (`FocusParams.auto_measured`).
* **B.2** As written; the mask is 1 over the flagged regions and feathers outside them (the one-frame buffer is grown
  into that band, so the edge never blends with nothing). On the synthetic moving patch the agreement map flags about
  a third of the patch: only that part is taken from one frame.
* **B.3** Half the depth of field from optics when Dual Pixel gives the step (0.849 / |s| steps for a 1 px blur
  radius); from the sharpness falloff otherwise.
* **B.7** `NAME.depth.npz` beside the result; Develop's depth (masks, haze, relight) uses it automatically
  (`pipeline.measured_depth`, cyan). Depth of field (B.6) not built (M7).
* **C** C.1-C.5, C.6 a-c, C.7, C.9 built. **C.8 decals not built** (the atlas exists; the cut-out of pieces standing
  out of a texture is the next small step). C.6 d (Dual Pixel scale calibration) is M7. Godot's height-map flip set
  from the class reference (false), not checked by rendering.
* **D** D.1-D.4 built; thickness is a constant (amber) until a back-lit frame is detected; pivot = the lowest point of
  the leaf (a click is not wired yet). Checked on Bill's bramble stack #20: subjects are made only of places sharp in
  some frame or measured both ways (`depthmask.strict_label`), and only small holes are filled, after the first run
  joined leaves to bokeh (docs/RESULTS.md). A stack whose Dual Pixel steps spread over 100% falls back to focus.
* **F** F.1, F.2, F.3 built; **not yet imported into Blender or Godot on Bill's PC** (the Blender connection's
  background mode has no Blender path; the Godot connection would add files to the game). First run:
  `studio-cli blender <NAME_surface_Studio.json>` or 3D Scan > Open in Blender. F.4 notes in every JSON.
* **G** `stacks`, `foliage`, `blender`, `cutout --depth-measured`, `dualpixel --make near-check`, `--maps`, `--size`,
  `--scale`, `--tileable cut|blend`, `--workers`, `--background`, `--no-cache`. `model` and `profile-dp` not built
  (M5-M7). The Forge's part is written in docs/CLI.md for its builder.
* **H** H.1 and H.2 (3D Scan > Sets, with Make them all, Open in Blender), H.6 partly (Depth view gaps, measured
  merge, moving parts, every game map; Dual Pixel Lab teach near; not yet: Sharpen gaps, Recover highlights, Depth
  of field, Cut out by measured depth button, Calibrate Dual Pixel scale). H.3-H.5 wait for M5-M6.

### Status, session 18 (2026-11-21 evening): M5 and M6 built; M7 not yet

Built and tested on synthetic views with known truth (`tests/scan_synth.py`, `tests/test_scan.py`: a textured, bumpy
log on textured ground, hand-held views with their own Dual Pixel depth affine in 1/z with an unknown scale and offset
per view). Where it differs from sections E-H, and why:

* **E.2** SIFT as planned (`engine/scan/sift.py`, RootSIFT). Plain SIFT stopped near 40 degrees on the orbit (the side
  of the log is squeezed about 2x between views 45 degrees apart: 1 good match of 10), so **affine simulation (ASIFT,
  Morel and Yu 2009)** was added: each picture also described as if seen tilted (tilts up to 2.8, 17 simulations);
  views 67 degrees apart went from 0 to 258 good matches. It is slow, so it runs only for views plain matching could
  not join to two others; the plain features stay the first block, so earlier matches stay valid. Depth-assisted
  guided matching (the last bullet of E.2) was not needed and is not built.
* **E.3** The essential matrix by 8-point RANSAC went wrong on curved patches seen from two sides (several motions fit
  the matches about equally, 50-70 degrees off): fixed by scoring hypotheses by the points in front of both cameras,
  local optimisation with 20-point samples from the inliers (LO-RANSAC, Chum 2003), and refining the leading
  candidates by their Sampson cost (MSAC). Even so, on wide pairs the depth path is the reliable one: when the two
  disagree, depth wins unless the matches have clearly more inliers. Planar: the homography explains 95 % of what the
  other models do.
* **E.4-E.5** As planned, with the bundle adjustment our own **Levenberg-Marquardt with analytic derivatives and the
  Schur complement** (`engine/scan/ba.py`): scipy's least squares with numerical derivatives stopped a few degrees from
  the truth on five views 72 degrees apart (the true poses fitted the matches at 0.17 px, its answer at 0.59 px). The
  pixel and depth noise are measured from the data (the depth had out-voted the pixels tenfold with a fixed 1 px).
  With fewer than 8 views the lens distortion is held at 0.
* **Focus breathing** (Bill, 18:58): a macro lens focused close is longer than marked (v = f (1 + m): 1.5x at 1:2), so
  each view's focal length is solved with a wide prior (35 %) when the lens records no distance, tied between stacks of
  the same lens (3 %), and the starting focal is found from the matches (the factor that lets the essential matrix
  explain the most matches). With a recorded distance the start is f (1 + m) and the prior 5 %.
* **E.6** As planned (`dense.py`): Dual Pixel depth to distance, a 21-hypothesis plane sweep round it (+-10 %) against
  the 2-3 best neighbours, NCC 7 x 7, fused by noise with an agreement map (red kept out of the mesh).
* **E.7** Scale from the lenses' recorded focus distance (amber) or relative; the coin and the Dual Pixel calibration
  (C.6 d) are not wired into models yet (C.6 d is M7).
* **E.8** TSDF + visual hull as planned (`fusion.py`); **surface nets** (Gibson 1998) in place of marching cubes (no
  256-case table, watertight, fewer slivers: a sphere's radius recovered to 0.02 voxel); Taubin smoothing; texture by
  the view seeing each face most squarely, charts smoothed by iterated conditional modes on a Potts energy (Besag 1986:
  57 000 speckle charts became 883), gains per chart over the seams; **simplification by quadric clustering** (Lindstrom
  2000, the Garland-Heckbert quadrics on a grid) in place of edge-by-edge collapse (array operations only; crack-free
  across seams); normal maps baked per LOD from the full mesh; collision is left to the importer (the JSON says so).
  Ground cut: RANSAC plane on the lowest tenth.
* **E.9** RealityScan cross-check: dropped by Bill (21:07).
* **E.10, H.3, H.4** The model view under the Sets list (`app/scanmodel.py`: a software turntable, colour or how-made,
  quality in plain words, the coach), "Make a 3D model" for the ticked stacks. The coach divides the surface by the
  direction it faces and says which side was never measured and how many stacks to add (hand-held steps).
* **Blender dataset** (Bill, 18:18: "only if they are of high enough quality"): `engine/scan/dataset.py`, behind a
  quality gate (3 views and 60 % placed, 1.5 px, 35 % measured, 25 % disagreement); cameras in Blender's convention
  (matrix_world) and OpenCV's, checked against each other in the tests; depth as float32 .npy.
* **G** `model` command (`NAME_model_Studio.json` "studio_model": 1, `NAME_cameras_Studio.json`, `NAME_model_job.json`);
  `schema` lists `contracts`. For the Forge: `stacks --run surface` also makes ESTIMATED surfaces (amber) from ordinary
  single photos and bursts, so every texture folder gets the same files.

Synthetic results (docs/RESULTS.md, session 18): 16 views 22 degrees apart: all placed, rotation within 0.23 degree,
position within 1.2 % of the cameras' spread (median 0.4 %; the worst view's focal length came out 2.5 % long: the
plan's 1 % of the object's size is met by the median, not by that view); with a stray view of a wall: 16 placed within 0.12 degree, the stray named; **5 views 72
degrees apart: all placed within 0.6 degree with the depth, 2 of 5 without it**; breathing (files 520 px, truth
627-675 px): all 8 placed within 0.5 degree, mean focal within 1 %; the mesh within 1 % of the object's size where
measured, the unseen underside violet.

**Not built yet (M7 and checks):** B.4 sharpen the gaps, B.5 highlights in a stack, B.6 depth of field after the fact,
C.6 d the Dual Pixel scale calibration and the self-distance scale (designed in Bill's answers below), E.9, the
Calibrate / Cut out by measured depth buttons; **models from Bill's real sets (the log #28-#34, the angel #4-#7, the
gnome #52-#56) not run in the workspace** (each stack's analysis takes minutes here; the model command and the 3D Scan
button are what to run on his PC first), the Blender and Godot imports. Next session: M7, then the real model checks.

### Bill's decisions, 2026-11-21 21:07

* **RealityScan is dropped** (E.9 and the Forge's hand-off): "it requires an epic account and most people I help won't
  want that. Lets improve our own stuff inhouse instead."
* **No per-lens distance calibration** (C.6 d's tape-measure frame, the self-distance scale for ring focusing): "The
  real world distance scale is unrealistic, i have over 100 lenses." Models are in relative units unless a lens
  records its focus distance (amber) or the photographer gives the object's length (`--length-m`); the shape, texture
  and cameras do not depend on it.
* **Build what the Forge needs to use the Studio**, tested on the shoot-2 data: `intake FOLDER` (one call per folder:
  objects become models, the rest textures; `NAME_intake_Studio.json`, "studio_intake": 1).

### Bill's answers to section L (2026-11-21, 16:28)

1. **Scale: no coins.** "I rarely have a coin with me ... with my back, I often have trouble getting that low without
   pain." Scale instead "by going off of the focus distance, which can indicate how far I am from an object and
   considering the maximum edge to edge view of the lens in question for that focal length". Built (C.6 c): when the
   lens records its focus distance, the magnification (thin lens) and the sensor size give the view edge to edge and
   the size of a pixel (amber: the lens's distance is coarse; `scale.view_mm` in the JSON). **The Makro-Planar ZE has
   contacts but does not record a distance on the R5** (A.2), so for it and for contactless lenses another way is
   needed ("perhaps you can come up with another solution?"). Proposed for the next build (replaces the coin in
   C.6 b-d and E.7): **the stack measures its own distance.** Every focus step's change in 1 / distance follows from
   the Dual Pixel step s and the lens (offset = 0.849 x (f / N) / 2 x v / pixel x (1/z - 1/z_focus)), and the line-up
   measures how much the picture's magnification changes per step; together they give the distance (focusing by
   moving the body: z = (change in magnification per step) / (change in 1/z per step); focusing with the ring: a
   lens-dependent version). First try on bark #17: magnification 0.77% per step, Dual Pixel 4.2e-5 per mm per step
   with the full-aperture baseline: about 18 cm if Bill moved his body; the real Dual Pixel baseline is narrower than
   the full aperture, so one calibration per lens and aperture is needed: one frame of a tape measure lying on a
   table, shot standing up (Bill was going to shoot it anyway), remembered in the camera's Dual Pixel profile. The coin
   finder stays in the command line for anyone who has one in the frame.
   Bill, 16:43, on how he focuses a stack: "I do either. Sometimes with some lenses I turn the focus ring. Sometimes I
   rock back and forth. Others, with some vintage medium format and large format lenses, I have to use bellows to
   focus anyways, so the lens focuses through movement. With the makro planar I used the focus ring." So the
   self-distance method needs all three cases: **body rocking** (the camera moves: magnification change per step =
   distance x change in 1/z per step), **bellows / extension** (the whole lens moves away from the sensor: the
   unit-focus formula, exact for bellows), and **the focus ring** (lens-dependent: unit focus for simple designs, a
   floating or internal focus changes the focal length; the one-time tape-measure frame per lens calibrates it). The
   Studio tells body rocking from the other two by the parallax it causes (near and far parts of the frame shift
   against each other between frames; turning a ring or racking bellows only rescales about the centre) and says
   which it found; a lens profile can say ring or bellows. Bill's shoot 2 (Makro-Planar): ring.
2. **3D Scan section: yes** ("each area gets its own section"). Built (H.1, H.2).
3. **Blender from the Forge's setting: yes.** Built (`engine/io/blender.py`, Help > Blender to override).
4. Not asked yet (after M5).
