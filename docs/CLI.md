# The Studio from the command line (for programs)

`studio/cli_tools.py` puts every editing tool of the Studio behind one command line, with results as JSON. It is
made for other programs: the RPG Forge, Claude, anybody's scripts. Photographers never need it; their commands
(`develop`, `import`, `calibrate`) are unchanged.

The Studio is GPL-3.0. A program under another licence (the Forge is MIT) uses it by **running it as a separate
program**, as below, not by importing its code. The pictures it makes belong to whoever made them.

## Running it

| Where | How |
|---|---|
| Windows (the Forge, scripts) | `"D:\Analyst_Toolkit\Photographic Denoiser\studio-cli.bat" COMMAND ...` |
| Any Python that has the packages in `requirements.txt` | `python -m studio COMMAND ...` with the Studio folder as the current folder or on `PYTHONPATH` |

Start with `schema --json`: it lists every setting with its default and what it does, every choice, every command,
and is built from the code, so it is never out of date. `schema light` or `schema light.shadows` shows one part.

## The contract

* `--json`: standard output carries exactly **one JSON object on one line**; progress and log lines go to standard
  error. `--result FILE` writes the same object to a file.
* The object: `{"ok", "command", "version", "outputs": [...], "problems": [...], "notes": [...], "seconds"}`.
  One entry in `outputs` per file written (`input`, `output`, `kind`, `width`, `height` ...), or per picture examined
  (`find`). `notes` say in plain words what was measured or decided. `problems` are `{text, input, error_report}`.
* Exit code: `0` everything worked, `1` something failed (the other pictures were still done), `2` the command line
  or a recipe was wrong.
* **Names are checked.** A mistyped setting is an error that names the nearest real ones. Nothing is silently ignored.
* **Nothing is overwritten.** Results are named `NAME_Studio.ext` (`--tag night` gives `NAME_night_Studio.ext`),
  never the picture's own name, and a file the Studio did not write is never replaced: the result goes beside it as
  `NAME_Studio-2.ext`. Always read the path from the JSON. Running a command again replaces its own earlier result.
* **The same picture and recipe give the same pixels.**
* Inputs: raws, TIFF, PNG, JPEG, FITS (astronomy; needs the optional astropy), and folders of them (`--recursive`). Points and boxes are fractions 0..1 of the
  picture (u to the right, v down).
* Default output: a raw becomes a 16-bit Adobe RGB TIFF; any other picture an 8-bit sRGB PNG; anything with
  transparency a PNG. Change with `--format --space --bits --quality --long-edge --sharpen-for`, or name the file
  with `--out-file`. `--format fits` writes 32-bit float linear light for Siril and PixInsight (needs astropy). `--out FOLDER` says where (default `outputs\cli`).
* `--preview N`: a quick sRGB JPEG, long edge N, by the fast path. Use it to look before a full-size run.

## Settings and recipes

The recipe is the Studio's own Edit, as JSON, holding only what differs from the defaults:

```json
{"studio_recipe": 1, "edit": {"exposure_ev": 0.3, "light": {"shadows": 25}, "colour": {"saturation": 10}}}
```

* `--set NAME=VALUE` (repeatable): `--set light.shadows=25 --set colour.bw=true`. Values are JSON when they parse.
* `--recipe FILE_OR_JSON` (repeatable, applied in order): a recipe file, inline JSON, or a photo's `.psb.json`.
* `--save-recipe FILE`: write the settings a command used, so one look can be reapplied to a hundred pictures.
* `--look NAME`, `--auto` (Make it good), `--use-sidecar` (start from the photo's own saved edit), `--fix-lens`.
* Order, later wins: sidecar, `--auto`, `--look`, recipes, the command's own tool, `--set`, what is measured on the
  picture (`--fix-lens`, negative, infrared), then `--match`.

## Commands

| Command | What it does |
|---|---|
| `edit` | Any picture + any settings: a finished file. `--dry-run` reports the settings and writes nothing. |
| `clean` | Noise removal. `--strength 1` `--chroma` `--method` (hybrid, fusion, starlet, nlm, mppca, collab, learned) `--keep-grain`. `--auto-strength`: each photo chooses its own strength, detail first (needs a calibrated camera; `--strength` then leans from that choice; a Dual Pixel photo is judged by its two halves). Best on raws. |
| `relight` | Move the light using the picture's depth (estimated). `--azimuth --elevation --amount --soft`. |
| `colourize` | Black and white to colour (a plausible guess). `--amount --warm --tint`. |
| `restore` | Old photos and scans: `--scratches` `--stains` `--descreen` `--jpeg` `--colourize`. |
| `jpeg-repair`, `descreen`, `negative`, `infrared` | One tool each; `negative --kind colour|bw`; `infrared --filter --preset` (raws only). |
| `match --to REF` | Colour match to a reference picture, or to a look profile. `--amount 0..1.5`. |
| `look-profile PICTURES --save FILE.json` | Measure a look profile (nine colour statistics) from reference pictures. |
| `depth --make map,view,normals,layers` | 16-bit depth map (near = white), coloured view, normal map (`--normals-y up|down`), or `--layers N` RGBA layers, 1 = farthest, that stack back into the picture. `--cuts` sets where they divide; `--fill` makes layer 1 a whole background plate. Estimated from one picture. |
| `cutout` | A subject with a soft edge as RGBA (straight alpha), or `--make mask`. Select with `--person [part]`, `--click U,V` / `--avoid U,V` (`--model classic|tiny|large`), `--depth-range`, `--luminance LO,HI`, `--colour HUE,WIDTH`, `--skin`, `--shape JSON`; `--combine`, `--invert`, `--feather`, `--hard 0.5` (firm edge for leaves and fences). |
| `find --what ...` | Data, no file: `size, info, stats, faces, people, bodies, dust, flare, crops, subject, red-eyes, blemishes, hdr, jpeg, infrared, depth`. `bodies`: each person's parts for body sculpting (centre line, outline distance each side, measured or estimated). |
| `stack focus|superres|pano FRAMES` | A focus bracket, a burst, or a panorama into one linear 32-bit file. Finish it with `edit`. Tool settings with `--set` (listed under `stack` in the schema). Focus only: `--make merge,depth,surface` (default `merge`): the surface files for games, below. |
| `dualpixel --make report,depth,half1,half2,hdr,anaglyph,sbs,wiggle,surface,near-check` | What a Canon Dual Pixel RAW can make. Its depth is measured, not estimated. `surface`: the same five files and JSON as a stack, from the one photo's own offsets. |
| `stacks FOLDER [--run surface,depth,foliage]` | The stacks in a folder or on a card (data), or every stack through jobs; resumable (`stacks_job.json`). See "Dual Pixel 3D". |
| `foliage FRAMES_OR_FOLDER --make cards,atlas,glb` | Leaf cards, an atlas and glb files cut out by measured depth. |
| `intake FOLDER --out DIR` | One call per folder (for the RPG Forge): objects become models, other stacks and photos textures; one JSON lists every asset. |
| `model FOLDER --make cameras,points,glb,dataset` | A 3-D model of one object from its hand-held stacks in any order (the Studio's camera solver with the Dual Pixel depth); a Blender dataset when the scan passes the quality gate. |
| `blender FILE.json|FILE.glb [--open]` | A Blender scene from a surface, model or glb (`NAME_Studio.blend`). |
| `cutout --depth-measured [--click U,V]` | The subject by its measured depth (a stack's result or a Dual Pixel RAW). |
| `schema [NAME]`, `looks` | What exists. |
| `batch JOBS.json` | Many commands in one start. Each job is a list of arguments, a command-line string, or `{"id", "args"}`; each job's result comes back under `jobs`. `--stop-on-error`. |

## Surfaces for games (Depth Stack)

`stack focus FRAMES --make surface` measures how far every place is, two ways when every frame is a Dual Pixel RAW
(which frame is sharpest, and each frame's own offset between its two halves), fuses the two into one height map and
writes, beside the frames or in `--out`:

| File | What it holds |
|---|---|
| `NAME_colour_Studio.png` | The all-in-focus merge (the Focus Stack method, default `auto`), developed with the default look, 8-bit sRGB. |
| `NAME_height_Studio.png` | 16-bit grey, **white = near**, spread over the measured range (filled-in places can go past it and are held at the ends). |
| `NAME_normal_Studio.png` | From the height. `--normals-y up` (OpenGL: Godot, Blender, the default) or `down` (DirectX). |
| `NAME_confidence_Studio.png` | 8-bit grey, 255 = sure; 0 where the height was filled in, not measured. |
| `NAME_agreement_Studio.png` | Grey + alpha: 255 = the measures agree, 128 = they stray half a focus step (the red line), 0 = a step or more; transparent where only one measure was there. |
| `NAME_surface_Studio.json` | How it was made, below. |

`NAME` is `FIRST_to_LAST` for a stack (the frames' own names), the photo's name for `dualpixel FILE --make surface`.
`--make depth` writes the height, confidence and agreement and the JSON only: no full-size merge, so it is quick. All
the files carry one geometry (the stack's common area, upright). Running again replaces the Studio's own earlier
files; when any one name is taken by a file the Studio did not write, the whole set moves to `NAME-2_...` together.

Settings (also under `stack.surface` in `schema`):

* `--ratio M`: the magnification as shot (`1:2`, `1:1`, `0.5`): with the camera's pixel size it gives the size of a
  pixel on the subject (`lateral.mm_per_pixel` in the JSON).
* `--step-mm X` (stack): how far the focus moved between frames, for a rail. With it the height is in millimetres.
  Without it the height is **relative**, and the JSON says so. (The lens's recorded focus distance is read when the
  lens records one: the Makro-Planar ZE does not.)
* `--near auto|first|last` (stack): which end of the stack is nearest. `auto` reads it from the Dual Pixel halves
  when the camera's reading is known: the camera's Dual Pixel profile (`dualpixel --make near-check`, taught on your
  own frames, cyan) or the Studio's table (the **Canon EOS R5** is in it since 2026-11-21, read from Bill's receding
  scenes); otherwise it takes the first frame in focus order and says so. One photo: `--swap`.
* `--normal-strength S`: when the height is relative, how steep the normal map is (1 = the steepest tenth tilts
  about 45 degrees). With mm height and pixel size both known the normals are physical and this is ignored.
* `--tileable [cut|blend]`: a repeating texture, one geometry for every file. `cut` (the default): a minimum-error
  cut through each seam band (structured textures: brick, planks); `blend`: the copies blended (even, random
  textures). The normal map and the maps computed from the height are worked out again from the tiled height.
  Before the seams are healed, the large scale is taken out (setting `tile_flatten`, on by default; 2026-11-21):
  the colour's light falloff (vignetting, uneven light: a smooth surface fitted to the log luminance, then lumps
  over 1/8 of the edge) and the height's overall tilt and broad bumps, so the copies meet without a step. The JSON's
  `tiling.flattened` says what was done; the height is then relief only.
* `--maps LIST` (see the next section) and `--size`, `--scale` (game size, real size).

The JSON (`"studio_surface": 2` since 2026-11-21; every key of version 1 is kept): `written_by`, `method` (`focus+dual_pixel`, `focus`, `dual_pixel`), `method_text`,
`best_for`, `weak_at`, `source` (frames, frames used, reference frame, left out), `camera`, `size`, `files`,
`height` (`white: near`, `units` relative or mm, `how_known`, `depth_units`, `range`, `near_end`, `near_measured`),
`lateral` (`mm_per_pixel`, `width_mm`, `height_mm`, `how_known`), `displacement.height_range_over_width` (for a
displacement setting; null unless both scales are known), `normal` (convention, scale), `dual_pixel` (`step_px` = s,
the offset change per focus step; `step_spread`; each frame's position in steps; each pair's own step; the measured
noise of one frame's vote and of focus), `coverage` (`measured_both`, `measured_one`, `filled`, `disagree`, shares
of the picture), `agreement` (median and 90th percentile in steps), `tileable`, `settings`, `notes`.

The command's own `--json` result has one entry per file, `kind` = `surface_colour`, `surface_height`,
`surface_normal`, `surface_confidence`, `surface_agreement`, `surface_json`; the JSON's entry carries `method`,
`height_units`, `near_end`, `coverage` and `files`, so a program needs nothing else to shelve the set.

How it works and what it is honest about is in `engine/stack/depthfuse.py` (the fusion) and
`engine/stack/surface.py` (the files); the measurements are in `docs/RESULTS.md` (2026-11-21).

## Dual Pixel 3D (BUILD_PLAN_DP3D, 2026-11-21)

### More maps (`--maps`), game sizes, real size

`--maps` picks the files (default: the five above; `all` = every map but roughness):

| Map | File | How it is made (the JSON's `maps.<name>.how` and colour) |
|---|---|---|
| `albedo` | `NAME_albedo_Studio.png` (sRGB) | **estimated** (amber): one light plus ambient fitted to the measured shape and divided out; `light` in the JSON gives the direction found and how well it fitted (`r2`) |
| `detail` | `NAME_detail_Studio.png` (16-bit, 0.5 = flat) | **estimated** (amber): fine relief from the shading left after de-lighting; never mixed into the height |
| `ao` | `NAME_ao_Studio.png` | computed from the measured height (horizon-based, 8 directions, reach 1/32 of the long edge) |
| `cavity` | `NAME_cavity_Studio.png` | computed from the measured height (small crevices darker) |
| `curvature` | `NAME_curvature_Studio.png` (16-bit, 0.5 = flat) | computed from the measured height (ridges bright) |
| `orm` | `NAME_orm_Studio.png` | occlusion (R) from the height, roughness (G) the estimate when asked else a constant 0.9, metallic (B) 0: Godot's ORMMaterial3D and glTF's order |
| `roughness` | `NAME_roughness_Studio.png` | **estimated** (amber) from the colour's local variation: nothing in these photos measures shine; asked for by name only |
| `gaps` | `NAME_gaps_Studio.png` | measured: places no frame had in focus (the focus coach below) |

* `--size 2048|4096|...`: a square power-of-two texture from the measured area's best part, every map cut and resized
  the same way (`game_size` in the JSON).
* `--scale coin` (or `coin:us_quarter`, `us_penny`, `euro_1` ...): a coin found in the picture gives the size of a
  pixel; `--scale click:U1,V1,U2,V2,MM`: two points (0..1 of the picture) a known distance apart. `scale` in the JSON
  says which.
* Version-2 keys: `maps` (each file: `how`, `colour` cyan / amber / violet, `space` srgb / linear), `scale`,
  `tiling`, `game_size`, `focus` (`dof_half_steps`, `coach`: plain sentences on where the focus jumped), `light`,
  `godot` (below), and `coverage.gap`.

### Godot 4 (F.4)

The JSON's `godot` block is what an import needs: `normal_map` "OpenGL (no flip)"; `heightmap_flip` **false** (the
height is white = near = higher; Godot's `heightmap_flip_texture` is for depth maps, where brighter is lower: from the
Godot 4 class reference, not yet checked by rendering in the game); `textures` (srgb for colour and albedo, linear for
everything else); `orm`; `repeat` (true when tileable). Getting Studio files into Godot 4: either import the PNGs and
build a StandardMaterial3D with those settings, or import the `.glb` files (foliage, and models when the solver is
built) directly: Godot reads glTF's materials, alpha scissor and double-sided flags itself; or go through Blender
(`blender` below) and export from there.

### Folders of stacks: `stacks`

```
studio-cli stacks FOLDER [--json]                          the stacks found (data: kinds, frames, Dual Pixel, left out)
studio-cli stacks FOLDER --run surface,foliage [--out DIR] every focus stack and single Dual Pixel photo through the jobs
```

Card to Stacks' sorter finds the stacks (bursts and ordinary single photos are listed and skipped, with the reason).
Each stack is analysed once: the **analysis cache** (`analysis_cache/` beside the app; Help > Analysis cache) keeps
pass A, and the full-size merge for a setting, so surfaces, then leaves, then a model cost one analysis. Results go to
a `Studio surfaces` folder beside the photos unless `--out`. **Resumable:** `stacks_job.json` in the output folder
records every item (`done`, `failed` with the reason, `skipped` with the reason, `pending`); run the same command again
and it carries on, reporting the earlier files with `"from_earlier_run": true`; `--restart` starts again. The surface
settings (`--maps`, `--size`, `--tileable`, `--scale`, `--ratio` ...) apply to every stack. `--run model` is listed
but says it needs the camera solver (not built yet).

Long jobs (G.2, G.4), on `stack`, `stacks` and `foliage`: `--workers N` (frames developed at once; default a third of
the cores, as memory allows: 4 on Bill's PC), `--background` (below-normal priority), `--no-cache`.

### Leaves: `foliage`

```
studio-cli foliage FRAMES_OF_ONE_STACK | ONE_DUAL_PIXEL_PHOTO | FOLDERS [--make cards,atlas,glb] [--size 2048]
                   [--pick N] [--triangles 600] [--ratio 1:2 | --scale coin] [--out DIR]
```

Each separate leaf is cut out by its measured depth (the bokeh behind measures nothing) and snapped to the colour
edges. Per leaf: `NAME_leafN_colour_Studio.png` (RGBA, straight alpha, colour bled into the transparent margin so
mipmaps never darken the edge), `_albedo` (estimated), `_normal` (from the measured relief), `NAME_leafN_Studio.glb`
(the bent leaf, 200-800 triangles, and the flat card; alpha scissor 0.5, double-sided; `COLOR_0` black at the stem,
white at the tip, for a wind shader). The atlas: `NAME_foliage_atlas_colour|albedo|normal_Studio.png` (power of two,
shelf packed, one layout for every map) and `NAME_foliage_atlas_Studio.glb`. `NAME_foliage_Studio.json`
(`"studio_foliage": 1`): each leaf's size in mm when known, its pivot (the stem: the leaf's lowest point), its atlas
rectangle in UV, how each map was made (thickness is a constant until a back-lit frame can measure it), Godot notes.

### Cut-outs by measured depth: `cutout --depth-measured`

`cutout PICTURE --depth-measured [--click U,V]`: the subject chosen by its **measured** depth (a focus stack's result
keeps its depth in `NAME.depth.npz` beside it; a Dual Pixel RAW is measured on the spot): the region clicked, else the
largest near one; holes it could not measure are filled; the outline follows the colour edges. The usual cutout
options (`--make rgba|mask|both`, `--feather`, `--hard`) apply.

### Which way is near: `dualpixel FILES --make near-check`

Every frame of a scene with ground receding at an angle votes (the offsets over a receding plane change steadily down
the upright frame); a scene votes when two thirds of its frames agree; the camera's sign is locked when at least
three scenes agree and none disagrees, and saved to this PC's Dual Pixel profile for the camera
(`calibration_local/profiles/<camera>_dualpixel.json`, numbers only). The result entry `dp_near_check` carries the
votes. The same in Dual Pixel Lab: "Teach the Studio which way is near".

### Blender: `blender`

`studio-cli blender NAME_surface_Studio.json | NAME.glb [--out DIR] [--open]` runs `studio/blender/build_material.py`
inside Blender: a surface becomes a plane its real size (1 m wide when the size is not known), subdivided and
displaced by the measured height, with a Principled BSDF (colour or albedo, AO multiplied in, the normal map in OpenGL
tangent space, the fine relief as a second bump at strength 0); a glb is imported. Saved as `NAME_Studio.blend` beside
the input (a .blend the Studio did not write is never replaced). Blender's place: the Studio's setting (Help >
Blender), else the RPG Forge's own (`forge/threed/blender/profiles.json`, read only). `--open` starts Blender with it.

### 3-D models: `model`

```
studio-cli model FOLDER_OF_ONE_OBJECT | RESULTS... | DUAL_PIXEL_PHOTOS... [--make cameras,points,glb,dataset]
                 [--triangles 30000] [--lods] [--ground-cut] [--atlas 4096] [--affine auto|on|off] [--out DIR]
                 [--no-gate] [--no-depth] [--workers N] [--background] [--no-cache]
```

Every focus stack in the folder (or every Focus Stack result with its `NAME.depth.npz`, or Dual Pixel photo) is one
view. Hand-held, any order: the Studio's own camera solver places them (features with affine simulation for views far
round, the pose from the matches and from each view's own Dual Pixel depth, bundle adjustment with depth terms; each
stack's focal length is solved, since a macro lens moves out to focus). Then each view's depth is measured two ways
(its Dual Pixel depth and stereo against its neighbours, with an agreement map), fused in a volume closed by the views'
outlines, and the surface is textured from the view that sees each part most squarely. `--make` (`all` = every one):

* `glb`: `NAME_model_Studio.glb`, the game mesh (`--triangles`; `--lods` adds `NAME_LOD1` and `NAME_LOD2` meshes),
  colour and baked normal map (OpenGL) embedded, Y up, metres when the scale is known.
* `mesh`: `NAME_model_full_Studio.glb`, the full-resolution mesh. `obj`: `NAME_model_Studio.obj` + `.mtl` +
  `NAME_model_colour|normal_Studio.png`.
* `points`: `NAME_points_Studio.ply` (colour; `how`: 2 triangulated, 1 from one view's depth).
* `cameras`: `NAME_cameras_Studio.json` ("studio_cameras": 1): every view, placed or not (and why in plain words),
  how it was placed (`both` cyan, `matches` / `depth` amber), pose (world-to-camera, x right, y down, z forward, in the
  model frame), focal length in pixels and where it came from, quality.
* `--length-m M`: the object's longest side in metres, when known (otherwise relative units; the shape is the same).
* `dataset`: the **Blender dataset** `NAME_dataset_Studio/`, written only when the scan passes the **quality gate**
  (at least 3 views placed and 60 % of those given, reprojection at most 1.5 px, at least 35 % of the surface
  measured, the two depth measures disagreeing on at most 25 %); otherwise nothing is written and the reasons are in
  the JSON's `gate.reasons` (`--no-gate` writes it anyway, reasons included). Its `dataset.json` (format
  `studio_blender_dataset`, version 1) describes: Blender axes (Z up), units, and per camera `blender`
  (`matrix_world` camera-to-world, `lens_mm` for a 36 mm sensor width, `shift_x/y`, `resolution`) and `opencv` (R, t,
  K); per camera `images/VIEW.png`, `depth/VIEW.npy` (float32 distance along the view axis, NaN = not measured),
  `depth/VIEW_preview.png`, `confidence/VIEW.png`, `agreement/VIEW.png` (cyan both, amber one, red disagree),
  `masks/VIEW.png`; `points.ply`, `model.glb`. Everything an add-on needs, read with NumPy and the standard library.
* always: `NAME_model_made_Studio.png` (the texture's "how made" map: cyan measured, violet filled) and
  `NAME_model_Studio.json` ("studio_model": 1): files, units and scale (`scale.how`, amber when from the lens's
  focus distance), views placed / not placed with reasons, `quality` (reprojection_px, depth_agreement,
  measured_share, filled_share, disagree_share), `gate`, `coach` (plain sentences: which side was never measured and
  how many stacks to add there, the underside, scale), `lods`, `godot` import notes.

Results go to a `Studio models` folder beside the photos unless `--out`; `NAME_model_job.json` there says `running`,
`done` or `failed` (with the reason) for a program polling it. A set with views of two objects models the larger and
says so. `--no-depth` is plain structure from motion, kept for comparison (on wide, few-view sets it places fewer
views: the reason the solver uses the depth).

### One call per folder: `intake` (for the RPG Forge)

```
studio-cli intake FOLDER [--out DIR] [--size 2048] [--no-tile] [--leaves] [--triangles 30000] [--no-lods]
                  [--affine auto|on|off] [--length-m M] [--workers N] [--background] [--json]
```

The Studio decides what each set of photos in the folder is for: every focus stack, Dual Pixel photo and Focus Stack
result (with its `NAME.depth.npz`) is a possible view, and the camera solver runs on all of them together; views it
places together (at least 3) are **one object** and become a model (as `model`: glb with LODs, cameras, points, the
Blender dataset behind its gate); every other view becomes a **measured texture**, every ordinary photo or burst an
**estimated texture** (amber); `--leaves` adds leaf cards from the remaining stacks. Everything is listed in
`NAME_intake_Studio.json` (`"studio_intake": 1`): `assets` (each with `kind` model / texture / foliage, `name`,
`files`, its own JSON, how it was made: `measured`, `colour`, and for models `quality`, `gate`, `units`, `scale`,
`coach`), `left_out` (with reasons), `summary` (counts), `notes`. `intake_job.json` in the output folder says
`running`, `done` or `failed`. Default output: a `Studio intake` folder inside the photos' folder; the Forge passes
its own (`--out <folder>/_intake/studio`), so nothing is written beside the photos.

**Front-only reliefs**: every view that does not join an object also becomes `NAME_relief_Studio.glb` (its measured
depth as a cut-out, textured relief mesh: the front of the subject as that stack saw it, not a full model),
`NAME_relief_made_Studio.png` and `NAME_relief_Studio.json` ("studio_relief": 1; `depth.ratio` is the depth against
the subject's width: estimated 0.5 in amber unless the camera solver measured it, or `--relief-depth`). `--no-reliefs`
leaves them out. `stacks <folder> --run relief` makes them for every stack.

**Scale**: models are in relative units unless a lens recorded its focus distance (amber) or `--length-m` gives the
object's longest side. The shape, texture and cameras are the same either way: only the size in metres is unknown.

### For the RPG Forge's builder (G.5; the Studio does not change the Forge)

* **One call per incoming folder:** `studio-cli.bat intake scans\incoming\<date>_<subject> --out
  scans\incoming\<date>_<subject>\_intake\studio --json`, then shelve every entry of `assets` from
  `NAME_intake_Studio.json` (textures: their `files` and `NAME_surface_Studio.json`; models: `files.glb`, with
  `units`, `scale` and `gate`; leaf cards: their glb and atlas). This replaces both of `forge/scans.py`'s present paths:
  its own guessed maps (the Studio now makes measured maps where the photos allow and estimated ones, labelled, where
  they do not) and the hand-off to RealityScan (Bill, 2026-11-21: RealityScan needs an Epic account most of the people
  he helps will not want; the Studio's own solver is the one used and improved). The Forge's "scan" / "texture" guess
  by frame count is no longer needed: the Studio decides by whether the views see one object.
* **What the Studio can do now (check it):** `studio-cli.bat schema --json` lists every command and, under
  `contracts`, the version of each JSON file it writes (`studio_surface` 2, `studio_foliage` 1, `studio_model` 1,
  `studio_cameras` 1, `studio_blender_dataset` 1, `studio_intake` 1, `studio_relief` 1). A reader checks the number it
  knows; later versions keep every key.
* **Single jobs** stay available: `stacks <folder> --run surface --maps all` (textures, now also from ordinary photos:
  `method` "estimated"), `model <folder>` (one object), `foliage` (leaves).
* **Raw development**: `edit` takes several raws (or a folder) in one call; one call for the folder is much faster
  than one per raw (the Studio starts once).
* Long jobs: poll `intake_job.json` (or `stacks_job.json`, `NAME_model_job.json`) in the output folder; pass
  `--background` and, while the Forge renders, a small `--workers`. The Studio has no graphics-card path yet, so it
  never takes the Forge's card lock (`forge/threed/cardlock.py`); when a GPU path is added it must take that lock first.
* Blender: the Forge's Blender runner can run `studio/blender/build_material.py` with `-- --surface <json> --out <dir>`
  (it prints `STUDIO_RESULT {json}` on its last line); models' glb files import directly.

## Flat Mosaic: `stack mosaic` (BUILD_PLAN_MOSAIC, 2026-11-23)

`stack mosaic FILES-OR-FOLDERS` stitches hand-held views of a flat or gently curved surface (a road, a path, a floor, a
wall) into game surfaces, with their height. The camera may move, tilt, turn and change height between views: every
view's camera is found over the ground and the whole set adjusted at once (`engine/stack/mosaic.py`). Views can be
raw focus stacks (one per spot: merged with their fused depth, Card to Stacks' grouping), Focus Stack results (a
picture with `NAME.depth.npz` beside it), single Dual Pixel RAWs, or ordinary photos and raws (colour only), mixed.

```
studio-cli.bat stack mosaic "D:\pavement\2026-11-24" --out "D:\pavement\out" --json
studio-cli.bat stack mosaic FOLDER --set resolution=fine --set surface=curve --ruler 0:0.20,0.50,0.80,0.50,300
studio-cli.bat stack mosaic FOLDER --pieces "0.10,0.20,0.25,0.40;0.60,0.10,0.75,0.30" --maps all
studio-cli.bat queue add stack mosaic FOLDER --out OUT        (a long set: it runs to its end)
```

Settings (`--set NAME=VALUE`; `schema` lists them under `stack.mosaic`, with what each is best for and weak at):
`method` auto | poses | free, `surface` plane | curve, `resolution` game (1 mm per pixel) | fine (0.5) | full | custom
(with `mm_per_px`), `scale` auto | ruler | focus | relative, `exposure`, `even_out`, `vignette` (on: a texture repeats
it as a blotch), `seams` smart | straight, `crop` rectangle | edge, `height`, `ortho`, `nudge`, `tileable` (on by
default; `--tileable cut|blend` picks the method), `size`, `focal_mm`. `--ruler VIEW:U1,V1,U2,V2,MM`: two points of a
tape in one view (0..1 of that view's picture; VIEW is its number from 0 or its name) and the distance between them.

Writes, beside the first view or in `--out`, the surface files of the Depth Stack above (same names, same JSON
contract, `"method": "mosaic"`), so the RPG Forge's intake reads them unchanged, plus:

| File | What it holds |
|---|---|
| `NAME_misfit_Studio.png` | Where the views still disagree in position: 255 = 4 output pixels or more. |
| `NAME_coverage_Studio.png` | Every view's footprint from above, cyan finest detail .. amber coarsest; red: gaps inside the set. |
| `NAME_pieceNN_{colour,height,normal}_Studio.png` | `--pieces` boxes cut out as decals (the colour with a soft transparent edge). |

The JSON gains `repeat_m` (the ground one copy of the texture covers, in metres: the game's tiling from a measurement,
not a guess; also written for every other surface that knows its size) and a `mosaic` block: the method and surface,
the solve (residual, drift and loop closures, scale and how it is known, lens distortion), each view's camera height,
tilt and ground detail, the relief (its size measured by parallax between overlapping views, the fit, and whether
near and far had to be swapped), whether it was orthorectified or nudged, the coach's advice and the shooting card.
`NAME` is `FIRST_to_LAST_Mosaic`. Output kinds: `surface_*` as above, `mosaic_misfit`, `mosaic_coverage`,
`mosaic_piece`. A set that cannot be made (fewer than two views, no overlap) is a usage error (exit 2) said plainly.

## MCP: an AI model connected directly

`studio/mcp_server.py` serves the same commands over the Model Context Protocol (stdio), so a model calls the Studio's
tools itself and the Studio stays loaded between calls. Setting it up for people is in the README ("Connect an AI
model"); `Connect Studio to Claude.bat` does it for the Claude desktop app, `tools/connect_claude.py --print` shows the
entry for any other client, `studio-mcp.bat` starts the server.

* **Tools are built from the command line's own parsers**: every command above except `batch` is a tool
  `studio_<command>` (dashes become underscores), every option a typed property with the option's help text. A new
  command or option appears as a tool by itself. Arguments use the option names with underscores (`long_edge`,
  `out_file`), `inputs` is a list of paths, flags are booleans, and an option with an optional amount
  (`scratches`, `person`) takes `true` or the amount.
* **`settings`** (on every command that takes a recipe): the recipe as an object,
  `{"exposure_ev": 0.3, "light": {"shadows": 25}}`. `set` (`["light.shadows=25"]`) and `recipe` (files) work too.
* **Answers**: the command's JSON result as text and as `structuredContent`; `isError` is true when the result has
  problems. A wrong argument or setting name comes back as a tool error that names the nearest real ones.
* **Seeing**: a call with `preview` returns the preview JPEG as an image in the answer; `studio_view`
  (`path`, `long_edge`, optional `settings` / `use_sidecar`) returns any picture as a small JPEG and writes no file.
* **Progress**: with a `progressToken`, each progress line is sent as `notifications/progress`. A call runs to its end
  (one at a time); a full-size stack can take minutes, so a client's tool time-out may need raising.
* **Protocol**: specification 2026-07-28 (per-request `_meta`, `server/discover`), and the `initialize` handshake of
  2025-11-25 and earlier for older clients. No extra package: the protocol is spoken directly.
* Standard output carries protocol messages only; logs go to standard error and `logs/studio.log`.
* Tests: `tests/test_mcp_server.py` (8 tests, both protocol eras, a real stdio run, the connect script).
  Not yet run: a real client (the Claude desktop app) against it on Windows.

### The job queue: long work that runs to its end (2026-11-21)

A client waits only so long for one tool call (the Claude desktop app's bridge: about 80 s), and a full-size focus
stack takes 10-20 minutes. Long work therefore goes through the queue (`studio/jobqueue.py`):

* **MCP:** `studio_queue_add` takes `jobs`, a list of `{tool, arguments, title}` (or one `tool` with its
  `arguments`): the same tool names and arguments as a direct call, checked at once, so a wrong argument is said
  straight away. It answers at once with each job's `id`. `hold: true` adds them held until `studio_queue_start`.
  `studio_queue_status` (no `id`: counts, the worker, the latest jobs with a running job's last progress lines;
  with `id`: that job in full, its progress lines and, once finished, the command's own JSON `result`).
  `studio_queue_cancel` (`id` or `all`: waiting jobs dropped, a running one stopped within seconds).
* **Command line** (the RPG Forge's `studio.run`): `studio-cli queue --json add [--hold] stack focus a.CR3 b.CR3
  --make surface --out DIR`, `queue --json status [ID]`, `queue --json start [IDS]`, `queue --json cancel ID|all`.
  Put `--json` before the action: everything after `add` is the job's own command line.
* **How it runs:** a worker started by itself in the background (below normal priority, no window) takes the jobs
  one after another, oldest first, each in a fresh Studio process, until none has waited for 90 s. A crash in one
  job fails that job only (the problem names the Windows code and the last progress lines); the next job runs.
  The worker's heartbeat (`jobs/worker.json`) says whether it is alive; a worker that died with the PC is
  replaced by the next `add` or `start`, and its unfinished job is marked failed so it can be added again.
* **Files:** `jobs/` beside the app (`STUDIO_JOBS_DIR` overrides): one folder per job with `job.json` (state:
  held, queued, running, done, failed, cancelled), `progress.log`, `stdout.txt`, `result.json`; the worker's
  log is `logs/queue.log`. Nothing is deleted; finished jobs stay listed.
* Tests: `tests/test_jobqueue.py` (the worker with stand-in jobs of known outcome: order, a crash, cancel, a dead
  worker, the MCP tools, the command line).

### An extension file for the Claude desktop app

`Make Claude Extension.bat` (`tools/make_extension.py`) writes `extension/photographic-studio.mcpb`: a zip whose
`manifest.json` starts this folder's own Python with `-m studio.mcp_server` (no Studio code inside; the paths are
this computer's, so it is made again on any other). Install it in Claude: Settings, Desktop app, Manage beside
Extensions, Advanced settings, Extension Developer, Install Extension. It is the way in when the settings-file entry
of `Connect Studio to Claude.bat` is not started by the app (the Microsoft Store install). Use one of the two, not
both (they share the name `photographic-studio`).

## Examples

```bat
rem One look for a whole folder of generated art, measured from the pictures that define it
studio-cli.bat look-profile "D:\refs\wasteland" --save "D:\looks\wasteland.json" --json
studio-cli.bat match "D:\art\incoming" --to "D:\looks\wasteland.json" --out "D:\art\matched" --json

rem A tree out of a photo, with a firm edge, as RGBA
studio-cli.bat cutout tree.CR3 --colour 140,60 --hard 0.5 --out cut --json

rem A backdrop split into three parallax layers with a filled background plate, plus its normal map
studio-cli.bat depth backdrop.png --make layers,normals --layers 3 --fill --out layers --json

rem Look before committing to a full-size render
studio-cli.bat edit photo.CR3 --recipe look.json --preview 1200 --json

rem A hand-held focus stack of Dual Pixel raws (bark, 50 mm macro at 1:2) into a game surface that repeats
studio-cli.bat stack focus "D:\textures\bark_01" --make surface --ratio 1:2 --tileable --json

rem Only the measured depth, quickly (no full-size merge)
studio-cli.bat stack focus "D:\textures\bark_01" --make depth --json

rem One Dual Pixel photo on its own
studio-cli.bat dualpixel stone.CR3 --make surface --ratio 1:2 --json
```

## What was run, and what was not (2026-11-17)

Run and tested (`tests/test_cli_tools.py`, 12 tests on synthetic pictures): schema, edit, recipes, match and look
profiles, cutout, depth files and layers, find, batch, the naming and never-overwrite rules, focus stack then edit.
Also run by hand: clean, relight, colourize, restore, jpeg-repair, negative, click cutout, super-res and panorama
stacks on small synthetic frames, `edit --auto` on a CR2, infrared on an 850 nm CR2, and Dual Pixel report, depth and
anaglyph on the R5 eagle file.

Not yet run: Dual Pixel `half1`, `half2`, `hdr` and `wiggle` (they follow the Dual Pixel Lab page's own calls);
`depth --fill` with LaMa (PatchMatch was run); the stacks on full-size raws; `cutout --person` on a photo with a
person in it; `studio-cli.bat` itself on Windows.

Depth Stack (2026-11-21): `stack focus --make surface|depth` and `dualpixel --make surface` run and tested
(`tests/test_cli_tools.py`: the file names, image modes, JSON keys, a second run replacing its own files, wrong values;
`tests/test_depth_stack.py`: the fusion on synthetic hand-held Dual Pixel stacks with known depth). Run by hand on
Bill's files: `dualpixel --make surface` on the R5 eagle (88 s, about 5 GB at its peak on the 2-core build machine)
and `stack focus --make depth` on the nine-frame hand-held fly stack (ordinary raws, focus alone; about 3 minutes,
3.3 GB). Not yet run: a real hand-held Dual Pixel stack (Bill is shooting one); `--make surface` on full-size real
raws (the colour is the full-size merge: as slow as Build full size); `studio-cli.bat` itself on Windows.

## Asked for, not built yet (needs new engine work, not a command)

Flatten light (take the shading out of a photo so it can be a surface colour), roughness maps, and day to night /
infrared to night as a batch look. (Tileable surfaces are built: `--tileable` above.)
