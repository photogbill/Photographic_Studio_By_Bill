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
* Inputs: raws, TIFF, PNG, JPEG, and folders of them (`--recursive`). Points and boxes are fractions 0..1 of the
  picture (u to the right, v down).
* Default output: a raw becomes a 16-bit Adobe RGB TIFF; any other picture an 8-bit sRGB PNG; anything with
  transparency a PNG. Change with `--format --space --bits --quality --long-edge --sharpen-for`, or name the file
  with `--out-file`. `--out FOLDER` says where (default `outputs\cli`).
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
| `clean` | Noise removal. `--strength 1` `--chroma` `--method` `--keep-grain`. Best on raws. |
| `relight` | Move the light using the picture's depth (estimated). `--azimuth --elevation --amount --soft`. |
| `colourize` | Black and white to colour (a plausible guess). `--amount --warm --tint`. |
| `restore` | Old photos and scans: `--scratches` `--stains` `--descreen` `--jpeg` `--colourize`. |
| `jpeg-repair`, `descreen`, `negative`, `infrared` | One tool each; `negative --kind colour|bw`; `infrared --filter --preset` (raws only). |
| `match --to REF` | Colour match to a reference picture, or to a look profile. `--amount 0..1.5`. |
| `look-profile PICTURES --save FILE.json` | Measure a look profile (nine colour statistics) from reference pictures. |
| `depth --make map,view,normals,layers` | 16-bit depth map (near = white), coloured view, normal map (`--normals-y up|down`), or `--layers N` RGBA layers, 1 = farthest, that stack back into the picture. `--cuts` sets where they divide; `--fill` makes layer 1 a whole background plate. Estimated from one picture. |
| `cutout` | A subject with a soft edge as RGBA (straight alpha), or `--make mask`. Select with `--person [part]`, `--click U,V` / `--avoid U,V` (`--model classic|tiny|large`), `--depth-range`, `--luminance LO,HI`, `--colour HUE,WIDTH`, `--skin`, `--shape JSON`; `--combine`, `--invert`, `--feather`, `--hard 0.5` (firm edge for leaves and fences). |
| `find --what ...` | Data, no file: `size, info, stats, faces, people, dust, flare, crops, subject, red-eyes, blemishes, hdr, jpeg, infrared, depth`. |
| `stack focus|superres|pano FRAMES` | A focus bracket, a burst, or a panorama into one linear 32-bit file. Finish it with `edit`. Tool settings with `--set` (listed under `stack` in the schema). |
| `dualpixel --make report,depth,half1,half2,hdr,anaglyph,sbs,wiggle` | What a Canon Dual Pixel RAW can make. Its depth is measured, not estimated. |
| `schema [NAME]`, `looks` | What exists. |
| `batch JOBS.json` | Many commands in one start. Each job is a list of arguments, a command-line string, or `{"id", "args"}`; each job's result comes back under `jobs`. `--stop-on-error`. |

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

## Asked for, not built yet (needs new engine work, not a command)

Flatten light (take the shading out of a photo so it can be a surface colour), make tileable (a seamless patch with
relief and roughness), and day to night / infrared to night as a batch look.
