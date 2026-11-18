# Photographic Studio By Bill: notes for Claude

Read these before changing anything in this folder:

- `BUILD_PLAN.md` section 0: the rules and Bill's decisions. It is the source of truth for the whole Studio.
- `BUILD_PLAN_TABS.md`: the build plan for the Dual Pixel Lab, Focus Stack and Super-Res Stack tabs (built, session 12).
- `BUILD_PLAN_PANO_IR.md`: the build plan for the Infrared / Full Spectrum and Panorama tabs (built, session 13, with
  flare ghosts in Develop > Lens and Fill in Retouch).
- `BUILD_PLAN_PLATE_GLOW_BURST_MACRO.md`: the next build (Bill, 2026-11-17): variable IR filter, then **card to
  finished stacks** (block F: mostly built in session 13b as Card to Stacks; F.8 says what is left), Best of burst,
  Macro planner, Plate Reader, Glow guard. Bill's real card for testing: `test_raws\DCIM\100CANON`.
- `BUILD_PLAN_NOISE.md`: Clean rebuilt on the camera's own calibration (Bill, 2026-11-18): truth harness with
  real noise, the residual test and detail guard, deep calibration, a network trained on the camera's own dark
  frames. Not started. Bill's R5 pile (449 raws) is in `calibration_local\`; section 0.3 has what it measured.
- `docs/PROGRESS.md`: where things stand. Keep it current.
- `docs/CLI.md`: the Studio from the command line for other programs (the RPG Forge, Claude): `studio/cli_tools.py`,
  `studio-cli.bat`. A new tool gets its command there, with a test in `tests/test_cli_tools.py`. The MCP server
  (`studio/mcp_server.py`) builds its tools from those commands by itself: do not add tools there by hand.

Standing rules that are easy to forget:

1. **Write down what changed, as you build it.** After ANY part is built or fixed, add one plain line for
   photographers to `docs/NEXT_UPDATE.md` (`- New: ...`, `- Fixed: ...`, `- Faster: ...`). `push.bat` sends those
   lines as the comment, `Release.bat` files them in `CHANGELOG.md` under the new version, and GitHub shows them on
   the release page that "What's new" in the Studio opens. Bill does not write comments; this is how they get written.
2. **Do not push.** Bill pushes with `push.bat`, which also makes the update every tester receives.
3. **Bill does not use the command line.** Everything he runs is a `.bat` he can double-click. Ask him questions in
   chat, in plain words, with a recommendation; never make him open a file to answer.
4. **Nothing is written to C:\ or the Windows user profile by default.** Data lives beside the app (`studio/paths.py`).
5. **Exports are named `NAME_Studio.ext`** (`studio/engine/io/export.py: export_name`), and nothing the Studio did
   not write is ever replaced (`free_target`). Never write a file under the name of a photo.
6. **Errors report themselves.** Inside any `except` that catches a fault of the Studio, call
   `studio.app.common.report_problem(what, exc)` (engine and CLI code: `studio.log.save_error_report`). It writes
   `error_reports/error_YYYY-MM-DD_HH-MM-SS.txt`. Problems that are not faults (an unreadable raw, a read-only
   drive, no internet) are said where they happen and write no report.
7. **Every method the person can choose says what it is best for and weak at** (plan 23.1), and results are
   colour-coded by how they were produced (cyan measured, amber estimated, violet shared).
8. **Run the tests before saying something works** (`Run Tests.bat`, or `python -m pytest`). New code gets a test on
   synthetic data with known truth. Tests never write outside `outputs/`.
9. **Line endings:** `.bat` files keep CRLF, `.sh` files keep LF (`.gitattributes`). A `.bat` with LF endings breaks.
10. **Commits:** no Claude-Session link; a Co-Authored-By line is fine.
