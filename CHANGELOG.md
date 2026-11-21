# Photographic Studio By Bill: what changed, update by update

Newest first. Written by tools/whats_new.py from docs/NEXT_UPDATE.md each time Release.bat makes an update.

## 0.1.13 (2026-11-21)

- New: Focus Stack measures depth. Shoot a focus stack in Dual Pixel RAW and every place is measured two ways (which frame is sharpest, and each frame's own two halves), then fused into one height map. Hand-held is fine: the stack measures its own focus steps, even uneven ones.
- New: the Depth view in Focus Stack shows that height, coloured by how each place was measured (cyan two ways, amber one way, violet filled in where nothing could be measured), with the places where the two measures disagree in red (something moved, a reflection).
- New: "Save surface for games" in Focus Stack writes colour, height, normal, confidence and agreement maps and a file saying how they were measured, for game textures and 3D. Ordinary raws give the same files from focus alone. Tick "Make it tileable" for a texture that repeats without a join.
- New: for other programs, `stack focus FRAMES --make surface` (or `--make depth`, the quick part) and `dualpixel FILE --make surface`, with `--ratio 1:2` for the size in millimetres and `--step-mm` for a focusing rail.
- Faster: reading a Dual Pixel photo's two halves is much quicker (the shading step on a 45 MP R5 file went from about 39 seconds to under a second).

## 0.1.12 (2026-11-18)

- New: Clean chooses its own strength for every photo (Develop > Clean > Auto). It works out, from the photo alone, how much noise each strength would leave and how much fine detail it would keep, and takes the gentlest one that does the job. It needs your camera calibrated on the Cameras page (lens-cap frames and flat pairs); until then your sliders decide, and it says so.
- New: Clean uses your camera's calibration before it removes noise: hot and warm pixels are healed from the camera's own map, the banding that is the same in every photo is subtracted exactly, and the black level is corrected per colour. Develop shows what was taken out.
- New: "Show what Clean removed" (Develop > Clean). Three pictures of a piece of your photo: before, after, and what was taken out. If the third looks like television static, only noise was removed; anything that was not noise is marked in magenta.
- New: Dual Pixel photos are judged by their own two halves: Clean cleans one half and checks it against the other, a measurement rather than an estimate, and reports whether the photo's noise matches the camera's profile.
- New: Clean cleans lifted shadows harder. When you brighten a photo or open its shadows, the parts that end up lifted are cleaned in proportion, so recovered shadows are not left grainy.
- New: two more Clean methods: Fusion (each method where it is best) and Collaborative (the slow BM3D-class yardstick the others are measured against).
- New: the Cameras page lists what Clean takes from your calibration, and what it looked at and left alone, with the reason (for example: a master dark frame would add noise unless made from many more frames).
- New: Train a general network (Cameras page): one network over every camera you have calibrated. A camera with no network of its own uses it, and a camera's own training starts from it and needs far fewer steps.
- New: the noise benchmark checks itself against your real flat frames, scores Clean with Auto on, and can score a folder of your own photos (drag it onto Run Noise Benchmark.bat).
- New: docs\CLEAN.md, a plain-words guide to Clean and to teaching it your camera; the Help page has the short version.
- Fixed: hot pixels are counted in electrons, so the same photosites are "hot" at every ISO (before, a faint one counted as hot at ISO 3200 and warm at ISO 400). Drop your calibration folder on the Cameras page again to recount.
- Fixed: at ISO settings where the camera's own noise is shared between neighbouring photosites (the R5 at ISO 400), Clean now measures that and allows for it.
- Fixed: Run Tests no longer fails over the optional training tools (PyTorch, ONNX Runtime), which are only installed when you train a camera's network.

## 0.1.10 (2026-11-18)

- Fixed: flat frames shot with a fast lens wide open are no longer left out. A flat has to be smooth, not perfectly even: the lens may darken toward the corners.
- Fixed: two dark frames a degree apart are compared as a pair even when they fall either side of a temperature step.
- Fixed: the Studio no longer reports "stopped unexpectedly last time" after a run that only opened a file dialog.
- Fixed: hot pixels are counted honestly. A hot pixel is a bright dot in any photo; a warm pixel only stands out of the noise at some ISO settings. The Cameras page shows the two apart (it used to call thousands of warm pixels hot at ISO 400).
- New: the Cameras page says what to shoot at the top: dark frames (needed), long dark frames (better) and flat pairs (for a Full profile), with what a flat pair is and why it takes two.
- New: the three What to shoot cards tick themselves: for each camera, green when that part is covered, amber when it is partly there, with what is missing.
- New: Detail guard in Clean (on by default). What Clean removed is tested against your camera's measured noise: structure that stands above the noise is given back, and where more was removed than noise can account for, the surplus is given back too, so an area keeps a little grain and all of its texture. Nothing may be added that noise could not have hidden. The panel shows how much was returned. Photos you already edited render as before until you tick it.
- New: Deep calibration (a button on each camera's card in Cameras). It reads the camera's dark frames once more and keeps patches of their real noise (the noise bank) and the part of the banding that is the same in every photo. Stays on this PC.
- New: Run Noise Benchmark.bat scores every Clean method on your own camera's real noise, on pictures whose right answer is known: noise left, texture kept, edges kept, colour blotches, and whether what was removed was only noise.
- New: teach Clean your camera. Each camera's card in Cameras has "Clean, taught by this camera": it shows what is ready (Full profile, noise bank, clean photos, trained network) and starts the training (Train Camera Model.bat) on your graphics card. Then choose the method "learned" in Develop > Clean. It is always followed by the detail guard and never invents detail.
- New: the Studio says how many clean photos the training needs (about 200, varied) and why, counts the ones in the folder you choose (raws at ISO 400 or lower), and will not train on too few.

## 0.1.9 (2026-11-18)

- New: Cameras page: drop everything in one pile. Shoot with the lens cap on (every ISO, at least twice, cool and warm if you can, a few long ones), then drop the whole card or folder on the Cameras page. Nothing needs sorting or renaming: every frame is measured one at a time (hundreds of frames are fine), ordinary photos in the pile are left out with the reason, and each camera in the pile gets its own profile.
- New: calibration frames add up. Drop more on another day and they join the ones already measured; a frame is never read twice, even from a copy in another folder. Stop keeps what was measured. Start this camera over forgets it.
- New: the camera's temperature counts. The Cameras page shows your dark frames by ISO and camera temperature (cyan: two or more, amber: one), read noise per temperature in a chart, and what would make the profile better. Clean now uses the read noise measured nearest the temperature the photo was taken at, and the extra noise your camera adds in long exposures.

## 0.1.8 (2026-11-17)

- New: Card to Stacks (its own place in the sidebar). Point it at a card or a folder, or right-click photos in the Library: Find focus stacks. It sorts everything into focus stacks, bursts and single shots, leaves out the frames that would hurt each stack with the reason (soft everywhere, adds nothing, shaken, camera moved too far, exposure jump, subject moved), and Stack all merges every stack in the background. Nothing is deleted; tick a frame to put it back. Results are saved beside the frames as FIRST_to_LAST_Focus_Studio.tif.
- New: the import summary has a Find focus stacks button that opens Card to Stacks on what was just copied.
- New: bursts shot with AI Servo AF (a dog running at you, birds in flight) are recognised as bursts, not focus stacks, from what the camera recorded.
- New: Show what's in focus (Card to Stacks). Click a stack: cyan wherever some frame got it sharp, amber where there is detail but no frame got it sharp, so a missed band of focus shows before you stack. Click a frame: cyan where that frame is in focus.
- New: Sort focus stacks when importing from a card. The frames that would hurt a stack are still copied, into a "Left out" folder inside that day's folder, so they are out of sight but never lost.
- New: In focus (F) in Review: cyan over the sharpest detail of the photo, so a missed focus shows at a glance while you rate. It works zoomed in and in Compare too.

## 0.1.7 (2026-11-17)

- New: the Studio can be driven by other programs (studio-cli.bat): every editing tool, colour match to a look, depth maps and layers, cut-outs and stacks, with results as data. Nothing changes in the Studio itself. See docs/CLI.md.
- New: an AI model can use the Studio directly (MCP). Double-click "Connect Studio to Claude.bat" to let the Claude desktop app develop, clean, match and cut out the pictures you point it to; the README says how to connect any other AI app, and what does and does not leave your PC.

## 0.1.6 (2026-11-17)

- Fixed: the Studio no longer keeps every photo you opened in memory. It used to grow by a few hundred MB per photo until it slowed down or stopped.
- New: every export is named after its photo with _Studio added (IMG_0001.CR2 becomes IMG_0001_Studio.tif), so an export can never take the place of your original or of the camera's own JPEG.
- New: when something goes wrong, the Studio saves an error report by itself: a dated file in the error_reports folder (Help has a button that opens it). Send that file to Bill.
- New: the Windows installer opens a window for you to pick where the Studio goes, and checks there is room before it downloads anything.
- Fixed: one photo that cannot be read no longer stops a whole export. The rest are exported and the problem photo is named.
- Fixed: a raw the Studio cannot open (Fujifilm X-Trans, Canon sRAW, Sigma Foveon, a damaged file) now says why, where the photo would be, instead of staying on Opening.
- Fixed: a photo you only looked at no longer gets a sidecar file, and no 'could not save' message on a read-only card.
- Faster: opening a photo no longer measures the lens's colour fringing twice.
- New: an empty Library folder says what to do next. Panasonic RW2 files show their camera and exposure.
- Fixed: the Develop panel no longer runs off its right edge before a photo is open, or with a composite subject selected.
- New: Focus Stack tab. Merges photos focused at different distances into one sharp photo, with a depth map, a confidence view and a halo guard. Four ways to merge (Auto, Depth, Pyramid, Average), each saying what it is good and weak at. Hand-held stacks line up locally.
- New: Super-Res Stack tab. Merges a burst into one cleaner photo, or a 1.5x or 2x larger one when the frames moved a little (drizzle from the raw data). Modes for hand-held, tripod, the moon and the night sky; a coverage chart says whether the extra resolution was really there; a view shows what was thrown out.
- New: Dual Pixel Lab tab. Opens Canon Dual Pixel RAW files (checked on the R5): the two half-views, a depth map with how sure it is, a focus report, a 3D wigglegram, anaglyph and side-by-side, a 16-bit depth map for other programs, and up to a stop of highlights the normal image clipped. A normal raw says plainly that it is not Dual Pixel.
- New: the R5's Dual Pixel CR3 is read directly. No need to split it with another program first.
- New: stacks are saved beside their photos as FIRST_to_LAST_..._Studio.tif, a linear file that opens in Develop with every tool working on it. The Library marks them with a stack chip.
- New: right-click photos in the Library: Send to Focus Stack, Send to Super-Res Stack, Open in Dual Pixel Lab (also in the Tools menu).
- Fixed: colouring a part of a black and white photo (Colour a part) now works without the DDColor model installed. It used to do nothing.
- Fixed: lining up frames of a plain grey card, fog or a lens cap no longer moves them by a random amount.
- Fixed: Composite, click-to-select, Depth haze and JPEG repair failed on installed copies (they needed a package the installer does not include). They now use the Studio's own code and give the same results.
- New: Infrared tab, for infrared and full-spectrum raws. It guesses the filter from the raw (and says how sure it is), sets white balance far beyond the camera's range (Auto foliage, click foliage, click a neutral, or a filter you measure once on a grey card), and offers looks: swap red and blue, blue sky (590), Goldie, a colour-infrared film look, and black and white from the best channel. Everything stays in the photo's own edit, so Open in Develop carries on with every tool, and Apply to the shoot does the rest of the strip.
- New: an 850 nm raw records no colour (the Studio measures this with a colour meter and says so), so it opens in black and white and can be developed without demosaicing: every photosite used, full resolution, no colour noise.
- New: black and white from the channels in the Infrared tab: exposure, contrast and separation per channel, then one mix (-300% to +300%), with a live noise meter that says what the mix costs. Save it as a B&W recipe; your last recipe for that camera and filter comes back on the next photo.
- New: Detail by scale (the shades of shadow) is in the Infrared tab too, judged in black and white. It is the same setting as in Develop.
- New: infrared hot spot correction. Measure the bright centre a lens shows in infrared from one photo, or learn it from several (Library > Tools > Learn IR hot spot). Off until you turn it on, like vignetting.
- New: colour by brightness for black and white infrared photos (ironbow, inferno, platinum, cyanotype, selenium). Labelled as added colour: the raw has none.
- New: visible + infrared fusion (infrared detail with visible colour, colour infrared, or an NDVI vegetation map) for a matched pair shot on a tripod.
- New: Channel masks in Develop: select by one raw channel or the ratio of two. Red / blue picks out infrared-bright foliage from sky and skin.
- New: right-click photos in the Library: Edit in Infrared tab (also in the Tools menu). Develop shows "IR white balance: set in Infrared" for those photos.
- New: Panorama tab. Give it overlapping frames (Library: right-click, Send to Panorama) and it works out where each one points, estimates the focal length when the lens records none (it found 138 mm for a 135 mm manual lens), measures the lens's distortion and vignetting from the overlaps, evens out exposure and colour, cuts the seams where the frames agree, and blends them so it looks like one photo. Rectilinear, cylindrical, spherical (360), Panini and Mercator, each saying what it is good for.
- New: Panorama flare guard. Lens flare and the infrared hot spot move with the camera, so a normal stitch repeats them; where two frames overlap, the Studio keeps the frame without the flare. Overexposed frames give way to frames that are not blown out.
- New: the Panorama tab shows the layout (each frame coloured by how well it is tied in), the seams, and where frames disagreed (something moved), and names any pair that does not line up (parallax or a moving subject). The result is saved as FIRST_to_LAST_Pano_Studio.tif and opens in Develop.
- New: lens flare removal. Develop > Lens > Find flare ghosts finds the round reflections a bright light leaves along a line through the centre of the photo (infrared shows them most: lens coatings are made for visible light). Subtract the glow takes the added light away and keeps the detail under it; untick any ghost that is a real round thing.
- New: Fill in Retouch (content-aware fill). Click or brush over a flare, a wire, a sign or a person and the Studio rebuilds it from what is around it, with real texture from your own photo. Rebuild ghosts with Fill (Develop > Lens) does this for strong or blown-out flare ghosts.
- New: LaMa, a generative fill model that runs on your PC, rebuilds what you Fill: it carries branches, horizons and edges through the area, even where nothing like them is elsewhere in the photo. Download lama_fp32.onnx from huggingface.co/Carve/LaMa-ONNX into the models folder and double-click Convert Models.bat. Without it Fill still works from your photo alone.
- Fixed: a panorama seam no longer runs along the very edge of a frame, where a thin bright line could show.
- New: panoramas now even out the broad light in every overlap (a flare veil in one frame, an infrared hot spot, light that changed between frames), so no step shows where one frame takes over from the next.
