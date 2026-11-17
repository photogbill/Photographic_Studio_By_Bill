# Photographic Studio By Bill: what changed, update by update

Newest first. Written by tools/whats_new.py from docs/NEXT_UPDATE.md each time Release.bat makes an update.

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
