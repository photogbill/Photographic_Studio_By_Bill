# What is in the next update

<!--
  The waiting list. One plain line per change, written for photographers, added as each part is built:
      - New: Focus Stack merges a focus bracket into one sharp photo.
      - Fixed: exports no longer replace the camera's own JPEG.
  Start each line with "- ". push.bat sends these lines as the comment; Release.bat then moves them into
  CHANGELOG.md under the new version number and empties this list. Leave this header where it is.
-->
- Fixed: scrolling down the Controls panel no longer moves the sliders and lists the pointer passes over. The mouse wheel now moves a slider only after you have clicked it.
- Fixed: photos opened in an earlier version, but never edited, kept that version's older look (for example brighter at ISO 50). Untouched photos now always use the newest rendering; photos you edited still keep theirs.
- Fixed: "Before (as shot)" now uses the same rendering version as the photo, so the before/after wipe shows only your edits.
- Fixed: Make it good now says plainly what it changed. It never changes brightness or colour, so on a clean, sharp, low-ISO photo it only adds gentle sharpening and the lens check.
- New: the preview is now colour-managed. The Studio sends the photo from its ProPhoto working space straight to your monitor's own colour profile, as Lightroom and Photoshop do, so a wide-gamut screen shows its full range and what you see matches what you print. A monitor without a profile is shown as Windows shows it, and the Studio tells you it needs calibrating.
- New: Tone > Curve colour. "Natural" (the default for new photos) keeps every colour's mix while contrast is added, so skin stays true and matches Lightroom and your prints; "Rich" is the old per-colour curve with more punch. Photos you already edited keep their look.
- New: soft proofing in Develop. Choose your printer and paper profile and the preview shows the print, with an option to mark in magenta the colours the paper cannot print.
- New: Save for web, a button next to Export and an option in the Export window: a 2048-pixel sRGB JPEG, sharpened for screens, at the smallest file size that still looks the same as the best-quality JPEG (checked on detail and colour). Saved as NAME_web_Studio.jpg.
- Changed: exports default to sRGB. Adobe RGB, ProPhoto RGB and Display P3 are still in the list for prints and archives.
- Fixed: scanned prints and film keep the Rich curve by default, and stacks the Studio made render exactly like the raws they came from.
- New: FITS files (astronomy cameras, Siril, PixInsight, DeepSkyStacker) open in the Studio, mono, colour or one-shot colour raws. Linear astro data opens dark: raise Exposure or press Auto. The Library shows them with an astronomy stretch so you can see the stars.
- New: Export > 32-bit FITS (astronomy): the finished picture as linear 32-bit float, ready for Siril or PixInsight.
- New: Night-sky stacks measure every frame's stars. The frame with the tightest stars is the reference, and frames with blurred or trailed stars are left out, with the reason given.
- New: the astronomy tools (astropy and photutils) are added once by themselves when the Studio starts with the internet on. The Studio works without them.
- New: Paint, a digital painting section in the sidebar, with "Start Paint.bat" (and start-paint.sh on Linux) to open straight into it.
- New: Paint > Tablet check shows live what your pen sends (pressure, tilt, speed, eraser end, keys) and says in plain words what to do when pressure is not arriving. Written first for the Huion HS64 on Linux.
- New: Paint's canvas zooms, pans, rotates and mirrors the view, and takes every pen event as it arrives, with nothing guessed ahead of the pen.
- New: Paint keeps your painting safe: unlimited undo (by memory), an autosave every minute beside the Studio, and an offer to recover a painting after a crash or power cut.
- New: Paint's brushes: Pencil, Ink pen, Round brush, Soft airbrush, Marker, Flat wash and two erasers, each saying what it is best for and weak at. Size, opacity and flow follow pressure; wash, build-up and airbrush painting.
- New: your own pressure curve for every brush in Paint (Linear, Soft, Firm, S-curve or drag your own), and seven smoothing modes, including Speed-adaptive: steady slow lines without lag on fast ones.
- New: Paint's layers (add, delete, move, merge, rename, opacity, seven blend modes, hide, lock, alpha lock), a colour wheel with recent colours, Alt + click to pick a colour, and a Values view to check light and dark.
- New: Paint saves and opens OpenRaster (.ora) paintings with their layers, which Krita, GIMP and MyPaint also read, and exports a flat PNG, TIFF or JPEG named NAME_Studio.
- New: Paint > Fill: fills an area by colour, reading the whole picture, the layer, or a reference layer (fill flats on their own layer under the line art), with gap closing so the fill stays inside line art that has small breaks.
- New: Paint > Gradient: drag to lay a linear or radial gradient, to the second colour or to see-through.
- New: Paint mixes colours three ways, chosen per brush and per gradient: Classic (as Photoshop and Krita do), Light (glows mix like real light) and Oklab (blends between hues without a muddy grey middle).
- New: any picture can be a brush tip in Paint: a scan of an ink mark, a leaf, a sponge.
- New: right-click (or the pen's side button) on the painting opens a pop-up palette with your brushes and recent colours.
- New: Paint's canvas is shown through your monitor's colour profile, the same way Develop shows photos, and soft proofing applies to paintings too.
