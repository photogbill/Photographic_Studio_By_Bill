# What is in the next update

<!--
  The waiting list. One plain line per change, written for photographers, added as each part is built:
      - New: Focus Stack merges a focus bracket into one sharp photo.
      - Fixed: exports no longer replace the camera's own JPEG.
  Start each line with "- ". push.bat sends these lines as the comment; Release.bat then moves them into
  CHANGELOG.md under the new version number and empties this list. Leave this header where it is.
-->
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
