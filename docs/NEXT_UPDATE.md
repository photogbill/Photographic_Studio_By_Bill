# What is in the next update

<!--
  The waiting list. One plain line per change, written for photographers, added as each part is built:
      - New: Focus Stack merges a focus bracket into one sharp photo.
      - Fixed: exports no longer replace the camera's own JPEG.
  Start each line with "- ". push.bat sends these lines as the comment; Release.bat then moves them into
  CHANGELOG.md under the new version number and empties this list. Leave this header where it is.
-->
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
