# What is in the next update

<!--
  The waiting list. One plain line per change, written for photographers, added as each part is built:
      - New: Focus Stack merges a focus bracket into one sharp photo.
      - Fixed: exports no longer replace the camera's own JPEG.
  Start each line with "- ". push.bat sends these lines as the comment; Release.bat then moves them into
  CHANGELOG.md under the new version number and empties this list. Leave this header where it is.
-->
- New: Cameras page: drop everything in one pile. Shoot with the lens cap on (every ISO, at least twice, cool and warm if you can, a few long ones), then drop the whole card or folder on the Cameras page. Nothing needs sorting or renaming: every frame is measured one at a time (hundreds of frames are fine), ordinary photos in the pile are left out with the reason, and each camera in the pile gets its own profile.
- New: calibration frames add up. Drop more on another day and they join the ones already measured; a frame is never read twice, even from a copy in another folder. Stop keeps what was measured. Start this camera over forgets it.
- New: the camera's temperature counts. The Cameras page shows your dark frames by ISO and camera temperature (cyan: two or more, amber: one), read noise per temperature in a chart, and what would make the profile better. Clean now uses the read noise measured nearest the temperature the photo was taken at, and the extra noise your camera adds in long exposures.
