# What is in the next update

<!--
  The waiting list. One plain line per change, written for photographers, added as each part is built:
      - New: Focus Stack merges a focus bracket into one sharp photo.
      - Fixed: exports no longer replace the camera's own JPEG.
  Start each line with "- ". push.bat sends these lines as the comment; Release.bat then moves them into
  CHANGELOG.md under the new version number and empties this list. Leave this header where it is.
-->
- New: Focus Stack measures depth. Shoot a focus stack in Dual Pixel RAW and every place is measured two ways (which frame is sharpest, and each frame's own two halves), then fused into one height map. Hand-held is fine: the stack measures its own focus steps, even uneven ones.
- New: the Depth view in Focus Stack shows that height, coloured by how each place was measured (cyan two ways, amber one way, violet filled in where nothing could be measured), with the places where the two measures disagree in red (something moved, a reflection).
- New: "Save surface for games" in Focus Stack writes colour, height, normal, confidence and agreement maps and a file saying how they were measured, for game textures and 3D. Ordinary raws give the same files from focus alone. Tick "Make it tileable" for a texture that repeats without a join.
- New: for other programs, `stack focus FRAMES --make surface` (or `--make depth`, the quick part) and `dualpixel FILE --make surface`, with `--ratio 1:2` for the size in millimetres and `--step-mm` for a focusing rail.
- Faster: reading a Dual Pixel photo's two halves is much quicker (the shading step on a 45 MP R5 file went from about 39 seconds to under a second).
