# What is in the next update

<!--
  The waiting list. One plain line per change, written for photographers, added as each part is built:
      - New: Focus Stack merges a focus bracket into one sharp photo.
      - Fixed: exports no longer replace the camera's own JPEG.
  Start each line with "- ". push.bat sends these lines as the comment; Release.bat then moves them into
  CHANGELOG.md under the new version number and empties this list. Leave this header where it is.
-->
- New: Paint > History, with branching undo. Go back a few steps and paint something new, and the old steps are kept as a branch instead of being lost; click any step on any branch to go back to it, so two ideas can be kept and compared.
- New: Paint > Layers has groups. Select a layer and press Group; groups nest, open and close, have their own opacity and blend mode (Pass-through keeps every layer blending as before, the others treat the group as one picture), move in and out with the arrows, merge into one layer, and are kept in .ora files both ways (Krita, GIMP and MyPaint groups open as groups). Painting with a group selected goes to the top layer inside it; a locked group stops painting on everything in it.
- New: Paint > Colour has three pickers: the wheel and square, a triangle (as Krita and MyPaint), and a wide-gamut picker in OKLCH that draws where sRGB and Display P3 end, so you can see which colours the canvas can hold.
- New: Paint > Colour has swatch sets: add the painting colour, take a palette of the painting's main colours (dark to light), save sets (as .gpl, which Krita, GIMP, Inkscape and MyPaint read) and load .gpl or Krita .kpl sets. Comes with the Zorn palette, an earth palette and ten value steps.
- New: Paint > Keys: give any action a key, or a tablet key (set the tablet's driver to send a key, then press it here). Each action can have a keyboard key and a tablet key; giving a key to a new action takes it from the old one and says so. Saved beside the app.
- New: Paint exports WebP (smaller than JPEG and keeps see-through areas), and every export now carries a colour profile: sRGB, Display P3, Adobe RGB or ProPhoto, with 16-bit PNG and TIFF. Exports are marked as the Studio's, so exporting again replaces only the Studio's own earlier file.
- New: Paint can save layers in 16-bit (Layers > Save layers in 16-bit), so very soft glazes and slow fades keep every shade instead of falling into steps. MyPaint and Krita read these files. Crash-recovery copies are now always 16-bit.
- New: Paint has a fourth blend style, LPM (Lorentz Pigment Mixing, adapted from Ivkovic et al. 2025): blends between two different colours get a touch more saturation than Classic. The effect is small; it says so in the Brush tab.
- New: Paint > History > Side by side: where you are next to another branch (or any step: right-click it), without changing the painting; then stay, or go to the other.
- New: Paint has smudge brushes: Smudge (smear) pulls paint and shapes along like a finger in wet paint, Blender (dull) blends two colours into a soft mix, and Oil (loaded brush) paints while mixing with the wet paint under it. Smudge length, colour rate and "read every layer" are in the Brush tab.
- New: Paint mixes colours like real paint: the Pigment blend style (Kubelka-Munk over 38 wavelengths, ported from spectral.js) makes blue and yellow go green, and a new Gouache (pigment) brush uses it. For now it is slower than the other blend styles on big brushes over detailed areas.
- New: Paint reads MyPaint brushes (.myb) and paints them with a port of MyPaint's own brush engine, curves and all (Brush tab > Load MyPaint brushes). Loaded brushes are kept beside the app and come back next time. A few MyPaint features are not ported yet, so some brushes look a little different from MyPaint.
