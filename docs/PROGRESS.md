# Build progress

Updated 2026-11-15 (session 9). Plan: BUILD_PLAN.md v1.7.

## Where things stand

| Phase | State | Notes |
|---|---|---|
| 0. Foundation | **Done, needs Bill's double-click** | Layout (2.2), `Install.bat`, `Run Tests.bat`, logging, `THIRD_PARTY.md`, true-ISO fix in the analyzer. Stack taken as recommended (Python + NumPy/SciPy + rawpy; PySide6 in Phase 4) since Bill was away. Acceptance: `Run Tests.bat` green **on Bill's PC** (green on the Linux test machine: 251/251 with the sample raws). |
| 1. Decode + develop engine | **Done, awaiting Bill's eye** | Stages 1, 2, 10, 11, 15 (basic), 21. `studio-cli` / `Develop Raws.bat` turns CR2/CR3 into 16-bit Adobe RGB TIFF. Export now also writes lens, exposure and the true ISO (EXIF in JPEG, XMP in TIFF/PNG). |
| 2. Calibration engine | **Engine done, awaiting frames** | `Calibrate Camera.bat` / `studio-cli calibrate`. Part G (exposure latitude) not yet. Acceptance needs Bill's Part A and B frames. |
| 3. Denoiser v1 | **Built, awaiting Bill's blind test** | Hybrid NLM + starlet default; MP-PCA and starlet candidates; blind-test tool (10 pairs ready). BM3D-class baseline still to write. Off by default until the blind test. |
| 4. Studio shell | **Built** | `Start Studio.bat`: Library, Develop (live render, wipe, loupe, histogram, panels, undo, sidecars, Make it good), Export, Cameras, Lenses, Help. |
| 5. Lens | **Built, awaiting Bill's eye** | Lateral CA and purple-fringe removal (earlier), plus this session: **Fix lens** button with a colour-coded report; distortion from lensfun (violet) or measured from straight lines (amber, strict gate: never on organic subjects); lens falloff measured (lensfun or estimated) with correction OFF by default; stage 13 geometry in one resample (distortion, vertical/horizontal perspective, straighten with **Auto level**, crop with aspect presets drawn on the photo, constrain-to-image); **lens tagging** (Library: Tag lens for a whole shoot; Develop: lens picker; shared starter list + your own private list; written into exports). Acceptance: fringing on Bill's vintage-lens files fixed by one button with no colour damage. |
| 6. Full develop | **Built, awaiting Bill's eye** | Session 4: **Microcontrast** (stage 16) and **capture sharpening** (stage 12). Session 5: the full **Light** panel (highlights, shadows, whites, blacks in stops; **Smart lighting**, region by region with a guided filter; **Auto** that sets highlights / whites / blacks only and keeps low-key photos low-key, the named test; a point **tone curve** per channel), the **Colour** stage (OKLCh: saturation, vibrance with skin protection, **Pick white** eyedropper, **Auto** white balance (grey-edge), a **DxO-style colour wheel** with several ranges, **HSL** in 8 bands, **colour grading** wheels with balance and blending, **black & white** with a colour mixer, **.cube LUTs**), **Looks** (8 starter looks incl. a draft **"Bill's Candlelight"**, Save look), **Copy / Paste settings** (Develop, and Library for many photos), **virtual copies**, a labelled **History** with **snapshots** (kept in the sidecar), **export presets**. Session 6: **highlight rebuild** (clipped channels get the colour of the clipped area's edge), **Dehaze** (dark-channel prior), **Texture**, **film grain** (size, roughness, mono or colour), **film toe** and **highlight roll-off** with a drawn film curve, a **parametric curve** (four regions), **Auto colour** (vibrance from measured colourfulness), **B&W filter presets**, a **camera-profile picker** reading DNG camera profiles (.dcp), and **output sharpening** for screen / glossy / matte in Export. Acceptance: Bill's microcontrast blind test against DxO; "Bill's Candlelight" tuned by Bill. |
| 7, 8 | Not started | Wizards; Dual Pixel Lab. |
| 9. Local + retouch | **Built (masks session 6, retouch session 7), awaiting Bill's eye** | **Masks** in Develop: Brush (feather, flow, erase, **auto mask**), Gradient, Radial, **Object** (one click selects a leaf; Alt-click takes a part away), **Colour**, **Brightness** and now **Skin** (skin tones of any complexion); shapes combine (add, subtract, only-where), invert, amount; inside each mask: exposure, contrast, highlights, shadows, whites, blacks, temperature, tint, saturation, vibrance, texture, clarity, dehaze, sharpness, noise reduction and now **Smooth skin** (softens blotches and uneven colour, keeps pores and edges). **Retouch** (its own group): **Heal**, **Clone** and **Red eye** spots, clicked or brushed along a hair, source chosen automatically and draggable; **Find dust** (one photo); **Learn dust map** in the Library (the dust on your sensor, learnt from several photos, kept on this PC) and **Dust map** in Develop to heal it in any photo (also `--dust-map` for batch develop). New **100% view** (Z, mouse wheel or the 100% button; drag to move) where every tool works. Still to do: depth and focus range masks (need Dual Pixel, Phase 8), denoise strength and microcontrast per mask, editing a drawn gradient or circle by its handles, content-aware fill for large removals. Acceptance: Bill selects one leaf in two clicks and adjusts it; dust removed automatically. |
| 9c. Person mask | **Built (session 9), awaiting a real portrait** | Masks > **Person**: everyone in the photo selected on this PC (MediaPipe person segmentation + BlazePose 33 body points), then a **Part**: whole person, background, skin, face, hair, clothes, arms, legs, torso; snapped to the photo's edges. Also in "Refine" (+ / - / x Person). |
| 9b. Portrait | **Built (session 9), awaiting a real portrait** | New **Portrait** group in Develop. **Find faces** (Google MediaPipe face detector and 468-point face mesh, run on this PC by the Studio's own NumPy code, no new install; a brightened copy is tried for candle-dark photos) or **Click the eyes** when a face is missed. **Remove blemishes** (one button: red spots and dark spots become Heal spots in Retouch, amber; moles and beauty marks are left alone). Sliders, all 0 by default: **Younger** (all of them at once), **Lines**, **Under eyes**, **Smile lines**, **Even skin**, **Jawline**, **Eyes**, **Teeth**. **Body sculpting**: Push / Shrink / Grow brushes and shape tools (**Breast lift & size**, **Butt improvement tool**, **Slim waist / tummy / arm / thigh**) with Lift, Size, Slim and width sliders; every warp capped. Acceptance: the person in a real portrait looks like themselves on a good day, judged by Bill and the sitter. |
| 10+ | Not started | |

## Waiting for you in the outputs folder (already generated)

* `outputs\blindtest\dxo_ready\index.html`: the Phase 3 blind test, 10 pairs (Studio vs your DxO exports, 100%).
* `outputs\develop\compare_20261114_080709\index.html`: all 17 sample raws, camera JPEG vs Studio, wipe slider.
* `outputs\develop\denoise_preview\index.html`: four high-ISO files denoised, with 100% crops beside Canon's own.
* `outputs\detail_preview\sample_raws\index.html`: microcontrast and Recover lens sharpness on three of the sample
  raws (fit-to-screen wipe and 100% crops). Run `Detail Preview.bat` for your Voigtlander folder: it puts your DxO
  export beside each crop.

## What Bill can double-click today

* `Install.bat`: builds the private environment inside the folder. Now also adds the optional lens database
  (lensfun) once; everything works without it.
* `Run Tests.bat`: every self-test, green/red summary.
* `Start Studio.bat`: the Studio. New this session in Develop: **Looks** at the top of the panel; the full **Light**
  panel (Auto, Highlights, Shadows, Whites, Blacks, Smart lighting) and, in Full, the **Tone curve**; **Colour** (Pick
  white, Auto, Saturation, Vibrance, Black & white with its mixer, LUT) and, in Full, the **Colour wheel**, **HSL** and
  **Colour grading** (these four start folded: click a header to open it); **History** with snapshots at the bottom;
  in the top bar the **version** menu (virtual copies), **Copy settings** and **Paste**. Library: **Paste settings**
  onto every selected photo. Export: presets (Archive, Print, Web, Social, and your own). Session 6 adds Dehaze and
  clipped-highlight rebuild (Light), parametric regions, Toe and Highlight roll-off with a drawn film curve (Tone
  curve), Auto colour, B&W filters and the camera-profile picker (Colour), Texture and film grain (Detail),
  output sharpening in Export, and the **Masks** group (Brush, Gradient, Radial, Object, Colour, Brightness).
  Session 7 adds the **Retouch** group (Heal, Clone, Red eye, Find dust, Dust map; Show spots), the **Skin** mask
  with **Smooth skin**, the **100%** view (Z or the mouse wheel; drag to look around), and in the Library **Learn dust
  map** (select 5-20 photos of different scenes from one camera; sky or a plain wall at f/11 or smaller is ideal).
  Session 8 adds **Light > Clipped highlights: measure from lens glow** (reads how bright and what colour a blown
  light was from the glow the lens spread around it) and, in the Library, **Measure lens glow** (once per lens: a
  bracket of one small light in a dark room; the Help button in the profile list explains how).
  Session 9 adds the **Portrait** group (below Retouch): Find faces or Click the eyes, Remove blemishes, Younger and
  the face sliders, and Body sculpting (brushes and shape tools). The face models are in `models\faces` (no install).
* `Detail Preview.bat` (new): drag raws or a folder on it, or double-click for the Voigtlander folder.
* `Develop Raws.bat`, `Denoise Preview.bat`, `Run Benchmark.bat`, `Analyze Raws.bat`, `Calibrate Camera.bat`,
  `Blind Test vs DxO.bat`: as before. `studio develop` also takes `--microcontrast 0.6 --sharpen --fix-lens --rotate`,
  `--look "Warm film"`, `--smart-lighting 40`, `--highlights rebuild --dehaze 30 --texture 20 --grain 15
  --profile NAME.dcp --sharpen-for screen`, and now `--dust-map` (heal the camera's learnt dust in every photo).

## Decisions taken while Bill was away (all reversible)

1. Stack: Python + NumPy/SciPy + rawpy, as the plan recommends (open question 1). No Rust.
2. Default demosaic: Menon 2007 (measured best of the three implemented). RCD next.
3. Studio Standard look: contrast 1.35, mid grey 0.11, tuned against the camera JPEGs of the R5, 5D IV and 6D.
4. Gamut compression added (not in the plan) after the UV-lit violet produced colours outside the spectral locus.
5. ICC profiles are generated by our code and named "... compatible".
6. Denoise off by default; the isolated-outlier filter is off by default (it ate real texture: -2 dB).
   Default luminance strength 0.6 (1.0 wiped leaf texture on the R5 at ISO 4000 even though PSNR liked it).
7. `raw_analyzer.py` stays at the top of the folder; it imports the Studio metadata reader for the true ISO.
8. **Recover lens sharpness is off by default** (a look decision, and many of Bill's lenses are character lenses).
   Make it good turns it on at 40% unless the photo's lens is marked as a character lens.
9. **Microcontrast: 2-stop range by default.** 1 stop was too weak on real photos (+6%); see RESULTS.md.
10. **lensfun is optional**, installed once by `tools\ensure_env.bat`. Its numbers are frozen into the sidecar when
    Fix lens uses them, so an edit renders the same on a PC without lensfun or with a newer database.
11. Geometry numbers (straighten, perspective, crop) are stored as they look on screen (upright frame); the crop is
    fractions of the frame left after "constrain to image". No upsampling: straightening makes the photo smaller.
12. Order deviations from plan section 3: purple-fringe removal runs before geometry (detected better on pixels not
    yet resampled), and capture sharpening runs right after it (still before geometry and tone), so a 100% loupe
    can sharpen only the pixels it shows.
13. Lens list: a shared starter list in `profiles/lenses/manual_lenses.json` (CC BY-SA; Laack, Heliar, Apo-Lanthar,
    Zeiss 1898 Anastigmat marked as character lenses, plus common vintage glass); your own lenses go to
    `settings/my_lenses.json` (private, never committed).
14. The "before" in the before/after wipe now uses the same straighten / perspective / crop as the edit, so it lines up.
15. **Colour stage order (deviation from plan section 3):** HSL, colour wheel, saturation / vibrance, B&W, grading and
    LUTs run *after* the tone curve, on display-referred values in OKLCh. Hue and chroma changes then act on the colours
    you see and never alter lightness; doing them before the curve made the curve re-shift what the wheel had set.
16. **Highlights / shadows work in stops before the base curve** (not as local masks): global, so no halos, and
    provably monotone at +-100. The local part is the separate **Smart lighting** control (guided filter, radius 1.5%
    of the long edge), off by default.
17. **Auto (Light) never touches exposure or shadows.** It only sets highlights (pull back clipping), whites (only
    when the frame is not low-key) and blacks. Bill's rule: no automatic brightness changes unless he turns them on.
18. **Looks never carry brightness.** Exposure, mid grey and "match camera brightness" are stripped from every look,
    shared or saved, so applying a look cannot brighten or darken a photo. The lens tag is never taken from a look.
19. **Auto white balance neutralises candle warmth.** It is a grey-edge estimate of the light's colour; for candle-lit
    portraits As shot usually looks truer. Its tooltip says so. Picked / Auto white balance is stored as exact
    multipliers, shown as the nearest temperature and tint; moving either slider carries on from there as Custom.
20. **"Bill's Candlelight (draft)"** is my starting point, not Bill's look: highlights -30, blacks -10, vibrance +10,
    orange and red slightly desaturated, a faint warm tint in the highlights, microcontrast 50% (shadows-first). It
    never changes exposure. Bill tunes it and saves it under his own name (Save look).
21. History: moving the same slider again within two seconds updates one step ("Exposure +0.5") instead of adding
    fifty; the last 40 steps are kept in the sidecar per version. Snapshots and virtual copies live in the same
    sidecar, so nothing is ever written outside the photo's folder or the app folder.
22. The Full-only groups (Tone curve, Colour wheel, HSL, Colour grading) start folded; each group remembers whether
    you left it open (settings\app.json).
23. **Clipped highlights stay "plain clip" by default**; "rebuild colour" is one choice away in the Light group (Full).
    Rebuild only shows when highlights or exposure are pulled down (the default white point is below sensor clip).
24. **Camera profiles: the profile's own tone curve is not used.** A .dcp sets colour only; the Studio's tone curve
    and Light panel keep control of brightness. Profiles you add are copied into settings\camera_profiles (private:
    commercial profiles must never be shared) and offered only for photos from the camera they were made for.
25. Grain is generated from the pixel's position, so a 100% view matches the export exactly; fit-to-screen previews
    show the grain a downsized export would have. Judge grain and Texture at 100%, like sharpening.
26. Output sharpening is off unless chosen in Export (the Web, Social and Print presets turn it on for their medium).
27. **Masks are attached to the photo, not the screen**: shapes are stored in the camera's own frame (before rotation
    and crop), so straightening or re-cropping later leaves them on their subject. They are recomputed at render time,
    so they are exact at any size and a 100% view matches the export.
28. **Click an object is not AI**: it learns the colours around your click and of everything else (GrabCut-style
    colour models), scores every pixel, smooths the score along the photo's edges, and keeps the connected part you
    clicked. Brightness counts less than colour, so shading across a leaf does not split it. It works best on the
    object's typical colour (clicking a blown highlight finds other blown areas). Alt-click marks what is not part
    of it. Its result is stored as a small bitmap in the sidecar and snapped to the full-size edges when rendering.
29. Local adjustments order: exposure / white balance / texture / clarity / dehaze in scene light (before the tone
    curve), tone controls by blending the global curve with a per-mask curve, colour after the global colour stage,
    sharpness and noise last (before grain). Several masks overlapping apply in list order.
30. "Paste settings" offers Masks as a group (off by default, since masks are drawn for one photo).
31. **Retouch spots are attached to the photo** like masks (camera frame, before rotation and crop). A spot's source is
    always read from the photo *before* retouching, so the order of spots does not matter and a 100% view matches the
    export exactly. Heal works on log values (light is matched by multiplication, as in the Photoshop Healing Brush);
    the membrane is second order with a feathered edge rather than Georgiev's fourth order (simpler; the feather hides
    the seam). Clone copies exactly, for hard edges and patterns.
32. Order: retouch is applied to the scene-linear photo right after capture sharpening and before exposure (plan
    section 3 lists it at stage 18), so every later slider (exposure, curves, colour, masks) acts on the repaired photo.
33. **Find dust is conservative** (at least 5% darker, six times above the local texture, round, not in deep shadow) and
    draws what it found in amber for review: one photo cannot tell dust from a bird in the sky. The dependable route is
    the **dust map**: at least 3 different scenes; a blob is kept only if it sits in the same place on the sensor in at
    least 3 photos (and 35% of them); the same scene twice counts once. The map stays in calibration_local (never
    shared, plan rule 2) and, when used, its spots are written into the edit, so re-learning the map later never
    changes an edit you already made.
34. The dust map heals only the spots this photo shows: each is looked for near its place with a size window from the
    aperture (the shadow spreads roughly as f-number(map) / f-number(photo)); for lenses that record no aperture (most
    of Bill's vintage glass) the window is wider.
35. **Smooth skin is a mask adjustment**, not a global slider (the research: blind smoothing destroys skin texture; good
    retouching is region-aware). Pores and fine hair are never touched (only the middle layer is softened); inside any
    mask it eases off on colours far from skin (eyes, brows, lips). Off unless Bill sets an amount on a mask.
36. **100% view**: Z, the mouse wheel or the 100% button; drag to move around (right- or middle-drag while a tool is
    on). Masks, retouch spots and the eyedroppers all work in it. The before/after wipe is fit-to-screen only.
37. "Paste settings" offers **Retouch** as a group (off by default) so dust spots can be synced across a shoot; looks
    never carry spots.
38. **Measured highlights (DESAT for cameras, session 8).** A third clipped-highlight mode reads a blown light's true
    brightness and colour from the glow the lens spread around it, as NASA's DESAT does for the Sun's flares with the
    telescope's diffraction. With a measured lens glow profile (cyan) every channel is measured; without one, only a
    light blown in some channels but not all can be (amber), from its unclipped channel and the colour of its glow;
    anything else keeps the colour rebuild. Blown areas larger than 0.3% of the frame, or not standing well above
    their surroundings, are surfaces, not lights, and are left to the rebuild. The extra light is laid in with the
    shape of an unclipped channel where there is one, otherwise as a dome: the glow tells how much light and what
    colour, never the fine detail inside. Off unless Bill picks it; nothing changes at all where nothing clipped.
39. Glow profiles are per lens (and aperture), kept in settings\glow on this PC, picked automatically by the lens
    (the hand-picked lens tag first) and nearest aperture. "Copy settings > Light" now also carries the clipped-
    highlight mode and profile; looks never carry a profile.

40. **Portrait faces are found on this PC** with Google's MediaPipe models (Apache 2.0), shipped unmodified in
    models\faces and run by our own NumPy code (tinynet.py): no TensorFlow, OpenCV or MediaPipe install, nothing
    downloaded, photos never leave the PC. OpenCV would have run them too but adds ~40 MB for one feature.
41. Faces are stored in the photo's sidecar as 468 points in the camera frame, like masks and spots, so straightening
    and cropping later keep them on the face. Faces found automatically draw amber; faces from your eye clicks cyan.
42. **Every portrait slider starts at 0 and is subtle at 100** (lines filled at most fully, folds lifted only part
    way, a jaw warp of about 2% of the face). New features default off, as always.
43. **Lines fill only elongated grooves** (oriented morphology), so freckles, moles and pores stay: they are part of
    the person. Remove blemishes also skips probable moles and beauty marks.
44. Blemishes become ordinary Heal spots (amber, auto = "blemish"), so they can be checked and removed one by one in
    Retouch, and Paste settings > Retouch carries them like dust spots.
45. Face work is computed once per face and setting as a smooth gain map attached to the photo (on a copy of the
    face at most 640 px wide), then applied to every render: the 100% view and the export match exactly, and the
    full-resolution pores are untouched.
46. **Body sculpting is a warp applied after retouching, in stroke order**, each step inverted exactly (no smearing);
    spots placed on a sculpted photo land on what was clicked. Caps: push 30% of the brush, lift 14% of the shape,
    size about 10%, slim 12% of the width.
47. Copy settings / looks carry the portrait *sliders* only; faces and sculpting stay with their own photo.
48. **Person mask parts are worked out, not labelled by a model**: the person mask and 33 body points come from
    MediaPipe's person segmentation and BlazePose (Apache 2.0, from the official mediapipe 0.10.9 wheel); skin,
    face, hair, clothes, arms, legs and torso come from the body points, the face map (if Find faces was used) and
    the person's own skin colour. Google's multiclass model would label hair / skin / clothes directly but its
    server is blocked here; Bill downloaded it (models\people\selfie_multiclass_256x256.tflite), and when it is
    there hair, skin, face and clothes come from it directly (the worked-out parts stay as the fallback).

## Questions for Bill (one at a time, in chat)

1. (Answered 2026-11-14: no automatic brightness changes unless turned on.)
2. Phase 0 stack: Python + PySide6 as the plan recommends - confirm.
3. Phase 2: have the calibration frames (checklist Parts A-C) been shot?
4. Phase 6: are the raws from the Heliar / Sigma 135 / Canon 180L same-leaf test still around? (To measure what
   the Heliar does to tonal gradients and make microcontrast reproduce it.)
5. Phase 6: should Recover lens sharpness be on by default for new edits, or stay a Make it good / manual choice?
6. Phase 6: "Bill's Candlelight (draft)": does the starting point look like your candle portraits? (Tune it, then
   Save look under your own name.)
7. Portrait: one or two raws of a face (ideally a candle-lit one, and one in ordinary light) to tune the face finder,
   the blemish finder and the Lines / Under eyes strengths on a real face.

## Open items for the next session

* Person mask: try on Bill's photos (candle-lit edges, long hair, hands); the 512 px hair model for finer hair; the
  MagicTouch click-to-select model for Object masks (int8: the runner needs integer layers first); snap sculpt shapes
  to the body points (one click for waist / thigh / arm ovals).
* Portrait: tune on Bill's real portraits (blemish thresholds, line and fold strengths, the candle-light face finder);
  editing a sculpt shape by handles (today: move by dragging, width and sliders; redraw to resize); the face outline
  shown after sculpting follows the unsculpted face; face tools on very small faces in group photos (the detector
  sees the whole frame at 192 px: tiling would find more); a "mirror" for paired shapes (both sides at once).

* Measured highlights next steps: Bill's first lens glow profile (the Laack, candle bracket) and a check on real candle
  portraits; then, if wanted, the star-filter route (Rouf et al. 2011: a cross-screen filter spreads the highlight into
  rays that can be read back tomographically, about 9 more stops with fine detail, at the cost of rays to remove).
* Phase 9 remainder: depth and focus range masks (with Dual Pixel), per-mask denoise and microcontrast, editing a
  drawn gradient or circle by its handles (today: delete and redraw), faster masks on 45 MP (two masks add about 16 s
  to an export on the test VM; a whole-frame Smooth skin about 40 s), content-aware fill (PatchMatch-style) for large
  removals, a noise-model-aware threshold for Find dust at extreme ISO (three false spots at ISO 32000), and an option
  to apply the dust map automatically on opening a photo.
* Phase 6 leftovers: a microcontrast blind test against DxO; the Heliar measurement once the Part E files are in;
  Bill's own Candlelight look; the Dual Pixel highlight mode (Phase 8); creative sharpening with masking (comes with
  masks in Phase 9).
* Speed: the colour stage costs about 7 s on a 45 MP export on the 2-core VM (OKLab both ways); Smart lighting 5.7 s.
  Both are strip-parallel, so Bill's 14 cores should be several times faster; port to compiled code if not.
* Capture sharpening: field-varying PSF (corners of vintage lenses are softer), the edge-preserving progressive
  RL candidate (section 22 change 8), and a deconvolution comparison beyond RL (section 21 change 6).
* Phase 5: auto vertical perspective ("upright") from converging verticals; use lensfun TCA as a cross-check for
  the measured lateral CA; a vintage-lens fringing test frame (branches against sky, checklist Part D).
* Phase 1: per-body baseline from calibration; RCD demosaic.
* Phase 3: BM3D-class baseline (our own implementation), auto strength from ISO, hot-pixel map.
* Speed: tone + output stages remain the slow part on a 45 MP frame; port hot kernels if Bill's PC is still slow.
* Nothing has been committed to git: the device shell cannot delete git's lock files, so `push.bat` should do the
  commit (it shows every file first).
