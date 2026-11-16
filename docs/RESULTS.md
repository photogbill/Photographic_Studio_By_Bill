# Results: what was tried and what the measurements were

Rule 0.3.10: honesty over polish. Negative results stay here.

## 2026-11-14: Phase 1 develop engine

### Demosaic quality (natural image)
Ground truth: the R5's own full-size embedded JPEG (3C9A6311), reduced 2x to suppress JPEG artefacts, cropped
1200 x 1800, re-mosaicked (RGGB) and demosaicked.

| Algorithm | PSNR all | R | G | B | Time (crop) |
|---|---|---|---|---|---|
| bilinear | 39.4 dB | 38.0 | 43.0 | 38.7 | 0.06 s |
| MHC 2004 | 44.2 dB | 42.8 | 47.1 | 43.7 | 0.08 s |
| **Menon 2007** (default) | **48.9 dB** | 48.4 | 50.9 | 47.9 | 0.22 s |

Synthetic natural-statistics scene (unit test): bilinear 36.7, MHC 40.4, Menon 51.6 dB.
Lesson: an early synthetic test with *random, uncorrelated* colour blocks ranked bilinear first. Colour-difference
demosaics rely on detail living mostly in luminance, as it does in photographs; the test scene now reflects that.
Next: add RCD and compare on the same harness.

### Studio Standard look vs the camera JPEG
Measured with `Develop Raws.bat` (comparison page). Brightness = how far our mid-grey sits from the camera's
(negative = Studio darker). Colour difference = median CIE76 delta E on ~400 px versions (so noise and sharpening
do not count). Saturation = median chroma ratio. Canon's picture style adds its own saturation and sharpening.

| File | Body | ISO | Studio vs camera | delta E | Saturation |
|---|---|---|---|---|---|
| 3C9A6198.CR3 | R5 | 2000 | -0.34 EV | 2.0 | 96% |
| 3C9A6311.CR3 | R5 | 125 | -0.42 EV | 4.9 | 92% |
| IMG_3001.CR2 | 6D | 5000 | +0.14 EV | 4.3 | 101% |
| 2I2A2584.CR2 | 5D Mark IV | 32000 | -0.22 EV | 6.6 | 81% |
| 2I2A3558.CR2 | 5D Mark IV | 100 | not comparable | 28 | - |

Tuning that got here: first pass (contrast 1.18, mid grey 0.125) was 0.4-0.6 EV darker than the camera and
10-17% less saturated. The camera's tone curve has a log-log slope of 1.30-1.46 at mid grey; contrast 1.35 and
mid grey 0.11 matched it best across the three bodies. Remaining differences vary by body (the 6D renders darker
than the R5), so a per-body baseline from calibration is the proper fix.

**2I2A3558 (UV-lit African violet): known difference, not a bug.** The light is narrow-band violet; the camera
matrix places the petals at xy (0.20, -0.05), *below the spectral locus* (imaginary colours, negative green even
in ProPhoto). Canon's JPEG renders them magenta-pink; LibRaw's own renderer gives the same blue-violet as ours
(sRGB 132, 0, 255). Hard clipping made flat, harsh patches; gamut compression (ACES RGC method) now handles them
smoothly. Matching Canon's look here needs a camera-specific look table from calibration (Phase 7).

### Black level
Optical-black measurement (masked left columns, sigma-clipped, per CFA position) vs LibRaw's value:
R5 ISO 125: 510.1-511.0 DN vs LibRaw 512 (1-2 DN lower). At ISO 125 that is a visible shadow colour cast in a
heavily pushed frame (section 21 change 1). Read noise from the same pixels: 2.9-3.1 DN.

### True ISO
Canon maker-note ISO (ShotInfo BaseISO x AutoISO) decoded independently and checked against EXIF on all five test
files: 100, 125, 2000, 5000, 32000 - all agree. The 65535 path is covered by synthetic CR2 and CR3 tests (no real
ISO 102400 file in the folder yet: please drop one in to confirm on real data).

### Speed (2-core test VM, not Bill's PC)
45 MP R5 frame, full develop + 16-bit TIFF: about 30 s (decode 0.4 s, demosaic about 8 s, tone and output the rest).
Half-size preview path: under 1 s for decode+demosaic. To be re-measured on the 14-core PC; Phase 4 has the
interactive budget.

## 2026-11-14: Phase 3 groundwork (denoiser, experimental)

### Noise model estimated from single images (amber tier)
Read noise pinned to the masked border; gain fitted from 2x2 Haar details in smooth blocks of all four planes.

| File | Body | ISO | Gain (e-/DN) | Read noise (DN) | Plausibility |
|---|---|---|---|---|---|
| 2I2A2584.CR2 | 5D Mark IV | 32000 | 0.0167 | 94-109 | ~5 e-/DN at ISO 100 / 320 = 0.016 |
| IMG_3001.CR2 | 6D | 5000 | 0.117 | 14.6-14.9 | ~5.7 e-/DN at ISO 100 / 50 = 0.11 |
| 3C9A6198.CR3 | R5 | 2000 | 0.177 | 11.2-11.8 | consistent with published R5 figures |

### VST inverse
At mean 0.0005 (deep shadow) the exact unbiased inverse returns 0.000502; the algebraic inverse returns
0.000232 (54% low). That bias is the lifted, tinted shadow problem; the exact inverse is used.

### Starlet candidate (c) on the harness
Clean crops of 3C9A6311 (R5 ISO 125) and 2I2A3558 (5D IV ISO 100), noise synthesised from the measured models above.
PSNR on sqrt-encoded mosaics (sanity check only).

| Clean | Noise model | Noisy | Starlet default (1 / 1.5) |
|---|---|---|---|
| 3C9A6311 | 5D IV ISO 32000 | 28.1 | 36.0 |
| 3C9A6311 | 6D ISO 5000 | 36.4 | 40.6 |
| 3C9A6311 | R5 ISO 2000 | 39.5 | 42.5 |
| 2I2A3558 | 5D IV ISO 32000 | 28.3 | 39.1 |
| 2I2A3558 | 6D ISO 5000 | 36.4 | 43.2 |
| 2I2A3558 | R5 ISO 2000 | 39.5 | 44.7 |

What was tried and dropped:
* Single-pass local-Wiener shrinkage: good PSNR but left isolated dark speckles at ISO 32000 (visible at 100%).
  Fixed with a two-pass empirical Wiener on the finest band only (+0.2 to +0.4 dB and the speckles are gone).
  Two-pass on all bands cost 1.6 dB at moderate noise (over-smoothed texture).
* Isolated-outlier (impulse) filter before the wavelet step: removed real fine texture, -2 dB on the R5 grasshopper.
  Off by default; hot pixels will come from the calibration map instead.

Visual (100% crops, Denoise Preview): the 6D ISO 5000 sky is cleaner than Canon's in-camera JPEG; fine leaf texture
is softer than Canon's (Canon also sharpens; the Studio has no sharpening stage yet). At ISO 32000 some colour
mottling remains where Canon is smoother. Not yet a DxO contender: that needs candidates a and b and Bill's blind test.

### Visual check that overruled PSNR (R5 ISO 4000, 3C9A6467)
At luma 1.0 the leaf-vein and cell texture that Canon's JPEG keeps was smoothed away, although PSNR rated 1.0 as
good as anything (section 22 change 2 in action). The harness had luma 0.6 slightly ahead at ISO 2000-5000 anyway
(42.64 vs 42.48 dB, 40.60 vs 40.56 dB) and behind only at ISO 32000 (35.5 vs 36.0). **Default changed to luma 0.6.**
A texture score (dead-leaves chart, section 22 change 1) is needed so the harness catches this by itself.

### All 17 sample raws vs the camera JPEG (comparison page, Studio Standard)
* R5 (13 files, ISO 125-4000): Studio is consistently 0.30-0.42 EV darker than Canon's JPEG; delta E 1.7-7.5.
  This is a systematic per-body offset, not scene-dependent: a per-body baseline exposure (from calibration, or a
  built-in +0.35 EV for the R5) would remove it. Left for Bill to decide (it is a look decision).
* 6D: +0.14 EV (IMG_3001); _MG_1544 (30 s night exposure) +1.6 EV: Canon renders that night frame much darker.
* 5D Mark IV: -0.22 EV (ISO 32000); the UV-lit violet is not comparable (see above).

## 2026-11-14: Phase 2 calibration engine (synthetic verification)
Synthetic frames with known answers (2 darks with 1.5 DN row banding and 3 hot pixels, 2 vignetted flats, 1 long
dark): read noise 3.0 DN recovered within 0.25 DN (after taking row banding out of the frame difference: with it
left in, read noise read 3.4 DN), gain 2.0 e-/DN within 8%, row banding 1.5 within 0.4 DN, all hot pixels found,
black within 0.3 DN. The calibrated profile replaces the image estimate in the pipeline automatically.
Real-frame acceptance is waiting for Bill's checklist frames.

## 2026-11-14: Optional "Match camera brightness"
Per-body offsets (R5 +0.35 EV, 5D Mark IV +0.22, 6D -0.14) applied only when the user turns the option on
(Bill: no automatic brightness changes). Default rendering is unchanged.

## 2026-11-14 (session 3): denoiser candidates, banding, lens

### Candidates on the harness (512 px crops of 3C9A6311 and 2I2A3558, noise from three measured models)
All within about 1 dB of each other by PSNR. Residual structure (how much real detail a method removes): NLM
lowest on the textured grasshopper; MP-PCA highest. MP-PCA best on the smooth flower at ISO 32000 (40.6 vs 39.0 dB
for starlet 0.6) but costs about 11 s/MP in NumPy (Lab only for now). Hybrid (NLM luminance + starlet colour)
vs starlet default, PSNR / residual structure:

| Clean | Noise | Starlet 0.6/1.5 | Hybrid 1/1.5 |
|---|---|---|---|
| 3C9A6311 | 5D IV ISO 32000 | 33.84 / 0.575 | 33.97 / 0.500 |
| 3C9A6311 | 6D ISO 5000 | 39.08 / 0.138 | 39.02 / -0.013 |
| 2I2A3558 | 5D IV ISO 32000 | 39.08 / 0.136 | 39.81 / 0.058 |
| 2I2A3558 | 6D ISO 5000 | 43.87 / 0.103 | 43.72 / -0.053 |

Hybrid removes less detail in every case at equal or better PSNR: **made the default**.

### Row banding (masked border, Wiener weight)
5D IV ISO 32000: true 9.7 DN, estimation noise 8.8 DN per row, weight 0.55 (raw subtraction would have swapped one
banding for another). 6D ISO 5000: 1.1 vs 1.8, weight 0.28. R5 ISO 2000: 0.9 vs 1.1, weight 0.43. R5 ISO 125:
0.8 vs 0.3, weight 0.89. Synthetic: 3 DN banding cut by more than half; a clean frame is left alone.

### Lateral CA (profile-free)
Synthetic (red +0.3%, blue -0.2% magnification): measured 91% and 106% of the truth; after correction no significant
residual. A cubic term was tried and dropped: it fitted noise on clean frames (false 0.8 px "CA"). On Bill's five
sample files no significant lateral CA was found (mostly shallow depth of field, little sharp detail off-centre);
the estimator is conservative by design. Needs a test frame with known fringing (branches against sky, Part D).

## 2026-11-14 (session 4): microcontrast, capture sharpening, Phase 5 geometry

### Microcontrast (stage 16): fast local Laplacian filter on log luminance
Synthetic, amount 100%: small medium-scale tonal variations (0.15 stop texture at ~8 px) +52%; pixel-level noise
+0.5% (left alone); deep-shadow noise in the boosted band +32% without the noise model, +7% with it.
Edges larger than the 2-stop range are left alone (named test "no halos at maximum"): a 3-stop edge changes by at
most 0.014 stop, a 6-stop edge by 0.004. A 2-stop step is a "shade" by definition and is boosted (0.06 stop).

What was tried and changed:
* First version used a 1-stop range. On a real R5 frame (3C9A6311 crop, band 8-40 px) it added only +6% (shadows)
  / +8% (midtones): real photos vary by more than a stop within 30 px, so most variation counted as "edges".
  Range 1.5: +16%; range 2.0: +27% / +22% but a 3-stop edge moved 0.039 stop. **Chosen: range 2.0 with a
  stronger edge guard** (band coefficients above a quarter of the range are protected): +20% / +17% on the real
  frame, 3-stop edge 0.012-0.014 stop.
* Without the edge guard the plain local Laplacian filter brightened the bright side of a 6-stop edge by 0.06
  stop (it is halo-free by construction only for the edge itself). The guard took that to 0.003.
* Whole-photo figure on the detail-preview page (band 6-30 px, amount 80%): +7% (R5 ISO 125), +8% (5D IV ISO 32000),
  +11% (R5 ISO 4000) "shades of shadow". Visible as more tonal depth in leaves and backgrounds, no halos around
  the backlit leaf or the antennae against black.
Speed (2-core test VM): 45 MP full frame 2.5-4.3 s; fit-to-screen preview well under 1 s (only levels 2-7 at full
resolution are processed, on a 1/16-size image).
Still to do (plan 9.3): measure what Bill's Heliar does to local tonal gradients against the Sigma and Canon on the
same leaf (checklist Part E) and make the control reproduce it; Bill's blind test is the acceptance.

### Capture sharpening (stage 12): measured blur, deconvolution to a 0.5 px target
Blur estimator on synthetic blurred steps (with pixel integration): within 0.05 px for 0.58-1.82 px when clean;
heavy noise reads up to 10% low at 1.8 px. Real files: R5 ISO 125 macro 0.98 px (621 clean edges, good);
5D Mark IV ISO 32000 0.78 px (249 edges); R5 ISO 4000 macro: no clean step edges, typical 0.9 px used.
Deconvolution (synthetic 0.95 px blur, amount 100%): error against the sharp truth -45%, measured blur 0.95 -> 0.65 px;
no pixel brighter than 1.02x its 5x5 neighbourhood maximum. Candle flame test: the darkest pixel next to an
8x-over-clip flame stays at >= 0.94x the true background (no dark ring).

What was tried and changed:
* Gradient-ratio estimate alone read 0.36 px on the R5 (texture and thin lines look "sharper than possible").
  Now a second, independent estimate (step contrast / peak gradient) must agree, with flat plateaus on both
  sides; estimates below 0.3 px are rejected. R5 then read 0.98 px.
* scipy's sampled derivative-of-Gaussian at sigma 0.5 reads a unit ramp as 0.86: gradients are now normalised.
* Dark-side clamp at 0.88x left a -0.22 EV rim around the candle flame: tightened to 0.94x.
* RL iterations scale with the kernel (6 + 8 sigma_k); 12 vs 25 iterations differed by under 0.03 px.
Speed: 45 MP deconvolution 17 s -> 8.5 s by running strips on all cores (2-core VM; Bill's 14 cores will be faster).
Visual (R5 ISO 125, 100%): crisper wing veins and speculars, no visible halos; out-of-focus areas barely change
(right: this is not a defocus fix).

### Phase 5 geometry, lensfun, falloff, straight lines
* lensfun numbers re-applied by our own code match lensfunpy's Modifier to 0.16 px (distortion, 6720 x 4480) and
  1e-4 (vignetting). 5D Mark IV + EF 100-300 f/5.6L: lensfun profile found (poly3 k = +0.0097 at 300 mm, slight
  pincushion). The 180L Macro is not in lensfun.
* Plumb-line distortion (synthetic grids): k -0.05 -> -0.043, +0.04 -> +0.040, -0.03 on a rotated grid -> -0.025;
  -0.01 rejected (too small to trust); random organic texture and circles -> no distortion (named test).
  Bill's three sample raws (macro, organic): left alone, as they should be.
* Auto level on synthetic grids: within 0.05 degrees; returns nothing when there are no trustworthy lines.
* Falloff estimate (synthetic, -0.7 and -1.5 EV corners): within 0.02 EV; none invented on an evenly lit scene.
  On Bill's macro frames it reads -0.3 to -1.0 EV with low confidence: the scenes are lit from the middle, which
  no single-image method can separate from lens falloff. Correction stays off by default.
* Full 45 MP export with sharpening + microcontrast + a 1.5 degree straighten: 57 s -> 39 s after threading the
  sharpening, the resampler and the tone stage (2-core VM). Peak memory 3.1 GB.
* Export metadata: lens (the user's tag for contactless glass), exposure, aperture, focal length and the true ISO
  (102400 stays 102400 via EXIF 2.3 fields) now written: EXIF in JPEG, XMP in TIFF and PNG.

### Phase 6 Light panel (stage 15) and Smart lighting
* Highlights / shadows curve (20,001 samples from -16 to +6 EV, all four corners of +-100): strictly increasing, so
  tones never swap order. Point curves (monotone cubic, Fritsch-Carlson / SciPy PCHIP) likewise.
* Grey ramp from -9 to +2.5 EV: Shadows +80 lifts the -6..-4 EV tones by more than 1.3x on screen while mid grey moves
  under 5% and the top end under 2%. Highlights -80 dims the top end by more than 5% and leaves deep shadow within 2%.
  Blacks -60 and Whites +60 move their ends; every control keeps the ramp in order.
* Smart lighting (synthetic: textured dark room beside a lit area, 3+ stops apart), amount 25 / 50 / 100: shadows lifted
  0.38 / 0.75 / 1.50 EV; texture in the shadows kept exactly (standard deviation ratio 1.000); no rim at the edge
  (0.000 EV); the lit side eased down 0.14 EV at amount 100 (deliberate highlight ease). A region render matches the
  full frame within 0.01.
* **Named test "low-key stays low-key":** synthetic candle-lit face (deep shadow, a 3x-over-clip flame): Auto sets whites
  0 and the median brightness moves under 0.1 EV. Bill's raws: 5D IV ISO 32000 -> highlights -28, median 0.00 EV;
  R5 ISO 125 macro -> nothing to change; R5 ISO 4000 leaf -> whites +50, blacks -36, median -0.06 EV (a small,
  conservative change: Auto is meant to be safe, not dramatic).

What was tried and changed:
* First shadows/highlights bumps were too narrow: the curve folded back on itself at +-100. Ramps widened until the
  slope never drops below 0.25.
* Smart lighting v1 also brightened midtones and highlights (wrong fade term); v2 was too weak to see; v3 fades out by
  -3 EV and lifts at most 1.5 EV.
* Blacks -100 crushed too much (black point at 6%): limited to 2% for deepening, 6% for a matte lift.

### Phase 6 Colour stage (stage 17, OKLCh)
* OKLab implementation: white -> L 1.0000, a, b < 1e-4; sRGB red -> L 0.628, a 0.225, b 0.126 (Ottosson's published
  values, within 2e-3); round trip error under 1e-5.
* Hue bands sum to exactly 1 at every hue; each band is 1.0 at its own centre. Neutral greys are untouched (under
  2e-3) by every hue control at +-80, saturation and vibrance. Saturation +50 multiplies chroma by 1.50 (+-0.02). B&W
  keeps OKLab lightness within 1e-3. An HSL band desaturates its colour and leaves blue unchanged (under 1e-3). The
  colour wheel shifts hue by exactly shift x chroma gate (within 0.5 degrees) and keeps lightness within 1e-3. Grading
  tints shadows (chroma > 0.03) and leaves bright tones nearly untouched (< 0.01).
* .cube LUTs: an identity LUT changes nothing (under 2e-3); a red->blue swap LUT turns red blue.
* Auto white balance (grey-edge, p = 6, sigma 1 on half-size camera RGB): finds a warm candle-like light on a synthetic
  scene. Bill's raws, as shot -> auto: 5D IV ISO 32000 4304 K -> 3871 K; R5 macro 5096 K -> 5024 K; R5 leaf 4812 K
  -> 5846 K. The eyedropper on a grey patch recovers the true multipliers within 3%.
* Speed (2-core test VM, 45 MP export with the Candlelight draft + Smart lighting 40): colour stage about 7 s wall
  (strip-parallel), Smart lighting 5.7 s, whole export 47.6 s.

What was tried and changed:
* Hue bands first doubled at their centres (weights summed to 2 there): fixed with a half-open interval.
* LUTs first clipped every colour to sRGB before the look: an identity LUT then changed ProPhoto colours by up to 0.19.
  Now the LUT sees the in-gamut part and its change is added back, so out-of-gamut colour survives a look.
* A grading wheel with only a lightness change did nothing (the code required a tint). Fixed and tested.
* A look file could have carried exposure: looks now never change brightness (decision 18).

### Phase 6 workflow (looks, copy / paste, virtual copies, history)
* 8 starter looks load and render; applying a look leaves exposure, white balance, crop and the lens tag alone;
  "Natural" returns colour and light to identity. Copy / paste by group keeps the destination's lens tag and shape.
* Sidecar round trip: master, a virtual copy, snapshots, library rating and history all survive saving the master.
* History: a four-step drag of Exposure becomes one step ("Exposure +0.5"); undo, redo and jumping back work; a new
  change after jumping back drops the redo steps.

### Phase 6 remainder: highlight rebuild, Dehaze, Texture, Grain, curves, profiles
* **Highlight rebuild** (synthetic orange light whose red clips on the sensor): hue error against the unclipped truth
  77 degrees with plain clip -> 1.3 degrees with rebuild; brightness 0.66x -> 0.98x of the truth. Pixels where nothing
  clipped are unchanged (1e-5) and a region rebuilds exactly as the full frame (1e-4). Bill's 5D IV ISO 32000 frame
  (backlit leaf, exposure -2, highlights -60): the grey blown patch becomes green leaf with its veins; a few cells
  where all three channels clipped stay pale (that information is gone). Maps computed once per photo: 7.3 s on that
  frame (clipping spread over most of it), 0.8 s on the R5 macro, nothing on frames without clipping.
  Tried and changed: the first version took colour from every pixel round the clipped area, so the dark background
  beside the leaf lent its edge a pink rim; donors must now be bright (over 25% of clip) and are weighted by
  brightness.
* **Dehaze** (synthetic haze thickening across the frame): contrast in the haziest third back to more than 1.8x, median
  within 35% of the haze-free truth; negative amounts add haze; a region matches the full frame within 0.002.
* **Texture**: fine detail +30% at +80 and -40% at -80 on log luminance; no rim at a 3-stop edge (under 0.05 EV);
  region matches the full frame within 0.002.
* **Grain**: unit-variance field (std 1.00 +- 0.08); a quarter-size preview has the std a downscaled render would
  (0.60); identical in a 100% region and the full render; zero in pure black. Strength reduced after the first look
  (grain 60 was heavier than intended): 0.25 stops per unit of grain at 100.
* **Film toe** monotone at +-100 and leaves +-1.5 EV around mid grey untouched (1e-5); **parametric curve** monotone at
  every corner of +-100 with black and white fixed (slope floor 0.25 enforced).
* **Camera profiles (.dcp)**: write/read round trip exact; a profile whose ForwardMatrix matches the camera matrix
  reproduces the default render within 2e-4; two-illuminant weighting picks illuminant 2 for daylight; a +30 degree
  hue table turns hues by 30 degrees (median within 1); a missing profile falls back to the camera matrix.
* **Auto colour**: a grey image has colourfulness 0; random colour noise > 60 (no change); a nearly grey photo gets
  vibrance up to +35. **Output sharpening**: screen < matte in local contrast, differences clamped at half a stop.

### Phase 9 masks (local adjustments)
* Masks off, empty, disabled or with no adjustment change nothing (bit-identical render). A radial +1 EV mask
  brightens its centre about one stop and leaves the corners exact; a gradient fades monotonically from full to none.
* A 100% region matches the full render within 0.003 with a radial (exposure, highlights, saturation, clarity) and a
  brush mask (texture, sharpness, temperature) on.
* Masks stay on their subject: on a raw stored rotated 90 degrees and straightened by 5 degrees, the mask's centre maps
  back to where it was drawn within 1% of the frame.
* Brush: flow 60% gives 0.60 +- 0.02; erasing removes it; the auto mask keeps a stroke across a green and a red area
  on the green side (> 0.8) and off the red (< 0.1).
* Brightness range selects only the bright half; colour range desaturates only the blue half.
* Click an object (synthetic: three leaves, two touching across a shadow line): the clicked leaf > 0.95 inside, the
  others < 0.1, the ground < 0.05, from one click. Bill's raws, one click each: the 5D IV backlit leaf is selected on
  its own; the R5 fly leaf is selected with the fly (holes up to 0.1% of the frame are filled) and without part of its
  lighter rim; on the low-key katydid frame a click on the lower leaf selects the plant and the insect cleanly, while a
  click on a leaf edge next to the black background also takes some background (Alt-click fixes it). 2-3 s per click.
* Speed (2-core test VM, R5 45 MP full render): no masks 17.1 s, with an object mask and an inverted radial mask
  33.3 s, peak memory 5.8 GB.

What was tried and changed for "click an object":
* Geodesic distance from the click with a cost budget (Bai and Sapiro's idea): noise added up along paths so it
  stopped half-way across a leaf; with a noise floor it leaked through every soft edge (98% of the frame).
  Automatic background seeds ("clearly different colour") landed inside the leaf's own shadows. Dropped.
* Colour-histogram models of "object" and "everything else", re-learnt four times, smoothed with a guided filter,
  connected part only: works on all the cases above. The first version came out about two pixels short of the edge
  everywhere; it now grows back up to two pixels where the score is only mildly against.

## 2026-11-14 (session 7): retouching, dust and Smooth skin

### Heal, clone, red eye (local/retouch.py)
* Synthetic dust spot (35% darker at its centre, 4 px wide) on textured skin-like grey: centre-to-surroundings ratio
  0.80 before, 1.007 healed, 1.006 for the same scene without dust. Pixels outside the spot are unchanged (< 1e-6).
* A source twice as far as needed and 1.6x brighter: heal matches the light at the edge (centre within 5% of its
  ring); clone copies the brightness (ratio > 1.2 after the tone curve), as it should.
* A plane fed in as edge values comes back exactly inside (1e-5): the membrane is harmonic.
* A 100% region and a half-size preview match the full render (1e-5 and 0.02) with a round spot, a brushed stroke and a
  clone spot on; spots stay on their subject on a raw stored rotated 90 degrees and straightened 4 degrees
  (S -> view -> S round trip 1e-4).
* The automatic source never picks another dust spot next to it, or the spot itself.
* Red eye: red/green 3+ in the pupil -> under 1.5; the grey around it moves under 0.02.
* Real raw (R5 fly, ISO 4000, f/5.6): heal of a 22 px spot is invisible at 100% and the noise grain carries straight
  through it. Region renders with spots 0.04 s; a 100% view is identical (0.0) to a larger region cropped.

### Find dust and the dust map (local/dust.py)
* Synthetic: the one dust spot is found (within 4 px), nothing is found in the same texture without dust, and nothing
  again once the spot is healed.
* Dust map: four different scenes sharing two sensor dust spots, one with an extra dark blob of its own: exactly the
  two dust spots are learnt; both are found again in a fifth scene at the same aperture. The same scene three times
  counts once (photos are compared by their fine detail, so frames of plain sky still all count).
* Bill's raws, Find dust (single photo, nothing known about the camera):
  - R5 katydid (ISO 125, f/6.3): nothing.
  - R5 fly (ISO 4000, f/5.6): one spot, a soft dark smudge about 20 px wide on the leaf that looks like sensor dust
    (one photo cannot prove it).
  - 5D IV backlit leaf (ISO 32000): three false spots, which are clumps of noise. At extreme ISO the found spots must
    be checked (they are drawn amber for that).
  - 5.5-9.5 s per photo on the 2-core test VM.
* What was tried and changed: the first detector (log before smoothing, texture from a Gaussian ring, 4.5x above it)
  marked 60 spots on both the ISO 32000 and the ISO 4000 frames: the log of noisy dark pixels has a long dark tail that
  looks like specks, and the fly's body counted as a blob. Now: smooth slightly before the log, a brightness floor
  (dust does not show in deep shadow anyway), 6x above the texture measured in a ring, at least 5% darker and at most
  0.85 stops (darker than that is an object, not a shadow). Integral images made it about twice as fast (15-18 s ->
  5.5-9.5 s).

### Smooth skin and the Skin mask (local/masks.py)
* Synthetic skin patch (shading, 25 blotches 5-10 px wide 0.25 stops deep, pores, an eye 1.6 stops dark) at 100:
  blotches 60% gone (40% kept), the pore layer kept (slope 0.78+), the eye keeps 95% of its contrast.
* Skin weight 1.0 for light, medium, dark and very dark skin tones (sRGB samples); 0 for the white of the eye and black
  hair; under 0.05 for a blue shirt; lips 0.5.
* A Skin mask leaves a blue area untouched (< 1e-3) and a 100% region matches the full render (0.003).
* Speed (2-core VM, R5 45 MP export of the green fly macro with a Skin mask): the mask selects only 2-4% of it, but in
  scattered specks across the whole frame, so all of it is processed: 62.9 s against 23.5 s without, peak memory
  4.6 GB. A brushed face costs a fraction of that (only the mask's box is processed).
* Tried and changed: a blemish scale of 0.4% of the long edge took out only a third of the blotches (a box that size
  still contains much of a blotch) -> 0.6%. The first version worked on the whole frame at once: 100 s, and the next
  attempt ran out of memory at 7 GB; it now works in overlapping strips (exact, the overlap covers the filter reach),
  shares the guide's statistics between filters, and runs up to four strips at once.

## 2026-11-14 (session 8): measured highlights from the lens glow (raw/glow.py)

* Synthetic candle (blown in every channel, 10 to 60x over the clip; the clip kept under a third of its light) with a
  known lens glow (6% of the light, power-law fall-off) and noise: with the matching profile, all three channels
  recovered to 92-93% of the true light and its colour within 1% (green/red 0.61 vs 0.60, blue/red 0.175 vs 0.167).
  The first lower-envelope fit gave 96-97% but 28% on blue (refused for low signal: the signal test counted one pixel,
  not the 10,000 used; fixed with a matched-filter test).
* A lamp blown in red and green but not blue, no profile: 97-99% from its unclipped blue and the colour of its glow.
* A bright wall right beside the candle: the light is not inflated (< 1.15x), the scene only adds light.
* A wrong profile scales the answer: a profile with half the true glow gives 1.86x the light, twice the glow 0.46x;
  the colour stays right either way. The measurement of the lens is what makes the brightness right.
* Measuring a lens from a synthetic 4-frame bracket (1/2000 s to 2 s, 24 stops): the profile matches the true glow
  within 3-12% from 2x the light's radius out to 200 half-size pixels, also with a lit room (3e-4 uniform ambient):
  the room's own level is fitted under the glow's tail (b + A r^-a). Without that fit the far radii fell to 9% (the
  median of the far frame still holds glow). A tiny unblurred test light under-sampled blue on the Bayer pattern
  (flux 2x off); real lights are blurred by the lens, so the test light is too.
* In the renderer, five stops under (rendered values): truth (0.89, 0.93, 0.08), glow (0.79, 0.72, 0.09), rebuild
  (0.48, 0.34, 0.04). Region renders identical (0.0). With nothing clipped the result is bit-identical to plain clip.
* Bill's raws (no lens profile yet, so only the "partly blown" route can act): backlit leaf ISO 32000, 40 blown areas,
  3 measured, the big sunlit areas left to the rebuild; katydid, 10 small glints, 2 measured; fly, none. The first
  version turned a bright spot on the lit leaf green (the "glow" around it was the leaf, not lens glow); now a light
  must be compact, its surroundings must fade away from it, and it must stand 8x above them. The remaining measured
  spots differ from the rebuild by a slight shift from pinkish to greenish white; no ground truth to say which is
  closer. The real test is a candle portrait with a profile of the Laack.
* Cost: 1.1 s for the glow part on a 30 MP frame with 40 blown areas (the colour rebuild it starts from is 9.6 s
  there, already the case in rebuild mode).
* Found while testing: the test run (and the Studio on closing) could crash on exit, after all tests had passed, when
  a background job was still running (exit code 139, about one run in five). The Studio and the tests now wait for
  background work before exiting; three full runs in a row then exited cleanly.

## 2026-11-15 (session 9): Portrait section (faces, blemishes, lines, de-ageing, body sculpting)

No portrait raw from Bill yet, so everything below was measured on synthetic faces: scikit-image's public-domain NASA
portrait ("astronaut", 512 px) turned into a synthetic raw (scaled up 3x or 8x, converted to camera RGB, mosaicked),
with blemishes and ageing drawn in. **These numbers are not a substitute for a real candle-lit portrait.**

**Face models on this PC, no new dependency.** HuggingFace and Google's model storage are blocked from the build
machine, MediaPipe's GitHub no longer hosts the files, and the mediapipe wheel ships without models; the unmodified
Google models were found inside the PyPI package face-detection-tflite 0.6.0 (Apache 2.0 models). OpenCV 5's TFLite
importer runs them but would add a ~40 MB dependency, so the Studio got its own runner (`tinynet.py`, 300 lines of
NumPy that read the .tflite file directly). Its outputs match OpenCV's: max difference 6.9e-5 on the detector's box
regressors (values up to 165) and 9.2e-5 on the 468 mesh points (values up to 145); the same on the classifier heads.
Time on the 2-core test VM: short-range detector 13 ms, full-range 60 ms, face mesh 16 ms.

| Check | Result |
|---|---|
| Faces found on the portrait | 1 of 1, mesh confidence 1.00 |
| Same portrait made candle-dark and orange (x0.08, warm cast) | found (by the brightened, half-balanced copy) |
| Two eye clicks instead of the detector (clicks 3 px off) | mesh within 1.2 px of the detector's |
| Raw stored sideways (orientation 3, 5, 6) | same face, same place on screen; eye-click refit within 1.0-1.4 px |
| 100% view vs full render, face sliders on | max difference 9e-8 |
| 100% view vs full render, sculpting on (orientation 0 and 6) | max difference 4.5e-8 |

**Lines (de-ageing).** Synthetic forehead lines, crow's feet, smile-line shadows and under-eye hollows drawn on the
portrait, Younger at 100, measured in scene-linear light:

| | removed |
|---|---|
| Forehead lines (on skin) | 45% (the part crossing the brows is left: brows are protected) |
| Crow's feet | 53% |
| Smile lines (shadow band) | 19-21% (lifted part way by design; also partly outside the zone where they were drawn) |
| Under-eye hollow | 13% (drawn partly below the mesh's under-eye zone; the zone was enlarged afterwards) |
| Unit test: two forehead lines and a crow's foot, Lines 100 | > 45% removed; a mole of the same darkness kept > 80%; pore grain kept > 85% |
| Unit test: under-eye hollows, Under eyes 100 | 35-95% lifted (the face keeps its shape) |

Negative results on the way (kept for the record):
* **Guided-filter layers removed only 5-14% of thin lines.** An edge-aware filter keeps a groove that runs next to
  a strong edge (eyes, brows, hair), which is exactly where crow's feet and forehead lines are. Replaced by grey-level
  closing, which fills narrow dark grooves whatever is next to them.
* **A square closing filled freckles and moles too.** Replaced by closings along lines at 6 angles: (largest closing)
  - (smallest closing) is the depth of elongated grooves only; round marks fill at every angle and cancel out.
* **Measuring through the tone curve** hid most of the change (the synthetic face sat on the shoulder of the curve);
  the numbers above are measured before the tone curve.
* Face gain took 9 s for a 735-px face on the VM; big blurs on a reduced copy, 6 angles instead of 8 and a 640-px
  working copy brought it to 2.5 s (computed once per face and setting, then reused by every render).

**Blemishes.** First version: 60 "blemishes", nearly all in the hair over the forehead and along the nostrils (the
zone reached into the hair, and hair strands look like dark specks). Fixed by keeping the search inside the face
outline, away from its edge, the eyes, brows, lips and the lower nose, and by two tests from the dust finder (the skin
right around a spot must be quiet; the spot must be round). Now on the synthetic portrait: the planted spots on the
cheeks are found, the ones planted on the lips and at the jaw edge are excluded by design, and 3-4 false finds remain
near the outer eye corners and hairline. Unit test: three red bumps found, a dark well-defined mole left alone,
at most 8 spots on clean textured skin. Moles are recognised by being dark through and through (> 0.75 stops below
the skin around them) and not red.

**Body sculpting.** Every warp is inverted to within 0.05 px (8 fixed-point steps; 4 are used in renders); a drag 7
times the cap moves exactly the cap (30% of the brush radius); the context a region render reads covers the largest
move. Breast lift at 100 moves the lower part of the oval up by 14% of its size, Size 100 enlarges by about 10%, Slim
100 narrows a limb by up to 12%: visible, not cartoonish. Not yet tried on a real body; straight lines right next to a
slimmed limb bend (shown in Help).

## 2026-11-15 (session 9, part 2): Person mask (people, body points, parts)

**Models.** The multiclass person model (hair, skin, clothes labelled directly) is on Google's model server, which
the build machine cannot reach (403) and the PC's VM has no network for. The official MediaPipe 0.10.9 wheel on PyPI
still carries selfie segmentation and BlazePose (detector + full landmark model), Apache 2.0; those are used. The
runner learnt seven more layer types; outputs agree with Google's own TensorFlow Lite runtime (LiteRT, installed on
the build machine for this check only):

| Model | Largest difference | Output scale |
|---|---|---|
| Selfie segmentation (transposed convolution, hard swish, average pooling) | 3.8e-5 | 1.0 |
| BlazePose detector (sparse weights, depth-to-space) | 7.9e-4 | 738 |
| BlazePose full: 33 points / presence / mask logits | 1.5e-3 / 9.5e-7 / 0.45 | 342 / 0.9 / 1650 |

Time on the 2-core VM: selfie segmentation 0.09 s, detector 0.07 s, pose 0.16 s; Find person on a whole photo about
1.4 s (both light variants), 3 s including the small render.

**On the test portrait:** one person, confidence 0.98; shoulders, elbows, the visible wrist, eyes, ears, nose and mouth
in the right places; hips (out of frame) guessed below the frame, as BlazePose does. Same result on a candle-dark,
orange copy. Parts by eye: whole person, background, face, hair, skin (face and neck) and clothes (the suit and its
collar ring) are right; arms follow the arm on each side; no legs (none in view); torso covers the chest. Two fixes
on the way: the long-hair zone first took in the dark collar ring (fixed: below the face only what matches the hair on
top counts), and dark brown hair was first called skin (same hue as skin; fixed with a stricter brightness test for
hair). A synthetic drawn person (unit test) puts every part on its region.

Limits, stated plainly: the masks come from 256-pixel model outputs, so fine hair and fingers rely on the edge
snapping; parts are inferred (body points + face map + colour), not labelled by a model, so a skin-coloured top
counts as skin; where a candle-lit body fades into black there is no edge to find. Not yet tried on Bill's photos.

**Multiclass parts model (added the same night, file downloaded by Bill).** Seven more layer types (softmax,
transpose, sum, nearest resize, TensorFlow Lite's own transposed convolution); output agrees with LiteRT to 7.8e-5
(logits up to 4). Its metadata gives the input scaling (mean 127.5, spread 127.5, so -1..1); the -1..1 run also looks
cleaner on the test portrait (fewer "accessory" specks on the collar). 0.4-0.6 s per run on the VM; run on the whole
photo and again on each person's pose crop (closer look), Find person about 5.5 s in all. On the test portrait the
model labels hair, face skin, neck and suit correctly; it also calls the helmet she is holding an accessory, so the
whole-person mask now includes the helmet (the person mask is the union of the pose mask and the parts model's
"not background").

## 2026-11-15 (session 9, part 3): first outside test, double chin, installers, updates

**First outside test (a friend, Sony raws).** Sony raws open and develop. The report "the filters aren't working on
the right-hand side" turned out to be speed: the only sign of work was a small "Rendering..." label in the photo's
bottom-left corner, far from the Controls panel on the right. Measured on the 2-core build machine, preview of a
45 MP photo (scale 0.26): first open 3.5 s, then per change: exposure 0.7 s, shadows 0.7 s, Smart lighting 1.0 s,
saturation 1.0 s, Texture 1.8 s, Microcontrast 2.2 s, Dehaze 4.7 s (each with the ones before it still on). Dragging
a slider also started a new full render on every pause, all running at once. Fixed: an obvious indicator (moving bar,
"Working... n s" pill, amber chip beside Controls) and one render at a time with only the newest settings queued.
A sweep of 23 sliders on a left-right mirrored test photo (so both halves should change identically) found the left
and right fifths of every change equal to within about 1%: no stage leaves part of the frame unprocessed. (Capture
sharpening shows no change at preview size, by design: it is judged at 100%.)

**Double chin.** On a face mesh placed on a synthetic skin image (face 420 px wide): nothing moves at or above the jaw
line (largest movement there 0.0 px), the largest movement is 4.5% of the face width at 16% of the face below the
chin, the warp's area ratio never drops below 0.51 (no folding), and a drawn crease 12% below the chin is lifted 59%
of its depth (test bounds 30-90%). With a dark collar 22% below the chin, the skin weight is under 0.1 on the collar
and over 0.6 on the skin above it, and the collar's edge moves less than a quarter of the skin's movement. On the test
portrait (no double chin, turtleneck right under the chin) the first version pulled the collar up and shortened the
neck; weighting by skin fixed that (the collar edge now moves under a pixel). Region renders match the full render
under the chin. Not yet tried on a real double chin: that, and the right strength for "one press" (60), need real
portraits.

**Hair model.** The runner's output on a fixed 512 x 512 input agrees with Google's MediaPipe ImageSegmenter to 0.0002
on average, 99% of pixels within 0.005, largest single difference 0.064 (at steep hair edges; MediaPipe runs it with
XNNPACK's arithmetic). This replaces an earlier note in the session that called the match "exact": it is close, not
exact. MediaPipe fills the model's fourth input channel (the hair prior) with zero; with 1 instead the output is
nonsense (mean hair probability 0.96), so the Studio feeds zero.

**Installers and updates: built, not yet run on Windows.** The build machine cannot run .bat files or build an MSI,
and python.org and GitHub downloads are blocked from it, so the installer, the update staging through
`Start Studio.bat`, Release.bat and the GitHub build are checked only by reading and by unit tests of the Python side
(version maths, how a copy was installed, staging a download, refusing a zip whose paths escape the folder, the
sidebar card). Bill's first install and first release are the real test.

**Composition overlays and suggested crops.** Each overlay's geometry is checked by a test: the golden spiral converges
at (5 + sqrt 5) / 10 = 0.7236 across and down, as theory says; the golden triangles' short lines meet the diagonal at
right angles in pixels; the diagonal method's lines are 45 degrees on a 3:2 photo; rabatment's lines are the squares on
the short sides; the harmonic armature's corner-to-midpoint lines cross the diagonals at the thirds. On the test
portrait (768 px), finding the subject takes 5.8 s on the 2-core build machine (face finder plus person finder), and
the suggestions take milliseconds after that. For a 3:2 crop the three suggestions were rabatment (eyes on the left
rabatment line, the shuttle as counterweight), rule of thirds (eyes on the upper right point) and rabatment again,
tighter; every one keeps her head whole. A synthetic subject turned right gets its eyes on the left of the frame
and turned left on the right (space in front of the face). Not yet tried on Bill's photos, so the weights between
the rules are first guesses.


## 2026-11-15 (session 10): Detail by scale, denoise safeguard, Auto HDR, auto white balance

### Detail by scale (detail/scales.py, plan 23.11)

**Why the first engine was invisible.** Measured on a sine texture (0.2 stops, features 32 px across, midtones):
the first engine ("Local contrast") lifts it x1.33 at 100%. Two causes: its taper stops being monotone at an
amplitude of 1.25, and its reference tones sit one stop apart, so about a third of the nominal gain arrives.

**What was tried for the new "Tonal" method, in order:**

| Remap | Result | Kept? |
|---|---|---|
| (1 - d/3)^2 taper, amplitude 2.2, tones 1.5 stops apart | +25-30% band energy on the R5 fly frame at 70%: still subtle. On that grid the gain goes negative beyond 1 stop from a reference tone, so most of it averaged away. | no |
| Power law (exponent 0.45), tones 0.5 stops apart | Plain to see (+35-74%). But the gain is a spike at zero difference: a clean 9-stop ramp came out rippled by +-13 millistops with slope down to 3% of the original. | no |
| Smooth Gaussian detail remap (sigma 0.5 stops) with the noise toe inside the remap | Ripple of +-15 millistops per pyramid level on a clean ramp: the toe is a notch at zero difference that the tone grid cannot follow. | no |
| **Smooth Gaussian remap, noise protection applied to the coefficients per level, tones 0.25 stops apart** | Clean ramp: 0 millistops change. Small differences x2.1-2.5 at 100% (x1.8-2.0 at 70% in midtones), 0.5-stop differences x1.7-1.9, 1-stop x1.1-1.25, finest scale relaxed to x0.76-0.9. | **yes** |

Numbers for the kept version (synthetic, flat tone weights, 6000 px frame; gain of a sine of that amplitude):

| Features | 0.05 stops | 0.2 | 0.5 | 1.0 |
|---|---|---|---|---|
| 1.5 px | x0.76 | x0.80 | x0.87 | x0.95 |
| 8 px | x1.64 | x1.63 | x1.28 | x0.94 |
| 16 px | x2.08 | x2.30 | x1.68 | x1.07 |
| 32-64 px | x2.10 | x2.48 | x1.85 | x1.2 |

* **Frame edges.** With the pyramid's plain mirror at the border, a steep gradient running off the frame (9 stops
  across the frame) came out 0.44-0.57 stops off within 16 px of the edge at Tonal 100%. Continuing the picture by
  point reflection instead: 0.003 stops (Tonal), 0.013 with Tonal + Local contrast + Crisp + Broad all at 100%.
* **Halos** (3-stop edge between flat areas, each method alone at 100%): flat areas stay within 0.025 stops (tonal),
  0.014 (local), 0.009 (crisp), 0.003 (separation), 0.011 (broad). All 15 pairs pass at 100%.
* **Candle flame** (6 stops above a dark room): centre moves at most 0.08 stops; no dark ring. Crisp first failed this
  (+0.30 centre, -0.25 ring: it treated the flame's slope as texture). Fixed by also guarding on the next two coarser
  scales.
* **Tonal separation.** First version (contrast-limited histogram equalisation per tile) shifted flat areas by
  0.4-0.6 stops near a big edge, tile by tile: a halo. Replaced by a mean-shift push away from the centre of each
  crowd of tones: flat areas now move 0.001-0.003 stops, a smooth gradient is untouched.
* **ProPhoto luminance is blind to blue.** On the UV-lit violet (2I2A3558) the median "luminance" is 10 stops below
  mid grey (blue's weight is 0.0001), so the first run did nothing (+1%). The new methods are driven by a lightness
  with weights 0.25 / 0.60 / 0.15, which equals luminance on neutrals. Same file after, Tonal 70%, fit to screen:
  +10-17% at the finer preview scales (a dark, soft-lit subject: a modest gain is the honest result).
* **High ISO (the 23.11 named test, R5 ISO 4000 noise model).** Gain of a 0.15-stop texture at 70%:

  | Tone | -4 EV | -3 EV | -2 EV | 0 EV | +1.5 EV | +2.5 EV |
  |---|---|---|---|---|---|---|
  | denoiser off | x1.20 | x1.49 | x1.67 | x1.58 | x1.37 | x1.28 |
  | denoiser on (hybrid 0.6) | x1.35 | x1.66 | x1.76 | x1.65 | x1.39 | x1.28 |

  Shadows now get at least as much as highlights, and more with the denoiser on (it was the reverse).
* **What the denoisers leave** (measured, flat patches, R5 ISO 4000 model; fraction of luminance noise left, by
  pyramid level 0-4): hybrid at 0.6: 0.16 / 0.28 / 0.49 / 0.74 / 0.89; starlet at 0.6: 0.20 / 0.36 / 0.47 / 0.60 /
  0.74. The old code assumed 0.70 at every level. MP-PCA is not measured yet (the starlet table is used for it).
* **Real frames.** R5 ISO 4000 fly (3C9A6467), Tonal 70%, fit to screen: veins, drops and the fly's body plainly more
  modelled; smooth background untouched; no rims. Balance number at 100% in the middle: 1.99 -> 2.13 with Tonal 70 +
  Crisp 30 (band energy on the preview: +22-46%). R5 ISO 400 low-key leaves (3C9A6347): +7-12% over the whole frame,
  which is mostly black; the shadow foliage separates.
* **The 100% view matches the export**: region against full-size render, every method and Auto HDR on: max difference
  0.0005 (16-bit: about 30 of 65535).
* **Speed** (2-core build machine): Tonal adds 1.1-1.5 s to a 1638 px preview; full-size 45 MP with Tonal + Crisp +
  Auto HDR: 30 s, peak memory 4.1 GB (the plain export was not measured for comparison). The reference tones are filtered in parallel, so more cores help directly.
* **Not judged by me:** whether 60-80% looks right to Bill, in colour and in black and white (the acceptance test).

### Denoise safeguard (raw/denoise.py, plan 23.12)

* Pure noise (384 px flat patch, starlet 1.0): at most 30 photosites kept; mean squared error within 3% of the
  unprotected result. The safeguard does not cost noise reduction.
* A spark 40 sigma above its surroundings on one photosite (starlet 1.5): 67% of its height survives without the
  safeguard, 82% with it. The guarantee is "within 4.5 to 6.75 sigma of where it started", not "untouched".

### Auto HDR (color/hdr.py, plan 23.6)

* **Halos.** Exposure fusion and Retinex both drifted inside the flat sides of a 6-stop edge (0.49 and 0.77 stops:
  the plan's "cannot make halos by design" is not true of exposure fusion). Fixed for every engine by re-expressing
  the change as one value per level of lighting (an edge-keeping average) before it is applied: 0.002-0.004 stops
  now, every engine, Heavy.
* **Amount** was not proportional with exposure fusion (50% gave a quarter of the lift); the engines now run at the
  strength's 100% and Amount scales the result.
* Synthetic backlit room (room -6 EV, window +3.5 EV), Medium: room +0.7 (fusion) to +1.6 stops (Laplacian, Retinex),
  window -0.5 to -1.5; the texture in both keeps at least 70% of its contrast.
* Low-key named test: black room unchanged within 0.1 stops, face lifted 0.1-1.6, flame still above +2 EV.
* Real frame (R5 3C9A6347, range 7.3 stops, shadows at -6.8 EV): Medium opens the shadow foliage by up to 1.5 stops,
  holds the sunlit leaves, no rims; 10% of the lift held back by the noise limit.
* **Not built:** denoising the shadows harder in proportion to the lift (the denoiser runs before this stage; the
  lift is limited by noise instead), Dual Pixel / bracket / burst sources.

### Auto white balance (color/autowb.py, plan 23.9)

* Grayness Index on a synthetic scene with grey patches under a warm light: within 3 degrees of the true light; two
  lights (candle left, window right) both found within 5 degrees.
* Real frames: R5 foliage shots (nothing neutral in them): grey edge 4870 K / 5850 K against grayness 4560 K / 5235 K,
  11-12 degrees apart: reported as "they disagree: check by eye". UV-lit violet: 1 degree apart, both far from any
  real light and pulled back to the band. This is the honest behaviour: the agreement is the confidence.
* **Not built:** the mixed-light blend and the skin-based estimate.

## 2026-11-15 (session 11): Phase 4b, Library and Import from card (plan section 24)

### Library speed on a 2,000-raw folder (tools/bench_library.py, `Library Speed Test.bat`)
Measured on the Linux test machine (SSD), offscreen, window 1126 x 708 (the Library's share of 1366 x 768). The test
folder is 2,000 hard links of the two sample raws (CR2 6D and CR3 R5), so after the first few files the operating
system's own file cache is warm: "cold" here means no thumbnail cache and no index yet, not a first read from a disk.

| Aim (24.7 B) | Cold | Second visit |
|---|---|---|
| Window answers within 0.5 s | 0.22 s (first import of the engine) | 0.001 s |
| First screen of thumbnails: 2 s SSD / 5 s USB HDD cold, 0.5 s cached | 0.64 s | 0.19 s |
| Ratings and details for all 2,000 | 0.44 s | 0.00 s (all from the index) |
| Longest wait for the window while loading | 46 ms | 78 ms |
| Scrolling never stalls (2,000 photos, a screen at a time) | longest stall 30 ms | |

Not measured yet: Bill's USB hard drive with a real, cold folder. Drag a folder from the archive drive onto
`Library Speed Test.bat` after a restart; the numbers land in outputs\bench_library\results.txt. The number of
thumbnail readers follows the drive: 2 on a spinning drive (Windows' seek-penalty query), up to 4 on an SSD.

### Card import (engine/io/ingest.py)
* All 14 named tests of 24.9 pass, plus 12 more (cancel, a copy that does not match, videos outside DCIM, JPEG
  capture times, the CLI, the record travelling with a moved Photos folder, source inside the destination refused).
* Capture times are read from the first 256 KB of each file (CR3 and CR2 keep their EXIF in the first 64 KB on all
  sample files), so scanning a 400-file card reads about 100 MB instead of the 9.6 GB the full metadata reader
  (24 MB per file) would have read.
* Check after copying: on Windows the read-back goes past the Windows file cache (FILE_FLAG_NO_BUFFERING), but only
  after it has proved itself on a 3 MB test file in the destination at the start of each import; if that test
  fails, the read-back after fsync is used and the log says which kind of check was done. Not yet run on Windows
  by me: the Windows-only test (`test_uncached_read_back_on_windows`) is the first proof.
* Raw files compared at the same name and size are hashed before anything is written: an identical photo already
  there costs one read of each, never a write.

### Review (app/review.py)
* Camera preview: the 6D and R5 embed full-size JPEGs (5472 x 3648 and 8192 x 5464), so 100% is real camera-JPEG
  pixels. The fitted view decodes a 2560 px copy in JPEG draft mode (fast); the full size only when zoomed, at most
  two kept in memory.
* From the raw on the 6D sample: 4.3 s to the first fitted render on the test machine; 100% windows render on
  demand like Develop's zoom.

### Demosaic picker (raw/demosaic.py, plan 23.8)
Harness: the camera's own embedded JPEG, halved (Lanczos) to suppress JPEG artefacts, made roughly linear (^2.2),
re-mosaicked (RGGB) and demosaicked; PSNR against it. "Noise" adds signal-dependent Gaussian noise of 0.01 / 0.03
at mid grey (high-ISO-like). Crops: the centre 1200 x 1800 of 3C9A6311 (R5, ISO 125, the earlier harness), and the
most detailed 800 x 800 of each file. (Absolute numbers differ from the Phase 1 table above: different linearisation.)

| Crop | Noise | MHC | Menon | RCD, ratio (RCD's own) | **RCD, additive (shipped)** | LMMSE |
|---|---|---|---|---|---|---|
| 3C9A6311 centre | 0 | 43.60 | 46.83 | 46.33 | **47.30** | 46.60 |
| 3C9A6311 centre | 0.01 | 41.05 | 42.08 | 42.23 | **42.58** | 42.29 |
| 3C9A6311 detailed | 0 | 39.32 | 43.60 | 42.41 | **43.80** | 43.29 |
| 3C9A6311 detailed | 0.01 | 38.12 | 40.59 | 40.16 | **40.92** | 40.62 |
| 3C9A6311 detailed | 0.03 | 33.56 | 33.78 | 34.05 | **34.23** | 34.17 |
| 3C9A6198 detailed (soft) | 0 | **49.62** | 49.01 | 49.44 | 49.43 | 49.00 |
| 3C9A6198 detailed | 0.01 | 36.19 | 35.85 | **36.68** | 36.59 | 36.32 |
| 3C9A6198 detailed | 0.03 | 27.07 | 26.76 | **27.61** | 27.52 | 27.25 |
| IMG_3001 detailed (6D, soft) | 0 | **41.02** | 39.43 | 40.36 | 40.49 | 39.77 |
| IMG_3001 detailed | 0.01 | 36.51 | 35.67 | **36.56** | 36.54 | 36.11 |
| IMG_3001 detailed | 0.03 | 28.62 | 28.21 | **29.06** | 28.99 | 28.71 |
| Synthetic scene (unit test) | 0 | 40.45 | 51.62 | 53.71 | **55.15** | 51.14 |

Decisions from this table:
* **RCD (additive) is the new default** for new edits: ahead of Menon on every crop, clean and noisy, and twice as
  fast (full 45 MP frame on the 2-core test machine: Menon 8.4 s, RCD 4.2 s, LMMSE 5.0 s, Dual 11.7 s). Sidecars store
  their algorithm, so every edit made before keeps Menon and renders exactly as before.
* **Negative result:** RCD's own low-pass *ratio* correction lost 1.2-1.4 dB on the detailed R5 crop (it amplifies
  errors at colour edges); it won by a hair only on the soft, noisy crops. Replaced by Hamilton-Adams' additive
  correction; the rest of RCD's method (refined directional discrimination, inverse-gradient weights, colour
  differences on the diagonals) is kept.
* **LMMSE is not chosen automatically at high ISO** (plan 23.8 suggested it): RCD scored higher than LMMSE with noise
  on all three files. LMMSE stays in the picker for its look (smoother, fewer maze patterns), as does Dual.
* MHC "wins" the two soft crops because the truth there is a smooth, camera-processed JPEG; it loses badly on detail.
* PSNR does not see maze patterns or zipper edges; the picker's texts say what each is for, and Bill's eye decides.
* Not done: the **PDAF line filter** for Sony and Nikon Z striping needs a raw that shows the striping (the first
  outside tester's Sony files) to be built and tested against; green equilibration is in (off by default).

### Eyes (local/eyes.py)
Iris finding. Drawn eyes (scene-linear, a real face mesh's eye outline, iris radius 22-30 px):

| Iris (true centre, radius) | Model alone | Classical fit alone | Shipped (model + edge search) |
|---|---|---|---|
| (214, 152), 28 | (216.1, 152.3), 26.5 | (215.0, 155.0), 29.4 | (215.0, 155.0), 29.1 |
| (195, 148), 26 | (196.2, 149.7), 23.3 | (194.9, 148.3), 26.6 | (194.9, 148.3), 26.5 |
| (228, 154), 30 | (227.1, 152.9), 27.5 | (228.4, 155.0), 29.4 | (228.4, 155.0), 30.2 |
| (210, 150), 22 | | | (211.7, 151.7), 22.1 |

Real eyes: NASA's public-domain astronaut portrait (scikit-image), irises about 4 px in radius at 512 px: model and
fit agree to 0.4 px in the centre and 0.1 px in the radius, both on the irises (picture checked). 0.02-0.05 s per
face with the model, 0.17 s per eye for the fit alone.

Eye colour on a drawn brown iris (radius 26 px), all at amount 100, OKLCh measured on the iris's middle band:
blue 245.0 deg / 0.20 (target 245 / 0.20), green 147.1 / 0.16 (150 / 0.16), hazel 84.6 / 0.15, grey 235.0 / 0.04,
violet 303 / 0.17. Skin and eyelids unchanged (largest change 3e-7), pupil within 0.004, catchlight within 0.03.
Two faults found by looking, both fixed: the colour first leaked onto the eyelids (the soft eye zone; now clipped to
the mesh's eyelid outline) and left a thin brown rim (the edge feather; now centred on the edge). On the astronaut's
4 px irises the colours come out weaker (the eyelid outline is coarse at that size); portraits at camera resolution
have irises of 30-80 px.

Red eye: a brown iris has red 3.4x the mean of green and blue in linear light (drawn) and about 3.8x in a typical
brown (sRGB 100, 60, 40); a red pupil 7-45x. Detection threshold 5x and brighter than the iris; Red eye spot
version 2 acts from 4x.

### Face mesh v1 against Face Landmarker v2 (local/faces.py)
The runner gives the same numbers as Google's own LiteRT interpreter on v2 (largest difference 2.4e-4 on outputs in
the hundreds, 3e-5 on the face score). Test: the astronaut portrait (eye spacing 43 px) and seven damaged copies;
each mesh starts from a deliberately off-centre, rotated, 10% too large crop, as a detector would give. No hand-made
ground truth exists for this face, so two measures:

Distance from the average of the two meshes' clean answers (mean over the 468 points, / eye spacing):

| Copy | v1 all | v2 all | v1 eyes+lips | v2 eyes+lips |
|---|---|---|---|---|
| clean | 0.027 | 0.021 | 0.027 | 0.014 |
| candle-dark (8% light, orange) | 0.027 | 0.020 | 0.025 | 0.013 |
| half shadow (3 stops) | 0.028 | 0.019 | 0.026 | 0.012 |
| noise (sigma 0.06) | 0.028 | 0.022 | 0.026 | 0.017 |
| blur (sigma 1.6 px) | 0.027 | 0.027 | 0.024 | 0.022 |
| rotated 20 | 0.022 | 0.021 | 0.022 | 0.017 |
| rotated -35 | 0.027 | 0.023 | 0.024 | 0.019 |
| small (0.4x) | 0.028 | 0.027 | 0.025 | 0.023 |

Steadiness: how far each mesh moves from its own clean answer (/ eye spacing; lower is steadier), and the iris centre
from v2's iris points against the shipped iris finder:

| Copy | v1 moves | v2 moves | v2 iris points | iris finder |
|---|---|---|---|---|
| clean (off crop only) | 0.011 | 0.006 | 0.007 | 0.008 |
| candle-dark | 0.014 | 0.008 | 0.003 | 0.007 |
| half shadow | 0.014 | 0.009 | 0.003 | 0.006 |
| noise | 0.014 | 0.010 | 0.008 | 0.008 |
| blur | 0.013 | **0.021** | 0.011 | 0.016 |
| rotated 20 | 0.012 | 0.005 | 0.007 | 0.007 |
| rotated -35 | 0.011 | 0.006 | 0.005 | 0.007 |
| small (0.4x) | 0.012 | **0.016** | 0.006 | 0.018 |

v1 and v2 agree to 1.6 px on average (3.5% of the eye spacing) on the clean photo; v2's irises and the iris model's
agree to 0.3 px in the centre and 0.1 px in the radius. Speed: v1 66 ms, v2 166 ms per face (two passes, each on the
crop as given and an evened-out copy). Limits of this test: one real face; the damage is simulated. Shipped: v2 when
present (decision 83).

### Zoom and pan (app/zoom.py, engine render_window)
Canon R5 raw (5463 x 8191), Clean on, Develop view 1240 x 795, after the whole-photo preview was made.

Before: the wheel went from fit straight to a rendered 100% window; each drag ended with the photo jumping back
until the next window was rendered, 2.2 s (1800 x 1100 window).

After: the photo is redrawn from the preview in memory on every wheel notch or drag step, 5.5-7.7 ms a frame
(5.1 ms at 400%); a drag shows no jump on release (screenshots mid-drag and after release identical). The sharp
render of the part on screen (plus 8%) follows:

| Zoom | Tile rendered | Time |
|---|---|---|
| 26% | 5151 x 3548 output px at 0.26 | 0.33 s |
| 37% | 3888 x 2493 at 0.37 | 0.23 s |
| 52% | 2767 x 1775 at 0.50 | 0.12 s (5.2 s before tiles up to 60% were cut from the preview's half-size frame) |
| 74% | 1945 x 1247 at 0.74 | 2.8 s |
| 100% | 1440 x 923 at 1.00 | 1.6-1.8 s |
| 200% | 720 x 462 at 1.00 | 0.7 s |

A 37% tile against the same part of the whole photo rendered at 37%: mean difference 0.0009 (0-1 scale). A tone
slider while zoomed: tile 0.14 s, then the preview 0.6 s (both from their own cache; with the old single cache
slot each render evicted the other's denoised data). Develop's minimum window width: 1,521 px before, 1,353 after
(a long file name now shortens instead of widening the window).

### Old photos: scans, negatives, dust and scratches (session 2026-11-16)
All in the container (2 CPU cores; Bill's PC is several times faster). Scans made from the R5 test raw rendered at
quarter size (1366 x 2048).

Scans render as the file (default edit, then encoded back to the file's space):

| File | Mean difference | Largest |
|---|---|---|
| 16-bit TIFF, Adobe RGB profile | 0.000% | 0.39% (one code value at 8 bits) |
| 8-bit JPEG / PNG, no profile (sRGB) | 0.096% (the export dither) | 0.39-0.78% |
| 8-bit grey JPEG | 0.098% | 0.39% |

Negatives (synthetic: orange base 0.72/0.42/0.24, layer contrast 0.55/0.60/0.65, then the turn back): log-brightness
correlation with the original 0.9995 (colour) and 0.9974 (B&W). The colour comes back neutral-balanced (a warm
candle scene comes back cooler, as from any auto-balanced scan); white balance takes it from there.

Dust and scratches (300 specks of radius 0.8-3.5 px and 15 scratches of 1-3 px, half bright and half dark, on the
2.8 MP candle scan):

| | Amount 30 | 50 | 70 |
|---|---|---|---|
| Damage found (pixels) | 100% | 100% | 100% |
| Damage left after the fill | 2.96 -> 0.06 stops | 0.07 | 0.07 |
| Pixels marked on the same photo with no damage (filter alone) | 0.12% | 0.32% | 0.77% |

Through the pipeline with the learned detector: damage 3.37 -> 0.09 stops, the rest changed by 0.004 stops on
average; a 300 x 240 region against the same part of the whole photo: largest difference 0.00002. Time: whole photo
5.7 s the first time (2.7 s of it the detector's map, made once per photo), half-size preview 0.7 s.

Half-size preview (the fit view): scratches 2.96 -> 0.06 stops, specks 2.41 -> 0.08, after two fixes: the detector's
verdict is taken a few pixels either side of a mark (its map is coarse), and a long thin mark standing at least
twice the threshold above its texture counts as a scratch even where the detector cannot see one that thin (before:
0.19 stops left on scratches at half size). The detector runs with the short side at 384 px from 1000 px up (one
thin scratch's median probability 0.10 at 256, 0.49 at 384; false marks on undamaged areas 0.98% -> 0.12%).

Scratch detector (NumPy): short side 256 px 2.7 s, 384 px 5.4 s on 2 cores. On the clean photo no pixel above 0.4 (its own
threshold); on the damaged one 70% of the damage pixels above it (the specks are below its resolution). 8-bit
weights against the float ones: the damage mask agrees on 99.92% of pixels (largest probability difference 0.14).

Colourize (DDColor paper tiny, 256 x 256): the Studio's NumPy runner against ONNX Runtime on three photos: largest
difference 0.0004 (a, b units), 1.1-1.5 s each on 2 cores (ONNX Runtime 0.6-1.6 s). 8-bit weights against float:
mean difference 0.15 / 0.30 / 0.65 Lab units (cat, coffee, astronaut), invisible side by side; one outlier pixel 24
units. Through the pipeline (1536 px and 1800 px grey JPEGs): 2.5-4.0 s, L* changed by 0.24-0.26 on average (99th
percentile under 1.1, from the taper and gamut limits), a zoomed region identical to the whole. The predicted
colour depends on how the black and white was made: the astronaut's orange suit comes back orange from true CIE
lightness and blue from a gamma-luma grey copy, which is the model's guess, not a bug.

Colour a part (astronaut, 1536 px grey JPEG): finding a part 0.4-0.55 s (geodesic distance on a 640 px copy).
Two clicks (one Shift) took the whole suit including its folds and left the badge out; one click on the wall took
the wall and leaked into the pale left cheek and the top of the hair; one Alt-click on the cheek fixed the cheek.
On the synthetic test (a dark-to-light ramp beside a sharp edge) the part fills the whole ramp and stops exactly at
the edge; brightness changed by under 0.8 L* anywhere (the hint test's limit; typical 0.0001).
