# Build plan: Clean, rebuilt on the camera's own measurements (noise reduction without smoothing)

Written 2026-11-18 with Bill, the day his R5 calibration pile went in (360 dark frames, 89 flat frames, every ISO
from 100 to 102400, camera 24 to 40 °C). For the builder (Claude Opus). Read section 0 before writing code.

**Bill's goal, in his words:** "the best noise reduction system possible without smoothing the image or losing
details." And: "I know that DxO uses AI to do that, and I'm sure Adobe does. How can we do the same, how can we
take this next level."

**The idea in one paragraph.** DxO and Adobe ship one network for every copy of every camera, trained on their own
lab pictures. They have never seen this sensor. We have: its read noise at every ISO, its gain, its fixed
pattern, its banding, its hot and warm pixels, how all of that moves with heat, and hundreds of real dark frames
to draw real noise from. So: (1) take out everything that is not random before denoising; (2) know the random part
exactly, per photosite, per photo; (3) train a network on this camera's real noise; (4) hold every method, learned
or not, to a test no AI product offers: **what was removed must be noise and nothing else, and it is checked and
shown.** "No smoothing" stops being a promise and becomes a number on the screen.

---

## 0. Read first

### 0.1 Rules (all of `BUILD_PLAN.md` section 0 and `CLAUDE.md` apply; these bite here)

1. **One raw in, one result out. No stacking in Clean, ever** (decision 0.4). Dual Pixel halves of the *same*
   exposure are allowed (block H).
2. **Nothing generative.** A method may remove noise; it may not invent detail. Block E enforces this for every
   method, the learned one most of all. Anything that cannot pass E stays in the Lab, labelled "synthetic".
3. **Measure, do not assume.** Every block ends with numbers on the harness (block A) and they go in
   `docs/RESULTS.md`. A step that does not measure better is switched off and the result written down. PSNR alone
   never decides (plan section 22): texture kept and the residual test count for more.
4. **More than one method is fine if the person can choose and understand** (23.1): each gets a "best for / weak
   at" text. Colour tiers: cyan measured on this body, violet shared, amber estimated, and "learned" its own tier.
5. **Bill's eye is the last judge**: the blind test against his DxO (`Blind Test vs DxO.bat`). Clean stays off by
   default until it passes (Phase 3 acceptance, never yet run).
6. Models and calibration data are never committed. Nothing on C:\. Bill runs `.bat` files only. One line per
   change in `docs/NEXT_UPDATE.md`. Tests on synthetic data with known truth; run them before saying it works.
7. Private stays private: dark frames, noise banks, master darks, hot-pixel maps live in `calibration_local\` and
   are never shared. A camera profile (numbers only) may be.

### 0.2 What exists (use it, do not rebuild)

| Piece | Where | State |
|---|---|---|
| Clean: VST, luminance/chroma rotation, starlet, MP-PCA, NLM, hybrid, keep-grain, safeguard (4.5 sigma) | `engine/raw/denoise.py` | built, never blind-tested; **no BM3D-class baseline** |
| Row banding from the photo itself | `engine/raw/banding.py` | built; ignores the profile |
| Noise model `NoiseModel(a, b)`, estimate from one image | `engine/calib/noise.py` | built |
| Drop-everything calibration: per-frame store, profiles by ISO and temperature, gain from flat pairs, thermal variance, hot / warm counts | `engine/calib/frames.py`, `builder.py`, `profile.py`, `app/cameras.py` | built 2026-11-18, run on Bill's real pile |
| `CameraProfile.noise_model(iso, range, temp_c, exposure_s)` | `profile.py`, called in `pipeline.py` | built: read noise near the photo's temperature + long-exposure variance |
| Hot / warm maps per ISO (short) and per ISO, seconds, temperature (long) | `calibration_local\<camera>\hot_pixels.json`, `hot_pixels_long.npz` | **saved, not used by Clean** |
| Benchmark harness (synthetic noise from a profile on clean raws; PSNR, residual structure) | `tools/bench.py`, `Run Benchmark.bat` | built; no texture score, formula noise only |
| Blind test page against DxO exports | `studio/blindtest.py` | built, not yet done by Bill |
| NumPy ONNX runner (small models only) | `engine/local/onnxnp.py` | too slow for a 45 MP denoiser |
| Neural runtime (ONNX Runtime, DirectML), learned candidates | plan 23.2, 23.3 | **decided, not built** |

The per-frame store keeps a 256-pixel centre crop and the hot list per frame, not the frame. Blocks A and B need
whole frames again: they re-read the raws (Bill's are in `calibration_local\`, 449 CR3, about 3 minutes on his PC).
Keep the raws' folder in the store's metadata so this is one click, and say plainly when the raws are no longer
where they were.

### 0.3 Facts measured on Bill's R5 on 2026-11-18 (do not re-derive; do re-check after block B)

* **Two sensors in one.** Read noise 4.9 to 6.0 e- at ISO 100 to 320, then 1.5 to 2.1 e- from ISO 400 up
  (the dual-gain switch). In raw units: 1.46 DN at ISO 100, 5.49 at 320, 1.78 at 400, 62 at 12800, 244 at 51200,
  426 at 102400. Black level 512 below ISO 320, 2048 from 320 up.
* **Gain is physical.** gain x ISO = 332 to 366 from ISO 100 to 32000 (e-/DN: 3.38 at ISO 100, 0.864 at 400,
  0.0284 at 12800), 378 at 40000, 395 at 51200, **442 at 102400** (a pushed setting: treat 51200 and 102400 as
  digitally scaled, check for clipping of the noise itself). Saturation about 45,700 e- at ISO 100.
* **Temperature does not move short-exposure read noise** (24 vs 28 °C agree within 1%). It moves long exposures:
  ISO 12800, 30 s at about 32 °C: 147 DN against 62 short. Every long frame so far is near 32 °C, so the
  doubling temperature is **not** measured yet (Bill may add cold 30 s frames).
* **Fixed pattern is small but not nothing** (centre crop, 7 to 14 frames per ISO): 7 to 9% of the temporal
  noise at ISO 100 to 320, **about 20% from ISO 400 up** (0.39 DN at 400, 13 DN at 12800, 47 DN at 51200). Taking
  it out lowers single-frame noise only about 2%, so the master dark (B.1) must earn its place on the harness.
  The fixed **row** part, measured over whole frames by deep calibration (0.5): 12% of the row-banding energy
  at ISO 400, 27% at 3200, 8% at 12800, 9% at 51200 is the same in every photo. (A first estimate from the centre
  crop said a quarter at 12800; the whole frame says 8%.) Worth subtracting at mid ISO, small elsewhere.
* **Neighbouring photosites are independent** (lag-1 correlation under 0.04) except at ISO 100 (+0.15) and
  ISO 102400 (+0.11). Methods that assume independent noise (Noise2Noise-style, SURE) are safe from 200 to 51200.
  **Corrected the same evening (0.6):** that figure was for *adjacent* photosites, which are different colours.
  Clean works in colour planes, where the neighbours are photosites of ONE colour, two apart. Those agree by
  **+0.08 to +0.10 in dark frames at ISO 400** (all four colours, both directions, three pairs, 2048 x 2048), and
  by nothing at ISO 3200 and 12800. In flat frames (the light's own noise) they agree by nothing at any ISO. So
  the sharing is in the read noise only, and matters only in very dark parts of low-ISO photos.
* **The noise is not a perfect bell curve at low ISO** (excess kurtosis +0.6 to +2.1 up to ISO 1600, near 0
  above 12800). Formula noise is a fair stand-in at high ISO and a poor one at low ISO: draw real noise (B.2).
* **Hot and warm pixels.** A few dozen truly hot photosites (66 at ISO 400: at least 1% of the range above
  black); about 6,000 warm ones that stand out only where read noise is low (median 20 DN above black at
  ISO 400). Both matter when shadows are pushed.
* **Flats are dim** (2 to 14% of the range) and mostly one pair per ISO. Good enough for gain; too dim to check
  the model near the top of the range (A.4 asks Bill for a brighter set only if the check fails).
* **Speed:** 2.5 s to open and measure one 45 MP frame on the 2-core build machine; 131 frames in 2.4 min on
  Bill's PC (14 cores, RTX 3080 Ti 16 GB, 64 GB RAM, Windows 11).

### 0.5 Status: first build, 2026-11-18 (same day; Bill: "Build everything... code everything needed for it to run")

**Built and tested** (tests/test_noise_plan.py, tests/test_lab.py; a subset of Bill's real R5 pile for the numbers):

* **B.2, B.4, part of B.1: Deep calibration** (`engine/calib/deep.py`; button on each camera's card; `studio-cli
  calibrate --deep`): the noise bank (up to 48 patches of 256 per ISO and exposure class), noise shape numbers,
  and the fixed row and column pattern per ISO. **Measured, not yet applied** (block C).
* **A.1, A.2, most of A.3: truth with real noise** (`engine/calib/synth.py`, `tools/noise_bench.py`, `Run Noise
  Benchmark.bat`): charts (dead leaves strong / in shadow / faint, flat patches, slanted edge) plus clean photo
  crops once the Lab has prepared some; scores noise left, texture kept, edges kept, colour blotches, residual,
  seconds; HTML page with crops. On Bill's R5 a synthetic flat matches the model within 1% at ISO 400, 3200,
  12800 and 51200.
* **E.1 to E.3: the residual test and the detail guard** (`engine/raw/guard.py`, on by default for new edits,
  sidecar stage version 3): structure above the noise is returned; the **noise budget** returns what was removed
  beyond what noise can account for in any neighbourhood and band; the envelope holds every scale, the smooth
  remainder included. Pure noise: 0.02% returned. Develop shows "Detail returned".
* **F.1 to F.4: the learned denoiser** (`engine/raw/learned.py`, `studio/lab/`, `Train Camera Model.bat`, method
  "learned" in Clean): runtime with tiles, training data from clean photos + the noise bank, the network (three
  sizes), training with held-out scoring, resume, ONNX export and a card; the Cameras card "Clean, taught by this
  camera" with what is ready, the clean-photo count and **why about 200 varied photos** (`lab.PHOTOS_WHY`, shown
  on the card, in the training window and when training is refused for too few).

**What the first numbers say (R5 real noise, quick charts):** Hybrid keeps 100% of fine texture at ISO 400 and
3200, 85% at 12800 and **50% at 51200**; it leaves 45 to 67% of the incoming error. The existing methods at their
defaults do not over-remove (residual 0.3 to 0.8), so the guard changes nothing there; on a deliberately too
strong setting it returns 9 points of texture at ISO 12800 and 7 at 51200. Detail fainter than the noise cannot be
recovered from one photo's residual by any test: keeping it needs a better prior, which is what block F is for.
The gap to close is high ISO texture.

**Not built yet** (in the plan's order): A.4 against Bill's real flats and A.5 real files with DxO; B.3 banding
spectrum, B.5 black by temperature, B.6 pushed ISOs, the per-photosite master dark; **all of C** (hot and warm
pixels from the map, subtracting the fixed pattern, profile-gated banding); D.2 BM3D-class baseline, D.4 SURE,
D.5 shadow-lift awareness, D.6; E.4's "What Clean removed" view (the number is shown, the view is not); F.5
published baselines, F.6 a general model with per-camera fine-tuning, F.7 joint demosaic; G; H; most of I.

**Never run:** `Train Camera Model.bat` on Windows (its install step), training on a graphics card, DirectML,
and any of this on real photographs. The training loop was run on the build machine's processor on synthetic
photos: 700 steps took the held-out noise left from 70% to 52% and texture kept from 84% to 92%, so it learns;
that says nothing yet about how good a real model will be.

**Hot / warm count, open question:** "1% of the range above black" counts the same warm photosite as warm at
ISO 400 (20 DN) and hot at ISO 3200 (160 DN). With gain measured, define it in electrons so the count does not
depend on ISO.

### 0.6 Status: second build, 2026-11-18 evening (Bill: "build everything on the Not Built Yet list. Make it easy for the user, make sure everything is explained as well")

**Built and tested** (tests/test_calibrated.py, test_denoise.py, test_referee.py, test_clean_ui.py, test_lab.py,
test_noise_plan.py; and run on Bill's real files in the build machine: 63 of his calibration raws, a candlelit
ISO 3200 photo, his Dual Pixel file):

* **C, all of it** (`engine/raw/calibrated.py`; `Edit.calibrated`, on for new edits; Develop > Clean > "Use this
  camera's calibration" with a cyan chip): black offset per colour (picture area against masked border), the
  fixed rows and columns (shared with the per-photo row step so nothing is taken out twice), hot photosites
  always and warm ones only where they show above the photo's local noise. On Bill's candlelit ISO 3200 photo: 42
  hot and 117 warm healed of 4,095 mapped, fixed rows 0.24 DN and columns 0.26 DN out, black corrected by up to
  0.33 DN.
* **B.3** banding spectrum (`deep._periodic`): on the R5 nothing periodic at ISO 400, 3200, 12800, 51200: the
  banding is random from row to row. **B.6** pushed ISOs flagged (gain x ISO well above the sensor's, or a comb
  in the histogram) and said in Develop. **Hot and warm in electrons** (`profile.HOT_ELECTRONS = 100`): the open
  question of 0.5 is closed.
* **The master dark, decided and not built:** per-photosite fixed pattern is 19 to 24% of the temporal noise on
  the R5. Subtracting a mean of n frames adds 1/n of the temporal variance and removes the fixed part: it only
  helps above about 20 frames per ISO; Bill has 8 to 16. The Cameras card says this with the camera's own numbers.
* **D.2** a BM3D-class method in our own code (`engine/raw/collab.py`, method "collab": block matching, two
  passes, hard threshold then Wiener). **D.4 Auto strength, detail first** (`engine/raw/auto_strength.py`,
  `pipeline.clean_auto`; `Denoise.auto`, on for new edits): Monte-Carlo SURE for the error, and **the same lemma
  on the finest band alone for the share of fine detail kept** (the Studio's own use of it; on charts the estimate
  is within 3 points of the truth). Plain least-error SURE was built first and measured: it gave up 4 points of
  texture for 2 of noise, so lost detail is now weighed `DETAIL_WEIGHT = 4` times more and the lightest strength
  within 3% of the best is taken. Only with a measured noise model (a profile with gain): with the noise put 30%
  too low a smooth photo was left 83% as noisy as it came, so an uncalibrated camera keeps its sliders and is told
  why. Read noise shared between neighbours is allowed for (generalised SURE: the probe is shaped like the
  noise, by the profile's figure times the photo's read-noise share). `Denoise.lean` leans from the choice.
* **D.5** shadow-lift awareness (`Edit.shadow_lift`, `Denoise.lift`). **D.6**, differently: no tuned tables
  per ISO regime; each photo chooses (D.4), "Make it good" keeps its ISO-scaled strengths for cameras without a
  profile, and pushed ISOs are said.
* **E.4** "What Clean removed" (`app/removed.py`, `pipeline.clean_removed`): before, after, removed (in units of
  the camera's noise), magenta where the residual test finds picture, a verdict in words and colour, with or
  without the guard. On Bill's candlelit photo at Auto: removed strength 0.96 and 0.89 of the noise, nothing that
  is not noise, guard returned 0.003%.
* **G** fusion (method "fusion": non-local means where there is texture, stronger wavelets where it is flat).
* **H.1, H.2 the Dual Pixel referee** (`engine/raw/referee.py`): one half cleaned and judged against the other
  for error and for fine detail kept; blocks where the halves disagree (out of focus), dark blocks (read noise
  rules) and photos whose halves share noise (measured in their flat places) are left out or give no verdict.
  Auto uses it for Dual Pixel CR3 files. On Bill's file (ISO 1600): 88% of the pieces usable, no shared noise
  found in its flat places.
* **F.6** a general network (`python -m studio.lab.train --general`, the Cameras card's "Train a general
  network"): trained over every calibrated camera, used by cameras with no network of their own (Clean says
  so), and the starting point of each camera's own training (`--start-from`, automatic when one exists).
* **A.4** the harness against real flat pairs (`noise_bench.honesty`, on the report page): on the R5 all six
  pairs within 1% (0.992 to 1.003). The flats are at 7 to 13% of the range; above that it is not checked.
  **A.5** real photos without truth (`noise_bench.real_photos`, a folder dragged onto the .bat): the Auto
  estimates per method. The blind test page already existed (Blind Test vs DxO.bat).
* **I**: all of the above in Develop with texts (best for / weak at), the Cameras card "What Clean takes from this
  calibration", Help cards, `docs/CLEAN.md`, `studio-cli clean --auto-strength` and the new methods.

**What the numbers say (R5 real noise, quick charts, guard on; noise left / texture kept / colour blotches left):**

| ISO | Hybrid, old defaults | Hybrid, Auto | Fusion, Auto |
|---|---|---|---|
| 400 | 66.9 / 100.0 / 38.5 | 66.3 / 100.0 / 31.1 | 65.1 / 100.0 / 31.1 |
| 3200 | 63.1 / 99.6 / 39.5 | 62.7 / 99.3 / 32.4 | 61.6 / 99.3 / 32.3 |
| 12800 | 54.4 / 85.2 / 37.4 | 55.8 / 91.1 / 30.6 | 54.4 / 91.2 / 30.6 |
| 51200 | 44.3 / 50.6 / 41.9 | 48.0 / 61.9 / 40.4 | 46.3 / 62.6 / 40.6 |

Auto buys texture where it was being lost (6 points at 12800, 11 to 12 at 51200) for 0 to 4 points of noise, and
takes a fifth off the colour blotches everywhere. Collaborative with Auto at 12800: 53.9 / 88.4 / 30.6. **Texture
at 51200 is still only 62%:** that is the gap a trained network is for.

**Not built, and why:**
* **F.5 published networks as baselines** (RawNIND, RawForge). Their weights, input conventions and licences were
  not checked on this machine, and a wrong convention gives a bad score that would be blamed on the model. Needs
  a session that can download and verify them.
* **F.7 joint denoise and demosaic.** It replaces the demosaic stage and needs a trained F.3 model first.
* **H.3** per-photo fine-tune and **H.4** depth-steered strength.
* **B.5** black level by temperature (Bill's short frames span 24 to 40 °C with no visible drift to model).
* **D.3** NL-Bayes and dual-domain candidates.
* **Removing a periodic banding line** (none was found on the R5; it is recorded if a camera has one).

**Never run:** `Train Camera Model.bat` on Windows, training on a graphics card, DirectML, the general network on
more than one camera, the 5D Mark IV's Dual Pixel CR2 (the referee reads CR3 only), the full test suite on Windows
after this build, and Bill's blind test against DxO. Auto and the calibration were run on two of Bill's photos
and judged by numbers, not by eye. On the build machine the whole suite passes except three tests that fail there
before this build too (two Portrait tests and one iris test that need model files the build machine lacks).

**A miss from the first build, fixed here:** `tests/test_imgops.py::test_every_import_is_installed_by_requirements`
has failed since 0.1.10, because the Lab's optional imports (torch, onnx, onnxruntime) were not listed as
optional. Run Tests.bat on Bill's PC would have shown it.

**Speed, measured on the build machine's two slow cores:** Auto costs 15 to 25 s once per photo and method
(remembered while the photo is open); on the cloud build machine 4 s. Not measured on Bill's PC.

### 0.4 Ask Bill once, at the start, in chat (defaults if he is away)

1. **Training on his PC.** The network (block F) is trained with PyTorch on his RTX 3080 Ti: an optional "Lab"
   install of about 2.5 GB beside the Studio on D:, never needed by anyone who only uses the Studio. Default: yes.
2. **Clean photos to learn from.** Which folders of his own base-ISO raws may be read (they are never copied or
   shared; only small patches enter training)? Default: `test_raws\` plus whatever folder he names; public
   RawNIND pairs added only after its licence is checked.
3. **Where the hot / warm line sits** (1% of the range today). Default: keep.

---

## A. The truth harness (first: everything else is judged on it)

Today's harness adds formula noise. That flatters methods tuned to formulas and hides exactly what this plan is
about. Rebuild it around real noise and around the question "was detail lost?".

A.1 **Clean truth.** Base-ISO raws, well exposed, reduced 2x by box average *in the mosaic's colour planes* (four
photosites of one colour into one), which cuts their own noise in half and keeps them a true Bayer mosaic. Crops
of 1024, chosen for content: fine texture (foliage, fabric, hair), smooth gradients (sky, skin), deep shadow,
candle flames and catchlights (the named tests), text. Plus generated charts with known truth: **dead leaves**
(texture, plan section 22), slanted edges, a zone plate, flat patches at 12 levels.

A.2 **Real noise onto truth.** For a target ISO, exposure and temperature: scale the clean mosaic to the
electrons that ISO collects, draw photon noise (Poisson) with the **measured gain**, convert to raw units, add a
**patch of a real dark frame** of that ISO, exposure class and temperature band from the noise bank (B.2), black
level and clipping as the camera does them. This carries true read noise, banding, fixed pattern, warm pixels
and the non-bell-curve tails. Formula-only noise stays as an option, for cameras with no pile.

A.3 **Scores** (per method, ISO and content class; one HTML page, colour-coded, zoomable crops, a leaderboard):
* error in display space (what the eye sees after the tone curve), luminance and colour apart;
* **texture kept**: the dead-leaves spectrum of the result over the truth's, as one number (100% = all kept);
  and the same for the real texture crops by band energy;
* **edge sharpness kept** (slanted edge, 10 to 90% rise);
* **colour blotches left**: low-frequency chroma power in flat patches;
* **the residual test** (block E) numbers: structure found in what was removed;
* **false detail**: energy in the result that is in neither the truth nor the input's noise envelope (catches a
  network that invents);
* time and memory for 45 MP.

A.4 **Is the harness honest?** Its synthetic flat at the level of each of Bill's real flats must have the same
variance as the real pair difference within 5%, at every ISO, and its dark patches the same row spectrum. If the
top of the range cannot be checked because the flats are dim, say so and ask Bill for six brighter pairs.

A.5 **Real files.** Bill's high-ISO folders with his DxO exports beside them, scored without truth: residual
test, blotch score in flat areas found automatically, and the blind-test page. `Run Benchmark.bat` runs all of it
and writes `outputs\bench\<date>\index.html`; `studio-cli bench` gives the numbers as data.

*Acceptance:* the page ranks the four existing methods at ISO 400, 3200, 12800, 51200 on Bill's R5 noise; A.4
passes; the run is repeatable to the last digit with a fixed seed.

## B. Calibration data, second pass (what the dark frames still hold)

Extend `frames.scan` with a second, optional pass over the raws ("Deep calibration", a button on the Cameras card,
with a line saying it needs the raw files and about how long it takes). Everything lands in
`calibration_local\<camera>\`, private.

B.1 **Master dark per ISO** (and per exposure class and temperature band where there are 4 or more frames): the
mean of the frames minus black, i.e. the fixed pattern. Store its row profile and column profile exactly (small),
and the per-photosite remainder quantised to 1/8 of the read noise, compressed. State the expected gain honestly
in the UI: it removes the part of the noise that is identical in every photo.

B.2 **Noise bank**: per ISO, exposure class and temperature band, up to 32 patches of 512 x 512 raw photosites
taken from different frames and places (CFA phase kept), with the master dark *not* removed. About 16 MB per ISO.
This is what A.2 and F.2 draw from.

B.3 **Banding signature** (Bill's cyclostationary idea, now testable on 360 frames): for each ISO, the spectrum
of row means and of column means across frames; which lines are present in every frame (fixed: subtract), which
are periodic with random phase (model: frequency, amplitude, how it is found in one photo), which are plain
1/f (estimate per photo). Report what was found, including "nothing periodic", per ISO.

B.4 **Noise shape and neighbours**: quantiles of the pair-difference distribution per ISO (the tails), lag-1 and
lag-2 correlation in both directions, per CFA plane. Stored in the profile (numbers only, shareable).

B.5 **Black level by temperature and exposure**, per CFA position, and the level of the masked border against the
visible area: a black error of 0.3 DN at ISO 400 is a colour cast in pushed shadows.

B.6 **Clipping and the pushed ISOs**: at 51200 and 102400, is the noise itself clipped at black, and is the
histogram combed (digital scaling)? Record the true analogue ISO behind each setting.

*Acceptance:* B.1 to B.6 built from Bill's pile in under 10 minutes; a held-out dark frame minus the master dark
has its fixed-pattern part reduced by at least 80%; the profile predicts the noise of held-out frames within 5%.

## C. Take out what is not random (before Clean sees the photo)

Order matters: defects, then fixed pattern, then banding, then black.

C.1 **Hot and warm pixels from the map**, matched to the photo's ISO, shutter time and camera temperature
(nearest measured cell; scale warm pixels by exposure time where the long frames show they grow). Heal from
same-colour neighbours with edge direction. Hot: always. Warm: when it stands above the photo's noise *after* the
shadow lift the edit applies (read the edit's exposure and shadows; a warm pixel invisible at 0 EV is a speck at
+3). The image-based detector stays for cameras with no map. Named tests: candle flame and catchlights survive.

C.2 **Subtract the master dark** (B.1) when one matches. Off automatically when the photo's temperature or
shutter time is outside what was measured, and said so.

C.3 **Banding**: subtract the fixed row and column profiles (exact), then the periodic lines B.3 found (fit phase
and amplitude on this photo, gated by the profile's amplitude so nothing is invented), then the existing
per-photo estimate for what remains. Never create a band on a clean frame (named test).

C.4 **Black level** per position from B.5 at the photo's temperature.

C.5 What C removed is shown in Develop as a chip per step with its measured size (cyan), and each can be switched
off in Full level.

*Acceptance on the harness:* banding energy left after C at ISO 12800 is at most half of today's; no score in
A.3 gets worse at any ISO; the named tests pass.

## D. The classical core, done properly

D.1 **Exact noise in.** With gain measured, the full model (a and b per position, per photo, with temperature
and long-exposure variance) replaces the per-photo estimate; the VST and its exact inverse use it. Chip: cyan
"measured on this body".

D.2 **A BM3D-class baseline in our own code** (collaborative filtering of grouped patches in a transform domain,
two passes; the `bm3d` package is not GPL-compatible). This is the bar plan 8.1a set and nothing has been measured
against it yet.

D.3 **NL-Bayes and dual-domain** (23.3) as candidates; keep what earns its place.

D.4 **Strength set by the photo, not by a slider default: SURE.** Because the noise is known exactly after the
VST, Stein's unbiased risk estimate gives the error of any setting *without a clean reference*, by denoising the
photo a second time with a small known perturbation (Monte-Carlo SURE; Ramani, Blu and Unser 2008; verify). Use
it to choose luminance and chroma strength, and scales, per photo and per region. Valid where neighbours are
independent (0.3: ISO 200 to 51200); elsewhere fall back to the profile's defaults. Show the choice (cyan).

D.5 **Shadow-lift awareness** (noted "not built" in RESULTS): Clean runs before tone, so it must be told how far
the edit lifts each brightness and clean the lifted parts in proportion.

D.6 **Near the two regimes and the pushed ISOs**: separate tuned defaults for 100 to 320, 400 to 40000, 51200
and up (0.3).

## E. The residual test and the detail guard (the centre of this plan)

What Clean removes is `removed = before - after`. If Clean did its job and nothing else, `removed` divided by
the known noise level at each photosite is **white noise of strength one**: no edges, no texture, no pattern, the
same everywhere. Any structure in it is detail that was taken. This is the signals test of a residual after
subtracting a model, applied to a photograph. (Related ideas to read first, verify each: "method noise", Buades,
Coll and Morel 2005; iterative regularisation, Osher et al. 2005; SOS boosting, Romano and Elad 2015; Tukey's
twicing.)

E.1 **Measure.** On the VST luminance and chroma planes, per region and per scale (starlet bands): the variance
of the normalised residual (should be 1: more means detail went with the noise, less means noise was left), its
autocorrelation at small lags (should be 0), and its correlation with the detail bands of the result (should be
0: a residual that looks like the picture's edges is the picture). Thresholds from the statistics of true white
noise at that region size, so a "detection" has a known false-alarm rate (set it at 1%).

E.2 **Give it back.** Where structure is detected, project the residual onto the local detail directions of the
result (structure tensor / the detected bands) and return that part to the image; what stays removed is the part
indistinguishable from noise. Repeat at most three times (it converges quickly; measure). This is done for every
method, classical or learned.

E.3 **The noise envelope** (the no-invention rule): at every scale, the result may differ from the input by no
more than a set number of that scale's noise sigmas (today's safeguard is the single-photosite case of this).
A network cannot hallucinate a texture that the noise could not have hidden.

E.4 **Show it.** In Develop: a view "What Clean removed" (it should look like television static; anything
recognisable is a fault), amber over regions where the test found structure before E.2 and cyan where it is
clean, and one number: "Detail returned: 0.4% of the removed energy". In Full level, the guard's strength. On
the harness: the A.3 residual numbers and texture-kept score before and after E.2, for every method.

*Acceptance:* on dead leaves at ISO 6400 and 51200, texture kept rises with the guard on for every method while
flat-patch noise rises by less than 5%; on pure noise the guard returns under 1% (false-alarm rate holds); the
"What Clean removed" view of Bill's candle portrait shows no face.

## F. The learned denoiser, trained on this camera

F.1 **Runtime** (plan 23.2): ONNX Runtime as an optional install, GPU by DirectML, CPU fallback, off until turned
on, tiles with overlap so 45 MP fits in 4 GB of video memory. Budget: 45 MP under 30 s on Bill's GPU.

F.2 **Training data = the harness generator** (A.1, A.2), run endlessly: random crops of clean truth, random ISO
across the profile, real noise from the bank, random exposure and white balance, the camera's clipping. Hold out
whole photos (not patches) for testing so the score is honest. Minimum clean set: 200 photos of varied content;
with fewer, say so on the training page (the risk is learning Bill's subjects, not his noise).

F.3 **Network.** A compact U-Net of NAFNet-style blocks on the packed mosaic (four planes at half size). Inputs
beside the image: the per-photosite noise level map (k-sigma normalisation, PMRID, 23.3: one network for every
ISO), a flag for the gain regime, shutter time and camera temperature. Output: the clean mosaic. Three sizes
(fast, balanced, best). Loss: error in display-encoded space plus a band-energy term that punishes *lost*
texture; **no adversarial loss** (rule 0.1.2). C runs before it and E after it, always.

F.4 **Training kit (Lab)**: `Train Camera Model.bat` on Bill's PC: checks the GPU, makes data, trains, reports
progress with the harness scores every so many steps, stops when the held-out score stops improving, exports
ONNX with its card (camera, date, data used, scores) into `calibration_local\<camera>\models\`. Resumable. Plan
for a first balanced model overnight.

F.5 **Baselines to beat** (23.3): RawNIND UtNet2 and RawForge TreeNet through the same runtime, scored on the
same harness. If a published model beats ours on Bill's R5 noise, say so and offer it.

F.6 **A personal model for any camera.** Train one general model; a user's own dark-frame pile then fine-tunes a
small add-on for their body (hours, their GPU, optional). The general model plus calibrated C, D and E is what
everyone else gets. General models we train ourselves are GPL-3 and may be shared with the profile library.

F.7 **Second stage: joint denoise and demosaic** (what DxO does) once F.3 is solid: output full colour, compared
against F.3 followed by RCD on the harness.

*Acceptance:* the balanced model beats every classical method on error and on texture kept at ISO 3200 to 51200
on held-out photos, passes E with less than 1% returned, shows zero false detail on the charts, and Bill prefers
it to his DxO export on at least half of ten blind pairs.

## G. Fusion (the best of each, by place)

Per region and per scale, blend candidates by which one the harness says wins for that local signal-to-noise and
content class (flat, edge, fine texture, deep shadow): typically learned for texture, wavelets for flat areas
and colour blotches at coarse scales, the original where the guard says nothing should move. One recommended
"Auto" on top; every single method still selectable with its text.

## H. Dual Pixel files (the same exposure, split in two)

H.1 In focused areas the two halves see the same scene with independent photon noise: their difference is a
**measured noise map for this photo** (check it against the profile; report disagreement).
H.2 **Referee:** denoise one half, score it against the other; choose strength and method per region from that
(real data's answer to D.4). Note the read-noise link between the halves (the file stores the sum and one half)
and test how much it hurts in deep shadow before trusting it there.
H.3 **Per-photo fine-tune** of the learned model on the two halves (Noise2Noise style), a few seconds.
H.4 Depth from the halves steers strength: more in defocused areas, least on the plane of focus.
Bill is shooting a Dual Pixel set for this (darks at four ISOs, high-ISO scenes, a clipped-highlight scene).

## I. In the Studio

* **Simple:** one Clean slider and Auto (D.4 / H.2 choose). The noise chip (cyan when measured). The detail
  number from E.4.
* **Full:** method picker with texts, luminance, colour, keep-grain, the guard, each C step, "What Clean
  removed", learned model picker (its own colour tier), per-mask Clean strength.
* **Cameras page:** Deep calibration (B) and Train camera model (F.4) on the camera's card, each with what it
  needs and how long it takes.
* Clean on by default only after the blind test passes.

## 6. Order of work

1. **A** harness with real noise (needs B.2, so build B.2 with it), and the scores. Record today's methods.
2. **E** residual test and guard: it improves what exists at once and judges everything after it.
3. **B** rest, then **C**. Measure each step; switch off what does not help.
4. **D.1, D.4, D.5**, then **D.2**. Now run **Bill's first blind test** against DxO: classical only.
5. **F.1** runtime and **F.5** baselines, then **F.2 to F.4** and the first trained model. Second blind test.
6. **G** fusion. **I** throughout.
7. **H** when the Dual Pixel set is in. **F.6, F.7** last.

After step 4 Clean is as good as it can be without a network and can be turned on for testers. After step 5 it is
a learned denoiser that can show it removed nothing but noise.

## 7. Risks, said plainly

* Beating DxO on every camera is not the aim and not likely. Matching or beating it on a calibrated body at the
  ISOs its owner shoots is plausible; the blind test decides, not this plan.
* Too few clean photos and the network learns subjects (F.2). The harness's held-out set and the false-detail
  score are the alarms.
* The master dark may measure as not worth its disk space (0.3). Then it is dropped, and only its row and column
  parts kept.
* The guard can return noise if its thresholds are loose. Its false-alarm rate is a tested number (E).
* Training needs PyTorch on Bill's PC only. If that install fails, F waits; A to E do not depend on it.
* Everything here is proven on one camera. The 5D Mark IV and 6D need their own piles before any claim is made
  for them.
