# Clean: removing noise without losing the photo

This is the guide to Develop > Clean and to what the Cameras page does for it. Plain words; no file here needs
editing. The plan and the measurements behind it are in `BUILD_PLAN_NOISE.md`.

## The idea

Noise is random. A lot of what spoils a high-ISO photo is not: hot pixels, banding that is the same in every
photo, a black level that is slightly off. And the random part can be known exactly: how strong it is at every
brightness, for your camera, at each ISO.

So Clean does three things, in this order:

1. **Takes out what is not random**, exactly, from what your camera's lens-cap frames showed.
2. **Removes the random noise** with a strength each photo chooses for itself, detail first.
3. **Checks its own work**: what it removed is tested against the camera's known noise, and anything that was
   not noise is given back. You can look at the proof.

## What to do, once per camera

On the **Cameras** page:

| Step | What | Why |
|---|---|---|
| 1 | Drop **lens-cap frames** (any ISO, any number, any order) | Read noise, banding, hot and warm pixels, black level |
| 2 | Drop **flat pairs** (two identical shots of an even, defocused surface, per ISO) | The gain: how much noise light itself brings. This is what turns on **Auto** |
| 3 | Press **Deep calibration** on the camera's card and give it the folder of those raws | The noise bank (patches of real noise) and the banding that is the same in every photo |

The camera's card then lists **what Clean takes from this calibration**, and what it looked at and left alone,
with the reason. Nothing leaves your PC.

Optional, for the best result at very high ISO: choose about **200 varied clean photos** (ISO 400 or lower) and
press **Train this camera's network**. It runs overnight on your graphics card. Why 200 and why varied is said on
the card. **Train a general network** makes one network over every camera calibrated on this PC; a camera with
none of its own then uses it, and each camera's own training starts from it.

## In Develop > Clean

| Control | What it does | Best for | Weak at |
|---|---|---|---|
| **Clean up noise** | Turns Clean on | | |
| **Auto: the photo chooses the strength** | Tries a handful of strengths on five pieces of the photo and takes the lightest one close to the best, counting lost detail five times as heavily as noise left in | Every raw from a calibrated camera | Needs the camera's noise measured (flat pairs). Without it, it says so and your sliders decide |
| **Strength** | With Auto: leans lighter or heavier from the photo's own choice (100% = as chosen). Without Auto: scales the two sliders | | |
| **Luminance / Colour** (Full) | The two strengths. With Auto on they are greyed and show what the photo chose | | |
| **Method** (Full) | Hybrid (recommended), Fusion, Starlet, Non-local means, MP-PCA, Collaborative (the slow yardstick), Learned. Each says below the list what it is best for and weak at | | |
| **Protect stars, catchlights and sparks** | A photosite Clean moved by far more than its noise was not noise: the original is kept | Stars, candle sparks | |
| **Detail guard** | What Clean removed is tested against the camera's noise; what was not noise is returned | Everything; keep it on | Detail fainter than the noise itself cannot be told apart from noise by any test |
| **Show what Clean removed...** | The proof (below) | | |
| **Clean lifted shadows harder** (Full) | Parts you brighten (Exposure, Shadows, Smart lighting, Blacks, Auto HDR) are cleaned harder in proportion | Underexposed photos, shadow recovery | Works in half-stop steps; does not know about masks or the tone curve |
| **Use this camera's calibration** (Full) | Step 1 above: hot and warm pixels, fixed banding, black level | Long exposures, high ISO, deep shadows | Needs the Cameras page steps; a camera without them is left alone |
| **Remove row banding** | The banding that changes from photo to photo, measured in each photo's own masked border | | |

### What the chips say (and their colours)

* **Cyan = measured.** "Calibration: 42 hot and 117 warm photosites healed (of 4,095 on this camera's map);
  fixed banding taken out; black level corrected". Or Auto on a Dual Pixel photo: "measured by the two halves".
* **Amber = estimated.** "Auto: luminance 2.00, colour 1.60 · keeps about 99% of the finest detail · estimated
  from this photo". The detail figure is worked out from the photo alone; on test charts it matches the truth
  within about three points.
* **Violet = shared or learned.** A profile or a network that was not measured on this very camera body.

### "Show what Clean removed"

Three pictures of the same piece of your photo: before, after, and **what was taken out**, made stronger so it
can be seen.

* If the third picture looks like **television static**, with nothing of the photo in it: only noise was removed.
* If you can see edges, hair or a weave in it: detail went with the noise. Lower the strength, or keep the detail
  guard on. What the test finds that is not noise is marked **magenta**, and the verdict is written under the
  pictures in green, amber or red.
* Tick **Show it without the detail guard** to see what the guard saves.

## Dual Pixel photos

A Dual Pixel RAW holds the photo twice: through the left and the right half of the lens. Where the photo is in
focus both halves show the same thing, each with its own noise. Clean's Auto uses that: it cleans one half and
checks it against the other, which is a measurement rather than an estimate. It also reports whether this photo's
noise matches the camera's profile. Out-of-focus parts are left out, and so is a photo too dark for the halves to
be compared fairly (they share the camera's read noise); then the single-photo estimate is used and the chip says
so.

## Checking it yourself

* **Run Noise Benchmark.bat**: every method scored on *your camera's real noise*, on pictures with a known right
  answer: noise left, texture kept, edges kept, colour blotches. It also checks itself against your real flat
  pairs ("Is this benchmark honest?"). Drag a folder of your own raws onto it to add them: they have no known
  answer, so their figures are estimates and say so.
* **Blind Test vs DxO.bat**: unlabelled pairs, you pick the better one. This is the test that decides.

## What it cannot do

* Detail fainter than the noise is gone before Clean starts. No method brings it back; a trained network can make
  a good guess from what photographs look like, which is why training is offered, and why its output is still
  checked by the detail guard.
* Auto for an uncalibrated camera: not offered, on purpose. With the noise guessed 30% too low, a smooth photo
  would be left four fifths as noisy as it came.
* A master dark frame is not subtracted: with the usual 8 to 16 frames per ISO it would add noise. The card says
  so with your camera's own numbers.
