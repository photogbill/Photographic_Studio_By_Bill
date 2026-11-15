<p align="center">
  <img src="assets/banner.svg" alt="Photographic Studio By Bill: a free raw studio for photographers" width="100%">
</p>

<p align="center">
  <img alt="Cost: free, forever" src="https://img.shields.io/badge/cost-free%2C%20forever-22C55E?style=flat-square">
  <a href="LICENSE"><img alt="Code licence: GPL-3.0" src="https://img.shields.io/badge/code-GPL--3.0-22D3EE?style=flat-square"></a>
  <a href="profiles/LICENSE.md"><img alt="Profiles licence: CC BY-SA 4.0" src="https://img.shields.io/badge/profiles-CC%20BY--SA%204.0-C084FC?style=flat-square"></a>
  <img alt="Status: early development" src="https://img.shields.io/badge/status-early%20development-F59E0B?style=flat-square">
  <img alt="Platform: Windows" src="https://img.shields.io/badge/platform-Windows-1C2A4A?style=flat-square">
</p>

---

## Why this exists

Good raw software costs money, often every year. **Photographic Studio By Bill** is for photographers who can't pay for it, or would rather not: free, open source, and built to get the most out of every raw file.

It starts with the hardest job: cleaning high-ISO noise **without smearing the detail** you shot. From there it grows into a full raw studio.

One raw file in, one finished file out. Your photos never leave your PC.

## What it does, and what's coming

| | Feature | What it means for you |
|:-:|---|---|
| ✅ | **Raw Analyzer** | Measures your camera's noise, banding and hot pixels from any raw file, and shows it all in a visual report. |
| ✅ | **Develop engine** | Turns CR2 and CR3 files into 16-bit Adobe RGB TIFFs, and builds a page comparing each result with your camera's own JPEG. |
| 🎨 | **Calibration wizard** | Teach the tool your camera: print a checklist, shoot about 120 frames, drop them in. |
| 🎨 | **Lens wizard** | Teach it your lenses, including vintage lenses with no electronic contacts. |
| 🧪 | **High-ISO denoiser** | Cleans noise in the raw data itself, before anything is clipped, smeared or tinted. |
| 🧪 | **Automatic lens correction** | One **Fix lens** button: colour fringing, distortion (from a lens profile or from straight lines in the photo) and falloff, fixed automatically, even for vintage lenses with no contacts. Purple-fringe removal is on by default; vignetting correction stays off unless you want it. |
| 🧪 | **Develop controls** | Exposure, white balance, straighten, perspective and crop work today; colour wheel and masks are next. |
| 🧪 | **Microcontrast** | More shades of shadow: the quality great lenses are loved for. No halos, and noise is not amplified. |
| 🧪 | **Recover lens sharpness** | Undoes part of the measured blur by deconvolution, without ringing. Off for lenses you mark as character lenses. |
| 🗺️ | **Render like my lens** | A shared library of real lens signatures. |
| 🗺️ | **Photoshop and PhotoCraft** | Send to Photoshop or PhotoCraft in one click, plus a Photoshop panel. |
| 🗺️ | **Every brand** | Canon first, then Nikon, Sony, Pentax, Fujifilm and more. |

✅ works today · 🧪 experimental preview · 🎨 designed · 🗺️ planned

**Output:** 16-bit TIFF in Adobe RGB by default, with DNG, PNG and JPEG export to come.

## Cameras

Canon **CR2** and **CR3** come first. The Raw Analyzer has been run on files from the **EOS R5**, **5D Mark IV** and **6D**. Other brands follow, and anyone with a different camera will be able to calibrate it with the wizard.

## Try it today

1. Install **Python 3.12** from [python.org](https://www.python.org/downloads/). Tick **"Add python.exe to PATH"** during setup.
2. Download this project: green **Code** button, then **Download ZIP**. Unzip it into a folder of its own.
3. Drag your **CR2** or **CR3** files, or a whole folder, onto **`Analyze Raws.bat`**.
4. The first run builds a private Python setup inside that folder (one time only). Then the report opens in your browser.
5. To develop raws: double-click **`Install.bat`** once, then drag raw files onto **`Develop Raws.bat`**. Your TIFFs and the comparison page land in the `outputs` folder.

Everything stays in that one folder. Nothing is installed anywhere else on your PC, and nothing is uploaded.

## Your photos stay yours

- Your photos are **never** uploaded.
- Later, you'll be able to share your camera's or lens's **profile** so others benefit. A profile is numbers, never pictures. Sharing is **off** until you turn it on, and you read the exact file before it goes.
- Never shared: photos, camera and lens serial numbers, your sensor's fingerprint, hot-pixel and dust maps, or your location.

## Help build it

- **Found a bug?** [Open an issue](../../issues) and attach the log from the `logs` folder.
- **Coming soon:** camera and lens profiles from the community. Every shared profile stays open under CC BY-SA 4.0, so the library belongs to everyone.

## Licence

- **Software:** [GNU GPL v3.0](LICENSE). Free to use, change and share. Anything built from it must stay free and open too.
- **Lens and camera profiles:** [CC BY-SA 4.0](profiles/LICENSE.md).

---

<p align="center">
  Made by Bill (<a href="https://github.com/photogbill">photogbill</a>), for every photographer who can't pay for the big names.
</p>
