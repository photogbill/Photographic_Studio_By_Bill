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
| 🧪 | **Import from card and the Library** | Copy a card into your own Photos folder, a folder per shooting day named by the date (or your name for the shoot), every file checked, the card never changed. Then look through the raws, rate and pick with the keyboard, check focus at 100%, compare two side by side, and export your picks in one go, without opening Windows Explorer. |
| ✅ | **Develop engine** | Turns CR2 and CR3 files into 16-bit Adobe RGB TIFFs, and builds a page comparing each result with your camera's own JPEG. |
| 🎨 | **Calibration wizard** | Teach the tool your camera: print a checklist, shoot about 120 frames, drop them in. |
| 🎨 | **Lens wizard** | Teach it your lenses, including vintage lenses with no electronic contacts. |
| 🧪 | **High-ISO denoiser** | Cleans noise in the raw data itself, before anything is clipped, smeared or tinted: that is where it works best, so shoot raw for the cleanest results. Scans and JPEGs can be cleaned too, from noise measured in the picture itself (good on grain and colour mottling; it cannot undo what the camera already smeared or compressed). |
| 🧪 | **Automatic lens correction** | One **Fix lens** button: colour fringing, distortion (from a lens profile or from straight lines in the photo) and falloff, fixed automatically, even for vintage lenses with no contacts. Purple-fringe removal is on by default; vignetting correction stays off unless you want it. |
| 🧪 | **Develop controls** | Exposure, white balance, straighten, perspective and crop work today; colour wheel and masks are next. |
| 🧪 | **Detail by scale (microcontrast)** | More shades of shadow: the quality great lenses are loved for. One Amount and a Balance from tonal to crisp, or six methods you can dial in and combine. No halos, noise is not lifted as detail, and you can judge it in black and white. |
| 🧪 | **Auto HDR** | One tick, then Light, Medium, Heavy, Highlights only, Low-key or Shadows only: opens shadows and holds highlights without flattening the photo. Never on unless you turn it on. |
| 🧪 | **Recover lens sharpness** | Undoes part of the measured blur by deconvolution, without ringing. Off for lenses you mark as character lenses. |
| 🧪 | **Old photos, scans and film** | Opens TIFF, JPEG and PNG scans beside your raws and shows them exactly as they are. Turns black and white or colour negatives into positives (from scans or camera copies), takes stains and corrosion out by unmixing them from the picture (where a stain has eaten the picture, it shows you instead of inventing), removes dust and scratches with Microsoft's scratch detector plus the Studio's own filter, and can colourize a black and white photo with DDColor (downloaded once, runs on your PC). Faces are never redrawn. |
| 🧪 | **Cognitive Adjustments** | The Studio's own answer to Photoshop's Neural Filters, built from research papers and run on your PC. Each tool measures what is in the photo and invents nothing: printing-dot removal for scans of printed pictures, colourize with evened input and colour balance, a heat map of what every automatic tool changed, depth from one photo (Depth haze, a depth view, and Masks > Depth to select by distance), click-to-select with Meta's SAM 2.1 (Tiny shipped; Large made on your PC by `Convert Models.bat`), colour match from another photo (solved as ordinary slider settings you can still move), JPEG repair (the smoothest picture that still matches the file exactly), Relight (move the light using the shapes in the depth map) and Composite (click a subject out of another photo, place it, match its colours and grain). |
| 🗺️ | **Render like my lens** | A shared library of real lens signatures. |
| 🗺️ | **Photoshop and PhotoCraft** | Send to Photoshop or PhotoCraft in one click, plus a Photoshop panel. |
| 🗺️ | **Every brand** | Canon first, then Nikon, Sony, Pentax, Fujifilm and more. |

✅ works today · 🧪 experimental preview · 🎨 designed · 🗺️ planned

**Output:** 16-bit TIFF in Adobe RGB by default, with DNG, PNG and JPEG export to come.

## Cameras

Canon **CR2** and **CR3** come first. The Raw Analyzer has been run on files from the **EOS R5**, **5D Mark IV** and **6D**. Other brands follow, and anyone with a different camera will be able to calibrate it with the wizard.

## Install

**Easiest (recommended):** download **[Install-Photographic-Studio.bat](https://github.com/photogbill/Photographic_Studio_By_Bill/releases/latest/download/Install-Photographic-Studio.bat)** and double-click it.
- It asks where to put the Studio (a folder on D: if your PC has one), then downloads everything into that one folder, including its own private copy of Python.
- No git, no Python install, no admin rights. Nothing is installed anywhere else on your PC; the only thing outside the folder is a desktop shortcut, and only if you say yes.
- Run it again at any time to update. Your settings, calibrations and photos are kept.

**Prefer a classic installer?** Download **[Photographic-Studio.msi](https://github.com/photogbill/Photographic_Studio_By_Bill/releases/latest/download/Photographic-Studio.msi)** from the [Releases page](../../releases/latest). It is bigger (Python and every package are inside), installs for your Windows account only, and shows up in Settings > Apps with an Uninstall button.

**Linux:** download **[install-linux.sh](https://github.com/photogbill/Photographic_Studio_By_Bill/releases/latest/download/install-linux.sh)**, then in a terminal run `bash install-linux.sh`.
- It asks where to put the Studio (`~/Photographic-Studio` unless you pick another folder) and downloads everything into that one folder. It uses your own Python 3.10 or newer, or downloads a private copy if yours is too old.
- No root for the Studio itself. If the window needs system libraries your PC lacks (often `libxcb-cursor0`), it names them and asks before installing them with your package manager; that one step asks for your password.
- It offers an applications-menu entry; otherwise start it with `start-studio.sh` in the folder. Run the installer again at any time to update.
- Works on glibc Linux for x86_64 and ARM64 (Ubuntu, Mint, Debian, Fedora, openSUSE, Arch and similar), including Bazzite and other Fedora Atomic systems: everything goes in your home folder, and use Desktop Mode, not Game Mode. Not on Alpine (musl).

Windows may show a blue **"Windows protected your PC"** box for either file. That is because the Studio is not code-signed (signing costs money every year, and this is a free project). Click **More info**, then **Run anyway**.

**Updates:** in the Studio, **Help > Check for updates**. It is off until you turn it on. When it is on, the Studio asks GitHub for the newest version number (nothing else is sent, never a photo) and can put the new version in place for you.

## Using it

- **Start Studio.bat** (or the desktop shortcut) opens the Studio; on Linux, `start-studio.sh` or the menu entry.
- Drag **CR2** or **CR3** files, or a whole folder, onto **Analyze Raws.bat** for a report on your camera's files.
- Drag raw files onto **Develop Raws.bat** to develop them. Your TIFFs and the comparison page land in the `outputs` folder.
- **Convert Models.bat** turns model downloads in the `models` folder into the Studio's format on your PC (SAM 2.1 Large for click-to-select, which is too big to ship; or a newer depth model). It checks each one runs and logs to `logs\convert_models.log`.

**Developers:** clone the repository with git and double-click **Install.bat** (on Linux: `bash install-linux.sh` inside the clone). It uses your own Python (3.10 or newer) in a private environment inside the folder, or downloads a private copy of Python if your PC has none. Installers never overwrite a git copy.

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
