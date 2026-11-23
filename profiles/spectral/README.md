# Camera colour data (Solomatov & Akkaynak 2023)

Used by `studio/engine/color/spectral.py`.

* `camera_sensitivities.json`: predicted red, green and blue spectral sensitivities of 1,012 cameras
  (400-700 nm every 10 nm), from the `data/predictions` folder of
  <https://github.com/COLOR-Lab-Eilat/Spectral-sensitivity-estimation>.
* `dual_matrices.json`: each camera's colour matrices for CIE illuminant A (tungsten) and D65 (daylight), from that
  repository's `data/color-matrices` folder (the paper describes them as the Adobe DNG Converter's colorimetric
  matrices; the Studio uses the pair only where the daylight one equals the matrix its raw reader already has).
* `cie_400_700_10nm.json`: CIE 1931 2-degree colour matching functions and the CIE daylight basis functions
  S0, S1, S2 (CIE 015), resampled to the same grid by the colour-science package (BSD-3-Clause).

**Attribution (CC BY 4.0):** Grigory Solomatov and Derya Akkaynak, "Spectral Sensitivity Estimation Without a
Camera", arXiv:2304.11549 (2023), licensed under Creative Commons Attribution 4.0 International
(<https://creativecommons.org/licenses/by/4.0/>). Changes made here: the CSV files were combined into JSON, the
sensitivities rounded to 6 decimals and the matrices to 4, and camera names taken from the file names with '-' read
as spaces. Nothing else was changed.
