# Third-party software and published methods

Rule 0.3.1: only GPL-3.0-compatible dependencies. Every one is listed here with its licence.

## Python packages (installed into `.venv` by `tools\ensure_env.bat`)

| Package | Used for | Licence | GPL-3.0 compatible |
|---|---|---|---|
| NumPy | all numerics | BSD-3-Clause | yes |
| SciPy | convolutions (demosaic), filters | BSD-3-Clause | yes |
| rawpy | Python binding to LibRaw (raw decoding) | MIT | yes |
| LibRaw (inside rawpy) | decoding CR2, CR3 and other raws | LGPL-2.1 or CDDL-1.0 (dual) | yes (LGPL-2.1) |
| tifffile | writing 16-bit TIFF | BSD-3-Clause | yes |
| Pillow | JPEG writing, resizing, previews | MIT-CMU (HPND) | yes |
| ExifRead | metadata in the Raw Analyzer (fallback) | BSD-3-Clause | yes |
| pytest | self-tests only (not needed to run the app) | MIT | yes |
| PySide6-Essentials (Qt 6) | the desktop interface (Phase 4) | LGPL-3.0 | yes |

Optional (installed once by `tools\ensure_env.bat`; the Studio works without it):

| Package | Used for | Licence | GPL-3.0 compatible |
|---|---|---|---|
| lensfunpy | Python binding to lensfun: distortion and vignetting numbers for modern lenses | MIT | yes |
| lensfun (inside lensfunpy) | lens correction library | LGPL-3.0 | yes |
| lensfun database (inside lensfunpy) | lens calibration data | CC BY-SA 3.0 | yes (data, read at run time; values used are frozen into the user's sidecar, not redistributed) |

Shipped data with its own licence (no package to install):

| Item | Used for | Licence | GPL-3.0 compatible |
|---|---|---|---|
| Google MediaPipe face models in `models/faces` (BlazeFace short and full range, Face Mesh 468 points), unmodified `.tflite` files, taken from the PyPI package face-detection-tflite 0.6.0 (its code MIT; not used) | Portrait: finding faces and their features | Apache-2.0 (NOTICE and LICENSE beside the files) | yes (Apache-2.0 into GPL-3.0) |
| Google MediaPipe Iris (`models/faces/iris_landmark.tflite`), unmodified, from Google's MediaPipe asset storage | Eyes: the iris centre and size, for eye colour, the iris tools and red eye | Apache-2.0 (NOTICE and LICENSE beside the files) | yes |
| Google MediaPipe Face Landmarker v2 / FaceMesh-V2 (`models/faces/face_landmarker_v2.tflite`): `face_landmarks_detector.tflite` taken unmodified out of Google's `face_landmarker.task` bundle, renamed | Portrait and Eyes: the face mesh (468 points) and the iris points, used in place of the first mesh when present | Apache-2.0 (NOTICE and LICENSE beside the files) | yes |
| Google MediaPipe people models in `models/people` (selfie segmentation, BlazePose detector, BlazePose full landmarks), unmodified `.tflite` files from the official mediapipe 0.10.9 package | Person mask: the person and 33 body points | Apache-2.0 (NOTICE and LICENSE beside the files) | yes |
| Microsoft scratch detector from *Bringing Old Photos Back to Life* (`models/scratches/scratch_detector.npz`): the detection U-Net's weights from `global_checkpoints.zip` (release v1.0), optimiser state dropped, batch norm folded in, 8-bit weights (`tools/convert_scratch_detector.py`, no PyTorch) | Old photo and film: decides which long marks are damage | MIT (NOTICE and LICENSE beside the file) | yes |
| DDColor "paper tiny" (Kang et al., ICCV 2023; weights `ddcolor_paper_tiny`), Qualcomm AI Hub's ONNX export (`ddcolor-onnx-float.zip`), converted to `models/colour/ddcolor.npz` by `tools/convert_onnx_model.py` (8-bit weights, 53 MB) | Old photo and film: Colourize | Apache-2.0 (NOTICE and LICENSE beside the file) | yes |
| Google MediaPipe multiclass selfie segmentation (`models/people/selfie_multiclass_256x256.tflite`, optional, unmodified, from Google's model page) | Person mask: hair, skin, face skin, clothes labelled directly | Apache-2.0 (model card, May 2023) | yes |
| Google MediaPipe hair segmentation (`models/people/hair_segmenter.tflite`, optional, unmodified, from Google's model page) | Person mask: the Hair part, at 512 pixels per head | Apache-2.0 (model card) | yes |
| Depth Anything V2 Small (Yang et al., NeurIPS 2024), the onnx-community ONNX export (`model.onnx`), folded to 518 x 518 and converted to `models/depth/depth_anything_v2_small.npz` by `tools/convert_onnx_model.py` (8-bit weights, 24 MB). Only the Small model is Apache-2.0; Base/Large/Giant are CC-BY-NC-4.0 and are not used | Cognitive Adjustments: Depth haze, Show depth map; Masks > Depth | Apache-2.0 (NOTICE and LICENSE beside the file) | yes |
| SAM 2.1 Hiera-Tiny (Ravi et al., Meta FAIR 2024), ONNX export (`sam2.1_hiera_tiny_20260221.zip`), encoder folded to 1024 x 1024, both graphs converted to `models/sam/sam2.1_tiny_*.npz` (16-bit weights, 61 + 8 MB). Hiera-Large is converted on each PC by `Convert Models.bat` and not shipped | Masks > Object and Colour a part: click to select | Apache-2.0 (NOTICE and LICENSE beside the files) | yes |

The models are run by our own NumPy code (`studio/engine/local/tinynet.py`, GPL-3.0); TensorFlow, TensorFlow Lite,
OpenCV and the MediaPipe library are not used or installed.

**Installers (not part of the Studio's code).** `Install Photographic Studio.bat` downloads, at install time, the
official Python "embeddable package" from python.org (Python Software Foundation License, GPL-compatible) and pip
(`get-pip.py` from bootstrap.pypa.io, MIT) into the app folder; nothing of theirs is in the repository. On Linux,
`install-linux.sh` uses the PC's own Python (3.10 or newer) in a private venv, or, when that cannot work, downloads a
**python-build-standalone** build of CPython (Astral, the builds `uv` uses; Python Software Foundation License for
Python, with the licences of the libraries it bundles listed inside it) into `python/`, checked against the release's
published SHA-256 sums; it is not in the repository either. The MSI is
built on GitHub's machines by **Briefcase** (BeeWare, BSD-3-Clause), which uses the **WiX Toolset** (source MS-RL;
its ready-made releases since v6 carry an "Open Source Maintenance Fee" that applies to revenue-generating use only,
which this free project is not). Neither tool is shipped inside the Studio; the MSI holds the Studio, Python and the
packages listed above.

Planned (not yet used): LittleCMS through Pillow's ImageCms (MIT; used today only in the tests to check our ICC profiles).

## Methods implemented from publications

| Method | Where | Source |
|---|---|---|
| Directional demosaic with a-posteriori decision (default demosaic) | `studio/engine/raw/demosaic.py` | Menon, Andriani, Calvagno, IEEE TIP 16(1), 2007. Code structure follows the BSD-3-Clause *colour-demosaicing* implementation (Colour Developers). |
| High-quality linear demosaic (preview) | same | Malvar, He, Cutler, ICASSP 2004 |
| Gamut compression | `studio/engine/color/gamut.py` | ACES Reference Gamut Compression (Academy of Motion Picture Arts and Sciences, 2021): published algorithm and default parameters |
| Planckian locus (temperature to xy) | `studio/engine/color/camera.py` | Kim et al., US patent 7,024,034 / 2002 cubic spline approximation |
| Correlated colour temperature from xy | same | McCamy, Color Res. Appl. 1992 |
| Bradford chromatic adaptation | `studio/engine/color/spaces.py` | Lam 1985; matrix as used by ICC |
| Camera matrix normalisation | `studio/engine/color/camera.py` | dcraw / LibRaw construction (public domain / LGPL), re-implemented |
| Sigma-clipped statistics for the black level | `studio/engine/raw/linearize.py` | standard astronomy practice |
| Poisson-Gaussian noise model | `studio/engine/calib/noise.py` | Foi, Trimeche, Katkovnik, Egiazarian, IEEE TIP 17(10), 2008 |
| Generalised Anscombe transform and exact unbiased inverse | same | Starck, Murtagh, Bijaoui 1998; Makitalo and Foi, IEEE TIP 2011/2013 (computed numerically) |
| Starlet (isotropic undecimated wavelet) | `studio/engine/raw/denoise.py` | Starck, Fadili, Murtagh, IEEE TIP 16(2), 2007 |
| Empirical Wiener second stage | same | idea from BM3D (Dabov et al., IEEE TIP 2007); no BM3D code used |
| MP-PCA (Marchenko-Pastur patch PCA) | same | Veraart, Fieremans, Novikov, NeuroImage 142, 2016 |
| Non-local means | same | Buades, Coll, Morel, CVPR 2005 (unit-noise statistics after the VST) |
| Wiener-gated row banding from optical black | `studio/engine/raw/banding.py` | our design; JWST 1/f reference-pixel practice |
| Gaussian / Laplacian pyramids | `studio/engine/detail/pyramid.py` | Burt and Adelson, IEEE Trans. Communications 31(4), 1983 |
| Local Laplacian filter, fast algorithm (microcontrast) | `studio/engine/detail/microcontrast.py` | Paris, Hasinoff, Kautz, ACM TOG 30(4), 2011; Aubry, Paris, Hasinoff, Kautz, Durand, ACM TOG 33(5), 2014. Remapping function (noise toe, tone-dependent strength, edge guard) is our design; no code reused |
| Blur from the gradient ratio of a re-blurred image | `studio/engine/detail/sharpen.py` | Zhuo and Sim, Pattern Recognition 44(9), 2011; combined with an edge-profile check (our design) |
| Richardson-Lucy deconvolution, to a target PSF | same | Richardson 1972, Lucy 1974; "deconvolve to a narrower PSF" from STARRED / Firedec (Millon et al. 2024) |
| Plumb-line distortion estimate | `studio/engine/lens/lines.py` | Devernay and Faugeras, Machine Vision and Applications 13(1), 2001 |
| Line support regions (segments) | same | in the spirit of LSD, von Gioi et al., IEEE TPAMI 32(4), 2010 (no LSD code used) |
| Distortion and vignetting models | `studio/engine/lens/geometry.py`, `falloff.py` | lensfun's documented ptlens / poly3 / poly5 and "pa" models, re-implemented and checked against lensfunpy |
| Single-image vignetting estimate (radial gradient symmetry) | `studio/engine/lens/falloff.py` | idea from Zheng, Yu, Kang, Lin, Kambhamettu, CVPR 2008 (simplified: median radial gradient per ring) |
| Oklab / OKLCh perceptual colour space (colour stage, colour wheels in the interface) | `studio/engine/color/grade.py`, `studio/app/widgets.py` | Ottosson, "A perceptual color space for image processing", 2020 (published matrices; reference code MIT). Band, range, vibrance and grading arithmetic is our design |
| Guided filter (Smart lighting) | `studio/engine/color/light.py` | He, Sun, Tang, IEEE TPAMI 35(6):1397-1409, 2013 (self-guided, box windows); lift and ease curves are our design |
| Grey-edge colour constancy (Auto white balance) | `studio/engine/color/autowb.py` | van de Weijer, Gevers, Gijsenij, IEEE TIP 16(9):2207-2214, 2007 (Minkowski p = 6, sigma 1); clipping and dark-pixel masks are ours |
| Monotone cubic interpolation (tone curves) | `studio/engine/color/light.py` | Fritsch and Carlson, SIAM J. Numer. Anal. 17(2):238-246, 1980, via SciPy's PchipInterpolator (BSD-3-Clause) |
| Dark channel prior (Dehaze) | `studio/engine/detail/finish.py` | He, Sun, Tang, CVPR 2009 and IEEE TPAMI 33(12):2341-2353, 2011; transmission refined with the guided filter instead of soft matting |
| Colourfulness metric (Auto colour) | `studio/engine/color/grade.py` | Hasler and Susstrunk, Proc. SPIE 5007:87-95, 2003 |
| DNG camera profiles (.dcp): matrices, illuminant interpolation, hue/sat/value tables | `studio/engine/color/dcp.py` | Adobe DNG Specification 1.6 (public; Adobe grants a royalty-free licence to implement it); no SDK code used |
| Hash noise for grain | `studio/engine/detail/finish.py` | MurmurHash3 32-bit finaliser (Austin Appleby, public domain) with Box-Muller |
| Highlight rebuild | `studio/engine/raw/highlights.py` | our design (log-chromaticity spread from the bright edge of clipped areas by normalised convolution); the idea of borrowing colour from the unclipped surroundings is shared with darktable's "inpaint opposed" (no code used) |
| Click an object (mask) | `studio/engine/local/masks.py` | idea of GrabCut (Rother, Kolmogorov, Blake, ACM TOG 23(3):309-314, 2004): iterated object / background colour models; ours uses 3-D colour histograms and a guided filter instead of GMMs and graph cuts |
| .cube 3D LUT format | `studio/engine/color/grade.py` | Adobe Cube LUT Specification 1.0 (2013), read by our own parser (3D, 0..1 domain only) |
| Heal (seamless cloning) | `studio/engine/local/retouch.py` | Perez, Gangnet, Blake, "Poisson image editing", SIGGRAPH 2003 (membrane by sparse LU through SciPy); multiplicative / log form after Georgiev, "Photoshop Healing Brush", ECCV workshop 2004 (no code used). Source search and the edge smoothing are our design |
| Measured highlights from the lens glow | `studio/engine/raw/glow.py` | idea from NASA SDO/AIA DESAT (Schwartz, Torre, Piana, ApJL 793:L23, 2014) and PSF-wing photometry of saturated stars; glare spread function after Talvala et al., ACM TOG 26(3), 2007. The lens-profile measurement, the lower-envelope fit, the partly-blown route and the light tests are our design (no code used) |
| Dust finder | `studio/engine/local/dust.py` | scale-space blob detection, Lindeberg, IJCV 30(2):79-116, 1998 (difference of Gaussians + Hessian roundness); the ring texture test, the dust map built from recurring blobs and its aperture-aware matching are our design |
| Running TensorFlow Lite models (face, person and pose models) | `studio/engine/local/tinynet.py` | our own reader of the TFLite FlatBuffer schema (Apache-2.0 schema, re-implemented from its field list) and NumPy layers, sparse weights (TFLite sparsity format), 8-bit weights (de-quantised on loading) and MediaPipe's own layers (transposed convolution, max pooling with indices, unpooling); outputs checked against OpenCV's TFLite importer, Google's LiteRT runtime and the MediaPipe library (all used only on the build machine, not shipped) |
| Person mask and body points | `studio/engine/local/people.py` | Bazarevsky et al., BlazePose, CVPR Workshop on CV for AR/VR 2020 (arXiv 2006.10204); MediaPipe selfie segmentation. Anchors, alignment crop and decoding follow MediaPipe's published calculators (no code copied); the parts (skin, face, hair, clothes, arms, legs, torso) from body points, the face map and the person's own colours are our design |
| Composition overlays and suggested crops | `studio/engine/local/composition.py` | the methods themselves (rule of thirds, golden ratio and spiral, golden triangles, baroque and sinister diagonals, Westhoff's diagonal method, Hambidge's dynamic symmetry, the harmonic armature, rabatment); spectral-residual saliency (Hou and Zhang, CVPR 2007) for photos without people; the crop rules are our design |
| Iris finding and the eye tools | `studio/engine/local/eyes.py` | Ablavatski et al., MediaPipe Iris, CVPR Workshops 2020 (the eye crop from the eye corners, 2.3 x their distance, mirrored for the left eye, follows MediaPipe's published graph; no code copied). The classical fit (the iris as the dark disc of the size the eye predicts, ~11.7 mm across), the hybrid (model for where, edge for how big), eye colour in OKLab relative to the eye whites, and the red-eye / blown-pupil detection are our design. Limbal ring: Peshek et al., Evolutionary Psychology 2011 |
| Face detection and face mesh | `studio/engine/local/faces.py` | Bazarevsky et al., BlazeFace, CVPR Workshops 2019; Kartynnik et al., face mesh, CVPR Workshops 2019; Google, FaceMesh-V2 model card (2023). SSD anchors, weighted NMS and the rotated face crop follow MediaPipe's published calculators (no code copied); the brightened candle-light copy and the two-click eye route are our design |
| Lines (oriented grey-level closings), folds, under-eyes, even skin, eyes, teeth | `studio/engine/local/portrait.py` | mathematical morphology (Serra 1982; oriented line operators, Soille and Talbot, IEEE TPAMI 23(11), 2001; DullRazor hair removal, Lee et al. 1997, as the closing idea); guided filter (He, Sun, Tang 2013) via Smooth skin; zones, strengths and the colour-relative whitening are our design |
| Blemish finder | same | scale-space blob detection (Lindeberg 1998) on brightness and redness, with the ring texture and roundness tests of the dust finder; the mole rule is ours |
| Body sculpting (warps) | same | smooth radial / elliptical displacement fields inverted by fixed-point iteration, Catmull-Rom resampling (Catmull and Rom 1974; Keys 1981); the shape tools and caps are our design (no code from any liquify tool) |
| Smooth skin and the Skin mask | `studio/engine/local/masks.py` | guided-filter layers (He, Sun, Tang 2013) softening only the middle layer, after the findings of Shafaei, Little, Schmidt (AutoRetouch, WACV 2021) and Liang, Jin, Li (IEEE T-Cybernetics 44(12), 2014); no code or data from either (the FFHQR dataset is CC BY-NC-SA and is not used) |
| Detail by scale (tonal, crisp, broad) | `studio/engine/detail/scales.py` | fast local Laplacian filter (Paris, Hasinoff, Kautz, SIGGRAPH 2011; Aubry, Paris, Hasinoff, Kautz, Durand, ACM TOG 33(5), 2014) with the paper's smooth detail remapping; Laplacian pyramid (Burt and Adelson 1983). Per-level noise protection, shared budget, edge guards and the whole-frame proxy are our design. No code copied. |
| Detail by scale (tonal separation) | same | mean-shift step on local tone histograms (Fukunaga and Hostetler 1975; Comaniciu and Meer 2002), tiles blended as in contrast-limited adaptive histogram equalisation (Zuiderveld 1994). Our formulation. |
| Detail by scale (veil) | same | veiling glare as a uniform stray-light floor (idea: Chambers, "Micro Contrast and the ZEISS Pop", Lenspire 2017; McCann and Rizzi on veiling glare). Our own estimate and removal. |
| Denoise safeguard | `studio/engine/raw/denoise.py` | the "restore" step of train / denoise / restore (solar Self2Self, arXiv 2502.16807; number not yet verified), applied in the variance-stabilised domain. Our code. |
| Auto HDR: exposure fusion | `studio/engine/color/hdr.py` | Mertens, Kautz, Van Reeth, "Exposure Fusion", Pacific Graphics 2007, on virtual exposures of one frame as in HDR+ (Hasinoff et al., SIGGRAPH Asia 2016). Our code. |
| Auto HDR: local Laplacian | same | Paris et al. 2011 tone-compression remapping in the fast form of Aubry et al. 2014. Our code. |
| Auto HDR: Multi-Scale Retinex | same | Jobson, Rahman, Woodell, IEEE TIP 6(7), 1997 (NASA Langley; Retinex theory: Land 1977), with edge-keeping surrounds (guided filter, He, Sun, Tang 2013) in place of Gaussian ones. Our code. |
| Auto HDR: noise-aware lift, upsampling | same | idea of a noise-aware shadow-up function (arXiv 1811.03280; number not yet verified) with our calibrated noise model; fast guided filter upsampling (He and Sun, arXiv 1505.00996, 2015). Our code. |
| Scans: ICC matrix/TRC and grey profiles | `studio/engine/io/scanfile.py` | ICC.1:2010 profile format (tags rXYZ/gXYZ/bXYZ, rTRC/gTRC/bTRC, kTRC; curv and para types), read field by field; our code. A scan is put into scene-linear light by inverting the Studio's own standard curve, our design |
| Film negatives to positives | `studio/engine/local/film.py` | density above the film base (Hurter and Driffield 1890); per-layer normalisation by the frame's own density range, as darkroom colour printers balance neutral shadows and highlights and as darktable's negadoctor does (Aurélien Pierre, after Kodak Cineon printing density); our formulation and code, no code from darktable |
| Dust and scratches | `studio/engine/local/scratches.py` | grey-level top-hats (Serra 1982), hysteresis thresholds (Canny 1986) against the local texture, harmonic fill by red-black successive over-relaxation; the fusion with Microsoft's learned detector (Wan et al., CVPR 2020) as the judge of long marks, and the grain-matched fill, are our design |
| Colourize | `studio/engine/local/colourize.py`, `studio/engine/local/onnxnp.py` | DDColor (Kang et al., ICCV 2023) predicts CIE Lab a, b, run by our own NumPy ONNX-graph runner (matches ONNX Runtime to 0.0004); guided filter upsampling (He, Sun, Tang 2010/2013); keeping every pixel's L* and placing the colour in scene-linear light through the inverse standard curve is our design |
| Colour a part (colour hints) | `studio/engine/local/colourize.py` | geodesic distance on the image grid (Dijkstra 1959, SciPy's csgraph) as in geodesic segmentation and matting (Bai and Sapiro, ICCV 2007; Criminisi, Sharp, Blake, ECCV 2008), with a squared-brightness-step cost so shading passes and sharp edges stop; the brightness-only affinity follows Levin, Lischinski, Weiss, "Colorization using Optimization" (SIGGRAPH 2004); the colour laid as reflectance times the photo's own shading is our design. No code copied |
| Auto white balance: Grayness Index | `studio/engine/color/autowb.py` | Qian, Kamarainen, Nikkanen, Matas, "On Finding Gray Pixels", CVPR 2019. Re-implemented from the paper; the two-light split and "keep the mood" are ours. |
| Atmospheric scattering model (haze added or cleared by distance) | `studio/engine/local/depth.py` | Koschmieder 1924; as used in single-image dehazing by He, Sun, Tang, CVPR 2009 (dark channel prior not used: the depth comes from the depth model) |
| Colour statistics matched by solving for slider settings (Gauss-Newton with Broyden updates) | `studio/engine/local/colourmatch.py` | statistics in the spirit of Reinhard, Ashikhmin, Gooch, Shirley, IEEE CG&A 2001; Broyden, Math. Comp. 1965; OKLab (Ottosson 2020) |
| JPEG decoding by projection onto the quantisation intervals with a total-variation prior | `studio/engine/local/jpegfix.py` | Zakhor, IEEE TCSVT 1992; Yang, Galatsanos, Katsaggelos, IEEE TCSVT 1993; Alter, Durand, Froment, JMIV 2005; Bredies, Holler, 2012 |
| Lighting estimated with first-order spherical harmonics; relit by shading ratio | `studio/engine/local/relight.py` | Ramamoorthi, Hanrahan, SIGGRAPH 2001; Basri, Jacobs, IEEE PAMI 2003 |
| Composite: cut-out by click-to-select, affine placement, noise matched by the MAD of a fine Laplacian | `studio/engine/pipeline/composite.py` | noise estimate after Donoho, Johnstone 1994 (MAD) |
| Night white balance SGP-LRD, ported (vectorised; setting picked by agreement instead of the learned agent) from the authors' Apache-2.0 reference code `utils/Algorithm.py` (Copyright 2026 Yuan-Kang Lee, Kuan-Lin Chen; licence text in `studio/engine/color/LICENSE.apache-2.0`) | `studio/engine/color/nightwb.py` | Lee, Chen, Chang, Liu, "RL-AWB", arXiv 2601.05249, 2026 |
| Clean for scans: noise level function measured per starlet band as a lower envelope of block deviations, removed by the raw denoiser's band-wise empirical Wiener rule | `studio/engine/local/scanclean.py` | Liu, Szeliski, Kang, Zitnick, Freeman, IEEE PAMI 2008 (noise level function); Shin, Park, Lee 2005 (block-based estimate); Starck, Fadili, Murtagh 2007 (starlet) |

## Data

* ICC profiles are generated by our own code (`studio/engine/color/icc.py`); no third-party profile files are shipped. "Adobe RGB (1998) compatible" names the colour space; Adobe RGB is a trademark of Adobe.
* Camera colour matrices come from LibRaw (Adobe DNG Converter matrices compiled into LibRaw).
* No third-party image data is in the repository. Tests generate synthetic images. `tests/data/face_mesh_sample.json`
  holds 468 face-mesh points (numbers only, no image) measured on scikit-image's public-domain NASA portrait; one test
  uses that portrait only when scikit-image happens to be installed (it is not a dependency).
* `profiles/lenses/manual_lenses.json`: a starter list of manual lens names for tagging (our own list, CC BY-SA 4.0).
* `profiles/looks/*.psblook`: eight starter looks (our own settings, CC BY-SA 4.0). No third-party LUTs are shipped.
