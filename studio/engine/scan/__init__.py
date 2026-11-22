"""3D Scan: models from hand-held Dual Pixel stacks with the Studio's own camera position solver
(BUILD_PLAN_DP3D E, F, H).

* ``views``: one view = one stack (or one Dual Pixel photo): its all-in-focus picture, fused depth, confidence,
  subject mask and camera.
* ``sift``: oriented, scale-normalised features (difference of Gaussians, gradient-histogram descriptors, RootSIFT).
* ``geom``: two-view and many-view geometry (essential matrix, triangulation, PnP, similarity).
* ``twoview``: a pair's relative pose from the matches and from the measured depth, compared.
* ``solve``: every view placed (incremental, order never assumed) and bundle-adjusted with depth terms.
* ``dense``, ``fusion``, ``mcubes``, ``texture``, ``decimate``: the mesh (M6).
* ``model``: the whole job and its files; ``raster``: a software turntable for the app; ``coach``: what to shoot next.

No Qt here.
"""
