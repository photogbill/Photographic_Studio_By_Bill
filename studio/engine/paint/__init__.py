"""Paint: the Studio's digital painting engine (BUILD_PLAN_PAINT.md). No Qt in this package.

tiles     the tiled canvas (256 px float32 premultiplied RGBA tiles), layers, the document and compositing
undo      undo and redo by the tiles a stroke changed
curves    pressure (and other input) curves
smoothing stroke smoothing modes (Basic, Weighted, Stabilizer, Dynamic, Pulled string, Speed-adaptive)
brush     the pixel brush engine: dabs, spacing, dynamics, wash / build-up / airbrush, eraser
colour    colour helpers (HSV, sRGB) and the colour picker
ora       OpenRaster (.ora) save and open
autosave  autosave and crash recovery, beside the app (studio/paths.py)
tablet    what the pen reports, and a plain-words diagnosis when pressure does not arrive
bench     the speed gate (plan 0.5)
"""
