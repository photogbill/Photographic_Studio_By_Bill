# Depth model (Depth Anything V2 Small)

`depth_anything_v2_small.npz` is Depth Anything V2 Small (Yang et al., NeurIPS 2024, Apache License 2.0, see
`NOTICE`), converted for the Studio's NumPy runner with 8-bit weights. It guesses how far away each part of a photo
is, from the one photo: an order (near to far), not metres. Develop > Cognitive Adjustments uses it for Depth haze
and the depth view, and Masks > Depth selects by distance. It runs on this PC; nothing is sent anywhere. About 3
seconds per photo, once; the map is kept with the photo.

Only the Small model is Apache 2.0; the bigger Depth Anything V2 models are non-commercial and are not used.

To rebuild it from the download (Convert Models.bat does this for you; needs onnx and onnxsim):

    python tools\convert_onnx_model.py model.onnx models\depth\depth_anything_v2_small.npz --fold pixel_values=1,3,518,518
