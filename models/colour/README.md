# Colour model (DDColor)

`ddcolor.npz` is DDColor (Kang et al., ICCV 2023, Apache License 2.0, see `NOTICE`), from Qualcomm's ONNX export,
converted for the Studio's NumPy runner with 8-bit weights. Develop > Old photo and film > Colourize uses it to add
colour to a black and white photo: only colour is added, every pixel keeps its brightness, and the colours are a
plausible guess, not a record. It runs on this PC; nothing is sent anywhere.

To rebuild it from Qualcomm's download (needs `pip install onnx` on that PC):

    python tools\convert_onnx_model.py ddcolor-onnx-float.zip models\colour\ddcolor.npz
