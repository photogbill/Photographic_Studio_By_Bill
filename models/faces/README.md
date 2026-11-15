# Face models (Portrait section)

Google MediaPipe models, Apache License 2.0 (see NOTICE and LICENSE in this folder). Unmodified files; run on this PC
by `studio/engine/local/tinynet.py`. Photos never leave the PC.

| File | Model | Input | SHA-256 |
|---|---|---|---|
| face_detection_short_range.tflite | BlazeFace (faces filling a good part of the frame) | 128 x 128 | 3bc182eb9f33925d9e58b5c8d59308a760f4adea8f282370e428c51212c26633 |
| face_detection_full_range.tflite | BlazeFace (smaller faces) | 192 x 192 | 99bf9494d84f50acc6617d89873f71bf6635a841ea699c17cb3377f9507cfec3 |
| face_landmark.tflite | Face Mesh, 468 points | 192 x 192 | 2efcb4f4de43c7614b80a3cc3e8a37354b3b3b40f75cce20f6f38f0f25d65493 |

If this folder is missing, Find faces says so; nothing else in the Studio needs it.
