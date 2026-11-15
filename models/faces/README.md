# Face models (Portrait section)

Google MediaPipe models, Apache License 2.0 (see NOTICE and LICENSE in this folder). Unmodified files; run on this PC
by `studio/engine/local/tinynet.py`. Photos never leave the PC.

| File | Model | Input | SHA-256 |
|---|---|---|---|
| face_detection_short_range.tflite | BlazeFace (faces filling a good part of the frame) | 128 x 128 | 3bc182eb9f33925d9e58b5c8d59308a760f4adea8f282370e428c51212c26633 |
| face_detection_full_range.tflite | BlazeFace (smaller faces) | 192 x 192 | 99bf9494d84f50acc6617d89873f71bf6635a841ea699c17cb3377f9507cfec3 |
| face_landmark.tflite | Face Mesh, 468 points | 192 x 192 | 2efcb4f4de43c7614b80a3cc3e8a37354b3b3b40f75cce20f6f38f0f25d65493 |
| iris_landmark.tflite | MediaPipe Iris: iris centre + 4 edge points, 71 eyelid and brow points | 64 x 64 (one eye) | d1744d2a09c25f501d39eba4faff47e53ecca8852c5ce19bce8eeac39357521f |
| face_landmarker_v2.tflite | Face Landmarker v2 (FaceMesh-V2): the same 468 face points plus 10 iris points | 256 x 256 | c7d54204ce0448474c7f3fa9af494787c0965cbdd6f20fc72867e43046bd43d5 |

If this folder is missing, Find faces says so; nothing else in the Studio needs it.

face_landmarker_v2.tflite is used in place of face_landmark.tflite when it is here (it is steadier on dark,
half-shadowed, noisy and tilted faces; see docs/RESULTS.md). Without it, or with the environment variable
STUDIO_FACE_MESH_V1 set, the first mesh is used as before. Its iris points are the first guess for the eye tools;
without them the iris model is used, and without iris_landmark.tflite a classical fit (the dark disc of the size the
eye predicts). In every case the iris edge itself decides the final size.

face_landmarker_v2.tflite is the file face_landmarks_detector.tflite taken unchanged out of Google's
`face_landmarker.task` bundle (a zip). The bundle itself, if it is here, is not used and is not part of the
repository: its other two models are a face detector (the Studio has its own two) and the blendshapes model
(expression scores, not needed).
