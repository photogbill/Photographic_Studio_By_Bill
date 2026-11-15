# People models (Person mask)

Google MediaPipe models, Apache License 2.0 (see NOTICE and LICENSE in this folder). Unmodified files; run on this PC
by `studio/engine/local/tinynet.py`. Photos never leave the PC.

| File | Model | Input | SHA-256 |
|---|---|---|---|
| selfie_segmentation.tflite | person / not a person, whole photo | 256 x 256 | 9ee168ec7c8f2a16c56fe8e1cfbc514974cbbb7e434051b455635f1bd1462f5c |
| pose_detection.tflite | BlazePose detector | 224 x 224 | 9ba9dd3d42efaaba86b4ff0122b06f29c4122e756b329d89dca1e297fd8f866c |
| pose_landmark_full.tflite | BlazePose full: 33 body points + person mask | 256 x 256 | e9a5c5cb17f736fafd4c2ec1da3b3d331d6edbe8a0d32395855aeb2cdfd64b9f |
| selfie_multiclass_256x256.tflite (optional) | hair, body skin, face skin, clothes, accessories | 256 x 256 | c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0 |
| hair_segmenter.tflite (optional) | hair alone, on a crop around each head | 512 x 512 | 2628cf3ce5f695f604cbea2841e00befcaa3624bf80caf3664bef2656d59bf84 |

Without the optional multiclass model the Person mask still works: hair, skin, face and clothes are then worked out
from the body points, the face map and the person's own colours. With the optional hair model, the Hair part comes
from it (512 pixels across each head, so finer strands than the 256-pixel models).

If this folder is missing, the Person mask button says so; nothing else in the Studio needs it.
