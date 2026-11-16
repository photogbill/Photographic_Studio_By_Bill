# Scratch detector

`scratch_detector.npz` is Microsoft's scratch detector from *Bringing Old Photos Back to Life* (MIT licence, see
`NOTICE`), converted for the Studio by `tools/convert_scratch_detector.py`. Develop > Old photo and film > Remove dust
and scratches uses it to decide which long marks are damage (a scratch, a crack, a hair) and which are part of the
picture (a collar, a wire). Without it the Studio still removes dust and scratches with its own filter, only more
cautiously.

To rebuild it from Microsoft's download:

    python tools\convert_scratch_detector.py path\to\global_checkpoints.zip
