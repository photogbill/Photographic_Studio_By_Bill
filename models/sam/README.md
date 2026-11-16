# Click-to-select models (SAM 2.1)

Meta's Segment Anything 2.1 (Apache License 2.0, see `NOTICE`), converted for the Studio's NumPy runner. Masks >
Object and Colour a part use it: click a thing and it is outlined, a second click adds to it, Alt-click takes a part
away. It only says where the mask goes; nothing is painted or invented. It runs on this PC; nothing is sent anywhere.

Choose the size in Masks > Click-to-select:

* **Tiny** (shipped): looks at a photo in about 10 seconds on a desktop CPU, once per photo; each click after that is
  instant.
* **Large**: finer outlines of hair, leaves and thin parts; looks at a photo several times slower. It is too big to
  ship, so it is made on your PC: put `sam2.1_hiera_large_20260221.zip` in the `models` folder and run
  `Convert Models.bat` (several minutes, once).
* **Classic**: the Studio's own colour-and-edge method; no model, instant.
