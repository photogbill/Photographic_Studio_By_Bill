# What is in the next update

<!--
  The waiting list. One plain line per change, written for photographers, added as each part is built:
      - New: Focus Stack merges a focus bracket into one sharp photo.
      - Fixed: exports no longer replace the camera's own JPEG.
  Start each line with "- ". push.bat sends these lines as the comment; Release.bat then moves them into
  CHANGELOG.md under the new version number and empties this list. Leave this header where it is.
-->
- New: the Studio now knows which way is near for the Canon EOS R5's Dual Pixel halves (read from three of Bill's own scenes of ground running away from the camera), so depth from an R5 stack no longer has to guess which end is closest.
- New: Dual Pixel Lab can learn which way is near for any camera: pick photos of a path, lawn or table running away from you and press "Teach the Studio which way is near".
- Faster: Dual Pixel focus stacks are about 2.4 times faster per frame, and several frames are worked on at once on a PC with the cores and memory for it.
- Faster: a focus stack is analysed once and kept (the analysis cache), so making surfaces, leaves or another merge of the same stack later starts at once. Help has a setting for its size and a button to clear it.
- New: Dual Pixel stacks where only some frames are Dual Pixel now use the ones that are.
- New: "Measured depth" merge for Dual Pixel stacks (now their default): each place is taken from the frames the measured depth puts it on, so a blurred bright edge can no longer win as "sharp" and leave a halo.
- New: "Moving parts from one frame": where something moved between frames (a leaf in the wind), that place comes from one frame instead of a ghostly blend.
- New: the Depth view shows places no frame had in focus (amber stripes) and says where the focus jumped, so you know where to rock slower next time.
- New: a focus stack's result keeps its measured depth beside it, and Develop's depth mask uses that measured depth instead of guessing.
- New: more game maps from a surface: ambient occlusion, cavity, curvature, de-lit colour (albedo), fine relief, an ORM map for Godot, and a map of focus gaps; each says whether it was measured or estimated.
- New: tileable textures can now use a seam cut that keeps bricks and planks crisp (the old blend is still there for even textures).
- New: game-size textures (a square 2048 or 4096 from the best part of the picture) and real sizes from a coin in the frame.
- New: cut out a subject by its measured depth, so a leaf separates from leaf litter of the same colour.
- New: leaf cards for games: each leaf cut out with clean edges, a normal map, its stem as the pivot and a gently bent mesh, packed into an atlas and saved as glb files for Blender and Godot.
- New: the Studio writes glTF (.glb), OBJ and PLY files itself, and can build a Blender scene from a surface (a displaced, textured plane) with "Open in Blender".
- New: a 3D Scan section in the sidebar: point it at a folder of stacks and "Make them all" makes surfaces and leaves for every stack, carrying on where it stopped if you stop it.
- New: Card to Stacks can make surfaces for every stack it found.
- New: DP Benchmark.bat times Dual Pixel frames the old and the new way on your own stack.
- Fixed: a Dual Pixel stack whose frames share too little sharp texture to measure the focus step (leaves against a soft background) now takes its depth from focus alone and says so, instead of a scrambled height.
- New: a texture shot as one ordinary photo (or a burst) now gives game maps too, every one marked amber as estimated, not measured; the RPG Forge's texture folders get the same files as a measured stack.
- New: 3D models from hand-held stacks, in any order. The Studio places the views itself with its own camera solver, using each stack's Dual Pixel depth, so a few wide views round an object are enough (5 stacks 72 degrees apart place where ordinary photogrammetry fails). 3D Scan > tick the stacks of one object > Make a 3D model.
- New: each 3D model gives a textured glb for Blender and Godot (with game LODs), the camera positions, a point cloud, and a map of which parts were measured (cyan) and which were filled in from the outline (violet).
- New: a Blender dataset folder (pictures, measured depth, masks, cameras in Blender's and OpenCV's terms, points, model) for your own add-on, written only when the scan is good enough; otherwise it says why in plain words.
- New: the coach under each model says which side was never measured and how many more stacks to shoot there, and whether the underside or top was missed.
- New: focus breathing is allowed for between stacks: a macro lens is longer than marked when focused close (about 1.5x at 1:2), so each stack's own focal length is found from the photos.
- New: 3D Scan has an Untick all button, so picking the stacks of one object out of a full card takes a few clicks.
- New: one call per folder of photos (intake): the Studio works out which stacks are views of one object and makes them a 3D model, makes every other stack or photo a game texture, and lists everything in one file, so the RPG Forge (or any program) can hand over a folder and get finished assets back.
- New: a 3D model can be given its real length (the object's longest side) when you know it; otherwise it stays in relative units, the same shape at an unknown size.
- New: front-only reliefs. Each Dual Pixel stack (or photo) can become a textured relief mesh for Blender and Godot: the front of the subject as that stack saw it, cut out at its edges. 3D Scan > tick Front reliefs > Make them all. When stacks round an object are too few or too far apart for a full model, the intake makes these instead.
