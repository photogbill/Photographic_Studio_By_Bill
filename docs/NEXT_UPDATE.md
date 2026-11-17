# What is in the next update

<!--
  The waiting list. One plain line per change, written for photographers, added as each part is built:
      - New: Focus Stack merges a focus bracket into one sharp photo.
      - Fixed: exports no longer replace the camera's own JPEG.
  Start each line with "- ". push.bat sends these lines as the comment; Release.bat then moves them into
  CHANGELOG.md under the new version number and empties this list. Leave this header where it is.
-->
- New: the Studio can be driven by other programs (studio-cli.bat): every editing tool, colour match to a look, depth maps and layers, cut-outs and stacks, with results as data. Nothing changes in the Studio itself. See docs/CLI.md.
- New: an AI model can use the Studio directly (MCP). Double-click "Connect Studio to Claude.bat" to let the Claude desktop app develop, clean, match and cut out the pictures you point it to; the README says how to connect any other AI app, and what does and does not leave your PC.
