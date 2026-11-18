# What is in the next update

<!--
  The waiting list. One plain line per change, written for photographers, added as each part is built:
      - New: Focus Stack merges a focus bracket into one sharp photo.
      - Fixed: exports no longer replace the camera's own JPEG.
  Start each line with "- ". push.bat sends these lines as the comment; Release.bat then moves them into
  CHANGELOG.md under the new version number and empties this list. Leave this header where it is.
-->
- New: Card to Stacks (its own place in the sidebar). Point it at a card or a folder, or right-click photos in the Library: Find focus stacks. It sorts everything into focus stacks, bursts and single shots, leaves out the frames that would hurt each stack with the reason (soft everywhere, adds nothing, shaken, camera moved too far, exposure jump, subject moved), and Stack all merges every stack in the background. Nothing is deleted; tick a frame to put it back. Results are saved beside the frames as FIRST_to_LAST_Focus_Studio.tif.
- New: the import summary has a Find focus stacks button that opens Card to Stacks on what was just copied.
- New: bursts shot with AI Servo AF (a dog running at you, birds in flight) are recognised as bursts, not focus stacks, from what the camera recorded.
- New: Show what's in focus (Card to Stacks). Click a stack: cyan wherever some frame got it sharp, amber where there is detail but no frame got it sharp, so a missed band of focus shows before you stack. Click a frame: cyan where that frame is in focus.
- New: Sort focus stacks when importing from a card. The frames that would hurt a stack are still copied, into a "Left out" folder inside that day's folder, so they are out of sight but never lost.
- New: In focus (F) in Review: cyan over the sharpest detail of the photo, so a missed focus shows at a glance while you rate. It works zoomed in and in Compare too.
