"""The Lab: training a camera's own denoising network on the user's PC (BUILD_PLAN_NOISE block F).

Nothing here is needed to use the Studio. Training needs PyTorch, installed by "Train Camera Model.bat" into
<app>/lab_packages (about 2.5 GB with a graphics card build), which can be deleted at any time.
"""

# How many clean photos a camera's network needs, and why. One place, so the Cameras page, the training window and
# the model's card all say the same thing (Bill, 2026-11-18: "make sure that in the software it lets them know to
# have around 200 varied photos and explain why").
PHOTOS_WANTED = 200
PHOTOS_MINIMUM = 40
PHOTOS_WHY = (
    "The network learns two things at once: what your camera's noise looks like, and what real detail looks like. "
    "The noise comes from your dark frames, so that part is covered. The detail comes from your clean photos. "
    "With only a few, or all of one kind, it learns those subjects instead of detail in general: it gets very "
    "good at your garden and then smears or invents texture on a face, a brick wall or a night sky. Around 200 "
    "photos that differ from each other fix that: fine texture (foliage, fabric, hair, fur), smooth areas (sky, "
    "skin, walls), hard edges (buildings, text), deep shadow and bright highlights, day and night, near and far. "
    "They should be sharp, well exposed and at your lowest ISO (100 to 400), because they are the 'right answer' "
    "the network is taught toward; a noisy or blurred one teaches it noise or blur. More than 200 helps a little; "
    "variety helps much more than number. They are only read, never copied, changed or shared.")


def photos_verdict(n: int) -> tuple:
    """(colour tier, sentence) for a count of usable clean photos: 'good', 'thin' or 'too few'."""
    if n >= PHOTOS_WANTED:
        return "good", f"{n} clean photos: enough. Variety matters more than adding more."
    if n >= PHOTOS_MINIMUM:
        return "thin", (f"{n} clean photos: training will run, but around {PHOTOS_WANTED} varied ones is the aim. "
                        f"With this few the network may favour the subjects it saw.")
    return "few", (f"{n} clean photos: too few to train on (at least {PHOTOS_MINIMUM}, around {PHOTOS_WANTED} "
                   f"varied ones is the aim).")
