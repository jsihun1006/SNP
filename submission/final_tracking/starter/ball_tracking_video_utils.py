"""Pure marker-colour checks shared by the retained ball tracking renderers.

Extracted unchanged from the historical alternative correction renderer; the
alternative AIRBORNE/ATD replacement package is not part of the final pipeline.
"""
import numpy as np


def color_pixels(patch, code):
    if patch.size == 0:
        return 0
    b, g, r = np.moveaxis(patch.astype(float), -1, 0)
    if code == 1:
        mask = (g > 145) & (g > r + 45) & (g > b + 35)
    elif code == 2:
        mask = (r > 175) & (g > 95) & (g < 225) & (r > b + 80)
    elif code == 3:
        mask = (b > 160) & (g > 130) & (b > r + 65) & (g > r + 40)
    elif code == 4:
        mask = (r > 175) & (g > 150) & (b < 155) & (r > b + 70)
    else:
        return 0
    return int(mask.sum())
