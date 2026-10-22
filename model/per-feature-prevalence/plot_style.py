"""Shared matplotlib styling for the per-feature-prevalence figures.

use_lexend() registers the Lexend font files installed on this machine
(~/Library/Fonts or /Library/Fonts) with matplotlib and makes Lexend the
default font. Lexend has no σ / β / θ glyphs, so DejaVu Sans is listed as a
fallback: matplotlib (>= 3.6) substitutes it glyph by glyph, and everything
else stays Lexend. If Lexend isn't installed, the defaults are left as-is.

Call it after importing matplotlib and before making figures:
    import plot_style; plot_style.use_lexend()
"""
import glob
import os

import matplotlib.pyplot as plt
from matplotlib import font_manager

_FONT_DIRS = [os.path.expanduser('~/Library/Fonts'), '/Library/Fonts']


def use_lexend():
    """Register Lexend and set it (with a DejaVu Sans glyph fallback) as the default font.
    Returns True if Lexend was found."""
    paths = [p for d in _FONT_DIRS for p in glob.glob(os.path.join(d, 'Lexend-*.ttf'))
             if 'Variable' not in os.path.basename(p)]      # static weights only
    if not paths:
        print('plot_style: Lexend not found; keeping matplotlib default font')
        return False
    for p in paths:
        font_manager.fontManager.addfont(p)
    plt.rcParams['font.family'] = ['Lexend', 'DejaVu Sans']
    return True
