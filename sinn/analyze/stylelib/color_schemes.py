from collections import namedtuple
import logging

import numpy as np
from cycler import cycler
import matplotlib as mpl

logger = logging.getLogger('sinn.analyze.stylelib.color_schemes')

def monochrome_palette(basecolor, nstops, s_range=(1, 0.3), v_range=(1, 1.3), absolute=False):
    """
    Produce an array of variations on the base colour by changing the
    saturation and/or the value. The result is a set of colours that have
    the same hue but different brightness.
    The `basecolor` is internally converted to HSV, where the S and V
    components are varied. The result is returned as RGB hex strings.

    Parameters
    ----------
    n_steps: int
        Number of colour stops to create in the palette.
    s_range: tuple of floats
        (begin, end) values for the *saturation* component of the palette.
        Range is inclusive.
        The values specified are relative to the base color's, so values greater than
        1 may be possible. Typical usage however has the base color as the brightest,
        which is achieved be setting `begin` to 1.
        Default: (1, 0.3)
    v_range: tuple of floats
        (begin, end) values for the *value* component of the palette.
        Range is inclusive.
        Values are relative, same as for `s_range`.
        Default: (1, 1.3)
    absolute: bool
        If True, range values are absolute rather than relative. In this case
        the saturation and value of the `basecolor` are discarded, and range values
        must be between 0 and 1.

    Examples
    --------
    Palette that varies towards white (Default):
        `s_range` = (1, 0.3)
        `v_range` = (1, 1.3)

    Palette that varies towards black:
        `s_range` = (1, 1)
        `v_range` = (1, 0.4)
    """
    def clip(val, varname):
        if val < 0:
            val = 0
            logger.warning("[monochrome_palette]: " + varname +
                           " was smaller than 0 and was clipped.")
        elif val > 1:
            val = 1
            logger.warning("[monochrome_palette]: " + varname +
                           " was greater than 1 and was clipped.")
        return val

    if isinstance(basecolor, tuple):
        if any( v>1 for v in basecolor ):
            raise ValueError("If you are defining the basecolor by an "
                             "RGB tuple, the values must be between 0 and 1. "
                             "Specified basecolor: {}.".format(str(basecolor)))
    basergb = mpl.colors.to_rgb(basecolor)
    h, s, v = mpl.colors.rgb_to_hsv(basergb)
    if absolute:
        s = 1; v = 1
    s_range = (clip(s_range[0] * s, 'saturation'), clip(s_range[1] * s, 'saturation'))
    v_range = (clip(v_range[0] * v, 'value'),      clip(v_range[1] * v, 'value'))

    slist = [a*s for a in np.linspace(s_range[0], s_range[1], nstops)]
    vlist = [a*v for a in np.linspace(v_range[0], v_range[1], nstops)]
    clist = [mpl.colors.to_hex(mpl.colors.hsv_to_rgb((h, s_el, v_el)))
                                                     for s_el, v_el in zip(slist, vlist)]
    return clist

property_cycles = {
    'dark pastel': ['#1e6ea7', '#9d3a11']
    }


# Heatmap color scheme structure
# Instead of populating this directly, we will generate a data dictionary, to allow some
# manipulation first
HeatmapColorScheme = namedtuple('HeatmapColorScheme',
                                ['white', 'black', 'min', 'max', 'accents', 'accent_cycles'])
    # white: colour to use instead of white (typically an off-white colour that complements the colour map)
    # black: colour to use instead of black
    # min:   colour corresponding to the low end of the colour map
    # max:   colour corresponding to the high end of the colour map
    # accent1: first colour of accent1_cycle. Provides good contrast with the colour map
    # accent1_cycle: variations on accent1 to provide a cycle of associated colours
    # accent2: different accent colour from colour 1; typically less contrasts less with the colour map
    # accent2_cycle: variations on accent2

# accent1 color list: First colour is 'bright' base. Subsequent have same hue and lightness, but decrease saturation by 10% / step
_cmaps_data = {
    'viridis': { 'white'   : '#F9DFFF',
                 'black'   : '#4f6066',
                 'accents'  : ['#EC093D', '#D6F5FF', '#BFBF00'] }
    }

# Create colour cycles from the accent colours
viridis_accents = _cmaps_data['viridis']['accents']
_cmaps_data['viridis']['accent_lists'] = [
    monochrome_palette(viridis_accents[0], 6, (1, 0.3), (1, 1.2)),
    monochrome_palette(viridis_accents[1], 6, (1, 1), (1, 1)),
    monochrome_palette(viridis_accents[2], 6, (1, 1), (1, 1.3))
    ]

# Interleave the accent lists to increase contrast between successive steps
# for key in _cmaps_data:
#     for clist in ['accent1_list', 'accent2_list']:
#         _cmaps_data[key][clist] = _cmaps_data[key][clist][::2] + _cmaps_data[key][clist][1::2]

cmaps = {}
for key, values in _cmaps_data.items():
    cmaps[key] = HeatmapColorScheme(
        white   = values['white'],
        black   = values['black'],
        min     = mpl.colors.to_hex(mpl.cm.get_cmap('viridis').colors[0]),
        max     = mpl.colors.to_hex(mpl.cm.get_cmap('viridis').colors[-1]),
        accents = values['accents'],
        accent_cycles = [cycler('color', clist) for clist in values['accent_lists']]
    )
