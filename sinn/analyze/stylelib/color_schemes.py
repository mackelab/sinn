from collections import namedtuple
from cycler import cycler

property_cycles = {
    'dark pastel': ['#1e6ea7', '#9d3a11']
    }


# Heatmap color scheme structure
# Instead of populating this directly, we will generate a data dictionary, to allow some
# manipulation first
HeatmapColorScheme = namedtuple('HeatmapColorScheme',
                                ['white', 'black', 'accent1', 'accent1_cycle', 'accent2', 'accent2_cycle'])
# accent1 color list: First colour is 'bright' base. Subsequent have same hue and lightness, but decrease saturation by 10% / step
_cmaps_data = {
    'viridis': { 'white'   : '#F9DFFF',
                 'black'   : '#4f6066',
                 'accent1_list' : ['#EC093D', '#C9143D', '#A81C3C', '#8A2239', '#6E2535', '#562530', '#3F2329', '#2C1E21' ],
                 'accent2_list' : ['#4ED73D', '#44C534', '#42A736', '#408B37', '#3C7135', '#365932', '#2E432C', '#252F24' ]}
    }

# Interleave the accent lists to increase contrast between successive steps
for key in _cmaps_data:
    for clist in ['accent1_list', 'accent2_list']:
        _cmaps_data[key][clist] = _cmaps_data[key][clist][::2] + _cmaps_data[key][clist][1::2]

cmaps = {}
for key, values in _cmaps_data.items():
    cmaps[key] = HeatmapColorScheme(
        white   = values['white'],
        black   = values['black'],
        accent1 = values['accent1_list'][0],
        accent1_cycle = cycler('color', values['accent1_list']),
        accent2 = values['accent2_list'][0],
        accent2_cycle = cycler('color', values['accent2_list'])
    )
