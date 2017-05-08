from collections import namedtuple

property_cycles = {
    'dark pastel': ['#1e6ea7', '#9d3a11']
    }


HeatmapColorScheme = namedtuple('HeatmapColorScheme',
                                ['white', 'black', 'accent'])
cmaps = {
    'viridis': HeatmapColorScheme(
                 white  = '#F9DFFF',
                 black  = '#4f6066',
                 accent = '#EC093D')
    }
