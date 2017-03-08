from collections import namedtuple

qualitative = {
    'dark pastel': ['#1e6ea7', '#9d3a11']
    }


HeatmapColorScheme = namedtuple('HeatmapColorScheme',
                                ['white', 'black'])
map = {
    'viridis': HeatmapColorScheme(
                 white = '#F9DFFF',
                 black = '#4f6066')
    }
