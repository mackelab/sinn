# This is module only for backwards compatibility

from .axisdata import *

HeatMap = ScalarAxisData
HeatMap.heatmap = ScalarAxisData.plot
HeatMap.heatmap2d = ScalarAxisData.plot_histogram2d
ml.iotools.register_datatype(HeatMap, 'HeatMap')
