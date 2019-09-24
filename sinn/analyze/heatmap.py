# This is module only for backwards compatibility

from sinn.axisdata import *

HeatMap = ScalarAxisData
HeatMap.heatmap = ScalarAxisData.plot
HeatMap.heatmap2d = ScalarAxisData.plot_histogram2d
mtb.iotools.register_datatype(HeatMap, 'HeatMap')
