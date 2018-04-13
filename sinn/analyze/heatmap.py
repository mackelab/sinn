# This is module only for backwards compatibility

from .griddata import *

# For backwards compatibility
HeatMap = ScalarGridData
HeatMap.heatmap = ScalarGridData.plot
HeatMap.heatmap2d = ScalarGridData.plot_histogram2d
ml.iotools.register_datatype(HeatMap, 'HeatMap')
