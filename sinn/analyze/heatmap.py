# -*- coding: utf-8 -*-
"""
Created Sat Feb 25 2017

author: Alexandre René
"""

import numpy as np
import matplotlib.colors as colors

import sinn.analyze.common as anlzcom

class HeatMap:

    ParameterAxis = anlzcom.ParameterAxis

    def __init__(self, function_label, data, param_axes, norm='linear'):
        self.color_label = function_label
        self.data = data
        self.param_axes = param_axes
        self.floor = -np.inf
        self.ceil = np.inf
        self.set_norm(norm)
        self.cmap = 'viridis'


    def max(self):
        return self.data.max()
    def min(self):
        return self.data.min()
    def set_floor(self, floor):
        self.floor = floor
    def set_ceil(self, ceil):
        self.ceil = ceil
    def set_cmap(self, cmap):
        self.cmap = cmap
    def set_norm(self, norm):
        # Some common aliases
        if norm == 'lin':
            norm = 'linear'
        elif norm == 'logarithmic':
            norm = 'log'
        assert(norm in ['linear', 'log', 'symlog'])
        self.norm = norm
    def get_norm(self):
        if self.norm == 'linear':
            return colors.Normalize(vmin=self.floor, vmax=self.ceil)
        elif self.norm == 'log':
            return colors.LogNorm(vmin=self.floor, vmax=self.ceil)
        elif self.norm == 'symlog':
            raise NotImplementedError
        else:
            raise RuntimeError("What are we doing here ?")

