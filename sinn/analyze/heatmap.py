# -*- coding: utf-8 -*-
"""
Created Sat Feb 25 2017

author: Alexandre René
"""

from copy import copy
from collections import OrderedDict
import numpy as np
import matplotlib.colors as colors

from . import common as com

class HeatMap:

    ParameterAxis = com.ParameterAxis

    def __init__(self, function_label, data, param_axes, norm='linear'):
        self.color_label = function_label
        self.data = data
        self.axes = param_axes
        self.floor = -np.inf
        self.ceil = np.inf
        self.set_norm(norm)
        self.cmap = 'viridis'

    def raw(self):
        # The raw format is meant for data longevity, and so should
        # seldom, if ever, be changed
        raw = OrderedDict([(ax.name, ax.stops) for ax in self.axes])
        raw['data'] = self.data
        raw['color_label'] = self.color_label
        return raw

    @classmethod
    def from_raw(cls, raw):
        param_axes = []
        for key, val in raw:
            if key not in ['data', 'color_label']:
                # This is a list of parameter stops
                # We default to a linear scale since that info is not saved
                param_axes.append(ParameterAxis(name = key,
                                                stops = val,
                                                idx = None,
                                                scale = 'linear',
                                                linearize_fn = lambda x:x,
                                                inverse_linearize_fn = lambda x:x))
        heatmap = cls(str(raw['color_label']), raw['data'], param_axes)
        return heatmap

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

    def slice(self, ax_slices):
        assert(len(ax_slices) == len(self.axes))
        new_map = copy(self)
        for i, slc in enumerate(ax_slices):
            new_map.axes[i] = new_map.axes[i]._replace(self.axes[i].stops[slc])

        if len(ax_slices) == 1:
            new_map.data = self.data[ax_slices[0]]
        elif len(ax_slices) == 2:
            new_map.data = self.data[ax_slices[0], ax_slices[1]]
        elif len(ax_slices) == 3:
            new_map.data = self.data[ax_slices[0], ax_slices[1], ax_slices[2]]
        else:
            raise NotImplementedError

    #####################################################
    # Operator definitions
    #####################################################
    # TODO: Operations with two heat maps ?

    def apply_op(self, new_label, op, b=None):
        if b is None:
            return HeatMap(new_label, op(self.data),
                           self.axes, self.norm)
        else:
            return HeatMap(new_label, op(self.data, b),
                           self.axes, self.norm)

    def __abs__(self):
        return self.apply_op('abs({})'.format(self.function_label),
                              operator.abs)
    def __add__(self, other):
        return self.apply_op(self.function_label + '+' + str(other),
                              operator.add, other)
    def __radd__(self, other):
        return self.apply_op(str(other) + '+' + self.function_label,
                              lambda a,b: b+a, other)
    def __sub__(self, other):
        return self.apply_op(self.function_label + '-' + str(other),
                              operator.sub, other)
    def __rsub__(self, other):
        return self.apply_op(str(other) + '-' + self.function_label,
                              lambda a,b: b-a, other)
    def __mul__(self, other):
        return self.apply_op(self.function_label + '*' + str(other),
                              operator.mul, other)
    def __rmul__(self, other):
        return self.apply_op(str(other) + '*' + self.function_label,
                              lambda a,b: b*a, other)
    def __matmul__(self, other):
        return self.apply_op(self.function_label + '@' + str(other),
                              operator.matmul, other)
    def __rmatmul__(self, other):
        return self.apply_op(str(other) + '@' + self.function_label,
                              lambda a,b: operator.matmul(b,a), other)
            # Using operator.matmul rather than @ prevents import fails on Python <3.5
    def __truediv__(self, other):
        return self.apply_op(self.function_label + '/' + str(other),
                              operator.truediv, other)
    def __rtruediv__(self, other):
        return self.apply_op(str(other) + '/' + self.function_label,
                              lambda a,b: b/a, other)
    def __floordiv__(self, other):
        return self.apply_op(self.function_label + '//' + str(other),
                              operator.floordiv, other)
    def __rfloordiv__(self, other):
        return self.apply_op(str(other) + '//' + self.function_label,
                              lambda a,b: b//a, other)
    def __mod__(self, other):
        return self.apply_op(self.function_label + '%' + str(other),
                              operator.mod, other)