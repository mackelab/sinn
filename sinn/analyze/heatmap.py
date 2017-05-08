# -*- coding: utf-8 -*-
"""
Created Sat Feb 25 2017

author: Alexandre René
"""

import copy
import operator
import numpy as np
import matplotlib.colors as colors

from . import common as com

__ALL__ = ['HeatMap']

class HeatMap:

    ParameterAxis = com.ParameterAxis

    def __init__(self, zlabel, data, param_axes, norm='linear'):
        self.zlabel = zlabel
        self.data = data
        self.shape = data.shape
        self.axes = param_axes
        self.floor = -np.inf
        self.ceil = np.inf
        self.set_norm(norm)
        self.cmap = 'viridis'

        self._marginals = None
        self._normalized_data = None

    def raw(self):
        # The raw format is meant for data longevity, and so should
        # seldom, if ever, be changed
        raw = {}
        raw['axes'] = np.array([(ax.name, ax.stops, ax.idx, ax.scale) for ax in self.axes],
                               dtype=[('name', object), ('stops', object), ('idx', object), ('scale', object)])
        raw['data'] = self.data
        raw['zlabel'] = self.zlabel
        return raw

    @classmethod
    def from_raw(cls, raw):
        param_axes = []
        for ax in raw['axes']:
            param_axes.append(HeatMap.ParameterAxis(name = ax['name'],
                                                    stops = ax['stops'],
                                                    idx = ax['idx'],
                                                    scale = ax['scale'],
                                                    linearize_fn = com.scales[ax['scale']].linearize_fn,
                                                    inverse_linearize_fn = com.scales[ax['scale']].inverse_linearize_fn))
        heatmap = cls(str(raw['zlabel']), raw['data'], param_axes)
        return heatmap

    def max(self):
        return self.data.max()
    def min(self):
        return self.data.min()
    def argmax(self):
        """Return the tuple of parameter values that correspond to
        the heatmap's maximum.
        """
        index_tup = np.unravel_index(np.argmax(self.data), self.shape)
        return tuple([ax.stops[idx] for ax, idx in zip(self.axes, index_tup)])
    def set_floor(self, floor):
        self.floor = floor
    def set_ceil(self, ceil):
        self.ceil = ceil
    def set_cmap(self, cmap):
        self.cmap = cmap
    def set_zlabel(self, label):
        self.zlabel = label
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

    def __getitem__(self, key):
        assert(len(key) == len(self.axes))
        key = tuple( com.get_index_slice(ax, k) for ax, k in zip(self.axes, key) )
        return self.data[key]

    def slice(self, ax_slices):
        assert(len(ax_slices) == len(self.axes))
        ax_slices = [ com.get_index_slice(ax, slc)
                      for ax, slc in zip(self.axes, ax_slices) ]
            # Converts value slices to indices
        new_map = copy.deepcopy(self)
        for i, slc in enumerate(ax_slices):
            new_map.axes[i] = new_map.axes[i]._replace(stops=self.axes[i].stops[slc])

        if len(ax_slices) == 1:
            new_map.data = self.data[ax_slices[0]]
        elif len(ax_slices) == 2:
            new_map.data = self.data[ax_slices[0], ax_slices[1]]
        elif len(ax_slices) == 3:
            new_map.data = self.data[ax_slices[0], ax_slices[1], ax_slices[2]]
        else:
            raise NotImplementedError

        return new_map

    def get_normalized_data(self, recompute=False):
        if self._normalized_data is None or recompute:
            self._normalized_data = self.data / self.sum()
        return self._normalized_data

    def get_marginals(self, recompute=False):
        """
        Compute the marginals. These are cached, so subsequent calls are cheap.
        The cache can be recomputed (which one would want to do if the data has
        changed) by setting recompute=True
        """
        if self._marginals is None or recompute:
            p = self.get_normalized_data()
            dimlist = tuple( range(len(self.axes)) )
            self._marginals = [p.sum(axis=dimlist[:i] + dimlist[i+1:])
                               for i in range(len(dimlist))]
        return self._marginals

    def mean(self):
        """Compute the mean along each axis. Data does not need to be normalized."""
        estμ = np.array([(pm * ax.stops).sum()
                         for pm, ax in zip(self.get_marginals(), self.axes)])
        return estμ

    def cov(self):
        """Compute the empirical covariance matrix."""
        # TODO: use np.cov for speed
        estμ = self.mean()
        p = self.get_normalized_data()
        estdiagΣ = [ (pm * (ax.stops - μ)**2).sum()
                     for pm, ax, μ in zip(self.get_marginals(), self.axes, estμ) ]
        estΣ = np.diag(estdiagΣ)
        for i in range(len(self.axes)):
            for j in range(i+1, len(self.axes)):
                estΣ[i,j] = (p * np.outer( (self.axes[i].stops - estμ[i]),
                                           (self.axes[j].stops - estμ[j]) ) ).sum()
                estΣ[j,i] = estΣ[i,j]
        return estΣ

    def sum(self, axis=None, dtype=None, out=None):
        # This allows calling np.sum() on a heat map.
        return np.sum(self.data, axis=axis, dtype=dtype, out=out)


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
        return self.apply_op('abs({})'.format(self.zlabel),
                              operator.abs)
    def __add__(self, other):
        return self.apply_op(self.zlabel + '+' + str(other),
                              operator.add, other)
    def __radd__(self, other):
        return self.apply_op(str(other) + '+' + self.zlabel,
                              lambda a,b: b+a, other)
    def __sub__(self, other):
        return self.apply_op(self.zlabel + '-' + str(other),
                              operator.sub, other)
    def __rsub__(self, other):
        return self.apply_op(str(other) + '-' + self.zlabel,
                              lambda a,b: b-a, other)
    def __mul__(self, other):
        return self.apply_op(self.zlabel + '*' + str(other),
                              operator.mul, other)
    def __rmul__(self, other):
        return self.apply_op(str(other) + '*' + self.zlabel,
                              lambda a,b: b*a, other)
    def __matmul__(self, other):
        return self.apply_op(self.zlabel + '@' + str(other),
                              operator.matmul, other)
    def __rmatmul__(self, other):
        return self.apply_op(str(other) + '@' + self.zlabel,
                              lambda a,b: operator.matmul(b,a), other)
            # Using operator.matmul rather than @ prevents import fails on Python <3.5
    def __truediv__(self, other):
        return self.apply_op(self.zlabel + '/' + str(other),
                              operator.truediv, other)
    def __rtruediv__(self, other):
        return self.apply_op(str(other) + '/' + self.zlabel,
                              lambda a,b: b/a, other)
    def __floordiv__(self, other):
        return self.apply_op(self.zlabel + '//' + str(other),
                              operator.floordiv, other)
    def __rfloordiv__(self, other):
        return self.apply_op(str(other) + '//' + self.zlabel,
                              lambda a,b: b//a, other)
    def __mod__(self, other):
        return self.apply_op(self.zlabel + '%' + str(other),
                              operator.mod, other)
