# -*- coding: utf-8 -*-
"""
Created Sat Feb 25 2017

author: Alexandre René
"""

import copy
import operator
from collections import Iterable
import numpy as np
import scipy as sp
import logging
logger = logging.getLogger('sinn.analyze')
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
except ImportError:
    pass

import mackelab as ml
from . import common as com
from .axis import Axis
from .stylelib import color_schemes

__ALL__ = ['HeatMap']

class HeatMap:
    # TODO: Change name to ScalarGridData
    # changes: HeatMap -> ScalarGridData
    #          hm -> scalargriddata
    #          heat map -> scalar grid data

    def __init__(self, zlabel, ztype=None, data=None, param_axes=None, norm='linear'):
        """
        Parameters
        ----------
        [...]
        ztype: str
            The type of scalar data (the 'z' axis). One of:
              - 'mass': Values represent number of counts within a bin. This
                is typically what we get from a histogram.
              - 'density': Values represent a function or density at a particular
                point. This is what we get when we normalize a histogram, or when
                we evaluate a function over a grid.
            Note that the ztype says nothing about whether the values are normalized.
        """
        if isinstance(zlabel, HeatMap):
            src = zlabel
            ztype = src.ztype if ztype is None else ztype
            data = src.data if data is None else data
            param_axes = src.axes if param_axes is None else param_axes
            self.__init__(src.zlabel, ztype, data, param_axes, src.norm)
        else:
            assert( all(arg is not None for arg in [ztype, data, param_axes]) )
            self.zlabel = zlabel
            self.ztype = ztype
            self.data = data
            self.shape = data.shape
            self.axes = param_axes
            self._floor = None
            self._ceil = None
            self.set_norm(norm)
            self.cmap = 'viridis'
            self.op_res_type = type(self)

            assert( all( len(ax.centers.stops) == s
                         for ax, s in zip(self.axes, self.data.shape ) ) )
            #self._marginals = None
            #self._normalized_data = None

    @property
    def repr_np(self):
        return self.raw()

    @classmethod
    def from_repr_np(cls, repr_np):
        return cls.from_raw(repr_np)

    def raw(self):
        """
        Current format version: 3
        """
        # The raw format is meant for data longevity, and so should
        # seldom be changed
        raw = {}
        raw['type'] = self.__class__.__name__
        raw['version'] = 3
        # raw['axes'] = np.array([(ax.name, ax.stops, ax.idx, ax.scale) for ax in self.axes],
        #                        dtype=[('name', object), ('stops', object), ('idx', object), ('scale', object)])
        raw['axes'] = np.array([(ax.label.name, ax.transformed_label.name, ax.idx,
                                 ax.stops, ax.format(), ax.to.desc,
                                 ax.back.desc) for ax in self.axes],
                               dtype=[('label', object), ('transformed_label', object),
                                      ('label_idx', object), ('stops', object),
                                      ('format', object), ('transform_fn', object),
                                      ('inverse_transform_fn', object)])
        raw['data'] = self.data
        raw['zlabel'] = self.zlabel
        raw['ztype'] = self.ztype
        return raw

    @classmethod
    def from_raw(cls, raw):
        if not isinstance(raw, np.lib.npyio.NpzFile):
            raise TypeError("'raw' data must be a Numpy archive.")
        if 'version' in raw:
            version = raw['version']
        else:
            if 'ztype' not in raw:
                version = 1
            else:
                version = 2
        param_axes = []
        for ax in raw['axes']:
            #if set(['name', 'stops', 'idx', 'scale']).issubset(ax.keys()):
            if version == 1:
                # Old format
                if ax['scale'] in ['lin', 'linear']:
                    to_desc = 'x -> x'
                    back_desc = 'x -> x'
                elif ax['scale'] == 'log':
                    to_desc = 'x -> np.log10(x)'
                    back_desc = 'x -> 10**x'
                else:
                    raise RuntimeError("Unrecognized scale '{}' in deprecated file type."
                                       .format(ax['scales']))
                axis = Axis(ax['name'], "f(" + ax['name'] + ")",
                            label_idx = ax['idx'],
                            stops = ax['stops'], format = 'centers',
                            transform_fn = to_desc,
                            inverse_transform_fn = back_desc)
            else:
                axis = Axis(**{name: ax[name] for name in ax.dtype.names})
            param_axes.append(axis)
        ztype = raw['ztype'] if version >= 2 else 'density'
        heatmap = cls(str(raw['zlabel']), ztype, raw['data'], param_axes)
        return heatmap

    @property
    def ndim(self):
        return len(self.axes)
    @property
    def floor(self):
        return self._floor if self._floor is not None else self.min()
    @property
    def ceil(self):
        return self._ceil if self._ceil is not None else self.max()
    def max(self):
        return self.data.max()
    def min(self):
        return self.data.min()
    def argmax(self):
        """Return the tuple of parameter values that correspond to
        the heatmap's maximum.
        This is defined as the center of the bin with the highest value.
        """
        index_tup = np.unravel_index(np.argmax(self.data), self.shape)
        return tuple([ax.stops.centers[idx] for ax, idx in zip(self.axes, index_tup)])
    def set_floor(self, floor):
        self._floor = floor
    def set_ceil(self, ceil):
        self._ceil = ceil
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

    def __str__(self):
        return self.zlabel

    def convert_ztype(self, ztype):
        if ztype == 'density':
            return self.density
        elif ztype == 'mass':
            return self.mass
        else:
            raise ValueError("Unrecognized ztype '{}'.".format(ztype))
    @property
    def density(self):
        if self.ztype == 'density':
            return self
        else:
            assert(self.ztype == 'mass')
            densitydata = self.data.astype(np.float64)
            # For each axis, divide by the width of each bin
              # Using broadcasting (rather than making an Nd weight array)
              # is memory efficient. 'reshape' aligns the axis widths with
              # the corresponding data dimension for broadcasting
            for i in range(self.ndim):
                densitydata /= self.axes[i].widths.reshape( (-1,) + (1,)*(self.ndim-i-1) )
            hm = type(self)(self.zlabel, 'density', densitydata,
                            self.axes, self.norm)
            if self._ceil is not None:
                hm.set_ceil(hm.max() / self.max() * self.ceil)
            if self._floor is not None:
                hm.set_floor(hm.min() / self.min() * self.floor)
            return hm
    @property
    def mass(self):
        if self.ztype == 'mass':
            return self
        else:
            assert(self.ztype == 'density')
            massdata = self.data.astype(np.float64)
            # For each axis, multiply by the width of each bin
            for i in range(self.ndim):
                massdata *= self.axes[i].widths.reshape( (-1,) + (1,)*(self.ndim-i-1) )
            hm = type(self)(self.zlabel, 'mass', massdata,
                            self.axes, self.norm)
            if self._ceil is not None:
                hm.set_ceil(hm.max() / self.max() * self.ceil)
            if self._floor is not None:
                hm.set_floor(hm.min() / self.min() * self.floor)
            return hm

    def axis_stops(self, format='current'):
        assert(all(isinstance(p, Axis) for p in self.axes))
        return [p.format(format).stops for p in self.axes]
    @property
    def axis_labels(self):
        assert(all(isinstance(p, Axis) for p in self.axes))
        def get_label(param):
            if param.label_idx is None:
                return param.name
            else:
                assert(isinstance(param.label_idx, (tuple, int)))
                return param.name + "[" + str(tuple(param.label_idx))[1:-1] + "]"
                    # [1:-1] removes the tuple parentheses
        return [get_label(p) for p in self.axes]

    def bounds(self, format='edges'):
        return [(stops[0], stops[-1]) for stops in self.axis_stops(format)]

    def slice(self, ax_slices):
        assert(len(ax_slices) == len(self.axes))
        ax_slices = [ com.get_index_slice(ax, slc)
                      for ax, slc in zip(self.axes, ax_slices) ]
            # Converts value slices to indices
        new_map = copy.deepcopy(self)
        for i, slc in enumerate(ax_slices):
            new_map.axes[i].stops = new_map.axes[i].stops[slc]

        if len(ax_slices) == 1:
            new_map.data = self.data[ax_slices[0]]
        elif len(ax_slices) == 2:
            new_map.data = self.data[ax_slices[0], ax_slices[1]]
        elif len(ax_slices) == 3:
            new_map.data = self.data[ax_slices[0], ax_slices[1], ax_slices[2]]
        else:
            raise NotImplementedError

        return new_map

    def _get_axis_idcs(self, axes=None):
        """
        Returns a tuple, such as to be compatible with np.sum & co.
        """
        if axes is None:
            axes = tuple(range(len(self.axes)))
        else:
            if isinstance(axes, str) or not isinstance(axes, Iterable):
                axes = (axes,)
            assert(len(np.unique(axes)) == len(axes))
                # Ensure there are no duplicate axes
            assert(len(axes) <= self.ndim and max(axes) < self.ndim)
            # Convert any names to indices
            axes = list(axes)
            axisnames = [ax.name for ax in self.axes]
            for i in range(len(axes)):
                if isinstance(axes[i], str):
                    axis_idcs = [j for j, nm in enumerate(axisnames) if nm == axes[i]]
                    if len(axis_idcs):
                        raise NotImplementedError("Heatmaps with multiple axes with same "
                                                  "name are not currently supported.")
                    else:
                        axes[i] = axis_idcs[0]
            axes = tuple(axes)
        return axes

    def meshgrid(self, axes=None, format='centers'):
        """
        Return a grid of the heat map's domain, in the same form as meshgrid.

        Parameters:
        -----------
        axes: list
            (Optional) The axes to include in the grid; each axis can be
            specified either with it's name or index. This argument may be
            used both to produce a grid for only a subset of axes and to
            change the order of the axes as they appear in the grid. Default
            is the reproduce this heat map's domain.
        format: str
            The format of the produced grid. Possible values are:
              - 'centers': (default) The grid values correspond to the centers
                of each discretization cell. This is the format used internally
                by heat map, and is best suited if the goal is to evaluate
                a function over its domain, or otherwise produce a new HeatMap.
              - 'edges': The grid values correspond to the edges of each
                discretization cell. This is the format expected by many
                plotting functions, such as matplotlib's `pcolormesh`.
            Some minor variations (such as the singular 'center') are also
            recognized but should not be relied up.

        Returns
        list of arrays
             If there are n axes, the returned list contains n arrays.
             Each has n dimensions, and is composed of tuples on length n.
        """

        axes = self._get_axis_idcs(axes)

        if format in ['centers', 'centres', 'center', 'centre']:
            gridvalues = [self.axes[idx].centers.stops for idx in axes]
        elif format in ['edges', 'edge']:
            gridvalues = [self.axes[idx].edges.stops for idx in axes]
        else:
            raise ValueError("Unrecognized mesh format '{}'. Please use one "
                             " of 'centers', 'edges'.".format(format))

        return np.meshgrid(*gridvalues, indexing='ij')

    def eval(self, f, label=None, ztype='density'):
        """
        Evaluate the scalar function `f` over this heat map's domain.
        Result is returned as another heat map.

        NOTE: The current implementation Make more efficient by using broadcasting rather than meshgrid.

        Parameters
        ----------
        f: callable
            Function taking as many inputs as there are axes to this heat map,
            and returning a scalar.
        """
        hm_shape = tuple(self.shape)
        #data = np.fromiter( (f(*x) for x in zip(*[grid.flat for grid in meshgrid(self)])),
        #                     dtype=float, count=np.prod(hm_shape) ).reshape(hm_shape)
        vecf = np.vectorize(f, otypes=[np.float64])
        data = vecf(*self.meshgrid(format='centers'))
        if label is None:
            label = ""
        return HeatMap(label, ztype, data, self.axes)

    def get_normalized_data(self, recompute=False):
        logger.warning("Deprecation warning: 'get_normalized_data' is planned for removal.")
        #if self._normalized_data is None or recompute:
        return self.data / self.sum()
        #return self._normalized_data

    def sum(self, axis=None, dtype=None, out=None):
        # Call signature allows calling np.sum() on a heat map.
        """
        Returns
        -------
        float or HeatMap
            Returns a scalar if all axes are summed, otherwise
            a heat map for the remaining axes.
        """
        axis = self._get_axis_idcs(axis)
            # Allow specifying summation axes by name
        res = np.sum(self.data, axis=axis, dtype=dtype, out=out)
        if res.ndim == 0:
            return res
        else:
            remaining_axes = [ax for i, ax in enumerate(self.axes) if i not in axis]
            return type(self)(self.zlabel, self.ztype, res,
                              remaining_axes, self.norm)

    def crop(self, bounds):
        """
        TODO: Check if flip of data if an axis or a bound is in descending
        order but not the other is OK.
        TODO: Merge with slicing

        Parameters
        ----------
        bounds: list of tuples
            List of (low, high) tuples, which what a call to `self.bounds` returns.

        Returns
        -------
        HeatMap
            The original data is not copied: the new instances data is a view of
            the original.
        """
        bound_arr = np.array(bounds)
        assert(bound_arr.ndim == 2 and bound_arr.shape[1] == 2)
        new_axes = []
        new_data = self.data

        for axi, (ax, axbounds) in enumerate(zip(self.axes, bounds)):
            # crop ax1 such that smallest element of ax1 >= smallest element of ax2
            # and largest element of ax1 <= largest element of ax2
            if ax.stops[0] < ax.stops[-1]:
                # axis is in ascending order
                left, right = 'left', 'right'
            else:
                left, right = 'right', 'left'
            lowi = np.searchsorted(ax.stops, min(axbounds), side=left)
            highi = np.searchsorted(ax.stops, max(axbounds), side=right)
            if (left == 'right') != (axbounds[0] > axbounds[-1]):
                # The bounds and axis are not in the same order
                step = -1
                start = max(lowi, highi)
                stop = min(lowi, highi)
            else:
                step = 1
                start = min(lowi, highi)
                stop = max(lowi, highi)
            new_axes.append( Axis(
                ax.label, ax.transformed_label,
                stops=ax.stops[start:stop:step],
                transform_fn = ax.to,
                inverse_transform_fn = ax.back) )
            new_data = np.swapaxes(
                np.swapaxes(new_data, axi, 0)[start:stop:step],
                0, axi)
                # Swap axis to known location, slice, swap back

        return type(self)(self, data=new_data, param_axes=new_axes)



    def collapse(self, axes, method=None):
        """
        Collapse the data such that only the specified axes remain.

        Derived classes may redefine this method to provide additional
        collapse method or change the default. This is how for example
        the Probability class allows to collapse by marginalizing.
        NOTE: When changing the default, do so dynamically, rather
        than in the signature (i.e., leave the argument as 'method=None').
        This allows the default to be selected even when calling from a
        method in the parent class.

        Parameters
        ----------
        method: str
            Which method to use to collapse. If None (default), an
            error is raised if there are any unplotted dimensions.
            The other possible value is:
              - 'sum': Sum along collapsed dimensions.
            Additional methods may be defined in derived classes.
        axes: list of axis indices (int) or names (str)
            The axes that should be left after collapsing.
        """
        axes = self._get_axis_idcs(axes)  # Convert all 'axes' to a list of ints
        if len(axes) < self.ndim:
            # Collapse the extra dimensions
            if collapse_method is None:
                raise ValueError("There are unplotted axes and no collapse method "
                                 "was specified.")
            elif collapse_method == 'sum':
                axes_to_sum = set(axes).difference(np.arange(self.ndim))
                return self.sum(axis=axes_to_sum)

        else:
            return self

    #####################################################
    # Operator definitions
    #####################################################
    # TODO: Operations with two heat maps ?

    def apply_op(self, new_label, op, b=None):
        """
        Apply the operator specified by `op` on the scalar value at every point
        of the domain. If `b` is given, it is provided as a second argument to `op`.
        `b` may be another HeatMap instance; in this case it is aligned with
        this heat map and the operation is carried out element-wise. Note that in
        this case the operation is not symmetric: this heat map's grid is used,
        and the other is interpolated onto it.

        The result is returned as a new HeatMap, whose data array is the result
        of the operation.

        FIXME: At present the result is always a heat map of same
        type as 'self'. This may not always be desired, for example when
        multiplying a probability.

        Parameters
        ----------
        new_label: str
            Label to assign to the new heat map.
        op: callable
            Operation to apply.
        b: (Optional)
            Second argument to provide to `op`, if required. If a HeatMap, it
            is cropped and aligned with this heat map's domain, and the operation
            performed element-wise.

        Returns
        -------
        HeatMap
            Will have the same (possibly cropped) axis stops as 'self', as well as
            the same norm.
        """

        if b is None:
            return self.op_res_type(new_label, self.ztype, op(self.data),
                                    self.axes, self.norm)
        else:
            if isinstance(b, HeatMap):
                # Check that heat map types are compatible:
                # subclasses of HeatMap can only operate on heat maps
                # of the same or of a parent type.
                if not (issubclass(type(b), type(self))
                        or issubclass(type(self), type(b))):
                    raise TypeError("Operations are not supported between operands "
                                    "of type '{}' and '{}'.".format(
                                        str(type(self)), str(type(b))))
                # Check that axes are of the same variables
                axisnames = [ax.name for ax in self.axes]
                othernames = [ax.name for ax in b.axes]
                bcopy = copy.deepcopy(b).convert_ztype(self.ztype)
                if set(axisnames) != set(othernames):
                    raise ValueError(
                        "To apply an operation between two heatmaps, the axes "
                        "must match. Here, the axes {} occur only in one."
                        .format(', '.join(["'"+name+"'" for name in set(axisnames).symmetric_difference(set(othernames))])))
                # Ensure that the axes are in the same order
                for i, name in enumerate(axisnames[:-1]):
                    if othernames[i] != name:
                        j = othernames.index(name)
                        assert(j > i) # All axes before i should already be ordered
                        np.swapaxes(bcopy.data, i, j)
                        bcopy.axes[i], bcopy.axes[j] = bcopy.axes[j], bcopy.axes[i]
                        othernames[i], othernames[j] = othernames[j], othernames[i]
                assert(axisnames[-1] == othernames[-1] == bcopy.axes[-1].name)

                # Crop the axis ranges
                # Otherwise if our parameter range is bigger than b, we can't interpolate
                cropped = self.crop(bcopy.bounds('edges'))

                # new_axes = copy.deepcopy(self.axes)
                # new_data = self.data.copy()

                # for axi, (ax1, ax2) in enumerate(zip(new_axes, bcopy.axes)):
                #     # crop ax1 such that smallest element of ax1 >= smallest element of ax2
                #     # and largest element of ax1 <= largest element of ax2
                #     if ax1.stops[0] < ax1.stops[-1]:
                #         # axis is in ascending order
                #         left, right = 'left', 'right'
                #     else:
                #         left, right = 'right', 'left'
                #     lowi = np.searchsorted(ax1.stops, ax2.stops.min(), side=left)
                #     highi = np.searchsorted(ax1.stops, ax2.stops.max(), side=right)
                #     start = min(lowi, highi)
                #     stop = max(lowi, highi)
                #     new_axes[axi] = new_axes[axi]._replace(
                #         stops = ax1.stops[start:stop])
                #     new_data = np.swapaxes(
                #         np.swapaxes(new_data, axi, 0)[start:stop],
                #         0, axi)
                #         # Swap axis to known location, slice, swap back

                # Interpolate the other heat map
                shape = tuple(len(ax.stops) for ax in cropped.axes)
                res_map = self.op_res_type(new_label, self.ztype, np.zeros(shape),
                                           cropped.axes, self.norm)
                eval_points = res_map.meshgrid(format='centers')
                interp_bdata = sp.interpolate.interpn(
                    [ax.stops for ax in bcopy.axes],
                    bcopy.data,
                    np.stack(a.flatten() for a in eval_points).T).reshape(shape)

                res_map.data = op(cropped.data, interp_bdata)
                return res_map

            else:
                return self.op_res_type(new_label, self.ztype, op(self.data, b),
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

    ########################
    # Plotting
    ########################

    def heatmap(self, axes=None, collapse_method=None, **kwargs):
        """
        TODO: Support plotting more than 2 axes

        Parameters
        ----------
        axes: list of str or ints
            Axes to plot, specified with either their index or their name.

        collapse_method: str
            What to do with unplotted dimensions. If None (default), an
            error is raised if there are any unplotted dimensions.
            The base HeatMap implementation only provides:
              - 'sum': Sum along collapsed dimensions.
            Derived data types (e.g. Probability) may provide additional methods.

        **kwargs: Other keyword arguments are passed on to `plt.pcolormesh`.

        Returns
        -------
        plot axis, colorbar axis
        """
        axes = self._get_axis_idcs(axes)
        if len(axes) == 2:
            return self.heatmap2d(axes, collapse_method, **kwargs)
        else:
            raise NotImplementedError("Only 2d heatmaps are currently implemented.")

    def heatmap2d(self, axes=None, collapse_method=None, **kwargs):
        assert(len(axes) == 2)
        axes = self._get_axis_idcs(axes)  # Convert all 'axes' to a list of ints
        datahm = self.collapse(collapse_method, axes)
        data = np.moveaxis(datahm.data, axes, np.arange(len(axes)))
            # Permute the data for the given axes

        ax1_grid, ax2_grid = datahm.meshgrid(axes=axes, format='edges')
        zmin = datahm.floor #max(datahm.floor, data.min())
        zmax = datahm.ceil  #min(datahm.ceil, data.max())
        quadmesh = plt.pcolormesh(ax1_grid, ax2_grid,
                                  data.clip(datahm.floor, datahm.ceil),
                                  cmap = datahm.cmap,
                                  norm = datahm.get_norm(),
                                  vmin=zmin, vmax=zmax,
                                  **kwargs)
        ax = plt.gca()
        plt.xlabel(datahm.axes[axes[0]].name)
        plt.ylabel(datahm.axes[axes[1]].name)

        cb = plt.colorbar()
        cb.set_label(datahm.zlabel)

        color_scheme = color_schemes.cmaps[datahm.cmap]
        ax.tick_params(axis='both', which='both', color=color_scheme.white,
                        top='on', right='on', bottom='on', left='on',
                        direction='in')
        cb.ax.tick_params(axis='y', which='both', left='off', right='off')

        for side in ['top', 'right', 'bottom', 'left']:
            ax.spines[side].set_color('none')
            cb.outline.set_visible(False)

        return ax, cb

    def plot_stddev_ellipse(self, width, **kwargs):
        """
        Add an ellipse to a plot denoting a heatmap's spread. This function
        is called after plotting the data, and adds the
        ellipse to the current axis.

        Parameters
        ----------
        width: float
            Amount of data to include in the ellipse, in units of standard
            deviations. A width of 2 will draw the contour corresponding
            to 2 standard deviations.
        **kwargs:
            Keyword arguments passed to maptplotlib.patches.Ellipse
        """
        # TODO: Deal with higher than 2D heatmaps
        # FIXME: Only works with Probability (requires mean, cov)

        eigvals, eigvecs = np.linalg.eig(self.cov())
        sort_idcs = np.argsort(abs(eigvals))[::-1]
        eigvals = eigvals[sort_idcs]
        eigvecs = eigvecs[:,sort_idcs]
        ax = plt.gca()
        w = width * np.sqrt(eigvals[0])
        h = width * np.sqrt(eigvals[1])
        color = kwargs.pop('color', None)
        if color is None:
            color_scheme = color_schemes.cmaps[self.cmap]
            color = color_scheme.accents[1]  # Leave more salient accents[0] for user
        e = mpl.patches.Ellipse(xy=self.mean(), width=w, height=h,
                                angle=np.arctan2(eigvecs[1,0], eigvecs[0,0])*180/np.pi,
                                fill=False, color=color, **kwargs)
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)

# The following two methods are deprecated; use the properties instead
def get_axes(param_axes):
    assert(all(isinstance(p, Axis) for p in param_axes))
    return [p.stops for p in param_axes]
def get_axis_labels(param_axes):
    assert(all(isinstance(p, Axis) for p in param_axes))
    def get_label(param):
        if param.label_idx is None:
            return param.name
        else:
            assert(isinstance(param.label_idx, (tuple, int)))
            return param.name + "[" + str(tuple(param.label_idx))[1:-1] + "]"
                # [1:-1] removes the tuple parentheses
    return [get_label(p) for p in param_axes]


#################################
#################################
#
# Specialized heat maps
#
#################################
#################################

class LogLikelihood(HeatMap):

    def __init__(self, zlabel, ztype=None, data=None, param_axes=None, norm='linear', depth=100):
        """
        Parameters
        ----------
        zlabel, ztype, data, param_axes, norm
            See `HeatMap`.
        depth: int
            Likelihoods this many orders of magnitude less than the maximum
            are considered zero. Effectively sets the difference between
            `ceil` and `floor`. Default is 100.
        """
        if isinstance(zlabel, LogLikelihood):
            src = zlabel
            ztype = src.ztype if ztype is None else ztype
            data = src.data if data is None else data
            param_axes = src.axes if param_axes is None else param_axes
            self.__init__(src.zlabel, ztype, data, param_axes, src.norm, src.depth)
        else:
            if not isinstance(zlabel, HeatMap):
                assert( all(arg is not None for arg in [ztype, data, param_axes]) )
            super().__init__(zlabel, ztype, data, param_axes, norm=norm)
            self._floor = None
            self._ceil = None
            self.depth = depth

    @property
    def floor(self):
        # TODO: Ignore _floor and always use depth ?
        if self._floor is None:
            return self.ceil - self.depth
    @property
    def ceil(self):
        if self._ceil is None:
            return self.data.max()

    def likelihood(self, depth=None, normalize=True):
        """
        Return the a new heat map corresponding to the likelihood.
        Parameters
        ----------
        depth: int
            (Optional) Number of (base e) orders magnitudes to consider.
        normalize: bool
            (Optional) If True, normalize the result such that it has
            total probability one. Default is True.
        """
        if depth is None:
            depth = self.depth

        if self.zlabel[:3].lower() == 'log':
            zlabel = self.zlabel[3:].strip()
        else:
            zlabel = "exp(" + self.zlabel + ")"
        Ldata = np.exp(self.data - self.data.max())
        L = Likelihood(zlabel, self.ztype, Ldata,
                       self.axes, norm='linear', normalized=normalize)
        L.set_ceil(L.max())
        L.set_floor(0)
        return L

class Probability(HeatMap):
    def __init__(self, zlabel, ztype=None, data=None, param_axes=None, 
                 norm='linear', normalized=True):
        """
        Parameters
        ----------
        zlabel, ztype, data, param_axes, norm
            See `HeatMap`.
        normalized: bool
            If set to True (default), the data is automatically normalized
            during creation such that the integral over the heat map is 1.
        """
        if isinstance(zlabel, Probability):
            src = zlabel
            ztype = src.ztype if ztype is None else ztype
            data = src.data if data is None else data
            param_axes = src.axes if param_axes is None else param_axes
            self.__init__(src.zlabel, ztype, data, param_axes, src.norm, src.normalized)
            self.op_res_type = HeatMap
        else:
            if not isinstance(self, HeatMap):
                assert( all(arg is not None for arg in [ztype, data, param_axes]) )
            super().__init__(zlabel, ztype, data, param_axes, norm)
            if normalized:
                self.data = self.data / super().mass.sum()
                    # Use super's `mass` property, which doesn't check that data is normalized
                self.normalized = True
            else:
                self.normalized = False

    @property
    def mass(self):
        res = super(Probability, self).mass
        if self.normalized:
            assert(np.isclose(res.sum(), 1))  # Essential that this always be true
        else:
            assert(0 <= res.sum() <= 1)
        return res

    def collapse(self, axes, method=None):
        if method is None:
            method = 'marginalize'
        axes = self._get_axis_idcs(axes)
        if len(axes) < self.ndim:
            if collapse_method == 'marginalize':
                return self.marginalize(axes)
            else:
                return super().collapse(axes, method)
        else:
            return self

    def marginalize(self, axis, warn_if_no_action=True):
        # 'axis' matches the signature of 'sum'
        """
        Return a new heat map over only the specified axes; the others
        are marginalized out.
        If no marginalization is performed (i.e. if every axis is listed in `axis`),
        a warning is sent to the logger. This can be disabled by setting `warn_if_no_action`
        to True.

        Parameters
        ----------
        axis: int, tuple of ints
            The axis/axes to keep. In other words, the marginalization is carried
            out over all *other* axes.

        warn_if_no_action: bool
            Set to False to disable printing a warning if no action is taken. Default is True.

        Returns
        -------
        HeatMap of dimension d, where d is the number axes in `axis`.
        """
        ax_idcs = self._get_axis_idcs(axis)
        if len(ax_idcs) == self.ndim:
            # Nothing to do
            if warn_if_no_action:
                logger.warning("Marginalizing over every axis does nothing.")
            return self
        remaining_idcs = set(range(self.ndim)).difference(ax_idcs)
        return self.mass.sum(axis=remaining_idcs).convert_ztype(self.ztype)

    def mean(self):
        """Compute the mean along each axis. Data does not need to be normalized."""
        # Same pattern as in 'density'
        # TODO: Use message passing to optimize sum-product ?

        return np.array([ (self.axes[i].centers.stops * self.mass.marginalize(i).data).sum()
                          for i in range(self.ndim) ])

    def cov(self):
        """Compute the empirical covariance matrix."""
        # TODO: use np.cov for speed
        estμ = self.mean()
        #p = self.get_normalized_data()
        marginals = [self.mass.marginalize(i).data for i in range(self.ndim)]
        assert( all(pm.ndim == 1 for pm in marginals) )
        estdiagΣ = [ (pm * (ax.centers.stops - μ)**2).sum()
                     for pm, ax, μ in zip(marginals, self.axes, estμ) ]
        estΣ = np.diag(estdiagΣ)
        for i in range(len(self.axes)):
            for j in range(i+1, len(self.axes)):
                p = self.mass.marginalize((i, j), warn_if_no_action=False).data
                estΣ[i,j] = (p * np.outer( (self.axes[i].centers.stops - estμ[i]),
                                           (self.axes[j].centers.stops - estμ[j]) ) ).sum()
                estΣ[j,i] = estΣ[i,j]
        return estΣ


class Likelihood(Probability):
    pass


try:
    import mackelab.iotools
except ImportError:
    pass
else:
    mackelab.iotools.register_datatype(HeatMap)
