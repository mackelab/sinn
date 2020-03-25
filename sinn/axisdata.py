# -*- coding: utf-8 -*-
"""
Created Sat Feb 25 2017

author: Alexandre René
"""

import copy
import operator
from collections import OrderedDict, Iterable, Sequence, namedtuple
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

from parameters import ParameterSet

import sinn
import mackelab_toolbox as mtb
import mackelab_toolbox.utils
import mackelab_toolbox.iotools
from mackelab_toolbox.stylelib import colorschemes

from . import common as com
from .axis import Axis

__ALL__ = ['ScalarAxisData']

class ScalarAxisData:
    # TODO: Change name to ScalarAxisData
    # changes: HeatMap -> ScalarArrayData
    #          hm -> scalararraydata
    #          heat map -> scalar array data
    # TODO: Remove 'cmap' attribute from data object;
    #       keep it within the plotting functions

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
        if isinstance(zlabel, ScalarAxisData):
            src = zlabel
            ztype = src.ztype if ztype is None else ztype
            data = src.data if data is None else data
            param_axes = src.axes if param_axes is None else param_axes
            self.__init__(src.zlabel, ztype, data, param_axes, src.norm)
        else:
            if any(arg is None for arg in [ztype, data, param_axes]):
                raise ValueError("Unless an ArrayData object is provided as first "
                                 "argument, 'ztype', 'data' and 'param_axes' are "
                                 "required arguments.")
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

            if len(self.axes) != self.data.ndim:
                # TODO: Better error type
                raise TypeError("There must be as many axes are there are "
                                "data dimensions.\n"
                                "Data dimensions: {}\n"
                                "# axes: {}".format(len(self.axes),
                                                    self.data.ndim))
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
        return tuple([ax.centers.stops[idx] for ax, idx in zip(self.axes, index_tup)])
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

    def meshgrid(self, axes=None, format='centers', transformed=False):
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
                a function over its domain, or otherwise produce a new ScalarAxisData.
              - 'edges': The grid values correspond to the edges of each
                discretization cell. This is the format expected by many
                plotting functions, such as matplotlib's `pcolormesh`.
            Some minor variations (such as the singular 'center') are also
            recognized but should not be relied upon.
        transformed: bool
            Whether to use the stops of the transformed or non-transformed
            axis. Default (False) is to use those of the non-transformed axis.

        Returns
        -------
        list of arrays
             If there are n axes, the returned list contains n arrays.
             Each has n dimensions, and is composed of tuples on length n.
        """

        axes = [self.axes[idx] for idx in self._get_axis_idcs(axes)]
        if transformed:
            axes = [axis.transformed for axis in axes]

        if format in ['centers', 'centres', 'center', 'centre']:
            gridvalues = [axis.centers.stops for axis in axes]
        elif format in ['edges', 'edge']:
            gridvalues = [axis.edges.stops for axis in axes]
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
        return ScalarAxisData(label, ztype, data, self.axes)

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
        float or ScalarAxisData
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
        ScalarAxisData
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
        `b` may be another ScalarAxisData instance; in this case it is aligned with
        this heat map and the operation is carried out element-wise. Note that in
        this case the operation is not symmetric: this heat map's grid is used,
        and the other is interpolated onto it.

        The result is returned as a new ScalarAxisData, whose data array is the result
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
            Second argument to provide to `op`, if required. If a ScalarAxisData, it
            is cropped and aligned with this heat map's domain, and the operation
            performed element-wise.

        Returns
        -------
        ScalarAxisData
            Will have the same (possibly cropped) axis stops as 'self', as well as
            the same norm.
        """

        if b is None:
            return self.op_res_type(new_label, self.ztype, op(self.data),
                                    self.axes, self.norm)
        else:
            if isinstance(b, ScalarAxisData):
                # Check that heat map types are compatible:
                # subclasses of ScalarAxisData can only operate on heat maps
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

    def plot(self, axes=None, collapse_method=None, transformed=False,
             ax=None, **kwargs):
        """
        TODO: Support plotting more than 2 axes

        Parameters
        ----------
        axes: list of str or ints
            Axes to plot, specified with either their index or their name.

        collapse_method: str
            What to do with unplotted dimensions. If None (default), an
            error is raised if there are any unplotted dimensions.
            The base ScalarAxisData implementation only provides:
              - 'sum': Sum along collapsed dimensions.
            Derived data types (e.g. Probability) may provide additional methods.

        **kwargs: Other keyword arguments are passed on to `plt.pcolormesh`.

        Returns
        -------
        plot axis, colorbar axis
        """
        if ax is None: ax = plt.gca()

        axes = self._get_axis_idcs(axes)
        if len(axes) == 2:
            return self.plot_histogram2d(axes, collapse_method, transformed,
                                         ax=ax, **kwargs)
        else:
            raise NotImplementedError("Only 2d heatmaps are currently implemented.")

    def plot_histogram2d(self, axes=None, collapse_method=None,
                         transformed=False, colorbar=True, ax=None, **kwargs):
        assert(len(axes) == 2)
        if ax is None: ax = plt.gca()

        # Use a default value of True for `rasterized`: vectorized heatmaps take up a lot
        # more memory (which can prevent compilation by LaTeX) and don't usually look
        # better than rasterized versions anyway.
        if 'rasterized' not in kwargs:
            kwargs['rasterized'] = True

        axes = self._get_axis_idcs(axes)  # Convert all 'axes' to a list of ints
        datahm = self.collapse(collapse_method, axes)
        data = np.moveaxis(datahm.data, axes, np.arange(len(axes)))
            # Permute the data for the given axes

        ax1_grid, ax2_grid = datahm.meshgrid(axes=axes, format='edges',
                                             transformed=transformed)
        zmin = datahm.floor #max(datahm.floor, data.min())
        zmax = datahm.ceil  #min(datahm.ceil, data.max())
        quadmesh = ax.pcolormesh(ax1_grid, ax2_grid,
                                  data.clip(datahm.floor, datahm.ceil),
                                  cmap = datahm.cmap,
                                  norm = datahm.get_norm(),
                                  vmin=zmin, vmax=zmax,
                                  **kwargs)
        ax.set_xlabel(datahm.axes[axes[0]].name)
        ax.set_ylabel(datahm.axes[axes[1]].name)

        color_scheme = colorschemes.cmaps[datahm.cmap]
        ax.tick_params(axis='both', which='both', color=color_scheme.white,
                       top='on', right='on', bottom='on', left='on',
                       direction='in')

        for side in ['top', 'right', 'bottom', 'left']:
            ax.spines[side].set_color('none')

        if colorbar:
            cb = ax.get_figure().colorbar(quadmesh, ax=ax)
            cb.set_label(datahm.zlabel)
            cb.ax.tick_params(axis='y', which='both', left='off', right='off')

            for side in ['top', 'right', 'bottom', 'left']:
                cb.outline.set_visible(False)

        if colorbar:
            return ax, cb
        else:
            return ax

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
            color_scheme = colorschemes.cmaps[self.cmap]
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

class LogLikelihood(ScalarAxisData):

    def __init__(self, zlabel, ztype=None, data=None, param_axes=None, norm='linear', depth=100):
        """
        Parameters
        ----------
        zlabel, ztype, data, param_axes, norm
            See `ScalarAxisData`.
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
            if not isinstance(zlabel, ScalarAxisData):
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
        else:
            return self._floor
    @property
    def ceil(self):
        if self._ceil is None:
            return self.data.max()
        else:
            return self._ceil

    def likelihood(self, depth=None, normalize=True):
        """
        Return a new heat map corresponding to the likelihood.
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
            zlabel = "exp " + self.zlabel
        Ldata = np.exp(self.data - self.data.max())
        L = Likelihood(zlabel, self.ztype, Ldata,
                       self.axes, norm='linear', normalized=normalize)
        L.set_ceil(L.max())
        L.set_floor(0)
        return L

class LogProbability(LogLikelihood):
    def probability(self, *args):
        # TODO: Analog to `likelihood()`
        raise NotImplementedError

class Probability(ScalarAxisData):
    # TODO: Make Probability subclass of Likelihood rather than other way around ?

    def __init__(self, zlabel, ztype=None, data=None, param_axes=None,
                 norm='linear', normalized=True):
        """
        Parameters
        ----------
        zlabel, ztype, data, param_axes, norm
            See `ScalarAxisData`.
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
            self.op_res_type = ScalarAxisData
        else:
            if not isinstance(self, ScalarAxisData):
                assert( all(arg is not None for arg in [ztype, data, param_axes]) )
            super().__init__(zlabel, ztype, data, param_axes, norm)
            if normalized:
                self.data = self.data / super().mass.sum()
                    # Use super's `mass` property, which doesn't check that data is normalized
                self.normalized = True
            else:
                self.normalized = False

    def logprobability(self, depth=None):
        """
        Return a new heat map corresponding to the log probability.

        Parameters
        ----------
        depth: float
            (Optional) Number of (base e) orders magnitudes to consider,
            i.e. sets the floor for what values are considered 'zero'.
            Any value more than `depth` orders of magnitudes less than the
            data's maximum is considered zero.
            If unspecified, the floor is set to the data's smallest non-zero value."
            Note: The log transformation requires a strictly positive floor.
        """
        # Ensure data is strictly positive by 'raising' zero elements to the floor
        if depth is not None:
            floor = self.max() * np.exp(-depth)
        else:
            floor = self.data[self.data.nonzero()].min()
        raiseddata = np.where(self.data > 0, self.data, floor)

        if self.zlabel[:3].lower() == 'exp':
            zlabel = self.zlabel[3:].strip()
        else:
            zlabel = "log " + self.zlabel
        logPdata = np.log(raiseddata)
        logP = LogProbability(zlabel, self.ztype, logPdata,
                              self.axes, norm='linear')
        logP.set_ceil(logP.max())
        logP.set_floor(logP.min())
        return logP

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
        ScalarAxisData of dimension d, where d is the number axes in `axis`.
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

    def plot_stddev_ellipse(self, width, ax=None, **kwargs):
        """
        Add an ellipse to a plot denoting a heatmap's spread. This function
        can be called after plotting the data, and adds the
        ellipse to the current axis.

        .. note:: This function assumes the probability to be a multivariate
                  Gaussian.

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
        if ax is None: ax = plt.gca()

        eigvals, eigvecs = np.linalg.eig(self.cov())
        sort_idcs = np.argsort(abs(eigvals))[::-1]
        eigvals = eigvals[sort_idcs]
        eigvecs = eigvecs[:,sort_idcs]
        w = width * np.sqrt(eigvals[0])
        h = width * np.sqrt(eigvals[1])
        color = kwargs.pop('color', None)
        if color is None:
            color_scheme = colorschemes.cmaps[self.cmap]
            color = color_scheme.accents[1]  # Leave more salient accents[0] for user
        e = mpl.patches.Ellipse(xy=self.mean(), width=w, height=h,
                                angle=np.arctan2(eigvecs[1,0], eigvecs[0,0])*180/np.pi,
                                fill=False, color=color, **kwargs)
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)


class Likelihood(Probability):
    def logprobability(self, *args):
        raise NotImplementedError("Log probability is undefined for likelihoods. "
                                  "Use `loglikelihood()`.")
    def loglikelihood(self, *args):
        # TODO: Should basically call super().logprobability but return
        #       a LogLikelihood rather than a LogProbability
        raise NotImplementedError("Function planned for future implementation.")


#################################
#################################
#
# Collection
#
#################################
#################################

class MarginalCollection:
    ParamDim = namedtuple('ParamDim', ['modelname', 'transformedname',
                                       'tracename', 'displayname',
                                       'desc', 'longdesc',
                                       'idx', 'flatidx',
                                       'to_desc', 'back_desc'])
    Marker = namedtuple('Marker', ['pos', 'color', 'α', 'size'])
    AxisFormat = namedtuple('AxesFormat', ['key', 'scale', 'visible', 'apply'])

    # TODO: Remove flat_idx ?

    def __init__(self, data, params, maxexpand=3, markers=None,
                 histogram_kwargs=None, colorscheme=None,
                 #key_sanitizer=None,
                 **kwargs):
        """
        Construct a collection of ScalarAxisData objects. Internally maintains a dictionary
        of the axes, so that specific axes can be formatted; this dictionary is an
        instance of mtb.utils.SanitizedOrderedDict, to allow "close-enough" indexing.

        Parameters
        ---------------
        data: sequence of ScalarAxisData | PyMC3 MultiTrace
            The data for which we want to compute marginals.
        params: dict
            Dictionary of ParamDim elements for which we want the marginals.
            Preferably a sanitized dictionary, to make accessing values easier.
        maxexpand: float
            If markers fall outside the boundary of the 2D histograms, these
            will be extended sufficiently to include the markers, up to a
            factor `maxexpand`. Bounds will not be expanded if not required to
            include a marker.
            Default value of 3 is useful for exploration – it allows to see
            how far true values may be from the posterior, without reducing
            the posterior to a single point.
            For publication plots, a value of 1 is recommended: this ensures
            that 2D plots line up with the 1D plots on the grid's diagonal.
        markers:
            List of points where a marker will be plotted on top of the heat maps
            (each point is an array with as many elements as the data has dimensions).
        colorscheme: str | HeatmapColorScheme
            Passed on to `set_colorscheme()`.
        histogram_kwargs: dictionary
            Dictionary of keyword arguments to pass to `numpy.histogramdd`.
        key_sanitizer: [REMOVED] function, or list of characters
            This argument is passed to mtb.utils.SanitizedOrderedDict, when creating the
            internal axes dictionary.
            Taken from `params`
        **kwargs:
            Additional keyword arguments are passed to the ScalarAxisData constructor.
        """

        # Try to import PyMC3 MultiTrace
        # We do this here because it's a relatively heavy import which most
        # of the time won't be used.
        try:
            import pymc3
            pymc_MultiTrace = pymc3.backends.base.MultiTrace
        except ImportError:
            pymc_MultiTrace = None

        # Set internal variables
        if maxexpand < 1:
            maxexpand = 1
            logger.warning("`maxexpand` argument must not be less than 1.")
        self.params = params

        # Get the parameters and data
        if isinstance(data, Sequence) and isinstance(data[0], ScalarAxisData):
            # Use Sequence instead of Iterable to ensure `data[0]` doesn't eat the first element
            raise NotImplementedError("Building a collection from ScalarAxisData objects is not yet implemented.")

        elif pymc_MultiTrace is not None and isinstance(data, pymc_MultiTrace):
            marginals1D, marginals2D = self.marginals_from_mcmc(data, params,
                                                                histogram_kwargs,
                                                                **kwargs)
        else:
            raise ValueError("Unsupported data type '{}'.".format(type(data)))

        self.marginals1D = marginals1D
        self.marginals2D = marginals2D
        self.maxexpand = maxexpand
        # Colorscheme must be set before markers
        if colorscheme is None:
            self.set_colorscheme()
        else:
            self.set_colorscheme(colorscheme)
        self.set_markers(markers)
        self.set_transformed(False)
        # Set dictionary of functions to apply to specific plot axes (aka spines)
        if isinstance(params, mtb.utils.SanitizedDict):
            self._axes_format = mtb.utils.SanitizedOrderedDict(sanitize=params.sanitize)
        else:
            self._axes_format = mtb.utils.SanitizedOrderedDict()

    @staticmethod
    def marginals_from_mcmc(traces, params, histogram_kwargs, threshold=1e-5, **kwargs):
        """
        Construct the 1D and 2D marginals for MCMC traces

        Parameters
        ---------------
        traces: PyMC3 MultiTrace
            The MCMC trace or traces for which we want to compute marginals.
        params: list of ParamDim
            Flat list of ParamDim
        histogram_kwargs: dictionary
            Dictionary of keyword arguments to pass to `numpy.histogramdd`.
        threshold: float
            To remove outliers, the range of the marginal is allowed to shrink. `Threshold`
            is the maximum total probability mass that may be removed, with either end of
            the interval removing at most have that value.
            This behaviour can be overridden by providing a value for 'range' in `histogram_kwargs`.
            NOTE: It's currently only possible to provide a single set of histogram keyword
            arguments, so if range is provided, all plots will have the same.
        **kwargs:
            Additional keyword arguments are passed to the ScalarAxisData constructor.
        """
        # TODO: Allow parameter-specific histogram_kwargs (e.g. range, weights)
        flat_params = params
        if threshold <= 0:
            raise ValueError("`threshold` must be strictly positive")

        histogram_kwargs = dict(histogram_kwargs) if histogram_kwargs is not None else {}


        # Get indices to the flattened array, skipping masked components
        flat_idcs = [0]
        keys = list(flat_params.keys())
        for i in range(1, len(keys)):
            if flat_params[keys[i]].tracename == flat_params[keys[i-1]].tracename:
                flat_idcs.append(flat_idcs[i-1] + 1)
            else:
                flat_idcs.append(0)

        # TODO: Rather than stack traces and create new array, just create
        #       a list of (varname, idx) tuples and access `traces` directly
        data = np.stack(traces[param.tracename][:,i]
                        for param, i in zip(flat_params.values(), flat_idcs)).T

        # Get 1D marginals
        marginals1D = params.newdict()
        for i, param in enumerate(flat_params.values()):
            # Get a first histogram to compute the bin range
            if histogram_kwargs.get('range', None) is None:
                # Set any parameter that might affect the calculation of the probability mass
                bins = histogram_kwargs.get('bins', 'auto')
                weights = histogram_kwargs.get('weights', None)
                freq, edges = np.histogram(data[:,i], bins=bins, weights=weights)
                pmass = freq / freq.sum()
                # Remove tails with probability mass below threshold
                # Left tail
                lefti = -1  # We are guaranteed at least one pass through the `while`,
                cum = 0     # so start lefti one below its maximum value
                while cum < threshold / 2:
                    lefti += 1
                    cum += pmass[lefti]
                # Right tail
                righti = len(edges) # Start one beyond the maximum possible value of righti
                cum = 0
                while cum < threshold / 2:
                    righti -= 1
                    cum += pmass[righti-1]   # -1 because `righti` corresponds to right of bin
                hist_kwds = histogram_kwargs.copy()
                hist_kwds['range'] = (edges[lefti], edges[righti])
            else:
                # 'range' is explicitly specified: just use that
                hist_kwds = histogram_kwargs
            # Now that we have the bins, compute the histogram we will actually use
            freq, edges = np.histogram(data[:,i], **hist_kwds)
            # Construct the associated axis
            axis = Axis(param.modelname, param.transformedname, param.idx,
                        transformed_stops=edges, format='edges',
                        transform_fn=param.to_desc,
                        inverse_transform_fn=param.back_desc)
            marginals1D[param.displayname] = Probability('p', 'mass', freq, [axis],
                                                         normalized=True, **kwargs)

        # TODO: Reuse the 1D axes
        # Get 2D marginals
        marginals2D = params.newdict()
        # Loop over all combinations with j != i
        flp_list = list(flat_params.values())
        for i, parami in enumerate(flp_list):
            for j, paramj in enumerate(flp_list):
                if i != j:
                    # Ensure that 2D histogram has same bins as 1D
                    # There are also more options for automatically setting bins for 1D histograms,
                    # which this allows to exploit for the 2D histogram
                    xedges = marginals1D[parami.displayname].axes[0].edges.transformed_stops
                    yedges = marginals1D[paramj.displayname].axes[0].edges.transformed_stops
                        # Transformed stops because the data are in the
                        # transformed space
                    # Get histogram
                    histogram_kwargs['bins'] = [xedges, yedges]
                    freq, edges = np.histogramdd(data[:,(i,j)], **histogram_kwargs)
                    assert((edges[0] == xedges).all() and (edges[1] == yedges).all())
                    # Recover the axis
                    axes = [Axis(param.modelname, param.transformedname, param.idx,
                                 transformed_stops=edgelst, format='edges',
                                 transform_fn=param.to_desc,
                                 inverse_transform_fn=param.back_desc)
                            for param, edgelst in zip( [parami, paramj],
                                                        edges )]
                    key = (parami.displayname, paramj.displayname)
                    assert(key not in marginals2D)
                    marginals2D[key] = (Probability('p', 'mass', freq, axes,
                                                    normalized=True, **kwargs))
                    if not np.isfinite(marginals2D[key].data).all():
                        logger.warning("Non finite entries present in 2D marginal for {}."
                                       .format(key))

        return marginals1D, marginals2D

    def set_colorscheme(self, scheme="viridis"):
        """
        Parameters
        ----------
        scheme: str | HeatmapColorScheme
            Either a string or a HeatmapColorScheme, as defined in mackelab.stylelib.colorschemes.
            If a string, the corresponding scheme in mackelab.stylelib.colorschemes.cmaps is selected.
        """
        if isinstance(scheme, str):
            self.colorscheme = mtb.stylelib.colorschemes.cmaps[scheme]
        else:
            self.colorscheme = scheme

    def set_markers(self, markers=None, colors=None, alphas=1., size=None):
        """
        All parameters can be passed as either iterable (same length) or scalar.
        size=None => compute marker size from axis size (0.2% plot area)
        """
        # Standardize `markers` format
        if isinstance(markers, (dict, ParameterSet)):
            # `markers` is treated as a list of dicts, one dict per marker
            markers = (markers,)
        elif markers is None:
            markers = ()

        # Standardize `colors` format
        if colors is None:
            try:
                colors = self.colorscheme.accents[0]
            except (AttributeError, KeyError):
                raise AttributeError("'colorscheme' was improperly set.")
        if isinstance(colors, str) or not isinstance(colors, Iterable):
            colors = (colors,)
        elif not isinstance(colors, Sequence):
            # `colors` is probably a consumable iterable
            colors = tuple(colors)
        if len(colors) != len(markers):
            if len(colors) != 1:
                raise ValueError("`colors` argument must either be of length 1 or "
                                 "of the same length as `markers`.")
            colors = list(colors)*len(markers)
        if not isinstance(alphas, Iterable):
            alphas = [alphas]*len(markers)
        else:
            alphas = list(alphas)  # In case alphas is consummable
            if len(alphas) != len(markers):
                if len(alphas) != 1:
                    raise ValueError(
                        "`alphas` argument must either be of length 1 or "
                        "of the same length as `markers`.")
                alphas = alphas*len(markers)
        if not isinstance(size, Iterable):
            sizes = [size]*len(markers)
        else:
            sizes = list(size)  # In case alphas is consummable
            if len(sizes) != len(markers):
                if len(sizes) != 1:
                    raise ValueError(
                        "`size` argument must either be of length 1 or "
                        "of the same length as `markers`.")
                sizes = sizes*len(markers)

        self.markers = [self.Marker(pos, color, α, s)
                        for pos, color, α, s
                        in zip(markers, colors, alphas, sizes)]

    def set_transformed(self, transformed_axes):
        """
        Parameters
        ----------
        transformed_axes: bool | list of bool
            Whether to plot the transformed or non-transformed axes. Can be
            specified independently for each axis by passing a list of bools,
            one bool per axis.
            This parameter only has an effect if `data` is not already a data grid,
            e.g. when `data` is an MCMC trace.
        """
        # TODO: Allow `transformed_axes` to be an unsanitized dict

        if not isinstance(transformed_axes, Iterable):
            transformed_axes = (transformed_axes,)*len(self.params)
        elif len(transformed_axes) == 1:
            transformed_axes = tuple(transformed_axes)*len(self.params)
        elif len(transformed_axes) != len(self.params):
            raise ValueError("`transformed_axes` must have as many elements as `params`, "
                             "or be a scalar.")

        self.transformed = self.params.newdict()
        for name, transformed in zip(self.params, transformed_axes):
            self.transformed[name] = transformed

    def set_axis(self, key, scale=None, visible=None, apply=None):
        """
        Set the formatting for an axis (aka matplotlib 'spine). If called
        multiple times with the same key, only the settings of the last call
        are kept.

        Parameters
        ----------
        key: str
            The spines associated to `key` will be affected. The special key
            'all' indicates to apply to all spines. Formatting set with 'all'
            is overwritten by formatting set with a specific key.
        scale: str
            Will be passed to `ax.set_[xy]scale` on the relevant axes.
        visible: list of str
            List of positions ('top', 'right', 'bottom', 'left') where
            the axis should be drawn.
        fn: callable
            Apply arbitrary operations to the axes.
            Required signature: `(mpl.Axes axes, mpl.Axis axis)`
            `axis` will be either an instance of XAxis or YAxis, which
            can be used to apply operations only to the x or y axis.

        Examples
        --------
        To override the tick locator and set axis stops and labels explicitly
        for the spines corresponding to key `X`:

        >>> marginals = MarginalCollection([…])
        >>> def format_plot_X(ax, axis):
        ...     axis.set_ticks([1, 4, 10])
        ...     axis.set_ticklabels(['A', 'B', 'C'])
        >>>> marginals.set_axis('X', apply = format_plot_X)
        """
        self._axes_format[key] = self.AxisFormat(key, scale, visible, apply)

    def _format_axis(self, key, axes, axis):
        """
        Internal function that applies the parameters specified in `set_axis()`.
        """
        if self._axes_format.sanitize(key) in self._axes_format:
            # Set which spines are visible
            format = self._axes_format[key]
            if format.visible is not None:
                for side, spine in axes.spines.items():
                    if (isinstance(axis, mpl.axis.XAxis)
                        or isinstance(axis, mpl.axis.YAxis)):
                        if side in format.visible:
                            spine.set_visible(True)
                        else:
                            spine.set_visible(False)

            # Set the axis scale
            if format.scale is not None:
                if isinstance(axis, mpl.axis.XAxis):
                    axes.set_xscale(format.scale)
                elif isinstance(axis, mpl.axis.YAxis):
                    axes.set_yscale(format.scale)

            # Arbitrary formatting
            if format.apply is not None:
                format.apply(axes, axis)

    def plot_marginal1D(self, key, ax=None):
        if ax is None: ax = plt.gca()

        hm = self.marginals1D[key].density
        param = self.params[key]
        parami_markers = [marker.pos[param.modelname].flatten()[param.flatidx]
                          for marker in self.markers]
        colors = [marker.color for marker in self.markers]
        if self.transformed[key]:
            to_desc = self.params[key].to_desc
            if to_desc is not None:
                transform = mtb.parameters.Transform(to_desc)
                parami_markers = [transform(marker) for marker in parami_markers]
        axis = hm.axes[0]
        if self.transformed[key]:
            axis = axis.transformed

        fig = ax.get_figure()
        stops = (axis.edges.transformed_stops[:-1] if self.transformed[key]
                 else axis.edges.stops[:-1])
        bar = ax.bar(stops, hm.data, axis.widths);
        for mark, color in zip(parami_markers, colors):
            ax.axvline(x=mark, color=color, zorder=2)
        ax.set_yticks([])

        tick_color = colorschemes.cmaps[hm.cmap].white
        ax.xaxis.set_tick_params(direction='in', color=tick_color)


        ax.draw(fig.canvas.get_renderer())
            # Force drawing of ticks and ticklabels

        # Apply custom user formatting, if any
        self._format_axis('all', ax, ax.xaxis)
        self._format_axis(key, ax, ax.xaxis)

        return ax

    def plot_marginal2D(self, keyi, keyj, stddevs=None, marker_size=None,
                        ax=None, **kwargs):
        """
        Parameters
        ----------
        ...
        stddevs: list of dictionaries  |  list of floats | float
            Each dictionary is a set of keyword arguments to plot_stddev_ellipse().
            'width' is required; all other parameters are optional. Default values are:
              - alpha: 0.85
              - color: self.colorscheme.accents[0]
              - zorder: 1
              - linewidth: 1.5
            For convenience, to use all the defaults, can specify a bare float instead
            of a dictionary.
            If there is only a single ellipse to draw, it does not need to be wrapped
            in a list.
        **kwargs:
            Keyword arguments passed to `ScalarAxisData.plot()`.
        """
        if ax is None: ax = plt.gca()

        hm = self.marginals2D[(keyj, keyi)].density
            # Invert keyi, keyj to put column parameter (j) on x axis
            # – it's easier to visually project the marginals this way
        parami, paramj = self.params[keyi], self.params[keyj]
        parami_markers = [marker.pos[parami.modelname].flatten()[parami.flatidx]
                          for marker in self.markers]
        paramj_markers = [marker.pos[paramj.modelname].flatten()[paramj.flatidx]
                          for marker in self.markers]
        if self.transformed[keyi]:
            to_desc = self.params[keyi].to_desc
            if to_desc is not None:
                transform = mtb.parameters.Transform(to_desc)
                parami_markers = [transform(marker) for marker in parami_markers]
        if self.transformed[keyj]:
            to_desc = self.params[keyj].to_desc
            if to_desc is not None:
                transform = mtb.parameters.Transform(to_desc)
                paramj_markers = [transform(marker) for marker in paramj_markers]
        # TODO: Don't recreate these lists every time we plot
        colors = [marker.color for marker in self.markers]
        αs = [marker.α for marker in self.markers]
        if marker_size is None:
            axsize = np.prod(ax.get_window_extent().bounds[2:])
            marker_size = axsize / 500
                # Base the marker size on the display size
                # `s` argument specifies marker _area_, so we use display area
        sizes = [marker.size if marker.size is not None else marker_size
                for marker in self.markers]
        maxexpand = self.maxexpand

        if self.transformed[keyi] != self.transformed[keyj]:
            raise NotImplementedError("Only setting 'transformed' globally is currently supported.")
        hm.set_cmap(self.colorscheme.name)
        ax = hm.plot(transformed=self.transformed[keyi], ax=ax, **kwargs);
        if isinstance(ax, tuple):
            ax, cb = ax
        ax.set_facecolor(self.colorscheme.min)
        #cb.remove()

        fig = ax.get_figure()

        # Draw stddev ellipses
        if stddevs is None:
            stddevs = ()
        elif not isinstance(stddevs, Iterable) or isinstance(stddevs, dict):
            stddevs = (stddevs,)
        for stddev in stddevs:
            # Check that required parameters are there
            if isinstance(stddev, (int, float)):
                stddev = {'width': stddev}
            if 'width' not in stddev:
                raise ValueError("You must specify at least a width for each stddev ellipse.")
            # Set defaults
            stddev_kwds = {'alpha': 0.85,
                           'color': self.colorscheme.accents[0],
                           'zorder': 1,
                           'linewidth': 1.5}
            # Replace defaults with passed parameters
            stddev_kwds.update(stddev)
            # Draw stddev ellipse
            hm.plot_stddev_ellipse(**stddev_kwds)

        # Possibly expand the plot region if it doesn't include some of the markers
        xlim, ylim = np.array(plt.xlim()), np.array(plt.ylim())
        xwidth, yheight = xlim[1]-xlim[0], ylim[1]-ylim[0]
        for marki, markj, color, α, s in zip(
              parami_markers, paramj_markers, colors, αs, sizes):
            ax.scatter(markj, marki, s=s, c=color, zorder=2, alpha=α,
                       edgecolors='none')
                # Recall that histograms set their x-axis to j-parameter
        newxlim, newylim = np.array(plt.xlim()), np.array(plt.ylim())
        if maxexpand*(xlim[1] - xlim[0]) < (newxlim[1] - newxlim[0]):
            if xlim[1] < newxlim[1]:
                ax.set_xlim(right=xlim[0]+maxexpand*xwidth)
            else:
                ax.set_xlim(left=xlim[1]-maxexpand*xwidth)
        if maxexpand*(ylim[1] - ylim[0]) < (newylim[1] - newylim[0]):
            if ylim[1] < newylim[1]:
                ax.set_ylim(top=ylim[0]+maxexpand*yheight)
            else:
                ax.set_ylim(bottom=ylim[1]-maxexpand*yheight)

        ax.draw(fig.canvas.get_renderer())
            # Force drawing of ticks and ticklabels

        # Apply custom user formatting, if any
        self._format_axis('all', ax, ax.xaxis)
        self._format_axis('all', ax, ax.yaxis)
        self._format_axis(keyj, ax, ax.xaxis)  # Remember, column j on x axis
        self._format_axis(keyi, ax, ax.yaxis)

        return ax

    def plot_grid(self, names_to_display, kwargs1D=None, kwargs2D=None, **kwargs):
        """
        Parameters
        ----------
        names_to_display: list of strings
            Variable names, corresponding to the keys of `self.params`.
        kwargs1D: dict
            Keyword arguments to pass to self.plot_marginal1D
        kwargs2D: dict
            Keyword arguments to pass to self.plot_marginal2D
        **kwargs: Keyword arguments passed on to self._plot_grid_layout
        """
        # def sanitize(s):
        #     # Ignore certain character
        #     return s.replace('{','').replace('}','').replace('$','')
        # # Select the parameters to display
        # if names_to_display is None:
        #     names = [sanitize(s) for s in self.params]
        # else:
        #     # Check that names are valid
        #     sanitized_params = {sanitize(s): s for s in self.params}
        #     sanitized_display_params = [sanitize(s) for s in names_to_display]
        #     unrecognized = [nm for nm in sanitized_display_params if nm not in sanitized_params]
        #     if len(unrecognized) > 0:
        #         raise ValueError("Variable names {} do not match any known dimensions. "
        #                          "Recognized dimension names: {}"
        #                          .format(unrecognized, list(self.params.keys())))

        #     names = [sanitized_params[name] for name in sanitized_display_params]
        # params = OrderedDict((name, self.params[name]) for name in names)

        if names_to_display is None:
            params = self.params
        else:
            params = self.params.newdict( (name, self.params[name])
                                          for name in names_to_display )

        # Wrap the plotting functions with functions that feed them the proper
        # keyword arguments
        if kwargs1D is None:
            kwargs1D = {}
        if kwargs2D is None:
            kwargs2D = {}
        def plot_marginal1D(*args, **kwargs):
            return self.plot_marginal1D(*args, **kwargs, **kwargs1D)
        def plot_marginal2D(*args, **kwargs):
            return self.plot_marginal2D(*args, **kwargs, **kwargs2D)

        self._plot_grid_layout(gridkeys=params,
                               plot_diagonal=plot_marginal1D,
                               plot_offdiagonal=plot_marginal2D,
                               **kwargs)

    #NOTE: Should be possible to split off the following function, if it can
    #      be useful elsewhere.
    def _plot_grid_layout(self, gridkeys, layout='upper right',
                  colwidth=3, rowheight=None, figsize=None,
                  xlabelpos=None, ylabelpos=None,
                  plot_diagonal=None, plot_offdiagonal=None):
        """
        Parameters
        ----------
        gridkeys: OrderedDict
            Currently of the form {key: ParamDim}.
            List of objects which identify the position in the grid of plots.
            These keys are iterated over and passed to `plot_diagonal()` and
            `plot_offdiagonal()` to generate each plot.
            #TODO: Allow to be any iterable of keys
        colwidth: float
            Width of colums, in inches. Ignored if `figsize` is given.
        rowheight: float
            Height of columns, in inches. If `None`, set to 2/3 of the value
            of `colwidth`.
        figsize: tuple of floats
            Equivelant to pyplot's `figsize` argument: specify the size of the
            full figure in inches. If given, overrides values of `colwidth`
            and `rowheight`; these are calculated from `figsize` and the
            numbers of columns and rows.
        xlabelpos: float, tuple
            Y position, in figure coordinates, of the xlabel. It will be
            placed at (0.5, `xlabelpos`). Typically between 1.1 and 1.6.
            If a tuple, xlabel is placed at `xlabelpos`.
        ylabelpos: float, tuple
            X position, in figure coordinates, of the ylabel. It will be
            placed at (`ylabelpos`, 0.5). Typically between 1.1 and 1.6.
            If a tuple, ylabel is placed at `ylabelpos`.
        plot_diagonal: function  |  (function, {})
            Function taking the key corresponding to a position along the
            diagonal, and producing the desired plot in the current axis.
            In the second form, a tuple with a function and keyword parameters.
        plot_offdiagonal: function  |  (function, {})
            Function taking the keys corresponding to x,y position in the
            plot grid (x: keyi, y: keyj), and producing the desired plot
            in the current axis.
            In the second form, a tuple with a function and keyword parameters.
        """
        def minorticks_off(axis):
            # ax.minorticks_off doesn't allow to specify the axis, and
            # ax.tick_params(…) doesn't seem to work on pcolormesh
            pass
            axis.set_minor_locator(mpl.ticker.LinearLocator(0))
                # LinearLocator(n) sets n-1 equally spaced minor ticks
                # Weirdly 1 doesn't seem to work, but 0 does
        if not isinstance(gridkeys, OrderedDict):
            raise TypeError("'gridkeys' argument must be an OrderedDict.")

        # Set the grid size
        nrows = len(gridkeys)
        ncols = nrows

        # Set the figure size
        if figsize is None:
            if rowheight is None:
                rowheight = 2/3 * colwidth
        else:
            if len(figsize) != 2:
                raise ValueError("`figsize` must be a tuple of size 2.")
            colwidth = figsize[0] / ncols
            rowheight = figsize[1] / nrows

        # Temporary hack
        self.grid_layouts = {
            'upper right': {'blank': lambda i,j: j < i},
            'lower left':  {'blank': lambda i,j: j > i},
            'lower right': {'blank': lambda i,j: i + j < nrows - 1}
        }

        # Start the plot
        plt.figure(figsize=(ncols*colwidth, nrows*rowheight))

        parami_lst = gridkeys.items()
        if layout in ('upper right', 'lower left'):
            paramj_lst = gridkeys.items()
        elif layout in ('lower right'):
            paramj_lst = [(key, self.params[key]) for key in list(gridkeys.keys())[::-1]]
        else:
            raise ValueError("Unrecognized grid layout '{}'.".format(layout))

        # TODO: Remove need for parami; then gridkeys can be any iterable
        for i, (keyi, parami) in enumerate(parami_lst):
            for j, (keyj, paramj) in enumerate(paramj_lst):
                if self.grid_layouts[layout]['blank'](i,j):
                    continue

                ax = plt.subplot(nrows, ncols, i*ncols + j + 1);

                if keyi == keyj:
                    if isinstance(plot_diagonal, tuple):
                        # `plot_diagnonal` is a tuple (function, {keywords})
                        plot_diagonal[0](**plot_diagonal[1])
                    else:
                        plot_diagonal(keyi, ax=ax)
                else:
                    assert(not self.grid_layouts[layout]['blank'](i,j))
                    if isinstance(plot_offdiagonal, tuple):
                        # `plot_diagnonal` is a tuple (function, {keywords})
                        plot_offdiagonal[0](**plot_offdiagonal[1])
                    else:
                        plot_offdiagonal(keyi, keyj, ax=ax)

                if 'right' in layout:
                    ax.yaxis.tick_right()
                    ax.yaxis.set_label_position('right')
                    if j == ncols - 1:
                        #ax.set_ylabel(parami.displayname, labelpad=15, rotation=0)
                        ax.set_ylabel(parami.displayname)
                        if ylabelpos is not None:
                            if isinstance(ylabelpos, Iterable):
                                if len(ylabelpos) == 1:
                                    coords = (ylabelpos[0], 0.5)
                                elif len(ylabelpos) == 2:
                                    coords = tuple(ylabelpos)
                                else:
                                    raise ValueError("`ylabelpos` must be of length 2.")
                            else:
                                coors = (ylabelpos, 0.5)
                            ax.yaxis.set_label_coords(ylabelpos, 0.5)
                    else:
                        ax.set_ylabel("")
                        ax.set_yticks([])
                        minorticks_off(ax.yaxis)

                if 'upper' in layout:
                    ax.tick_params(axis='x', top='on', bottom='off')
                elif 'lower' in layout:
                    ax.tick_params(axis='x', top='off', bottom='on')
                if 'left' in layout:
                    ax.tick_params(axis='y', left='on', right='off')
                elif 'right' in layout:
                    ax.tick_params(axis='y', left='off', right='on')
                if layout == 'upper right':
                    if i == 0:
                        ax.set_title(paramj.desc)
                    if i == j:
                        ax.set_xlabel(parami.displayname)
                        ax.yaxis.set_label_position('left')
                    else:
                        ax.set_xlabel("")
                        ax.set_xticks([])
                        minorticks_off(ax.xaxis)
                elif layout == 'lower left':
                    if i == j:
                        ax.set_title(parami.desc)
                    if j == 0:
                        ax.set_ylabel(parami.displayname, rotation=0)
                    else:
                        ax.set_ylabel("")
                        ax.set_yticks([])
                        minorticks_off(ax.yaxis)
                    if i == nrows - 1:
                        ax.set_xlabel(paramj.displayname)
                    else:
                        ax.set_xlabel("")
                        ax.set_xticks([])
                        minorticks_off(ax.xaxis)
                elif layout == 'lower right':
                    if i + j == nrows - 1:
                        ax.set_title(parami.desc)
                    if i == nrows - 1:
                        ax.set_xlabel(paramj.displayname)
                    else:
                        ax.set_xlabel("")
                        ax.set_xticks([])
                        minorticks_off(ax.xaxis)
                else:
                    raise ValueError("Unrecognized grid layout '{}'.".format(layout))


mtb.iotools.register_datatype(ScalarAxisData)
