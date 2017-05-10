# -*- coding: utf-8 -*-
"""
Analysis and plotting functions for the SINN package.

Created Tue Feb 21 2017

author: Alexandre René
"""

__all__ = ['smooth', 'subsample', 'plot', 'get_axes', 'get_axis_labels']

import logging
import collections
from collections import namedtuple
import itertools
import numpy as np
import scipy as sp
logger = logging.getLogger('sinn.analyze')

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    logger.warning("Unable to import matplotlib. Plotting functions "
                   "will not work.")

import theano_shim as shim
import sinn.histories as histories
from . import common as com
from . import heatmap
from .stylelib import color_schemes as color_schemes

# ==================================
# Data types
ParameterAxis = com.ParameterAxis

# ==================================
# Data manipulation

def smooth(series, amount=None, method='mean', **kwargs):
    """Smooth the `series` and return as another Series instance.
    Data is smoothed by averaging over a certain number of data points
    (determined by `amount`) at each time point.

    Parameters
    ----------
    series: Histories.Series instance
        The data we want to smooth.
    amount: int | float
        A numerical parameter controlling the degree of smoothing.
        Its interpretation depends on the chosen method.
    method: string
        One of:
          + 'mean': A running average along the data. The number
                    of bins is determined by `amount`.
          + 'gaussian': Not implemented
          + 'convolve': Not implemented
        Default is 'mean'.

    Return
    ------
    Series instance
        Note that in general the time range will be shorter than
        the original data, due to the averaging.
    """
    if series.use_theano:
        if series.compiled_history is not None:
            series = series.compiled_history
        else:
            raise ValueError("Cannot smooth a Theano array.")
    if method == 'mean':
        assert(amount is not None)
        res = histories.Series(name = series.name + "_smoothed",
                               t0 = series.t0 + (amount-1)*series.dt/2,
                               #tn = series.tn - (amount-1)*series.dt/2,
                               tn = series.t0 + (amount-1)*series.dt/2 + (len(series) - amount)*series.dt,
                                   # Calculating tn this way avoids rounding errors that add an extra bin
                               dt = series.dt,
                               shape = series.shape)
        assert(len(res) == len(series) - amount + 1)
        res.pad(series.t0idx, len(series._tarr) - len(series) - series.t0idx)
            # Add the same amount of padding as series
        res.set(_running_mean(series[:], amount))

        return res

    else:
        raise NotImplementedError


def subsample(series, amount):
    """Reduce the number of time bins by averaging over `amount` bins.

    Parameters
    ----------
    history: Series instance
    amount: integer
        The factor by which the number of bins in `history` is reduced.

    Returns
    -------
    Series instance
        The result will be `amount` times shorter than `history`. The result
        of each new bin is the average over `amount` bins of the original
        series. Bins are identified by the time at which they begin.
    """
    assert(np.issubdtype(np.asarray(amount).dtype, np.int))
    if series.use_theano:
        if series.compiled_history is not None:
            series = series.compiled_history
        else:
            raise ValueError("Cannot subsample a Theano array.")
    newdt = series.dt * amount
    nbins = int( (series.tn - series.t0) // newdt )
        # We might chop off a few bins, if the new dt is not commensurate with
        # the original number of bins.
    res = histories.Series(name = series.name + "_subsampled_by_" + str(amount),
                           t0   = series.t0,
                           tn   = series.t0 + (nbins - 1) * newdt,
                               # nbins-1 because tn is inclusive
                           dt   = newdt,
                           shape = series.shape)
    data = series.get_trace()[:nbins*amount]
        # Slicing removes bins which are not commensurate with the subsampling factor
    t0idx = series.t0idx
    res.set(np.sum(data[i : i+nbins*amount : amount] for i in range(amount)))
    return res

# ==================================
# Plotting

def plot(data, **kwargs):
    """
    Parameters
    ----------
    data: a sinn data structure
        This parameter's type will determin the type of plot.
        - Series: produces a line plot (w/ plt.plot(.))
        - HeatMap: produces a 2D density (w/ pcolormesh(.))
        - Spiketimes: produces a raster plot (not implemented)

    **kwargs: keyword arguments
        These will be forwarded to the underlying plotting function;
        all are optional. The following keywords are preprocessed:
        - `label`
          Can be specified as a single string or a list of strings.
          In the former case, a subscript is added to indicate components;
          in the latter,  strings are used as-is and the list should be of
          the same length as the number of components.
          If not specified, the data's `name` attribute is used, with
          components indicated as a subscript.
    Returns
    -------
    A list of the created axes.
    """

    if isinstance(data, histories.Series):
        if data.use_theano:
            assert(hasattr(data, 'compiled_history'))
            if data.compiled_history is None:
                raise ValueError("You need to compile a Theano history before plotting it.")
            data = data.compiled_history

        label = kwargs.pop('label', None)
        comp_list = list( itertools.product(*[range(s) for s in data.shape]) )
        if label is None or isinstance(label, str):
            name = label if label is not None else data.name
            # Loop over the components
            def cleanname(_name):
                s = _name.strip('$')
                # wrap underscored elements with brackets
                s_els = s.split('_')
                s = s_els[0] + ''.join(['_{' + el + '}' for el in s_els[1:]])
                # wrap the whole string in brackets, to allow underscore with component
                return '{' + s + '}'
            if len(comp_list) > 1:
                labels = [ "${}_{{{}}}$".format(cleanname(name), str(comp).strip('(),'))
                           for comp in comp_list ]
            else:
                labels = [ "${}$".format(cleanname(data.name)) ]
        else:
            assert(isinstance(label, collections.Iterable))
            labels = label

        ax = plt.gca()
        # Loop over the components, plotting each separately
        # Plotting separately allows to assign a label to each
        for comp, label in zip(comp_list, labels):
            plt.plot(data.get_time_array(), data.get_trace(comp), label=label, **kwargs)
        return ax

    elif isinstance(data, heatmap.HeatMap):
        # TODO: Override keyword arguments with **kwargs
        ax1_grid, ax2_grid = np.meshgrid(_centers_to_edges(data.axes[0]), _centers_to_edges(data.axes[1]), indexing='ij')
        zmin = max(data.floor, data.min())
        zmax = min(data.ceil, data.max())
        quadmesh = plt.pcolormesh(ax1_grid, ax2_grid,
                                  data.data.clip(data.floor, data.ceil),
                                  cmap = data.cmap,
                                  norm = data.get_norm(),
                                  vmin=zmin, vmax=zmax,
                                  **kwargs)
        ax = plt.gca()
        plt.xlabel(data.axes[0].name)
        plt.ylabel(data.axes[1].name)
        ax.set_xscale(data.axes[0].scale)
        ax.set_yscale(data.axes[1].scale)

        cb = plt.colorbar()
        cb.set_label(data.zlabel)

        color_scheme = color_schemes.cmaps[data.cmap]
        ax.tick_params(axis='both', which='both', color=color_scheme.white,
                       top='on', right='on', bottom='on', left='on',
                       direction='in')
        cb.ax.tick_params(axis='y', which='both', left='off', right='off')

        for side in ['top', 'right', 'bottom', 'left']:
            ax.spines[side].set_color('none')
        cb.outline.set_visible(False)

        return ax, cb

def plot_stddev_ellipse(data, width):
    """
    Add an ellipse to a plot denoting a heatmap's spread. This function
    is called after plotting the data, and adds the
    ellipse to the current axis.

    Parameters
    ----------
    data:  heatmap_like
        A data object which provides a .cov method
    width: float
        Amount of data to include in the ellipse, in units of standard
        deviations. A width of 2 will draw the contour corresponding
        to 2 standard deviations.
    """
    # TODO: Deal with higher than 2D heatmaps
    eigvals, eigvecs = np.linalg.eig(data.cov())
    ax = plt.gca()
    w = width * np.sqrt(eigvals[0])
    h = width * np.sqrt(eigvals[1])
    color_scheme = color_schemes.cmaps[data.cmap]
    e = mpl.patches.Ellipse(xy=data.mean(), width=w, height=h,
                            angle=np.arctan2(eigvecs[0][1], eigvecs[0][0]),
                            fill=False, color=color_scheme.accent)
    ax.add_artist(e)
    e.set_clip_box(ax.bbox)


def get_axes(param_axes):
    assert(all(instance(p, ParameterAxis) for p in param_axes))
    return [p.stops for p in param_axes]

def get_axis_labels(param_axes):
    assert(all(instance(p, ParameterAxis) for p in param_axes))
    def get_label(param):
        if param.idx is None:
            return param.name
        else:
            assert(isinstance(param.idx, tuple))
            return param.name + "[" + str(param.idx)[1:-1] + "]"
                # [1:-1] removes the tuple parentheses
    return [get_label(p) for p in param_axes]

# ===============================
# Private functions

def _running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0, axis=0), axis=0)
    return (cumsum[N:] - cumsum[:-N]) / N

def _linear_centers_to_edges(arr):
    """
    Convert an axis where the stops identify bin centers to one where
    stops identify bin edges.

    Parameters
    ----------
    arr: 1D array of length N

    Returns
    -------
    1D array of length N+1
    """
    dxs = (arr[1:]-arr[:-1])/2
    newarr = np.concatenate(((arr[0]-dxs[0],), (arr[1:] - dxs), (arr[-1] + dxs[-1],)))
    return newarr

def _centers_to_edges(centers, linearization_function=None, inverse_linearization_function=None):
    """Same as _centers_to_edges, but for logarithmic axes."""
    if isinstance(centers, ParameterAxis):
        arr = centers.stops
        linearization_function = centers.linearize_fn
        inverse_linearization_function = centers.inverse_linearize_fn
    elif (linearization_function is None
          or inverse_linearization_function is None):
        raise TypeError("(inverse)_linearization_function only optional if "
                        "`centers` is a ParameterAxis instance.")
    else:
        arr = centers
    return inverse_linearization_function(_linear_centers_to_edges(linearization_function(arr)))
