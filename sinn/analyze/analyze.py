# -*- coding: utf-8 -*-
"""
Analysis and plotting functions for the SINN package.

Created Tue Feb 21 2017

author: Alexandre René
"""

__all__ = ['mean', 'diff', 'smooth', 'subsample',
           'window_mean', 'window_variance',
           'plot', 'get_axes', 'get_axis_labels']

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

from mackelab.stylelib import colorschemes
import theano_shim as shim

import sinn.histories as histories
from . import common as com
from . import heatmap

# ==================================
# Data types
ParameterAxis = com.ParameterAxis

# ==================================
# Data manipulation

def mean(hist, pop_slices=None, time_array=None, **kwargs):
    """
    Compute the time-dependent mean, by applying `np.mean` to every time slice
    of `hist`.

    Parameters
    ----------
    hist: History
        Can also be a plain numpy array

    pop_slices: list of slices
        (Optional) If specified, timeslices are subdivided with theses slices
        and the mean is computed for each slice separately.
        TODO: 2d pop_slices ?

    time_array: 1d array
        When `hist` is given as a an ndarray, this parameter specifies the
        corresponding time stops. It is ignored when `hist` is given as a
        history.

    **kwargs:
        Forwarded to `np.mean`. Can be use e.g. to limit the mean to one axis.

    Returns
    -------
    History, of same length as hist.
    """

    if isinstance(hist, np.ndarray):
        assert(hist.ndim > 1)
        data_arr = hist
        if time_array is None:
            time_array = np.arange(len(data_arr))
        hist = histories.Series(time_array = time_array, shape=data_arr.shape[1:])
        hist.set(data_arr)
        hist = histories.DataView(hist)
    elif isinstance(hist, histories.HistoryBase):
        hist = histories.DataView(hist)
        try:
            data_arr = hist.get_trace()
        except AttributeError:
            raise ValueError("Histories of type '{}' don't provide a `get_trace` method. "
                             "You can try passing a raw NumPy array, or better yet, defining "
                             "`get_trace` for this history type.".type(hist.hist))
    else:
        raise ValueError("Unrecognized History type '{}'.".format(type(hist.hist)))

    if len(data_arr.shape) < 2:
        raise ValueError("Data must be at least 2 dimensional: time + data axes.")

    # If 'axis' is specified as a keyword, it relates to a time slice, so we add the time axis
    if 'axis' in kwargs:
        if not isinstance(kwargs['axis'], collections.Iterable):
            kwargs['axis'] = axis + 1
        else:
            kwargs['axis'] = tuple(ax+1 for ax in kwargs['axis'])
    else:
        # Take mean over all but the time axis
        if data_arr.ndim == 2:
            kwargs['axis'] = 1 # Sparse arrays – which are always 2D – don't accept axis as a tuple
        else:
            kwargs['axis'] = tuple(range(1, len(data_arr.shape)))

    # NOTE: Essential to use the mean method (rather than np.mean): data_arr may be sparse
    if pop_slices is not None:
        if 'keepdims' not in kwargs and not shim.sparse.issparse(data_arr):
            kwargs['keepdims'] = True
        res_data = np.concatenate( [ data_arr[:, pop_slice].mean( **kwargs )
                                     for pop_slice in pop_slices ],
                                   axis = 1 )
    else:
        if 'keepdims' not in kwargs and not shim.sparse.issparse(data_arr):
            kwargs['keepdims'] = False
        res_data = data_arr.mean(**kwargs)

    if isinstance(res_data, np.matrix):
        res_data = res_data.A

    res = histories.Series(hist.hist, name = "<" + hist.name + ">",
                           shape = res_data.shape[1:],
                           iterative = False)
    res.set( res_data )
    res.lock()
    return res

def diff(hist, mode='centered'):
    """
    Compute the numerical derivative of `hist`

    Parameters
    ----------
    hist: History
        Any history, but only really makes sense for Series
    mode: str
        Determines the method used to compute the derivative. One of
        - 'centered' : Centered differences (default)
           ( hist[l+1] - hist[l-1] ) / (2Δt)
           This consumes two time bins and is O(Δt²).
        - 'forward'  : Forward-difference
           ( hist[l+1] - hist[l] ) / Δt
           This consumes one time bin and is O(Δt)
        - 'backward' : Backward-difference
           ( hist[l] - hist[l-1] ) / Δt
           This consumes one time bin and is O(Δt)

    Returns
    -------
    Series
       The length of the series will be slightly reduced by the number of bins
       consumed by the differentiation method.
    """

    hist = histories.DataView(hist)

    if mode == 'centered':
        # Remove consumed bins, but only if there's no padding
        if hist.t0idx >= 1:
            t0 = hist.t0
            startidx = hist.t0idx - 1
        else:
            t0 = hist.t0 + hist.dt
            startidx = hist.t0idx
        if len(hist._tarr) > hist.t0idx + len(hist):
            tn = hist.tn
            endidx = hist.t0idx + len(hist) + 1
        else:
            tn = hist.tn - hist.dt
            endidx = hist.t0idx + len(hist)

        # Possibly shorten the length of data, if series was not computed to end
        endidx = min(hist._cur_tidx.get_value(), endidx)

        res = Series(hist, name = "D " + hist.name,
                     t0 = t0, tn = tn,
                     iterative = False)
        res.set( (hist[startidx+2:endidx] - hist[startidx:endidx-2]) / (2*hist.dt) )

    if mode == 'forward':
        # Remove consumed bins, but only if there's no padding
        t0 = hist.t0
        startidx = hist.t0idx

        if len(hist._tarr) > hist.t0idx + len(hist):
            tn = hist.tn
            endidx = hist.t0idx + len(hist) + 1
        else:
            tn = hist.tn - hist.dt
            endidx = hist.t0idx + len(hist)

        # Possibly shorten the length of data, if series was not computed to end
        endidx = min(hist._cur_tidx.get_value(), endidx)

        res = Series(hist, name = "D " + hist.name,
                     t0 = t0, tn = tn,
                     iterative = False)
        res.set( (hist[startidx+1:endidx] - hist[startidx:endidx-1]) / (hist.dt) )

    if mode == 'backward':
        # Remove consumed bins, but only if there's no padding
        if hist.t0idx >= 1:
            t0 = hist.t0
            startidx = hist.t0idx - 1
        else:
            hist.t0 + hist.dt
            startidx = hist.t0idx

        tn = hist.tn
        endidx = hist.t0idx + len(hist) + 1

        # Possibly shorten the length of data, if series was not computed to end
        endidx = min(hist._cur_tidx.get_value(), endidx)

        res = histories.Series(hist, name = "D " + hist.name,
                               t0 = t0, tn = tn,
                               iterative = False)
        res.set( (hist[startidx+1:endidx] - hist[startidx:endidx-1]) / (hist.dt) )

    res.lock()
    return res

def smooth(series, amount=None, method='mean', name = None, **kwargs):
    """Smooth the `series` and return as another Series instance.
    Data is smoothed by averaging over a certain number of data points
    (determined by `amount`) at each time point.

    Parameters
    ----------
    series: Histories.Series instance
        The data we want to smooth.
        If provided as a plain NumPy array, a surrogate series is created
        with step size (dt) of 1.
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
    name: string
        Name to assign to the smoothed series. Default is to append
        '_smoothed' to the series' name.

    Return
    ------
    Series instance
        Note that in general the time range will be shorter than
        the original data, due to the averaging.
    """

    if isinstance(series, np.ndarray):
        series_data = series
        series = histories.Series(name = name,
                                  t0 = 0, tn = len(series)-1, dt = 1,
                                  shape = series.shape[1:],
                                  iterative=False)
        series.set(series_data)

    series = histories.DataView(series)
         # FIXME: make DataView inherit type, in case one is passed as argument

    # # TODO: Update Theano check
    # if hasattr(series, 'use_theano') and series.use_theano:
    #     if series.compiled_history is not None:
    #         series = series.compiled_history
    #     else:
    #         raise ValueError("Cannot smooth a Theano array.")
    if method == 'mean':
        # TODO: Don't move t0 or tn if there is enough padding
        assert(amount is not None)
        if name is None:
            name = series.name + "_smoothed"

        # Possibly shorten the length of data, if series was not computed to end
        datalen = series._cur_tidx.get_value() - series.t0idx + 1
        if datalen < amount:
            raise ValueError("The smoothing window is wider than the length "
                             "of data ({} bins vs {} bins)."
                             .format(amount, datalen))
        # Create the result (smoothed) series
        t0 = series.t0 + (amount-1)*series.dt/2
        res = histories.Series(name = name,
                               time_array = np.arange(datalen-amount+1) * series.dt + t0,
                               # t0 = series.t0 + (amount-1)*series.dt/2,
                               # #tn = series.tn - (amount-1)*series.dt/2,
                               # tn = series.t0 + (amount-1)*series.dt/2 + (datalen - amount)*series.dt,
                               #     # Calculating tn this way avoids rounding errors that add an extra bin
                               # dt = series.dt,
                               shape = series.shape,
                               iterative = False)
        assert(len(res) == datalen - amount + 1)
        res.pad(series.t0idx, len(series._tarr) - len(series) - series.t0idx)
            # Add the same amount of padding as series
        res.set(_running_mean(series[:series.t0idx+datalen], amount))
        res.lock()

        return res

    else:
        raise NotImplementedError

def subsample(series, amount):
    """Reduce the number of time bins by averaging over `amount` bins.
    TODO: add mode parameter to allow bins being identified by their centre or end time

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
    series = histories.DataView(series)

    assert(np.issubdtype(np.asarray(amount).dtype, np.int))
    # if hasattr(series, 'use_theano') and series.use_theano:
    #     if series.compiled_history is not None:
    #         series = series.compiled_history
    #     else:
    #         raise ValueError("Cannot subsample a Theano array.")
    newdt = series.dt * amount
    nbins = int( (series.tn - series.t0) // newdt )
        # We might chop off a few bins, if the new dt is not commensurate with
        # the original number of bins.
    res = histories.Series(name = series.name + "_subsampled_by_" + str(amount),
                           time_array = np.arange(nbins) * newdt + series.t0,
                           #t0 = series.t0,
                           #tn   = series.t0 + (nbins - 1) * newdt,
                           #    # nbins-1 because tn is inclusive
                           #dt   = newdt,
                           shape = series.shape,
                           iterative = False)
    data = series.get_trace()[:nbins*amount]
        # Slicing removes bins which are not commensurate with the subsampling factor
    t0idx = series.t0idx
    res.set(np.sum(data[i : i+nbins*amount : amount] for i in range(amount))/amount)
        # Can't use np.mean on a generator
    res.lock()
    return res

def window_mean(series, window_len, name=None):
    if name is None:
        if hasattr(series, 'name'):
            name = series.name + "_avg"
        else:
            name = "data_avg"

    return smooth(series, amount=window_len, method='mean', name=name)

def window_variance(series, window_len, name=None, mean_series=None):
    if name is None:
        if hasattr(series, 'name'):
            μname = series.name + "_avg"
            sqrname = series.name + "_sqr"
        else:
            μname = "data_avg"
            sqrname = "data_sqr"

    if mean_series is None:
        mean_series = smooth(series, amount=window_len, method='mean', name=μname)
    else:
        assert(len(mean_series) == len(series))
    # Return <X²> - <X>²
    return ( smooth(series**2, amount=window_len, method='mean', name=sqrname)
             - mean_series**2 )

# ==================================
# Plotting

def cleanname(_name):
    s = _name.strip('$')
    # wrap underscored elements with brackets
    s_els = s.split('_')
    s = s_els[0] + ''.join(['_{' + el + '}' for el in s_els[1:]])
    # wrap the whole string in brackets, to allow underscore with component
    return '{' + s + '}'

def plot(data, **kwargs):
    """
    Parameters
    ----------
    data: a sinn data structure
        This parameter's type will determin the type of plot.
        - Series: produces a line plot (w/ plt.plot(.))
        - HeatMap: produces a 2D density (w/ pcolormesh(.))
        - Spiketimes: produces a raster plot (not implemented)

    TODO: Organize all keywords by data type
          Include Spiketrain keywords

    **kwargs: keyward arguments
        These depend on the data type
        Possible keywords for History data:
        - start: float
            Time at which to start plotting the data. Default is to plot
            from the beginning. Ignored for pure NumPy data, as it has no associated
            time array.

        - end: float
            Inclusive time at which to stop plotting the data. Default is to
            plot until the end. Ignored for pure NumPy data, as it has no associated
            time array.

    Other keyword arguments
        These will be forwarded to the underlying plotting function;
        all are optional. The following keywords are preprocessed:
        - `label`
          Defined for: Series
          Can be specified as a single string or a list of strings.
          In the former case, a subscript is added to indicate components;
          in the latter,  strings are used as-is and the list should be of
          the same length as the number of components.
          If not specified, the data's `name` attribute is used, with
          components indicated as a subscript.
        - `component`
          Defined for: Series
          Restrict plotting to the specified components.
          TODO: Implement for Heatmap
    Returns
    -------
    A list of the return values of the plotting calls.

    """
    if isinstance(data, histories.History):
        data = histories.DataView(data)
        start = kwargs.pop('start', data.t0idx)
        stop = kwargs.pop('stop', data.tnidx+1)
        stop = min(stop, data.tnidx+1)
        tslice = slice(start, stop)

    # TODO: Collect repeated code

    if isinstance(data, np.ndarray):
        comp_list = kwargs.pop('component', None)
        label = kwargs.pop('label', None)

        if comp_list is None:
            if len(data.shape) > 1:
                comp_list = list( itertools.product(*[range(s) for s in data.shape[1:]]) )
        else:
            # Component list must be a list of tuples
            if not isinstance(comp_list, collections.Iterable):
                comp_list = [(comp_list,)]
            elif not isinstance(comp_list[0], collections.Iterable):
                if isinstance(comp_list, list):
                    comp_list = [tuple(c) for c in comp_list]
                elif isinstance(comp_list, tuple):
                    comp_list = [comp_list]
                else:
                    comp_list = [tuple(c) for c in comp_list]

        if comp_list is not None:
            if label is None or isinstance(label, str):
                name = label if label is not None else "y"
                # Loop over the components
                #if len(comp_list) > 1:
                labels = [ "${}_{{{}}}$".format(cleanname(name), str(comp).strip('(),'))
                           for comp in comp_list ]
                #else:
                #    labels = [ "${}$".format(cleanname(data.name)) ]
            else:
                assert(isinstance(label, collections.Iterable))
                labels = label

        # Loop over the components, plotting each separately
        # Plotting separately allows to assign a label to each
        lines = []
        if comp_list is None:
            lines.append( plt.plot(np.arange(len(data)), data) )

        else:
            for comp, label in zip(comp_list, labels):
                idx = (slice(None),) + comp
                lines.append( plt.plot(np.arange(len(data)), data[idx], label=label, **kwargs) )
        return lines

    elif ( isinstance(data, histories.Series)
           or ( isinstance(data, histories.DataView)
                and isinstance(data.hist, histories.Series) ) ):
        # Second line catches a DataView of a Series
        # TODO: Make DataView a derived class of its self.hist within __new__;
        #       then 'isinstance' would work.
        # if hasattr(data, 'use_theano') and data.use_theano:
        #     assert(hasattr(data, 'compiled_history'))
        #     if data.compiled_history is None:
        #         raise ValueError("You need to compile a Theano history before plotting it.")
        #     data = data.compiled_history

        comp_list = kwargs.pop('component', None)
        if comp_list is None:
            comp_list = list( itertools.product(*[range(s) for s in data.shape]) )
        else:
            if not isinstance(comp_list, collections.Iterable):
                comp_list = [comp_list]

        label = kwargs.pop('label', None)
        if label is None or isinstance(label, str):
            name = label if label is not None else data.name
            # Loop over the components
            #if len(comp_list) > 1:
            labels = [ "${}_{{{}}}$".format(cleanname(name), str(comp).strip('(),'))
                       for comp in comp_list ]
            #else:
            #    labels = [ "${}$".format(cleanname(data.name)) ]
        else:
            assert(isinstance(label, collections.Iterable))
            labels = label

        # Loop over the components, plotting each separately
        # Plotting separately allows to assign a label to each
        lines = []
        for comp, label in zip(comp_list, labels):
            lines.append( plt.plot(data.get_time_array(time_slice=tslice),
                                   data.get_trace(comp, time_slice=tslice),
                                   label=label, **kwargs) )
        return lines

    elif ( isinstance(data, histories.Spiketrain)
           or ( isinstance(data, histories.DataView)
                and isinstance(data.hist, histories.Spiketrain) ) ):

        lineheight = kwargs.pop('lineheight', 1)
        markersize = kwargs.pop('markersize', 1)
        alpha = kwargs.pop('alpha', 0.3)
        linestyle = kwargs.pop('linestyle', 'None')
        baselabel = kwargs.pop('label', 'Population')
        if baselabel[-1] != ' ':
            # Make sure there's a space to separate the population number
            baselabel += ' '

        tarr = data._tarr[data._data.row]
        tstart = data._tarr[data.get_t_idx(tslice.start)]
        tend = data._tarr[data.get_t_idx(tslice.stop)-1]
            # We do 'tarr[idx-1]' to avoid indexing beyond the end of _tarr
        tidcs = np.where(np.logical_and(tstart <= tarr, tarr <= tend))[0]
        lines = []
        for i, popslice in enumerate(data.pop_slices):
            popidcs = np.where(np.logical_and(popslice.start <= data._data.col,
                                            data._data.col < popslice.stop))[0]
            idcs = np.intersect1d(tidcs, popidcs)

            lines.append( plt.scatter(data._tarr[data._data.row[idcs]],
                                      data._data.col[idcs]*lineheight,
                                      s = markersize,
                                      linestyle = linestyle,
                                      label = baselabel + str(i),
                                      alpha = alpha )  )

        # Set the axis limits to see the whole time range
        # (if the beginning or end is empty, it would otherwise be truncated)
        margin = (tend - tstart) / 20
        plt.xlim( (tstart-margin, tend+margin) )

        return lines

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

        color_scheme = colorschemes.cmaps[data.cmap]
        ax.tick_params(axis='both', which='both', color=color_scheme.white,
                       top='on', right='on', bottom='on', left='on',
                       direction='in')
        cb.ax.tick_params(axis='y', which='both', left='off', right='off')

        for side in ['top', 'right', 'bottom', 'left']:
            ax.spines[side].set_color('none')
        cb.outline.set_visible(False)

        #return ax, cb
        return [quadmesh]

    else:
        logger.warning("Plotting of {} data is not currently supported."
                       .format(type(data)))

def plot_stddev_ellipse(data, width, **kwargs):
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
    **kwargs:
        Keyword arguments passed to maptplotlib.patches.Ellipse
    """
    # TODO: Deal with higher than 2D heatmaps
    data = histories.DataView(data)

    eigvals, eigvecs = np.linalg.eig(data.cov())
    ax = plt.gca()
    w = width * np.sqrt(eigvals[0])
    h = width * np.sqrt(eigvals[1])
    color = kwargs.pop('color', None)
    if color is None:
        color_scheme = colorschemes.cmaps[data.cmap]
        color = color_scheme.accents[1]  # Leave more salient accents[0] for user
    e = mpl.patches.Ellipse(xy=data.mean(), width=w, height=h,
                            angle=np.arctan2(eigvecs[0][1], eigvecs[0][0]),
                            fill=False, color=color)
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
    if arr.ndim != 1:
        raise ValueError("Axis stops must be given by a 1D array. (Given axis is {}D.)"
                         .format(arr.ndim))
    if len(arr) == 0:
        raise ValueError("Provided axis has length 0.")
    elif len(arr) == 1:
        logger.warning("Computing the 'edges' for a length 1 axis. Unless you are debugging, "
                       "you probably made a mistake when producing the data.")
        dx = 10**int(np.log10(abs(arr[0]))) / 2
        newarr = np.array([arr[0]-dx, arr[0]+dx])
    else:
        dxs = (arr[1:]-arr[:-1])/2
        newarr = np.concatenate(((arr[0]-dxs[0],), (arr[1:] - dxs), (arr[-1] + dxs[-1],)))
    return newarr

def _centers_to_edges(centers, linearization_function=None, inverse_linearization_function=None):
    """[TODO]"""
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
