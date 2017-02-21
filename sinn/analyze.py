# -*- coding: utf-8 -*-
"""
Analysis and plotting functions for the SINN package.

Created Tue Feb 21 2017

author: Alexandre René
"""

import logging
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import theano_shim as shim
import sinn.history as histories

# ==================================
# Data manipulation

def smooth(series, amount=None, method='mean', **kwargs):
    """Smooth the `series` and return as another Series instance.
    Data is smoothed by averaging over a certain number of data points
    (determined by `amount`) at each time point.

    Parameters
    ----------
    series: History.Series instance
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


# ==================================
# Plotting

def plot(history):

    if isinstance(history, histories.Series):
        plt.plot(history.get_time_array(), history.get_trace())

# ===============================
# Private functions

def _running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0, axis=0), axis=0)
    return (cumsum[N:] - cumsum[:-N]) / N
