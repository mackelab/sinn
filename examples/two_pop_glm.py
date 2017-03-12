# -*- coding: utf-8 -*-
"""
Created Wed Jan 25 2017

author: Alexandre René
"""

import logging
import os.path
import numpy as np
import scipy as sp
from collections import namedtuple, OrderedDict

try:
    import matplotlib.pyplot as plt
except ImportError:
    logging.warning("Unable to import matplotlib. Plotting won't work.")

import theano_shim as shim
import sinn
import sinn.histories as histories
from sinn.models.glm import GLM_exp_kernel as GLM
import sinn.models.noise as noise
import sinn.iotools as io
import sinn.analyze as anlz
import sinn.analyze.sweep as sweep

logger = logging.getLogger('two_pop_srm')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

_DEFAULT_DATAFILE = "2-pop-glm.dat"
name, ext = os.path.splitext(_DEFAULT_DATAFILE)
_DEFAULT_POSTERIORFILE = name + "_logposterior" + ext

#import dill
#dill.settings['recurse'] = True

def init_activity_model():
    model_params = GLM.Parameters(
        N = np.array((500, 100)),
        c = (2, 2),
        J = ((3, -6), (6, 0)),
        τ = (0.01, 0.08)
        )
    Ahist = histories.Series(name='A',
                             shape=(len(model_params.N),),
                             t0 = 0,
                             tn = 4,
                             dt = 0.001)

    # Noisy input

    rndstream = shim.RandomStreams(seed=314)
    noise_params = noise.GaussianWhiteNoise.Parameters(
        std = (.000003, .000003),
        shape = (2,)
        )
    noise_hist = histories.Series(Ahist, name='ξ', shape=model_params.N.shape)
    noise_model = noise.GaussianWhiteNoise(noise_params, noise_hist, rndstream)
    def input_fn(t):
        res = 2 + 2*shim.lib.sin(t*2*np.pi) + noise_hist[t]
        return res

    input_hist = histories.Series(Ahist, name='I', shape=model_params.N.shape, iterative=False)
    input_hist.set_update_function(input_fn)

    # GLM activity model

    activity_model = GLM(model_params, Ahist, input_hist, rndstream)

    return activity_model

def generate(filename = _DEFAULT_DATAFILE):
    logger.info("Generating data...")
    try:
        # Try to load precomputed data
        activity_model = io.load(filename)
    except FileNotFoundError:
        # Data don't exist, so compute them
        activity_model = init_activity_model()
        activity_model.A.compute_up_to(-1)
        activity_model.A.set()  # Ensure all time points are computed

        # Save the new data
        io.save(filename, activity_model)

        logger.info("Done.")
    else:
        logger.info("Data already exists. Skipping.")

    return activity_model

def plot_activity(model):

    # Plot the activity
    plt.subplot(2,1,1)
    anlz.plot(model.A)

    # Plot the input
    plt.subplot(2,1,2)
    anlz.plot(model.I)

    plt.show(block=False)


def plot_posterior(model_filename = _DEFAULT_DATAFILE,
                   posterior_filename = _DEFAULT_POSTERIORFILE):
    plt.style.use('../sinn/analyze/plot_styles/mackelab_default.mplstyle')

    target_dt = 0.002
        # The activity timestep we want to use.
    if posterior_filename is None:
        name, ext = os.path.splitext(model_filename)
        posterior_filename = name + "_posterior" + ext

    try:
        # See if the posterior has already been computed
        logposterior = io.load(posterior_filename)
    except:
        logposterior = compute_posterior(model_filename, posterior_filename,
                                         target_dt)

    # Plot the posterior
    logposterior.cmap = 'viridis'
    logposterior.set_ceil(logposterior.max())
    logposterior.set_floor(logposterior.min())
    logposterior.set_norm('linear')
    ax, cb = anlz.plot(logposterior)
        # analyze recognizes logposterior as a heat map, and plots accordingly
        # anlz.plot returns a tuple of all plotted objects. For heat maps there
        # are two: the heat map axis and the colour bar
    # ax.set_xlim((2, 8))
    # ax.set_ylim((0.01, 1))

    plt.show(block=False)
    return


def compute_posterior(target_dt,
                      activity_model = None,
                      output_filename=_DEFAULT_POSTERIORFILE,
                      ipp_url_file=None, ipp_profile=None):
    """
    […]
    Parameters
    ----------
    […]
    ipp_url_file: str
        Passed to ipyparallel.Client as `url_file`.
    ipp_profile: bytes
        Passed to ipyparallel.Client as `profile`. Ignored if ipp_url is provided.

    Returns:
    --------
    sinn.HeatMap
    """
    if activity_model is None:
        try:
            # Try to load precomputed data
            activity_model = io.load(_DEFAULT_DATAFILE)
        except FileNotFoundError:
            raise FileNotFoundError("Unable to find data file {}. To create one, run "
                                    "`generate` – this is required to compute the posterior."
                                    .format(_DEFAULT_DATAFILE))

    logger.info("Computing log posterior...")
    try:
        logposterior = io.load(output_filename)
    except FileNotFoundError:
        pass
    else:
        logger.info("Log posterior already computed. Skipping.")
        return logposterior

    Ihist = activity_model.I

   # Construct the arrays of parameters to try
    fineness = .5
    burnin = 0.5
    data_len = 4.0
    param_sweep = sweep.ParameterSweep(activity_model)
    J_sweep = sweep.linspace(-1, 10, fineness)
    τ_sweep = sweep.logspace(0.0005, 0.5, fineness)
    param_sweep.add_param('J', idx=(0,0), axis_stops=J_sweep)
    param_sweep.add_param('τ', idx=(1,), axis_stops=τ_sweep)
    def f(model):
        return model.loglikelihood(burnin, burnin + data_len)
    param_sweep.set_function(f, 'log $L$')

    # timeout ipp.Client after 3 seconds
    if ipp_url_file is not None:
        import ipyparallel as ipp
        try:
            with sinn.timeout(3):
                ippclient = ipp.Client(url_file=ipp_url_file)
        except TimeoutError:
            logger.info("Unable to connect to ipyparallel controller.")
            ippclient = None
        else:
            logger.info("Connected to ipyparallel controller.")
    elif ipp_profile is not None:
        import ipyparallel as ipp
        try:
            with sinn.timeout(3):
                ippclient = ipp.Client(profile=ipp_profile)
        except TimeoutError:
            logger.info("Unable to connect to ipyparallel controller with "
                        "profile '" + ipp_profile + ".'")
            ippclient = None
        else:
            logger.info("Connected to ipyparallel controller for profile '" + ipp_profile + "'.")
    else:
        ippclient = None

    if ippclient is not None:
        # Initialize the environment in each cluster process
        ippclient[:].use_dill().get()
            # More robust pickling

    # Compute the posterior
    logposterior = param_sweep.do_sweep(output_filename, ippclient)
            # This can take a long time
            # The result will be saved in output_filename

    return logposterior

def main():
    activity_model = generate()
    plt.ioff()
    plt.figure()
    plot_activity(activity_model)
    true_params = {'J': activity_model.params.J[0,0],
                   'τ': activity_model.params.τ[0,1]}
        # Save the true parameters before they are modified by the sweep
    activity_model.A.lock = True
    activity_model.I.lock = True
        # Locking A and I prevents them from being cleared when the
        # parameters are updated (by default updating parameters
        # reinitializes histories). It also triggers a RuntimeError if
        # an attempt is made to modify A or I, indicating a code error.
    logposterior = compute_posterior(0.002, activity_model, ipp_profile="default")
    plt.figure()
    plot_posterior(logposterior)
    color = anlz.plot_styles.color_schemes.map[logposterior.cmap].white
    plt.axvline(true_params['J'], c=color)
    plt.axhline(true_params['τ'], c=color)
    plt.show()

    return logposterior

if __name__ == "__main__":
    main()
