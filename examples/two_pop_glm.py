# -*- coding: utf-8 -*-
"""
Created Wed Jan 25 2017

author: Alexandre René
"""

import logging
import os.path
import time
import numpy as np
import scipy as sp
from collections import namedtuple, OrderedDict

try:
    import matplotlib.pyplot as plt
except ImportError:
    logging.warning("Unable to import matplotlib. Plotting won't work.")
    do_plots = False
else:
    do_plots = True

import theano_shim as shim
import sinn
import sinn.histories as histories
from sinn.models.glm import GLM_exp_kernel as GLM
import sinn.models.noise as noise
import sinn.iotools as io
import sinn.analyze as anlz
import sinn.analyze.sweep as sweep

logger = logging.getLogger('two_pop_glm')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

#sinn.config.load_theano()
#shim.theano.config.exception_verbosity='high'
#shim.theano.config.optimizer='fast_compile'
#shim.theano.config.optimizer='none'

def add_before_ext(s, suffix):
    name, ext = os.path.splitext(s)
    return name + suffix + ext

theano_str = ""#"_theano" if sinn.config.use_theano() else ""
_DEFAULT_DATAFILE = add_before_ext("2-pop-glm.dat", theano_str)
_DEFAULT_LIKELIHOODFILE = add_before_ext(_DEFAULT_DATAFILE, "_loglikelihood" + theano_str)

def init_activity_model(activity_history=None, input_history=None):
    model_params = GLM.Parameters(
        N = np.array((500, 100)),
        c = (5, 2),
        J = ((4, -3), (3, 0)),
        τ = (0.01, 0.02)
        )
    memory_time = 0.553 # What we get with τ=(0.01, 0.08)
    if activity_history is not None:
        Ahist = activity_history
        if Ahist.use_theano:
            assert(Ahist.compiled_history._cur_tidx >= Ahist.t0idx + len(Ahist) - 1)
        else:
            assert(Ahist._cur_tidx >= Ahist.t0idx + len(Ahist) - 1)
        Ahist.lock()
    else:
        Ahist = histories.Series(name='A',
                                 shape=(len(model_params.N),),
                                 t0 = 0,
                                 tn = 4,
                                 dt = 0.001)

    # Noisy input
    rndstream = shim.RandomStreams(seed=314)
    noise_params = noise.GaussianWhiteNoise.Parameters(
        std = (.06, .06),
        shape = (2,)
    )
    noise_hist = histories.Series(Ahist, name='ξ', shape=model_params.N.shape)
    noise_model = noise.GaussianWhiteNoise(noise_params, noise_hist, rndstream)

    def input_fn(t):
        import numpy as np
        # import ensures that proper references to dependencies are pickled
        # This is only necessary for scripts directly called on the cli – imported modules are fine.
        res = np.array([12,4]) * (1 + shim.sin(t*2*np.pi)) + noise_hist[t]
        return res
    if input_history is not None:
        Ihist = input_history
        if Ihist.use_theano:
            assert(Ihist.compiled_history._cur_tidx >= Ihist.t0idx + len(Ihist) - 1)
        else:
            assert(Ihist._cur_tidx >= Ihist.t0idx + len(Ihist) - 1)
        Ihist.lock()
    else:
        Ihist = histories.Series(Ahist, name='I', shape=model_params.N.shape, iterative=False)

    Ihist.set_update_function(input_fn)
    Ihist.add_inputs([noise_hist])

    # GLM activity model
    activity_model = GLM(model_params, Ahist, Ihist, rndstream,
                         memory_time=memory_time)
    return activity_model

def generate(filename = _DEFAULT_DATAFILE, autosave=True):
    logger.info("Generating data...")
    try:
        # Try to load precomputed data
        activity_model = io.load(filename)
    except FileNotFoundError:
        # Data don't exist, so compute them
        activity_model = init_activity_model()
        if activity_model.A.use_theano:
            logger.info("Compiling activity model...")
            activity_model.A.compile()
            logger.info("Done.")
            Ahist = activity_model.A.compiled_history
            Ihist = activity_model.I.compiled_history
        else:
            Ahist = activity_model.A
            Ihist = activity_model.I
        t1 = time.perf_counter()
        Ahist.set()
        Ihist.set()
            # In theory Ihist is computed along with Ahist, but it may e.g.
            # leave the last data point uncomputed if Ahist does not need it.
            # Ihist.set() here ensures that the input is also computed all the
            # way to the end, which avoids when building the graph for the likelihood.
        t2 = time.perf_counter()
        logger.info("Data generation took {}s.".format((t2-t1)))

        if autosave:
            # Save the new data. Using the raw format allows us make changes
            # to the sinn library and still use this data
            fn, ext = os.path.splitext(filename)
            io.saveraw(fn + "_A" + ext, activity_model.A)
            io.saveraw(fn + "_I" + ext, activity_model.I)

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


def plot_likelihood(model_filename = _DEFAULT_DATAFILE,
                   likelihood_filename = _DEFAULT_LIKELIHOODFILE):
    plt.style.use('../sinn/analyze/stylelib/mackelab_default.mplstyle')

    target_dt = 0.002
        # The activity timestep we want to use.
    if likelihood_filename is None:
        name, ext = os.path.splitext(model_filename)
        likelihood_filename = name + "_likelihood" + ext

    try:
        # See if the likelihood has already been computed
        loglikelihood = io.load(likelihood_filename)
    except:
        loglikelihood = compute_likelihood(model_filename, likelihood_filename,
                                         target_dt)

    # Convert to the likelihood. We first make the maximum value 0, to avoid
    # underflows when computing the exponential
    likelihood = (loglikelihood - loglikelihood.max()).apply_op("L", np.exp)

    # Plot the likelihood
    likelihood.cmap = 'viridis'
    likelihood.set_ceil(likelihood.max())
    likelihood.set_floor(0)
    likelihood.set_norm('linear')
    ax, cb = anlz.plot(likelihood)
        # analyze recognizes loglikelihood as a heat map, and plots accordingly
        # anlz.plot returns a tuple of all plotted objects. For heat maps there
        # are two: the heat map axis and the colour bar
    # ax.set_xlim((2, 8))
    # ax.set_ylim((0.01, 1))

    plt.show(block=False)
    return


def compute_likelihood(target_dt,
                      activity_model = None,
                      output_filename=_DEFAULT_LIKELIHOODFILE,
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
                                    "`generate` – this is required to compute the likelihood."
                                    .format(_DEFAULT_DATAFILE))

    logger.info("Computing log likelihood...")
    try:
        loglikelihood = io.load(output_filename)
    except FileNotFoundError:
        pass
    else:
        logger.info("Log likelihood already computed. Skipping.")
        return loglikelihood

    Ihist = activity_model.I

   # Construct the arrays of parameters to try
    fineness = 10
    burnin = 0.5
    data_len = 3.5
    param_sweep = sweep.ParameterSweep(activity_model)
    J_sweep = sweep.linspace(-1, 10, fineness)
    τ_sweep = sweep.logspace(0.0005, 0.5, fineness)
    param_sweep.add_param('J', idx=(0,0), axis_stops=J_sweep)
    param_sweep.add_param('τ', idx=(1,), axis_stops=τ_sweep)

    # Define the loglikelihood function
    # if sinn.config.use_theano():
    #     # TODO Precompile function
    #     def l(model):
    #         lcompiled = shim.theano.function([], model.loglikelihood(burnin, burnin + data_len))
    #         return lcompiled()
    # else:
    #     def l(model):
    #         return model.loglikelihood(burnin, burnin + data_len)
    param_sweep.set_function(activity_model.get_loglikelihood(start=burnin,
                                                              stop=burnin+data_len),
                             'log $L$')

    ippclient = sinn.get_ipp_client(ipp_profile, ipp_url_file)

    # # timeout ipp.Client after 2 seconds
    # if ipp_url_file is not None:
    #     import ipyparallel as ipp
    #     try:
    #         with sinn.timeout(2):
    #             ippclient = ipp.Client(url_file=ipp_url_file)
    #     except TimeoutError:
    #         logger.info("Unable to connect to ipyparallel controller.")
    #         ippclient = None
    #     else:
    #         logger.info("Connected to ipyparallel controller.")
    # elif ipp_profile is not None:
    #     import ipyparallel as ipp
    #     try:
    #         with sinn.timeout(3):
    #             ippclient = ipp.Client(profile=ipp_profile)
    #     except TimeoutError:
    #         logger.info("Unable to connect to ipyparallel controller with "
    #                     "profile '" + ipp_profile + ".'")
    #         ippclient = None
    #     else:
    #         logger.info("Connected to ipyparallel controller for profile '" + ipp_profile + "'.")
    # else:
    #     ippclient = None

    if ippclient is not None:
        # Initialize the environment in each cluster process
        ippclient[:].use_dill().get()
            # More robust pickling

    # Compute the likelihood
    t1 = time.perf_counter()
    loglikelihood = param_sweep.do_sweep(output_filename, ippclient)
            # This can take a long time
            # The result will be saved in output_filename
    t2 = time.perf_counter()
    logger.info("Calculation of the likelihood took {}s."
                .format((t2-t1)))

    return loglikelihood

def main():
    # activity_model = generate()
    # if do_plots:
    #     plt.ioff()
    #     plt.figure()
    #     plot_activity(activity_model)
    # del activity_model
    sinn.inputs.clear()
    Ahist = histories.Series.from_raw(io.loadraw(add_before_ext(_DEFAULT_DATAFILE, '_A')))
    Ihist = histories.Series.from_raw(io.loadraw(add_before_ext(_DEFAULT_DATAFILE, '_I')))
    activity_model = init_activity_model(Ahist, Ihist)
    true_params = {'J': activity_model.params.J.get_value()[0,0],
                   'τ': activity_model.params.τ.get_value()[0,1]}
        # Save the true parameters before they are modified by the sweep
    activity_model.A.lock()
    activity_model.I.lock()
        # Locking A and I prevents them from being cleared when the
        # parameters are updated (by default updating parameters
        # reinitializes histories). It also triggers a RuntimeError if
        # an attempt is made to modify A or I, indicating a code error.
    loglikelihood = compute_likelihood(0.002, activity_model, ipp_profile="default")
    if do_plots:
        plt.figure()
        plot_likelihood(loglikelihood)
        color = anlz.stylelib.color_schemes.map[loglikelihood.cmap].white
        plt.axvline(true_params['J'], c=color)
        plt.axhline(true_params['τ'], c=color)
        plt.show()

    return loglikelihood

if __name__ == "__main__":
    main()
