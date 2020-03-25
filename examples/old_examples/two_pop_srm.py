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
import sinn.models.srm as srm
import sinn.models.noise as noise
import sinn.iotools as io
import sinn.analyze.analyze as anlz
import sinn.analyze.sweep as sweep

logger = logging.getLogger('sinn.examples.two_pop_srm')

_DEFAULT_DATAFILE = "2-pop-srm-example.dat"

#import dill
#dill.settings['recurse'] = True

def init_spiking_model():
    model_params = srm.Spiking.Parameters(
        N = np.array((500, 100)),
        c = (4, 2),
        Js = ((3, -6), (6, 1)),
        Jr = (3, 3),
        τabs = (0.002, 0.002),
        τm = (0.01, 0.01),
        τs = (0.002, 0.002)
        )
    hist_params = {'t0': 0,
                   'tn': 0.1,
                   'dt': 0.0001} # ideally 1000th the smallest time constant

    # Noisy input

    rndstream = shim.RandomStreams(seed=314)
    noise_params = noise.GaussianWhiteNoise.Parameters(
        std = (2, 2),
        shape = (2,)
        )

    input_hist = histories.Series(name='I', shape=model_params.N.shape, **hist_params)
    input_model = noise.GaussianWhiteNoise(noise_params, input_hist, rndstream)


    # Full spiking model

    spike_hist = histories.Spiketimes(name='spikehist', pop_sizes=model_params.N, **hist_params)
    #Ahist = histories.Spiketimes(**hist_params, shape=(len(model_params.N),))

    spiking_model = srm.Spiking(model_params, spike_hist, input_hist, rndstream)
    #Ahist = spiking_model.A


    # Activity model

    # ahist = histories.Series(**hist_params, shape=params.N.shape)
    # Ahist = histories.Series(**hist_params, shape=params.N.shape)

    # activity_model = srm.Activity(model_params, ahist, Ahist, input_hist, rndstream)

    return spiking_model

def generate(filename = _DEFAULT_DATAFILE):
    compute_data = True
    try:
        # Try to load precomputed data
        spiking_model = io.load(filename)
    except FileNotFoundError:
        pass
    else:
        compute_data = False

    if compute_data:
        # Data don't exist, so compute them
        spiking_model = init_spiking_model()
        spiking_model.spikehist.compute_up_to(-1)
        spiking_model.A.set()  # Ensure all time points are computed

        # Save the new data
        io.save("2-pop-srm-example.dat", spiking_model)

    # try:
    #     plot_activity(spiking_model.A)
    # except Exception as e:
    #     logger.warn("Unable to plot output. You may need to "
    #                 "install matplotlib or one of its backends.")
    #     logger.error("The error raised was: \n{}".format(str(e.args)))

def plot_activity(A_history):

    Asmooth = anlz.smooth(A_history, 20)

    # Plot the result
    anlz.plot(Asmooth)


def plot_posterior(model_filename = _DEFAULT_DATAFILE,
                   posterior_filename = None):
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
    logposterior.set_floor(logposterior.max() - 10)
    logposterior.set_norm('linear')
    ax, cb = anlz.plot(logposterior)
        # analyze recognizes logposterior as a heat map, and plots accordingly
        # anlz.plot returns a tuple of all plotted objects. For heat maps there
        # are two: the heat map axis and the colour bar
    ax.set_xlim((2, 8))
    ax.set_ylim((0.01, 1))

    plt.show()
    return


def compute_posterior(output_filename, target_dt, model_filename=_DEFAULT_DATAFILE, ipp_url_file=None, ipp_profile=None):
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
    try:
        # Try to load precomputed data
        spiking_model = io.load(model_filename)
    except FileNotFoundError:
        raise FileNotFoundError("Unable to find data file. To create one, run "
                                "`generate`")

    # Typically, spiking is generated with a finer time resolution
    # than we need for activity, so we need to smooth/subsample the result
    subsamplefactor = int(round(target_dt / spiking_model.A.dt))
    Ahist = anlz.subsample(spiking_model.A, subsamplefactor)
        # This is our data
    ahist = histories.Series(Ahist, 'a')
        # This will compute and store the expected mean, which depends
        # on the data and the parameters
    Ihist = anlz.subsample(spiking_model.I, subsamplefactor)
        # The input also has to be subsampled to match the data

    # Initialize the SRM activity model
    params = spiking_model.params
    activity_model = srm.Activity(params,
                                  activity_history = Ahist,
                                  activity_mean_history = ahist,
                                  input_history = Ihist)

    activity_model.set_init_occupation_numbers('quiescent')
    # Construct the arrays of parameters to try
    fineness = 1
    burnin = 0#0.5
    data_len = 4.0
    param_sweep = sweep.ParameterSweep(activity_model)
    Js_sweep = sweep.linspace(-1, 10, fineness)
    # τa_sweep = sweep.logspace(0.001, 1, fineness)
    τm_sweep = sweep.logspace(0.0005, 0.5, fineness)
    param_sweep.add_param('Js', idx=(0,0), axis_stops=Js_sweep)
    param_sweep.add_param('τm', idx=(1,), axis_stops=τm_sweep)
    def f(model):
        return model.loglikelihood(burnin, burnin + data_len)
    param_sweep.set_function(f, 'log $L$')


    # Try to connect to the ipyparallel cluster
    if ipp_url_file is not None:
        import ipyparallel as ipp
        ippclient = ipp.Client(url_file=ipp_url_file)
    elif ipp_profile is not None:
        import ipyparallel as ipp
        ippclient = ipp.Client(profile=ipp_profile)
    else:
        ippclient = None
    if ippclient is not None:
        # Initialize the environment in each cluster process
        ippclient[:].use_dill().get()
            # More robust pickling
        # ippclient[:].execute("import sinn")
        # ippclient[:].execute("import sinn.models.srm")
        # ippclient[:].execute("import sinn.kernels")
        # ippclient[:].execute("import sinn.histories")
        # ippclient[:].execute("from sinn.models.srm import Kkernel")

    # Compute the posterior
    logposterior = param_sweep.do_sweep(output_filename, ippclient)
        # This can take a long time
        # The result will be saved in output_filename

    return logposterior


if __name__ == "__main__":
    #generate()
    compute_posterior("logposterior.dat", 0.002, "2-pop-srm-example.dat", ipp_profile="default")
