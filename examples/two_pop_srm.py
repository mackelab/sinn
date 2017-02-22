# -*- coding: utf-8 -*-
"""
Created Wed Jan 25 2017

author: Alexandre René
"""

import logging
import numpy as np
import scipy as sp
from collections import namedtuple, OrderedDict

import theano_shim as shim
import sinn
import sinn.config as config
import sinn.history as history
import sinn.model.srm as srm
import sinn.model.noise as noise
import sinn.iotools as io

logger = logging.getLogger('sinn.two_pop_srm')

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
                   'tn': 40,
                   'dt': 0.00001} # 100th the smallest time constant

    # Noisy input

    rndstream = shim.RandomStreams(seed=314)
    noise_params = noise.GaussianWhiteNoise.Parameters(
        std = (2, 2),
        shape = (2,)
        )

    input_hist = history.Series(name='I', shape=model_params.N.shape, **hist_params)
    input_model = noise.GaussianWhiteNoise(noise_params, input_hist, rndstream)


    # Full spiking model

    spike_hist = history.Spiketimes(name='spikehist', pop_sizes=model_params.N, **hist_params)
    #Ahist = history.Spiketimes(**hist_params, shape=(len(model_params.N),))

    spiking_model = srm.Spiking(model_params, spike_hist, input_hist, rndstream)
    #Ahist = spiking_model.A


    # Activity model

    # ahist = history.Series(**hist_params, shape=params.N.shape)
    # Ahist = history.Series(**hist_params, shape=params.N.shape)

    # activity_model = srm.Activity(model_params, ahist, Ahist, input_hist, rndstream)

    return spiking_model

def do_it():
    compute_data = True
    filename = "2-pop-srm-example.dat"
    try:
        # Try to load precomputed data
        spiking_model = io.load(filename)
    except EOFError:
        logger.warning("File {} is corrupted or empty. A new "
                       "one is being computed, but you should "
                       "delete this one.".format(filename))
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

    try:
        plot_activity(spiking_model.A)
    except:
        logger.warn("Unable to plot output. You may need to "
                    "install matplotlib or one of its backends.")

def plot_activity(A_history):
    import matplotlib.pyplot as plt
    import sinn.analyze as anlz

    Asmooth = anlz.smooth(A_history, 20)

    # Plot the result
    anlz.plot(Asmooth)


if __name__ == "__main__":
    do_it()
