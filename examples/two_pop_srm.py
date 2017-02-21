# -*- coding: utf-8 -*-
"""
Created Wed Jan 25 2017

author: Alexandre René
"""

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

model_params = srm.Spiking.Parameters(
    N = np.array((50, 10)),
    c = (4, 2),
    Js = ((3, -6), (6, 1)),
    Jr = (3, 3),
    τabs = (0.002, 0.002),
    τm = (0.01, 0.01),
    τs = (0.002, 0.002)
    )
hist_params = {'t0': 0,
               'tn': 2,
               'dt': 0.001}  # Make this smaller

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
Ahist = spiking_model.A


# Activity model

# ahist = history.Series(**hist_params, shape=params.N.shape)
# Ahist = history.Series(**hist_params, shape=params.N.shape)

# activity_model = srm.Activity(model_params, ahist, Ahist, input_hist, rndstream)

def do_it():
    # Compute the model
    spike_hist.set()
    Ahist.set()

    # Save it
    io.save("2-pop-srm-example.dat", spiking_model)

    # Plot the result
    plt.plot(Ahist.get_time_array(), Ahist.get_trace())


if __name__ == "__main__":
    do_it()
