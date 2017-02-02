# -*- coding: utf-8 -*-
"""
Created Wed Jan 25 2017

author: Alexandre René
"""

import numpy as np
import scipy as sp
from collections import namedtuple, OrderedDict

import sinn
import sinn.config as config
import sinn.lib as lib
import sinn.history as history
import sinn.model.srm as srm
import sinn.model.noise as noise
import sinn.theano_shim as shim

model_params = srm.Activity.Parameters(
    N = (500, 100),
    c = (4, 2),
    Js = np.array(((3, 5), (5, 3))),
    τabs = np.array((0.002, 0.002)),
    τm = np.array((0.01, 0.01))
    )

hist_params = {t0: 0,
               tn: 2,
               dt: 0.002,
               shape: (2,)}

noise_params = noise.GaussianWhiteNoise.Parameters(
    std = (1, 1),
    shape = (2,)
    )

input_hist = history.Series(hist_params.t0, hist_params.tn, hist_params.dt, hist_params.shape)
ahist = history.Series(hist_params)
Ahist = history.Series(hist_params)

shim.RandomStreams(seed=314)

input_model = noise.GaussianWhiteNoise(noise_params, input_hist)
activity_model = srm.Activity(model_params, ahist, Ahist, input_hist)



plt.plot(Ahist.get_time_array(), Ahist.get_trace())

