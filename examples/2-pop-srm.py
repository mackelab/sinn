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


model_params = srm.Activity.Parameters(
    N = (500, 100),
    c = (4, 2),
    Js = np.array(((3, 5), (5, 3))),
    τabs = 0.002,
    τm = 0.01)

hist_params = {t0: 0,
               tn: 2,
               dt: 0.002,
               shape: (2,)}

ahist = history.Series(hist_params)
Ahist = history.Series(hist_params)
activity_model = srm.Activity(model_params, ahist, Ahist)

plt.plot(Ahist.get_time_array(), Ahist.get_trace())
