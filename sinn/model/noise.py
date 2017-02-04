# -*- coding: utf-8 -*-
"""
Created Wed Feb 2 2017

author: Alexandre René
"""

import numpy as np
import scipy as sp
from collections import namedtuple, OrderedDict

import sinn.config as config
from . import common as com
Model = com.Model
Parameter = com.Parameter

class GaussianWhiteNoise(Model):
    """
    Values will limited to the range ±clip_limit;
    """

    Parameter_info = OrderedDict{'std':        (config.cast_floatX, 1.0),
                                 'shape':      (np.int, (1,)),
                                 'clip_limit': (np.int, 87)}
        # exp(88) is the largest value scipy can store in a 32-bit float
    Parameters = com.define_parameters(Parameter_info)
        # TODO: Move to Model constructor

    def eval(self, t):
        return sp.clip(sp.random.normal(0, self.std/sp.sqrt(dt),  shape),
                       -self.clip_limit, clip_limit)


class Step(Model):
    Parameter_info = OrderedDict{'height': config.cast_floatX,
                                 'begin':  config.cast_floatX,
                                 'end':    config.cast_floatX}
    Parameters = com.define_parameters(Parameter_info)

    def eval(self, t):
        return self.params.height if self.params.begin < t < self.params.end else 0
