# -*- coding: utf-8 -*-
"""
Created Wed Feb 2 2017

author: Alexandre René
"""

import numpy as np
#import scipy as sp
from collections import namedtuple, OrderedDict

import sinn.config as config
import sinn.theano_shim as shim
import sinn.common as com
import sinn.model.common
Model = sinn.model.common.Model

class GaussianWhiteNoise(Model):
    """
    Values will limited to the range ±clip_limit;
    """

    Parameter_info = OrderedDict( ( ( 'std',        (config.cast_floatX, 1.0) ),
                                    ( 'shape',      (np.int, (1,)) ),
                                    ( 'clip_limit', (np.int, 87)   ) ) )
        # exp(88) is the largest value scipy can store in a 32-bit float
    Parameters = com.define_parameters(Parameter_info)
        # TODO: Move to Model constructor

    def __init__(self, params, history, random_stream):
        self.rndstream = random_stream
        super().__init__(params, history)

    def eval(self, t):
        return np.clip(self.rndstream.normal(avg  = 0,
                                             std  = self.params.std/np.sqrt(self.history.dt),
                                             size = self.params.shape),
                       -self.params.clip_limit,
                       self.params.clip_limit)


class Step(Model):
    Parameter_info = OrderedDict( ( ( 'height', config.cast_floatX ),
                                    ( 'begin',  config.cast_floatX ),
                                    ( 'end',    config.cast_floatX ) ) )
    Parameters = com.define_parameters(Parameter_info)

    def eval(self, t):
        return self.params.height if self.params.begin < t < self.params.end else 0