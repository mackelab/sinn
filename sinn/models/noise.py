# -*- coding: utf-8 -*-
"""
Created Wed Feb 2 2017

author: Alexandre René
"""

import numpy as np
#import scipy as sp
from collections import namedtuple, OrderedDict

import theano_shim as shim
import sinn.config as config
import sinn.common as com
import sinn.models.common

Model = sinn.models.common.Model

class GaussianWhiteNoise(Model):
    """
    Values will limited to the range ±clip_limit;
    """

    Parameter_info = OrderedDict( ( ( 'std',        (config.floatX, 1.0, False) ),
                                    ( 'shape',      ('int32', (1,), False) ),
                                    ( 'clip_limit', ('int8', 87,   False)   ) ) )
        # exp(88) is the largest value scipy can store in a 32-bit float
    Parameters = com.define_parameters(Parameter_info)
        # TODO: Move to Model constructor

    def __init__(self, params, history, random_stream):
        self.rndstream = random_stream
        super().__init__(params, history)
        history._iterative = False # It's white noise => no dependence on the past

    def eval(self, t):
        # 'shape' and 'clip_limit' are not parameters we want to treat
        # with Theano, and since all parameters are shared, we use
        # `get_value()` to get a pure NumPy value.
        if shim.isscalar(t):
            outshape = self.params.shape.get_value()
        else:
            assert(t.ndim==1)
            outshape = shim.concatenate((t.shape, self.params.shape.get_value()))
                # Shape is an array, not a tuple
        return shim.clip(self.rndstream.normal(avg  = 0,
                                                   std  = self.params.std/np.sqrt(self.history.dt),
                                                   size = outshape),
                             -self.params.clip_limit.get_value(),
                             self.params.clip_limit.get_value())


class Step(Model):
    Parameter_info = OrderedDict( ( ( 'height', (config.floatX, None, False) ),
                                    ( 'begin',  (config.floatX, None, False) ),
                                    ( 'end',    (config.floatX, None, False) ) ) )
    Parameters = com.define_parameters(Parameter_info)

    def __init__(self, params, history):
        super().__init__(params, history)
        history._iterative = False # Value at t does not depend on past

    def eval(self, t):
        return self.params.height if self.params.begin < t < self.params.end else 0
