# -*- coding: utf-8 -*-
"""
Created Fri Mar 10 2017

author: Alexandre René
"""

import numpy as np
import scipy as sp
from collections import namedtuple, OrderedDict
import logging

import theano_shim as shim
import sinn
import sinn.config as config
import sinn.histories as histories
import sinn.kernels as kernels
import sinn.models as models

logger = logging.getLogger("sinn.models.glm")

Model = models.Model
ModelKernelMixin = models.ModelKernelMixin
Kernel = kernels.Kernel

# This seems to be a bug in dill, that K can't be a member of the model class
class ExpK(ModelKernelMixin, kernels.ExpKernel):
    @staticmethod
    def get_kernel_params(model_params):
        return kernels.ExpKernel.Parameters(
            height = 1,
            decay_const = model_params.τ,
            t_offset = 0)



class GLM_exp_kernel(Model):

    # Entries to Parameter_info: ( 'parameter name',
    #                              (dtype, default value, shape_flag) )
    # If the shape_flag is True, the parameter will be reshaped into a 2d
    # matrix, if it isn't already. This is ensures the parameter is
    # consistent with kernel methods which assume inputs which are at least 2d
    # The last two options can be omitted; default flag is 'False'
    # Typically if a parameter will be used inside a kernel, shape_flag should be True.
    Parameter_info = OrderedDict( ( ( 'N', np.int ),
                                    ( 'c', config.cast_floatX ),
                                    ( 'J', (config.cast_floatX, None, True) ),
                                    ( 'τ', (config.cast_floatX, None, True) )) )
    Parameters = sinn.define_parameters(Parameter_info)

    def __init__(self, params, activity_history, input_history,
                 random_stream=None, memory_time=None):
        self.A = activity_history
        self.I = input_history
        self.rndstream = random_stream
        # This runs consistency tests on the parameters
        Model.same_shape(self.A, self.I)
        Model.same_dt(self.A, self.I)
        Model.output_rng(self.A, self.rndstream)

        super().__init__(params)
        # NOTE: Do not use `params` beyond here. Always use self.params.

        self.ρ = histories.Series(self.A, "ρ") # track hazard function

        # Compute Js*A+I before convolving, so we only convolve once
        # (We shamelessly abuse of unicode support for legibility)
        self.JᕽAᐩI = histories.Series(self.A, "JsᕽAᐩI",
                                      shape = self.A.shape,
                                      f = self.JᕽAᐩI_fn)

        self.add_history(self.A)
        self.add_history(self.I)
        self.add_history(self.ρ)
        self.add_history(self.JᕽAᐩI)

        self.A.add_inputs([self.ρ])
        self.ρ.add_inputs([self.JᕽAᐩI])
        self.JᕽAᐩI.add_inputs([self.A, self.I])

        self.A.set_update_function(self.A_fn)
        self.ρ.set_update_function(self.ρ_fn)

        κshape = self.params.N.shape
        self.κ = ExpK('κ', self.params, κshape, memory_time=memory_time)
        self.add_kernel(self.κ)

        self.JᕽAᐩI.pad(self.κ.memory_time)

    def JᕽAᐩI_fn(self, t):
        return shim.lib.dot(self.params.J, self.A[t]) + self.I[t]
                              # NxN  dot  N   +  N

    def ρ_fn(self, t):
        return self.params.c * shim.exp(self.κ.convolve(self.JᕽAᐩI, t))

    def A_fn(self, t):
        p_arr = sinn.clip_probabilities(self.ρ[t] * self.A.dt)
        return self.rndstream.binomial(size = self.A.shape,
                                       n = self.params.N,
                                       p = self.ρ[t] * self.A.dt) / self.params.N / self.A.dt

    def A_range_fn(self, t_array):
        shim.check( t_array[0] < t_array[-1] )
            # We don't want to check that the entire array is ordered; this is a compromise
        shim.check( t_array[0] == self.a._cur_tidx + 1 )
            # Because we only store the last value of occN, calculation
            # must absolutely be done iteratively

        if not shim.is_theano_object(self.A._data):
            def loop(t):
                res = self.A_fn(t)
                # If you have any shared variables, update them here
                return res

            A_lst = [loop(t) for t in t_array]
            return A_lst

        else:
            res, upds = theano.scan(self.A_onestep,
                                    sequences = t_array,
                                    non_sequences = [self.JᕽAᐩI],
                                    name = 'A scan')


    def update_params(self, new_params):
        if np.all(new_params.J == self.params.J):
            self.JᕽAᐩI.lock = True
        else:
            self.JᕽAᐩI.lock = False
        super().update_params(new_params)

    def loglikelihood(self, start=None, stop=None):

        # We deliberately use times here (instead of indices) for start/
        # stop so that they remain consistent across different histories
        if start is None:
            start = self.A.t0
        else:
            start = self.A.get_time(start)
        if stop is None:
            stop = self.A.tn
        else:
            stop = self.A.get_time(stop)

        A_arr = self.A[start:stop]

        # Binomial mean: Np = Nρdt.  E(A) = E(B)/N/dt = ρ
        ρ_arr = self.ρ[start:stop]
        # if self.A.lock and self.I.lock:
        #    self.JᕽAᐩI.lock = True
        # TODO: lock JᕽAᐩI in a less hacky way

        #------------------
        # True log-likelihood

        # Number of spikes
        k_arr = (A_arr * self.A.dt * self.params.N).astype('int16')
        # Spiking probabilities
        p_arr = sinn.clip_probabilities(ρ_arr * self.A.dt)

        # loglikelihood: -log k! - log (N-k)! + k log p + (N-k) log (1-p) + cst
        # We use the Stirling approximation for the second log
        l = shim.lib.sum( -shim.lib.log(sp.misc.factorial(k_arr, exact=True))
                     -(self.params.N-k_arr)*shim.lib.log(self.params.N - k_arr)
                     + self.params.N-k_arr
                     + k_arr*shim.lib.log(p_arr) + k_arr*shim.lib.log(1-p_arr) )
            # with exact=True, factorial is computed only once for whole array

        return l

        #-----------------
        # Gaussian approximation to the log-likelihood

        # Binomial variance: Np(1-p)  V(A) = V(B)/(Ndt)² = ρ(1/dt - ρ)/N
        v_arr = ρ_arr * (1/self.A.dt - ρ_arr) / self.params.N

        # Log-likelihood: Σ -log σ + (x-μ)²/2σ² + cst
        l = shim.lib.sum( -shim.lib.log(shim.lib.sqrt(v_arr)) + (A_arr - ρ_arr)**2 / 2 / v_arr )

        return l




