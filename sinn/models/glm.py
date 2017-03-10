# -*- coding: utf-8 -*-
"""
Created Fri Mar 10 2017

author: Alexandre René
"""

import numpy as np
import scipy as sp
from collections import namedtuple, OrderedDict

import theano_shim as shim
import sinn
import sinn.config as config
import sinn.histories as histories
import sinn.kernels as kernels
import sinn.models as models

lib = shim.lib
Model  = models.Model
ModelKernelMixin = models.ModelKernelMixin
Kernel = kernels.Kernel

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
    Parameters = sinn.define_parameters

    class K(ModelKernelMixin, kernels.ExpKernel):
        pass

    def __init__(self, params, activity_history, input_history,
                 random_stream=None, memory_time=None):
        self.A = activity_history,
        self.I = input_history
        self.rndstream = random_stream
        # This runs consistency tests on the parameters
        Model.same_shape(self.A, self.I)
        Model.same_dt(self.A, self.I)
        Model.output_rng(self.A, self.rndstream)

        super().__init__(params)

        self.ρ = histories.Series(self.A, "ρ") # track hazard function

        # Compute Js*A+I before convolving, so we only convolve once
        # (We shamelessly abuse of unicode support for legibility)
        self.JᕽAᐩI = histories.Series(self.A, "JsᕽAᐩI",
                                     shape = self.A.shape,
                                     f = lambda t: lib.dot(params.J, self.A[t]) + self.I[t])
                                                              # NxN  dot  N   +  N

        self.add_history(self.A)
        self.add_history(self.I)
        self.add_history(self.ρ)
        self.add_history(self.JᕽAᐩI)

        self.A.set_update_function(self.A_fn)
        self.ρ.set_update_function(self.ρ_fn)

        κshape = np.max(params.J.shape)
        κparams = kernels.ExpKernel.Parameters(
            height = params.J,
            decay_const = params.τ,
            t_offset = 0)
        self.κ = kernels.ExpKernel('κ', params, κshape)
        self.add_kernel(κ)

    def ρ_fn(t):
        return self.params.c * lib.exp(self.κ.convolve(self.params.J.dot(self.A), t))
    def var_fn(t):
        return self.params.self.ρ[t] * (1 - self.ρ[t])

    def A_fn(t):
        return shim.binomial(size = self.A.shape,
                             n = self.params.N,
                             p = self.ρ[t] * self.A.dt) / self.params.N / self.A.dt

    def loglikelihood():
        # We approximate the pdf at each bin by a Gaussian
        # (Poisson would probably be better, but it's less convenient)

        A_arr = self.A.get_trace()

        # Binomial mean: Np = Nρdt.  E(A) = E(B)/N/dt = ρ
        ρ_arr = self.ρ.get_trace()

        # Binomial variance: Np(1-p)  V(A) = V(B)/(Ndt)² = ρ(1/dt - ρ)/N
        v_arr = ρarr * (1/self.A.dt - ρarr) / self.params.N

        # Log-likelihood: Σ -log σ + (x-μ)²/2σ² + cst
        l = lib.sum( -lib.log(lib.sqrt(v_arr)) + (A_arr - ρ_arr)**2 / 2 / v_arr )

        return l




