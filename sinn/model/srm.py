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

if config.use_theano:
    import theano
    import theano.tensor as T
    from theano.tensor.shared_randomstreams import RandomStreams  # CPU only
    #from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams  # CPU & GPU

Model  = com.Model
Kernel = com.Kernel

# =============================================================================
#
# Renewal model for activity (no adaptation)
#
# =============================================================================

class Activity(Model):

    # Parameters tuple defined after kernels

    #################
    # kernel definitions

    Parameters = namedtuple('Parameters',
                            set(['N', 'c', 'Js']).union( set(K.Parameters._fields),
                                                         set(H.Parameters._fields) ) )

    def η2_fn(self, t):
        """The refractory kernel coming after the absolute refractory period"""
        return self.params.Jr * lib.exp(-(t-self.params.τabs)/self.params.τm)

    def __init__(self):

        lib.seed(314)

        ###########################################
        # Excitatory component h

        # Initialize series objects used in computation
        # (We shamelessly abuse of unicode support for legibility)
        lib.check(A.dt == I.dt)
        JsᕽAᐩI = history.Series(A.t0,
                                A.tn,
                                A.dt,
                                A.shape,
                                lambda t: self.Js*A[t] + self.I[t])
        κ = com.ExpKernel('κ',
                          1, params.τm,
                          t0=params.τ)
        JsᕽAᐩI.pad(κ.memory_time)
        self.h = history.Series(A.t0, A.tn, A.dt, A.shape)
        if A._cur_tidx >= len(A):
            # A already has all the data; we can calculate h in one go



        ##########################################
        # Inhibitory component θ

        M = int(round((self.κ.memory_time + 1) / A.dt))
            # num of bins of history to use in computations

        ########################
        # Refractory component η
        tarr_η = np.arange(A.dt,
                           (M + 2)*A.dt - sinn.config.abs_tolerance,
                           A.dt)[::-1]
            # Less numerical errors if we start at A.dt and flip array
            # +1 because we want to include memory_idx_time
            # np. b/c there is no need to use a Theano variable here
        tarr_η[0] = inf
        η2 = η2_fn(tarr_η)
        if not config.use_theano:
            # If τabs is not a multiple of dt, we will have important numerical errors
            lib.check(np.isclose(params.τabs % A.dt, 0,
                                 rtol=config.rel_tolerance,
                                 atol=config.abs_tolerance ) )
        abs_refrac_idx_len = lib.cast_varint16( lib.round(params.τabs / A.dt) ) - 1
            # -1 because tarr_η starts at dt
        self.η = lib.ifelse(abs_refrac_idx_len > 0,
                            lib.set_subtensor(η2[-abs_refrac_idx_len:] = lib.inf),
                            η2)


    def a_fn(self, tidx):

        # Update θ
        θ = self.η

        # Update h
        h = self.κ.convolve(self.JsᕽAᐩI, tidx)

        # Update ρ
        ρ = self.c * lib.exp(h - θ)

        # Update occupation numbers
        # (occN[0] is a superfluous bin used to store an
        #  intermediate value – don't use it later)
        # ( superfluous bin => len(occN) = len(ρ) + 1 )
        self.occN[:-1] = self.occN[1:] * (1 - self.ρ*self.A.dt)
        self.occN[0] += self.occN[1]
        self.occN[-1] = self.A[tidx - 1]

        # Compute the new a
        return lib.sum( self.ρ * self.occN[1:], axis=1 )

    def A_fn(self, tidx):
        if sinn.config.use_theano:
            return self.rndstream.normal(self.shape,
                                         avg=self.a[tidx],
                                         std=T.sqrt(self.a[tidx]/self.params.N/dt))
        else:
            return np.random.normal(loc=self.a[tidx],
                                    scale=np.sqrt(self.a[tidx]/self.params.N/dt),
                                    size=self.shape)


    def compute_range_a(self, start, stop):

        if not use_theano:
            lib.check( isinstance(start, int)
                       and isinstance(stop, int)
                       and 0 <= start < stop )
            #range_len = stop - start
            #a = np.zeros( (range_len,) + self.shape )

            for tidx in range(start, stop):
                a_fn(tidx)
        else:
            res_a, res_ρ, res_occN = theano.scan(self._update_function,
                                                 )

            if start == stop == None:
                #Avoid Theano indexing call if possible
                return res_a, res_ρ[-1], res_occN[-1]
            else:
                return res_a[start:stop], res_ρ[-1], res_occN[-1]
