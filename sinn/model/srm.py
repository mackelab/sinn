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
import sinn.model.common as com

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

    def __init__(self, params, activity_history, activity_mean_history):

        lib.seed(314)
        self.A = activity_history
        self.a = activity_mean_history

        ###########################################
        # Excitatory component h

        # Initialize series objects used in computation
        # (We shamelessly abuse of unicode support for legibility)
        lib.check(self.A.dt == I.dt)
        JsᕽAᐩI = history.Series(self.A.t0,
                                self.A.tn,
                                self.A.dt,
                                self.A.shape,
                                lambda t: self.Js*A[t] + self.I[t])
        κ = com.ExpKernel('κ',
                          1, params.τm,
                          t0=params.τ)
        JsᕽAᐩI.pad(κ.memory_time)
        self.h = history.Series(A.t0, A.tn, A.dt, A.shape)

        self.a.set_update_function(self.a_fn)
        self.A.set_update_function(self.make_A_fn(self.a))

        if A._cur_tidx >= len(A):
            # A already has all the data; we can calculate h in one go



        ##########################################
        # Inhibitory component θ

        M = int(round((self.κ.memory_time + 1) / A.dt))
            # num of bins of history to use in computations
        self._occN_arr = np.zeros(M+1, dtype=config.floatX)
            # M+1 because the extra bin occN[0] is used for
            # an intermediate calculation
            # Although we never use it, _occN_arr underlies occN,
            # and thus must remain in memory for the entire lifetime
            # of this class.
        self._occN_arr[1] = params.N
        self.occN = shim.shared(self._occN_arr)

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
            # By setting η this way, we ensure that a call to set_subtensor
            # is only made if necessary


        self.a.pad()
        self.A.pad()

    def init_occupation_numbers(self, init_occN):
        if init_occN == 'fully quiescent':
            self.occN[:] = 0
            self.occN[1] = self.params.N
        else:
            self.occN[1:] = init_occN
            shim.check(lib.sum(self.occN[1:]) == self.params.N)

    def a_fn(self, tidx, lastA, JsᕽAᐩI, occN):

        # Update θ
        θ = self.η

        # Update h
        h = self.κ.convolve(JsᕽAᐩI, tidx)

        # Update ρ
        ρ = self.c * lib.exp(h - θ)

        # Update occupation numbers
        # (occN[0] is a superfluous bin used to store an
        #  intermediate value – don't use it except in next line)
        # ( superfluous bin => len(occN) = len(ρ) + 1 )
        new_occN = lib.concatenate(
                           (occN[1:] * (1 - self.ρ*self.dt),
                            lastA.dimshuffle('x', 0)),
                           axis=0)
        # Combine bins 0 and 1 into bin 1
        new_new_occN += shim.inc_subtensor(new_occN[1], new_occN[0])

        # Compute the new a
        return (lib.sum( self.ρ * new_new_occN[1:], axis=1 ),
                {occN: new_new_occN})

    def compute_range_a(self, start, stop):

        lib.check( isinstance(start, int)
                   and isinstance(stop, int)
                   and 0 <= start < stop )
        lib.check( start == self.cur_tidx + 1 )
            # Because we only store the last value of occN, calculation
            # must absolutely be done iteratively

        if not use_theano:

            occN = self.occN
            updates = None # Declare outside loop

            def loop(tidx):
                nonlocal occN
                nonlocal updates
                res_a, updates = a_fn(tidx, occN)
                occN = updates[self.occN]
                return res_a

            return [loop(tidx) for tidx in range(start, stop)]

        else:
            (res_a, updates) = theano.scan(
                                          self.a_fn,
                                          sequences=[np.arange(start, stop),
                                                     self.A[start:stop]]
                                          outputs_info=[None],
                                          non_sequences=[self.JsᕽAᐩI, self.occN],
                                          name='a scan')
                # NOTE: time index sequence must be specified as a numpy
                # array, because History expects time indices to be
                # non-Theano variables
            return res_a, updates

    def A_fn(self, tidx):
        if sinn.config.use_theano:
            return self.rndstream.normal(self.shape,
                                         avg=self.a[tidx],
                                         std=T.sqrt(self.a[tidx]/self.params.N/dt))
        else:
            return np.random.normal(loc=self.a[tidx],
                                    scale=np.sqrt(self.a[tidx]/self.params.N/dt),
                                    size=self.shape)
