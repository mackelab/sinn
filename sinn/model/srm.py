# -*- coding: utf-8 -*-
"""
Created Wed Jan 25 2017

author: Alexandre René
"""

import numpy as np
import scipy as sp
from collections import namedtuple, OrderedDict

import sinn
import sinn.com as com
import sinn.config as config
import sinn.theano_shim as shim
import sinn.history as history
import sinn.kernel as kernel
import sinn.model as model
import sinn.model.com

lib = shim.lib
Model  = model.com.Model
Kernel = com.Kernel
#Parameter = com.Parameter

# =============================================================================
#
# Renewal model for activity (no adaptation)
#
# =============================================================================

class Activity(Model):

    # Parameters tuple defined after kernels

    #################
    # kernel definitions

    Parameter_info = OrderedDict( ( 'N'   , np.int ),
                                  ( 'c'   , config.cast_floatX ),
                                  ( 'Js'  , config.cast_floatX ),
                                  ( 'τa'  , config.cast_floatX ),
                                  ( 'τm'  , config.cast_floatX ),
                                  ( 'τabs', config.cats_floatX ) )
    Parameters = com.define_parameters(Parameter_info)

    def η2_fn(self, t):
        """The refractory kernel coming after the absolute refractory period"""
        return self.params.Jr * lib.exp(-(t-self.params.τabs)/self.params.τm)

    def __init__(self, params,
                 activity_history, activity_mean_history, input_history,
                 memory_time=None, init_occupation_numbers=None):

        self.A = activity_history
        self.a = activity_mean_history
        self.I = input_history

        ###########################################
        # Excitatory component h
        ###########################################

        κ = kernel.ExpKernel('κ',
                          height      = 1,
                          decay_const = params.τm,
                          t0          = params.τabs,
                          memory_time = memory_time)

        # Initialize series objects used in computation
        # (We shamelessly abuse of unicode support for legibility)
        shim.check(self.A.dt == self.I.dt)
        self.JsᕽAᐩI = history.Series(self.A.t0,
                                    self.A.tn,
                                    self.A.dt,
                                    self.A.shape,
                                    lambda t: self.Js*A[t] + self.I[t])
        self.JsᕽAᐩI._cur_tidx
            # FIXME: Set this internally, maybe by defining +,* ops

        self.JsᕽAᐩI.pad(κ.memory_time)
        #self.h = history.Series(A.t0, A.tn, A.dt, A.shape)

        self.a.set_update_function(self.a_fn)
        self.a.set_range_update_function(self.compute_range_a)

        if self.A._cur_tidx >= len(A):
            # A already has all the data; we can calculate h in one go
            # Caching mechanism takes care of actually remembering the result
            self.κ.convolve(JsᕽAᐩI)
        else:
            self.A.set_update_function(self.make_A_fn(self.a))


        ##########################################
        # Model memory time & padding
        #########################################
        self.memory_time = κ.memory_time
        #self.a.pad()
        #self.A.pad()

        ##########################################
        # Inhibitory component θ
        #
        # Indices go as:
        # η[0]   -> η[(M-1)dt]
        # ...
        # η[M-2] -> η[dt]
        ##########################################

        M = int(round((self.memory_time + 1) / self.A.dt))
            # num of bins of history to use in computations
        tarr_η = np.arange(M-1, 0, -1) * self.A.dt
            # np. b/c there is no need to use a Theano variable here

        if init_occupation_numbers is not None:
            self.set_init_occupation_numbers(init_occupation_numbers)

        ########################
        # Refractory component η
        η2 = η2_fn(tarr_η)
        lib.check(np.isclose(params.τabs % self.A.dt, 0,
                             rtol=config.rel_tolerance,
                             atol=config.abs_tolerance ) )
            # If τabs is not a multiple of dt, we will have important numerical errors
        abs_refrac_idx_len = shim.cast_varint16( lib.round(params.τabs / self.A.dt) ) - 1
            # -1 because tarr_η starts at dt
        self.η = lib.ifelse(abs_refrac_idx_len > 0,
                            lib.set_subtensor(η2[-abs_refrac_idx_len:] = lib.inf),
                            η2)
            # By setting η this way, we ensure that a call to set_subtensor
            # is only made if necessary

        ########################
        # Adaptation component θ

        # TODO

    def set_init_occupation_numbers(self, init_occN='quiescent'):
        """
        Parameters
        ----------
        init_occN: string, ndarray
            If a string, one of
                - 'quiescent':
                    No neurons have ever fired (i.e. they are all
                    binned as having fired at -∞.
            If an array, it should have the following structure:
                - occN[0] -> no. of neurons that fired at time -∞
                - occN[1] -> no. of neurons that fired at time t0 - (M-1)dt
                - ...
                - occN[M-1] -> no. of neurons that fired at time t0 - dt
            where `M` is the number of bins we keep in memory before
            lumping all neurons in the ∞-bin.
        """

        self._occN_arr = np.zeros(M+1, dtype=config.floatX)
            # M+1 because the extra bin occN[0] is used for
            # an intermediate calculation
            # Although we never use it, _occN_arr underlies occN,
            # and thus must remain in memory for the entire lifetime
            # of this class.
            #
            # occN structure:
            # occN[0] -> undefined
            # occN[1] -> occN[-∞]
            # occN[2] -> occN[t0 - (M-1)dt]
            # ...
            # occN[M] -> occN[t0 - dt]
        if init_occN in ['quiescent', None]:
            self._occN_arr[:] = 0
            self._occN_arr[1] = self.params.N
        else:
            self._occN_arr[1:] = init_occN
            shim.check(lib.sum(self.occN[1:]) == self.params.N)
        self.occN = shim.shared(self._occN_arr)

#        self.rndstream = shim.RandomStreams(seed=314)

    def check_indexing(t):
        # On reason this test might fail is if A and a have different padding
        # and indices are used (rather than times)
        assert(self.a._tarr[self.a.get_t_idx(t)] == self.A._tarr[A.get_t_idx])

    def a_onestep(self, t, lastA, JsᕽAᐩI, occN):

        # Update θ
        θ = self.η

        # Update h
        h = self.κ.convolve(JsᕽAᐩI, t)

        # Update ρ
        ρ = self.c * lib.concatenate( ( np.ones(self.a.shape), # ∞-bin
                                        lib.exp(h - θ)),       # rest
                                     axis=0)

        # Update occupation numbers
        # (occN[0] is a superfluous bin used to store an
        #  intermediate value – don't use it except in next line)
        # ( superfluous bin => len(occN) = len(ρ) + 1 )
        new_occN = lib.concatenate(
                           (occN[1:] * (1 - self.ρ*self.a.dt),
                            lastA.dimshuffle('x', 0)),
                           axis=0)
        # Combine bins 0 and 1 into bin 1
        new_new_occN += shim.inc_subtensor(new_occN[1], new_occN[0])

        # Compute the new a
        return (lib.sum( self.ρ * new_new_occN[1:], axis=1 ),
                {occN: new_new_occN})

    def a_fn(self, t):
        # Check that the indexing in a and A match
        self.check_indexing(t)
        # Compute a(t)
        return a_onestep(self, t, self.A[t - self.A.dt], self.JsᕽAᐩI, self.occN)

    def compute_range_a(self, t_array):
        """
        Parameters
        ----------
        t_array: ndarray
            Array of floats or ints. If floats, interpreted as times; if ints,
            interpreted as *indices* to a._tarr.
            The elements of `t_array` must be in ascending order and correspond
            to successive steps (no time bins are skipped). For performance
            reasons this is not enforced, so do be careful.
        """

        lib.check( t_array[0] < t_array[-1] )
            # We don't want to check that the entire array is ordered; this is a compromise
        lib.check( t_array[0] == self.cur_tidx + 1 )
            # Because we only store the last value of occN, calculation
            # must absolutely be done iteratively
        self.check_indexing(t_array[0])
            # Check that the indexing in a and A match

        if not use_theano:

            def loop(t):
                res_a, updates = a_fn(t)
                occN.set_value(updates[self.occN])
                return res_a

            return [loop(t) for t in t_array]

        else:
            (res_a, updates) = theano.scan(self.a_onestep,
                                           sequences=[t_array,
                                                      self.A[start:stop]]
                                           outputs_info=[None],
                                           non_sequences=[self.JsᕽAᐩI, self.occN],
                                           name='a scan')
                # NOTE: time index sequence must be specified as a numpy
                # array, because History expects time indices to be
                # non-Theano variables
            return res_a, updates

    def A_fn(self, t):
        return self.rndstream.normal(size=self.A.shape,
                                     avg=self.a[t],
                                     std=lib.sqrt(self.a[t]/self.params.N/dt))

    def update_params(self, new_params):
        """Change parameter values, and refresh the affected kernels."""

        #TODO: change params

        #TODO: recompute affected kernels

        #TODO: delete all discretizations of affected kernels

        raise NotImplementedError
