# -*- coding: utf-8 -*-
"""
Created Wed Jan 25 2017

author: Alexandre René
"""

import numpy as np
import scipy as sp
from collections import namedtuple, OrderedDict

import sinn
import sinn.common as com
import sinn.config as config
import sinn.theano_shim as shim
import sinn.history as history
import sinn.kernel as kernel
import sinn.model as model
import sinn.model.common

lib = shim.lib
Model  = model.common.Model
Kernel = kernel.Kernel
#Parameter = com.Parameter

# =============================================================================
#
# Renewal model for activity (no adaptation)
#
# =============================================================================

class Activity(Model):
    """
    State variables
    ---------------
    occN:
        Occupation number of each lag bin
        At the start of the computation for a(t), occN holds the
        occupation number for each bin at t-2.
        It is first updated to t-1, which allows the computation
        of the _expeBdbQuit:cted_ spike count at t for each bin
    normalized_E_bin_spikes:
        Expected number of spikes for each bin.
        At the start of the computation for a(t), normalized_E_bin_spikes
        the expected number of spikes at t-1 for each bin,
        normalized to the total number of spikes (such that
        sum(normalized_E_bin_spikes) == 1 is always true).
        The new value for t is calculated at the very end, after a(t).
    """

    # Parameters tuple defined after kernels

    #################
    # kernel definitions

    Parameter_info = OrderedDict( ( ( 'N'   , np.int ),
                                    ( 'c'   , config.cast_floatX ),
                                    ( 'Js'  , config.cast_floatX ),
                                    ( 'Jr'  , config.cast_floatX ),
                                   #( 'τa'  , config.cast_floatX ),
                                    ( 'τs' , config.cast_floatX ),
                                    ( 'τm' , config.cast_floatX ),
                                    ( 'τabs', config.cast_floatX ) ) )
    Parameters = com.define_parameters(Parameter_info)

    def η2_fn(self, t):
        """The refractory kernel coming after the absolute refractory period"""
        kernel_dims = shim.asarray(self.params.τabs).ndim
            # Shape of the η kernel
        shim.check(kernel_dims == shim.asarray(self.params.τm).ndim)
        t = shim.add_axes(t, kernel_dims, 'right')
            # In order to broadcast properly, we need to add to t
            # the dimensions corresponding to the kernel

        # FIXME: This is apparently the same for both populations ??
        retval = self.params.Jr * lib.exp(-(t-self.params.τabs)/self.params.τm)
            # Both Jr and the τ should be indexed as [to idx][from idx], so
            # we just use broadcasting to multiply them
        if kernel_dims > 1:
            # If there's more than one kernel dimension, we need to contract the result
            if kernel_dims > 2:
                raise NotImplementedError
            retval = lib.sum(retval, axis=-1)

        return retval

        # TODO: Pull out the dimensions testing and branching into a reusable function `distributed_dot`
        # TODO: Stress-test this function
        # Jr_dims = shim.asarray(self.params.Jr).ndim
        # if kernel_dims == 0:
        #     shim.check(Jr_dims = 0)
        #     return self.params.Jr * lib.exp(-(t-self.params.τabs)/self.params.τm)

        # else:
        #     if kernel_dims == 1:
        #         if Jr_dims == 1:
        #             # Just do the element-wise product
        #             return (shim.add_axes(self.params.Jr, 1, 'before')
        #                     * lib.exp(-(t-self.params.τabs)/self.params.τm))
        #         elif Jr_dims == 2:
        #             # The following effectively distributes the dot product
        #             # np.dot(Jr, exp(…)) along the time axis
        #             return lib.tensordot(
        #                 lib.exp(-(t-self.params.τabs)/self.params.τm),
        #                 self.params.Jr,
        #                 (-1, -1))
        #         else:
        #             raise ValueError("Incoherent dimensions")
        #     elif kernel_dims == 2:
        #         if Jr_dims == 2:
        #             # The following effectively distributes the dot product
        #             # np.dot(Jr, exp(…)) along the time axis
        #             return shim.moveaxis(
        #                       lib.tensordot(
        #                           lib.exp(-(t-self.params.τabs)/self.params.τm),
        #                           self.params.Jr,
        #                           (-2, -1)),
        #                       -2, -1)
        #         else:
        #             raise NotImplementedError
        #     else:
        #         raise NotImplementedError

    def get_M(self, memory_time):
        """Convert a time into a number of bins."""
        # TODO: move to History
        return int(shim.round(self.memory_time / self.A.dt) + 1)

    def __init__(self, params,
                 activity_history, activity_mean_history, input_history,
                 random_stream,
                 memory_time=None, init_occupation_numbers=None):

        super().__init__(params, history)

        self.A = activity_history
        self.a = activity_mean_history
        self.I = input_history
        assert(self.A.shape == self.a.shape == self.I.shape)

        self.rndstream = random_stream

        self.spike_counts = self.A * (self.A.dt * params.N)
        self.occN = None
        self.normalized_E_bin_spikes = None

        ##########################################
        # Parameter sanity checks
        ##########################################

        # TODO

        ###########################################
        # Excitatory component h
        ###########################################

        κparams = kernel.ExpKernel.Parameters(
            height      = 1,
            decay_const = params.τm,
            t_offset    = params.τs )
        self.κ = kernel.ExpKernel('κ',
                             params      = κparams,
                             t0          = np.min(params.τs),
                             memory_time = memory_time,
                             shape       = (1,) + self.A.shape )
            # Here the kernel doesn't actually mix the populations
            # (that's done by Js), so it's the same shape as A
            # (plus the time dimension for the θ lags)

        # TODO:
        # assert(self.κ.compatible_with(Ahist))

        # Initialize series objects used in computation
        # (We shamelessly abuse of unicode support for legibility)
        shim.check(self.A.dt == self.I.dt)
        self.JsᕽAᐩI = history.Series(self.A.t0,
                                    self.A.tn,
                                    self.A.dt,
                                    shape = self.A.shape,
                                    f = lambda t: lib.dot(params.Js, self.A[t]) + self.I[t])
                                                              # NxN  dot  N   +  N
        self.JsᕽAᐩI._cur_tidx
            # FIXME: Set this internally, maybe by defining +,* ops

        self.JsᕽAᐩI.pad(self.κ.memory_time)
        #self.h = history.Series(A.t0, A.tn, A.dt, A.shape)

        self.a.set_update_function(self.a_fn)
        self.a.set_range_update_function(self.compute_range_a)

#        import pdb; pdb.set_trace()
        if self.A._cur_tidx >= len(self.A) - 1:
            # A already has all the data; we can calculate h in one go
            # Caching mechanism takes care of actually remembering the result
            self.κ.convolve(self.JsᕽAᐩI)
        else:
            self.A.set_update_function(self.A_fn)


#        import pdb; pdb.set_trace()
        ##########################################
        # Model memory time & padding
        #########################################
        self.memory_time = self.κ.memory_time
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

        M = self.get_M(self.memory_time)
            # num of bins of history to use in computations
        tarr_η = np.arange(M-1, 0, -1) * self.A.dt
            # np. b/c there is no need to use a Theano variable here

        if init_occupation_numbers is not None:
            self.set_init_occupation_numbers(init_occupation_numbers)

        ########################
        # Refractory component η
        η2 = self.η2_fn(tarr_η)
        shim.check(np.isclose(params.τabs % self.A.dt, 0,
                              rtol=config.rel_tolerance,
                              atol=config.abs_tolerance ).all() )
            # If τabs is not a multiple of dt, we will have important numerical errors
        abs_refrac_idx_len = shim.largest(shim.cast_varint16( shim.round(params.τabs / self.A.dt) ) - 1, 0)
            # -1 because tarr_η starts at dt
            # clip at zero because otherwise τabs=0  =>  -1
        start_idcs = len(tarr_η) - abs_refrac_idx_len
            # The time array is flipped, so the start idx of the absolute refractory
            # period is at the end.
        self.η = shim.ifelse(lib.any(shim.gt(abs_refrac_idx_len, 0)),
                             lib.stack( [ shim.set_subtensor(η2[start_idx:,i], shim.inf)
                                          for i, start_idx in enumerate(start_idcs) ] ),
                             η2)
        # self.η = shim.ifelse(lib.any(shim.gt(abs_refrac_idx_len, 0)),
        #                      lib.stack(
        #                          [ lib.stack(
        #                             [ shim.set_subtensor(η2[start_idx:,i,j], shim.inf)
        #                                for j, start_idx in enumerate(start_idx_row) ] )
        #                             for i, start_idx_row in enumerate(start_idcs) ] ),
        #                      η2)
            # By setting η this way, we ensure that a call to set_subtensor
            # is only made if necessary

        ########################
        # Adaptation component θ

        # TODO

    def set_init_occupation_numbers(self, init_occN='quiescent'):
        """
        […]
        Also instantiates the normalized_E_bin_spikes variable to
        zero (this allows assigning to it later)

        The number of bins to keep is determined by self.memory_time,
        which is set in the model's initializer.

        Parameters
        ----------
        init_occN: string, ndarray
            If a string, one of
                - 'quiescent':
                    No neurons have ever fired (i.e. they are all
                    binned as having fired at -∞.)
                - 'constant':
                    Neurons are uniformly binned over the memory time; none
                    have fired at -∞. Note that this depends on the number
                    of bins, and especially on the memory time.
            If an array, it should have the following structure:
                - occN[0] -> no. of neurons that fired at time -∞
                - occN[1] -> no. of neurons that fired at time t0 - (M-1)dt
                - ...
                - occN[M-1] -> no. of neurons that fired at time t0 - dt
            where `M` is the number of bins we keep in memory before
            lumping all neurons in the ∞-bin.
        """
        # TODO: I don't you actually need to keep _occN_arr

        # self.memory_time corresponds to the time after which, if a
        # neuron still hasn't fired, it is considered equivalent to one
        # which last fired at -∞. So we use that here to determine the
        # number of bins.
        M = self.get_M(self.memory_time)

        self._occN_arr = np.zeros((M+1,) + self.A.shape, dtype=config.floatX)
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
        elif init_occN == 'constant':
            self._occN_arr[0] = 0
            self._occN_arr[1:] = self.params.N / M
        else:
            self._occN_arr[1:] = init_occN
            shim.check(lib.sum(self.occN[1:]) == self.params.N)
        self.occN = shim.shared(self._occN_arr)

        # Instantiated the normalized_E_bin_spikes
        # It has one less element along the 0 axis
        shape = (self._occN_arr.shape[0] - 1,) + self._occN_arr.shape[1:]
        self.normalized_E_bin_spikes = shim.shared(np.zeros(shape))

    def check_indexing(self, t):
        # On reason this test might fail is if A and a have different padding
        # and indices are used (rather than times)
        assert(self.a._tarr[self.a.get_t_idx(t)]
               == self.spike_counts._tarr[self.spike_counts.get_t_idx(t)])

    def a_onestep(self, t, last_spike_count, JsᕽAᐩI, occN, last_normalized_E_bin_spikes):

        # TODO: Adjust A shape before, so we don't need to add_axes

        # Update occupation numbers to t-1

        # (occN[0] is a superfluous bin used to store an
        #  intermediate value – don't use it except in next line)
        # ( superfluous bin => len(occN) = len(ρ) + 1 )
        new_occN = lib.concatenate(
                           (occN[1:] - last_spike_count * last_normalized_E_bin_spikes,
                            shim.add_axes(last_spike_count, 1, 'before')),
                           axis=0)
        # Combine bins 0 and 1 into bin 1
        new_new_occN = shim.inc_subtensor(new_occN[1], new_occN[0])

        E_spikes, normalized_E_bin_spikes = self.a_onestep_2nd_half(t, JsᕽAᐩI, new_new_occN)

        return (E_spikes,
                {occN: new_new_occN,
                 last_normalized_E_bin_spikes: normalized_E_bin_spikes})

    def a_onestep_2nd_half(self, t, JsᕽAᐩI, occN):
        # Update θ
        θ = self.η

        # Update h
#        h = lib.sum( self.κ.convolve(JsᕽAᐩI, t), axis=1 )
            # Kernels are [to idx][from idx], so to get the total contribution to
            # population i, we sum over axis 1 (the 'from indices')
#        import pdb; pdb.set_trace()
        h = self.κ.convolve(JsᕽAᐩI, t)

        # Update ρ
        ρ = self.params.c * lib.concatenate( ( shim.add_axes(np.ones(self.a.shape), 1, 'before'), # ∞-bin
                                               lib.exp(h - θ)),       # rest
                                             axis=0)

        # Compute the new a(t)
        # In the writeup we set axis=1, because we sum once for the entire
        # series, so t is another dimension
        E_bin_spikes = self.a.dt * ρ * occN[1:]
        E_spikes = lib.sum(E_bin_spikes, axis=0, keepdims=True)
        normalized_E_bin_spikes = E_bin_spikes / E_spikes

#        import pdb; pdb.set_trace()
        return E_spikes[0], normalized_E_bin_spikes
            # Index E_spikes[0] to remove the dimension we kept

    def a_fn(self, t):
        # Check that the indexing in a and A match
        self.check_indexing(t)
        # Compute a(t)
        spikes, updates = self.a_onestep(t, self.spike_counts[self.spike_counts.get_t_idx(t) - 1], self.JsᕽAᐩI,
                                         self.occN, self.normalized_E_bin_spikes)
        res_a = spikes / self.a.dt / self.params.N
            # a_onestep returns spike counts, so we must convert into activities
        return res_a, updates

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

        shim.check( t_array[0] < t_array[-1] )
            # We don't want to check that the entire array is ordered; this is a compromise
        shim.check( t_array[0] == self.a._cur_tidx + 1 )
            # Because we only store the last value of occN, calculation
            # must absolutely be done iteratively
        self.check_indexing(t_array[0])
            # Check that the indexing in a and A match

        new_a, new_nEbs = self.a_onestep_2nd_half(
              t_array[0], self.JsᕽAᐩI, self.occN)
        new_a = new_a / self.a.dt / self.params.N
            # a_onestep_2nd_half actually returns a spike count: convert to activity

        if not config.use_theano:

            self.a.update(self.a.get_t_idx(t_array[0]), new_a)
            self.normalized_E_bin_spikes.set_value(new_nEbs)

            def loop(t):
                res_a, updates = self.a_fn(t)  # Already converts spike counts to activities
                self.occN.set_value(updates[self.occN])
                self.normalized_E_bin_spikes.set_value(updates[self.normalized_E_bin_spikes])
                return res_a

            a_lst = [loop(t) for t in t_array[1:]]
            # Make sure to include the first a we calculated
            assert(new_a.shape == a_lst[0].shape)
                # Until recently (version 1.11?) NumPy was ignoring the keepdims argument
                # on subclasses of ndarray (such as shim.shared variables), which lead to
                # problems here. If this fails, try upgrading NumPy.
            a_lst.insert(0, new_a)
            return a_lst

        else:
            # The A required for a_onestep is the one before t, so we offset by one
            start = self.A.get_t_idx(t_array[0])
                # no -1 because we are actually starting at t_array[0] + 1
            stop = self.A.get_t_idx(t_array[-1])
                # no -1 because stop is an exclusive bound, which cancels the offset
            (res_spikes, updates) = theano.scan(self.a_onestep,
                                                sequences=[t_array[1:],
                                                           self.A[start:stop]],
                                                outputs_info=[new_a],
                                                non_sequences=[self.JsᕽAᐩI, self.occN, new_nEbs],
                                                name='a scan')
                # NOTE: time index sequence must be specified as a numpy
                # array, because History expects time indices to be
                # non-Theano variables
            res_a = res_spikes / self.a.dt / self.params.N
                # Convert spike counts to activities

            # Reassign the output to the original shared variable
            updates[self.normalized_E_bin_spikes] = updates.pop(new_nEbs)
            # Include the originally calculated a in the returned value
            returned_a = lib.concatenate((shim.add_axes(new_a, 1, 'before'), res_a),
                                       axis = 0)

            return returned_a, updates

    def A_fn(self, t):
        return self.rndstream.normal(size=self.A.shape,
                                     avg=self.a[t],
                                     std=lib.sqrt(self.a[t]/self.params.N/self.A.dt))

    def update_params(self, new_params):
        """Change parameter values, and refresh the affected kernels."""

        #TODO: change params

        #TODO: recompute affected kernels

        #TODO: delete all discretizations of affected kernels

        raise NotImplementedError
