# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 2017

Author: Alexandre René
"""

import numpy as np
import scipy as sp
#from scipy.integrate import quad
#from collections import namedtuple
import logging
logger = logging.getLogger(__name__)
from collections import OrderedDict, Sequence, Iterable
from inspect import isclass
from itertools import chain

import theano_shim as shim
import mackelab.utils as utils
import mackelab.theano
from mackelab.theano import GraphCache, CompiledGraphCache
import sinn.config as config
import sinn.common as com
import sinn.histories
import sinn.kernels
import sinn.diskcache as diskcache

_models = {}
registered_models = _models.keys()
    # I don't really like this, but it works. Ideally it would be some kind
    # of read-only property of the module.
    # Registering models allows us to save the model name within a parameter
    # file, and write a function which can build the correct model
    # automatically based on only on that parameter file.

expensive_asserts = True

failed_build_msg = (
        "Failed to build the symbolic update. Make sure that the "
        "model's definition of State is correct: it should include "
        "enough histories to fully define the model's state and "
        "allow forward integration. If you are sure the problem is "
        "not your model, you may need to workaround the issue by "
        "defining a `symbolic_update` in your model class. "
        "Automatic construction of symbolic updates is still work in "
        " progress and not always possible.")

def register_model(model, modelname=None):
    """
    Register a subclass of Model.
    Typically this is called from the module which implements the subclass.
    If `modelname` is unspecified, `model`'s class name is used.
    """
    global _models
    assert(isclass(model))
    if modelname is None:
        modelname = model.__name__
    assert(isinstance(modelname, str))
    _models[modelname] = model

def is_registered(modelname):
    """Returns True if a model is registered with this name."""
    global _models
    return modelname in _models

def get_model(modelname, *args, **kwargs):
    """Retrieves the model associated with the model name. Same arguments as dict.get()."""
    global _models
    return _models.get(modelname, *args, **kwargs)

def make_placeholder(history, name_suffix=' placeholder'):
    """
    Return a symbolic variable representating a time slice of
    :param:history.
    TODO: Add support for >1 lags.
    """
    return shim.tensor(history.shape, history.name + name_suffix, history.dtype)

def _graph_batch_bounds(model, start, stop, batch_size):
    """ Internal function for :func:batch_function_scan. """
    start = model.get_tidx(start)
    if stop is None and batch_size is None:
        raise TypeError("Batch function requires that `start` and"
                        "one of `stop`, `batch_size` be specified.")
    elif batch_size is None:
        stop = model.get_tidx(stop)
        batch_size = stop - start
    elif stop is None:
        batch_size = model.index_interval(batch_size)
        stop = start + batch_size
    else:
        logger.warning("Both `stop` and `batch_size` were provided "
                       "to a batch function. This is probably an "
                       "error, and if it isn't, make sure that "
                       "they are consistent.")
    return start, stop, batch_size

def batch_function_scan(*inputs):
    """
    To be used as a decorator. Uses `scan` to construct the vectors
    of values constituting a batch, iterating from `start` to `stop`.

    Parameters
    ----------
    inputs: list of str
        Each string must correspond exactly to the identifier for one
        of the model's histories. (They are retrieved with °gettattr`.)
        The slice corresponding to a batch for each defined input will
        be passed to the function, in the order defined by :param:inputs.

    Example
    -------

    >>> import numpy as np
    >>> from Collections import namedtuple
    >>> from odictliteral import odict
    >>> import sinn
    >>> from sinn.histories import Series
    >>> from sinn.models import Model
    >>> class MyModel(Model):
            requires_rng = True
            Parameter_info = odict['θ': 'floatX']
            Parameters = sinn.define_parameters(Parameter_info)
            State = namedtuple
            def __init__(self, params, random_stream=None):
                self.A = Series('A', shape=(1,), time_array=np.arange(1000))
                self.rndstream = random_stream
                super().__init__(params=params, reference_history=self.A)
                self.A.set_update_function(
                    lambda t: self.rndstream.normal(avg=self.θ))

            @batch_function_scan('A', 'a')
            def logp(A):
                # Squared error
                return ((A - self.θ)**2).sum()

    >>> θ = 0
    >>> model = MyModel(MyModel.Parameters(θ=θ))
    >>> model.A.set(np.random.normal(loc=θ)
    >>> model.logp(start=200, batch_size=500)
    >>> model.params.θ.set_value(1)
    >>> model.logp(start=200, batch_size=500)

    """
    def decorator(f):
        def wrapped_f(self, start, stop=None, batch_size=None):
            """Either :param:stop or :param:batch are required."""
            start, stop, batch_size = \
                _graph_batch_bounds(self, start, stop, batch_size)
                # Returns integer indices

            # Define a bunch of lists of histories and indices to be able to
            # permute inputs between the order of `self.State` and that defined
            # by `inputs`.
            statehists = list(self.unlocked_statehists)
            locked_statehists = list(self.locked_statehists)
            inputhists = [getattr(self, name) for name in inputs]
                # The set of histories which are included in the function
                # May include both state and non-state histories
            lockedstateinputs = [h for h in inputhists
                                   if h in locked_statehists]
            nonstateinputs  = [h for h in inputhists
                                 if h not in self.statehists]
            stateinputs     = [h for h in inputhists if h in statehists]
            stateinput_idcs = [(i, statehists.index(h))
                               for i, h in enumerate(inputhists)
                               if h in statehists]

            # Construct the initial values
            if not shim.is_theano_object(start):
                assert all(h.cur_tidx >= self.get_tidx_for(start-1, h)
                           for h in chain(nonstateinputs, statehists))
                assert all(h.cur_tidx >= self.get_tidx_for(stop-1, h)
                           for h in locked_statehists)
            initial_values = [h._data[self.get_tidx_for(start-1, h)]
                              for h in chain(nonstateinputs, statehists)]

            if shim.cf.use_theano:
                def onestep(tidx, *args):
                    for x, name in zip(
                        utils.flatten(
                            tidx, *args, terminate=shim.cf._TerminatingTypes),
                        utils.flatten(
                            'tidx (scan)',
                            #(h.name + ' (scan)' for h in lockedstateinputs),
                            (h.name + ' (scan)' for h in nonstateinputs),
                            (h.name + ' (scan)' for h in statehists),
                            terminate=shim.cf._TerminatingTypes)):
                        if getattr(x, 'name', None) is None:
                            x.name = name
                    m = len(nonstateinputs)
                    _nonstateinputs = args[:m]
                    _state = args[m:]
                    assert(len(_state) == len(statehists))
                    _stateinputs = [_state[j] for i,j in stateinput_idcs]
                    state_outputs, updates = self.symbolic_update(tidx, *_state)
                    nonstate_outputs, nonstate_updates = self.nonstate_symbolic_update(
                        tidx, nonstateinputs, _state, state_outputs)
                    assert len(set(updates).intersection(nonstate_updates)) == 0
                    updates.update(nonstate_updates)
                    return nonstate_outputs + state_outputs, updates

            else:
                def onestep(tidx, *args):
                    # There are no symbolic state updates if we are using NumPy
                    return ([h[self.get_tidx_for(tidx, h)] for h in inputhists],
                            OrderedDict())

            # Accumulate over the batch
            if batch_size == 1:
                # No need for scan
                outputs, updates = onestep(start, *initial_values)
                # Add the batch dimension which scan would have created
                outputs = [o[np.newaxis,...] for o in outputs]
            else:
                outputs, updates = shim.scan(
                    onestep, sequences=shim.arange(start, stop),
                    outputs_info=initial_values,
                    return_list=True)
            assert(len(outputs) == len(nonstateinputs) + len(statehists))

            # Permute the outputs so they are in the order expected by `f`
            finputs = [None]*len(inputs)
            m = len(nonstateinputs)
            for h in lockedstateinputs:
                i = inputhists.index(h)
                _start = self.get_tidx_for(start, h)
                _stop = self.get_tidx_for(stop, h)
                finputs[i] = h._data[_start:_stop]
            for h, o in zip(nonstateinputs, outputs[:m]):
                i = inputhists.index(h)
                finputs[i] = o
            for h, o in zip(statehists, outputs[m:]):
                if h in inputhists:
                    i = inputhists.index(h)
                    finputs[i] = o
            assert all(i is not None for i in finputs)

            # Evaluate `f`.
            return f(self, *finputs)
                # `f` is still a function while being decoratod, so we need
                # to explicitly pass `self`

        return wrapped_f
    return decorator


class Model(com.ParameterMixin):
    """Abstract model class.

    A model implementations should derive from this class.
    It must minimally provide:
    - A `Parameter_info` dictionary of the form:
        (See sinn.common.Parameterize)
        ```
        Parameter_info = OrderedDict{ 'param_name': Parameter([cast function], [default value]),
                                      ... }
        ```
    - A class-level (outside any method) call
        `Parameters = com.define_parameters(Parameter_info)`

    Implementations may also provide class methods to aid inference:
    - likelihood: (params) -> float
    - likelihood_gradient: (params) -> vector
    If not provided, `likelihood_gradient` will be calculated by appyling theano's
    grad method to `likelihood`. (TODO)
    As class methods, these don't require an instance – they can be called on the class directly.
    """
    def __init__(self, params, public_histories, reference_history=None):
        # History is optional because more complex models have multiple histories.
        # They should keep track of them themselves.
        # ParameterMixin requires params as a keyword parameter
        """
        Parameters
        ----------
        params: self.Parameters instance

        public_histories: Ordered iterable
            List of histories that were explicitly passed to the model
            initializer. This allows to recover the histories that were passed
            (as usually they contain the desired output) without needing to
            know their internal name.

        reference_history: History instance
            If provided, a reference is kept to this history: Model evaluation
            may require querying some of its attributes, such as time step (dt).
            These attributes are allowed not to be constant.
            If no reference is given, the first history in `public_histories`
            is used.
        """
        self.graph_cache = GraphCache('sinn.models.cache', type(self),
                                      modules=(sinn.models.common,))
        self.compile_cache = CompiledGraphCache('sinn.models.compilecache')
            # TODO: Add other dependencies within `sinn.models` ?

        # Format checks
        if not hasattr(self, 'requires_rng'):
            raise SyntaxError("Models require a `requires_rng` bool attribute.")

        super().__init__(params=params)
        self.kernel_list = []
        self.history_set = set()
        self.history_inputs = sinn.DependencyGraph('model.history_inputs')
        self.compiled = {}  # DEPRECATED
        self._pymc = None  # Initialized with `self.pymc` property

        if not isinstance(public_histories, (Sequence, OrderedDict)):
            raise TypeError("`public_histories` (type: {}) must be an ordered "
                            "container type. Recognized types: Sequence, "
                            "OrderedDict.".format(type(public_histories)))
        self.public_histories = public_histories

        if reference_history is not None:
            self._refhist = reference_history
            #self.add_history(reference_history)  # TODO: Possible to remove ?
        elif len(public_histories) > 0:
            self._refhist = self.public_histories[0]
        else:
            self._refhist = None

        # Ensure rng attribute exists
        if not hasattr(self, 'rng'):
            self.rng = None

        # Create symbolic variables for batches
        # Any symbolic function on batches should use these, that way
        # other functions can retrieve the symbolic input variables.
        self.batch_start_var = shim.symbolic.scalar('batch_start',
                                                    dtype=self.tidx_dtype)
        self.batch_start_var.tag.test_value = 1
            # Must be large enough so that test_value slices are not empty
        self.batch_size_var = shim.symbolic.scalar('batch_size',
                                                   dtype=self.tidx_dtype)
            # Must be large enough so that test_value slices are not empty
        self.batch_size_var.tag.test_value = 2


    def __getattribute__(self, attr):
        """
        Retrieve parameters if their name does not clash with an attribute.
        """
        # Use __getattribute__ to maintain current stack trace on exceptions
        # https://stackoverflow.com/q/36575068
        if (attr != 'params' and hasattr(self, 'params')
            and isinstance(self.params, Iterable)
            and attr in self.params._fields):
            return getattr(self.params, attr)
        else:
            return super().__getattribute__(attr)

    def set_reference_history(self, reference_history):
        if self._refhist is None:
            raise RuntimeError("Reference history for this model is already set.")
        self._refhist = reference_history

    @property
    def statehists(self):
        return utils.FixedGenerator(
            ( getattr(self, varname) for varname in self.State._fields ),
            len(self.State._fields) )

    @property
    def unlocked_statehists(self):
        return (h for h in self.statehists if not h.locked)

    @property
    def locked_statehists(self):
        return (h for h in self.statehists if h.locked)

    # TODO: Put the `if self._refhist is not None:` bit in a function decorator
    @property
    def t0(self):
        return self._refhist.t0

    @property
    def tn(self):
        return self._refhist.tn

    @property
    def t0idx(self):
        return 0
        #return self._refhist.t0idx

    @property
    def tnidx(self):
        return self._refhist.tnidx - self._refhist.t0idx + self.t0idx

    @property
    def tidx_dtype(self):
        return self._refhist.tidx_dtype

    @property
    def dt(self):
        if self._refhist is not None:
            return self._refhist.dt
        else:
            raise AttributeError("Reference history for this model is not set.")

    @property
    def cur_tidx(self):
        if self._refhist is not None:
            return self._refhist._cur_tidx - self._refhist.t0idx + self.t0idx
        else:
            raise AttributeError("Reference history for this model is not set.")

    @property
    def unlocked_histories(self):
        return (h for h in self.history_set if not h.locked)

    @property
    def pymc(self):
        """
        The first access should be done as `model.pymc()` to instantiate the
        model. Subsequent access, which retrieves the already instantiated
        model, should use `model.pymc`.
        """
        if getattr(self, '_pymc', None) is None:
            import sinn.models.pymc3
                # Don't require that PyMC3 be installed unless we need it
            return sinn.models.pymc3.PyMC3ModelWrapper(self)
                # PyMC3ModelWrapper assigns the PyMC3 model to `self._pymc`
        return self._pymc

    def get_tidx(self, t, allow_rounding=False):
        """
        Returns the time index corresponding to t, with 0 corresponding to t0.
        """
        if self._refhist is not None:
            if shim.istype(t, 'int'):
                return t
            else:
                return self._refhist.get_tidx(t, allow_rounding) - self._refhist.t0idx + self.t0idx
        else:
            raise AttributeError("The reference history for this model was not set.")
    get_t_idx = get_tidx

    def get_tidx_for(self, t, target_hist, allow_fractional=False):
        if self._refhist is not None:
            ref_tidx = self.get_tidx(t) - self.t0idx + self._refhist.t0idx
            return self._refhist.get_tidx_for(
                ref_tidx, target_hist, allow_fractional=allow_fractional)
        else:
            raise AttributeError("Reference history for this model is not set.")

    def index_interval(self, Δt, allow_rounding=False):
        if self._refhist is not None:
            return self._refhist.index_interval(Δt, allow_rounding)
        else:
            raise AttributeError("The reference history for this model was not set.")

    def get_time(self, t):
        if self._refhist is not None:
            if shim.istype(t, 'float'):
                return t
            else:
                assert(shim.istype(t, 'int'))
                tidx = t - self.t0idx + self._refhist.t0idx
                return self._refhist.get_time(tidx)
        else:
            raise AttributeError("The reference history for this model was not set.")

    # Simple consistency check functions
    @staticmethod
    def same_shape(*args):
        assert(all(arg1.shape == arg2.shape for arg1, arg2 in zip(args[:-1], args[1:])))
    @staticmethod
    def same_dt(*args):
        assert(all(sinn.isclose(arg1.dt, arg2.dt) for arg1, arg2 in zip(args[:-1], args[1:])))
    @staticmethod
    def output_rng(outputs, rngs):
        """
        Parameters
        ----------
        outputs: History
            Can also be a list of Histories
        rngs: random stream, or list of random streams
            The random stream(s) required to generate the histories in
            `outputs`
        """
        if isinstance(outputs, sinn.histories.History):
            outputs = [outputs]
        else:
            assert(all(isinstance(output, sinn.histories.History) for output in outputs))
        try:
            len(rngs)
        except TypeError:
            rngs = [rngs]

        if any( not shim.isshared(outhist._data) for outhist in outputs ):
            # Bypass test for Theano data
            return

        if ( any( outhist._cur_tidx.get_value() < len(outhist) - 1 for outhist in outputs )
             and any( rng is None for rng in rngs)
             and not all( outhist.locked for outhist in outputs ) ) :
            raise ValueError("Cannot generate {} without the required random number generator(s).".format(str([outhist.name for outhist in outputs])))
        elif ( all( outhist._cur_tidx.get_value() >= len(outhist) - 1 for outhist in outputs )
             and any( rng is not None for rng in rngs) ) :
            logger.warning("Your random number generator(s) will be unused, "
                           "since your data is already generated.")

    def cache(self, obj):
        """
        Call this function on all Kernel and History objects that should be
        saved to the disk cache.
        This function is cheap to call: the object is only written out when
        it's removed from program memory.
        """

        if isinstance(obj, sinn.kernels.Kernel):
            logger.warning("Deprecated. Use add_kernel instead.")
            self.kernel_list.append(obj)
        else:
            assert(isinstance(obj, sinn.histories.History))
            logger.warning("Histories aren't written to disk. Use add_history instead")
            self.history_inputs.add(obj)

    def add_history(self, hist):
        assert(isinstance(hist, sinn.histories.History))
        self.history_set.add(hist)
        if hist not in self.history_inputs:
            self.history_inputs.add(hist)
    def add_kernel(self, kernel):
        assert(isinstance(kernel, sinn.kernels.Kernel))
        if kernel not in self.kernel_list:
            self.kernel_list.append(kernel)

    def eval(self):
        """
        Remove all symbolic dependencies by evaluating all ongoing updates.
        If the update is present in `shim`'s update dictionary, it's removed
        from there.
        """
        # Get the updates applied to the histories
        tidx_updates = {h._original_tidx: (h, h._cur_tidx)
                        for h in self.history_set
                        if h._original_tidx is not h._cur_tidx}
        data_updates = {h._original_data: (h, h._data)
                        for h in self.history_set
                        if h._original_data is not h._data}
        updates = OrderedDict( (k, v[1])
                               for k, v in chain(tidx_updates.items(),
                                                 data_updates.items()) )
        # Check that there are no dependencies
        if not shim.graph.is_computable(updates.values()):
            non_comp = [str(var) for var, upd in updates.items()
                                 if not shim.graph.is_computable(upd)]
            raise ValueError("A model can only be `eval`ed when all updates "
                             "applied to its histories are computable.\n"
                             "The updates to the following variables have "
                             "symbolic dependencies: {}.".format(non_compu))
        # Get the comp graph update dictionary
        shimupdates = shim.get_updates()
        for var, upd in updates.items():
            logger.debug("Evaluating update applied to {}.".format(var))
            if var in shimupdates:
                if shimupdates[var] == upd:
                    logger.debug("Removing update from CG update dictionary.")
                    del shimupdates[var]
                else:
                    logger.debug("Update differs from the one in CG update "
                                 "dictionary: leaving the latter untouched.")
            var.set_value(shim.eval(shim.cast(upd, var.dtype)))
        # Update the histories
        for orig in tidx_updates.values():
            h = orig[0]
            h._cur_tidx = h._original_tidx
        for orig in data_updates.values():
            h = orig[0]
            h._data = h._original_data

        # Ensure that we actually removed updates from the update dictionary
        assert len(shimupdates) == len(shim.get_updates())

    def theano_reset(self):
        """Put model back into a clean state, to allow building a new Theano graph."""
        for hist in self.history_inputs:
            if not hist.locked:
                hist.theano_reset()
        for kernel in self.kernel_list:
            kernel.theano_reset()

        if self.rng is not None and len(self.rng.state_updates) > 0:
            logger.warning("Erasing random number generator updates. Any "
                           "other graphs using this generator are likely "
                           "invalidated.\n"
                           "RNG: {}".format(self.rng))
            self.rng.state_updates = []
        #sinn.theano_reset() # theano_reset on histories will be called twice,
                            # but there's not much harm
        shim.reset_updates()

    def update_params(self, new_params):
        """
        Update model parameters. Clears all histories except those whose `locked`
        attribute is True, as well as any kernel which depends on these parameters.

        TODO: Make `new_params` a dict and just update parameters in the dict.

        Parameters
        ----------
        new_params: same type as model.params | dict
            # TODO: Allow also **kwargs
        """
        def gettype(param):
            return type(param.get_value()) if shim.isshared(param) else type(param)
        if isinstance(new_params, self.Parameters):
            assert(all( gettype(param) == gettype(new_param)
                        for param, new_param in zip(self.params, new_params) ))
        elif isinstance(new_params, dict):
            assert(all( gettype(val) == gettype(getattr(self.params, name))
                        for name, val in new_params.items() ))
        else:
            raise NotImplementedError
        logger.monitor("Model params are now {}. Updating kernels...".format(self.params))

        # HACK Make sure sinn.inputs and models.history_inputs coincide
        sinn.inputs.union(self.history_inputs)
        self.history_inputs.union(sinn.inputs)

        # Determine the kernels for which parameters have changed
        kernels_to_update = []
        if isinstance(new_params, self.Parameters):
            for kernel in self.kernel_list:
                if not sinn.params_are_equal(
                        kernel.get_parameter_subset(new_params), kernel.params):
                    # Grab the subset of the new parameters relevant to this kernel,
                    # and compare to the kernel's current parameters. If any of
                    # them differ, add the kernel to the list of kernels to update.
                    kernels_to_update.append(kernel)
        else:
            assert(isinstance(new_params, dict))
            for kernel in self.kernel_list:
                if any(param_name in kernel.Parameters._fields
                       for param_name in new_params):
                    kernels_to_update.append(kernel)

        # Now update parameters. This must be done after the check above,
        # because Theano parameters automatically propagate to the kernels.
        sinn.set_parameters(self.params, new_params)

        # Loop over the list of kernels whose parameters have changed to do
        # two things:
        # - Remove any cached binary op that involves this kernel.
        #   (And write it to disk for later retrievel if these parameters
        #    are reused.)
        # - Update the kernel itself to the new parameters.
        for obj in list(self.history_inputs) + self.kernel_list:
            if obj not in kernels_to_update:
                for op in obj.cached_ops:
                    for kernel in kernels_to_update:
                        if hash(kernel) in op.cache:
                            diskcache.save(op.cache[hash(kernel)])
                            # TODO subclass op[other] and define __hash__
                            logger.monitor("Removing cache for binary op {} ({},{}) from heap."
                                        .format(str(op), obj.name, kernel.name))
                            del op.cache[hash(kernel)]

        for kernel in kernels_to_update:
            diskcache.save(kernel)
            logger.monitor("Updating kernel {}.".format(kernel.name))
            kernel.update_params(self.params)

        self.clear_unlocked_histories()
        # TODO: If parameters change value but all keep their id (same shared
        #       variable), we don't need to call `clear_advance_function`.
        #       We should only call it when necessary, since it forces a
        #       recompilation of the graph.
        self.clear_advance_function()

    def clear_unlocked_histories(self):
        """Clear all histories that have not been explicitly locked."""
        #for hist in self.history_inputs.union(sinn.inputs):
        for hist in self.history_inputs:
            # HACK: Removal of sinn.inputs is a more drastic version attempt
            #       at correcting the same problem as fsgif.remove_other_histories
            if not hist.locked:
                self.clear_history(hist)

    def clear_advance_function(self):
        """
        Removes the compiled advance function, if present, forcing it to be
        recompiled if called again.
        We need to do this if any of the parameters change identity (e.g.
        replaced by another shared variable).
        """
        if hasattr(self, '_advance_fn'):
            del self._advance_fn

    def clear_other_histories(self):
        """
        Clear unlocked histories that are not explicitly part of this model
        (but may be inputs).
        """
        # Implemented as a wrapper around clear_unlocked_histories:
        # first lock of this model's histories, clear histories, and then
        # revert to the original locked/unlocked status
        old_status = {hist: hist.locked for hist in self.history_set}
        for hist in self.history_set:
            if not hist.locked:
                hist.lock(warn=False)
        self.clear_unlocked_histories()
        for hist, status in old_status.items():
            if status == False:
                hist.unlock()

    def clear_history(self, history):
        # Clear the history, and remove any cached operations related to it
        # In contrast to `update_params`, we don't write these operations to
        # disk, because histories are data structures: there's no way of knowing
        # if they're equivalent to some already computed case other than comparing
        # the entire data.
        logger.monitor("Clearing history " + history.name)
        history.clear()
        #if history in self.history_inputs.union(sinn.inputs):
        if history in self.history_inputs:
            # HACK: Removal of sinn.inputs is a more drastic version attempt
            #       at correcting the same problem as fsgif.remove_other_histories
            for obj in list(self.history_inputs) + self.kernel_list:
                for op in obj.cached_ops:
                    if hash(history) in op.cache:
                        del op.cache[hash(history)]
        else:
            for obj in list(self.history_inputs) + self.kernel_list:
                for op in obj.cached_ops:
                    if hash(history) in op.cache:
                        logger.error("Uncached history {} is member of cached "
                                     "op {}. This may indicate a memory leak."
                                     .format(history.name, str(op)))

    def remove_other_histories(self):
        """HACK: Remove histories from sinn.inputs that are not in this model.
        Can remove this once we store dependencies in histories rather than in
        sinn.inputs."""
        histnames = [h.name for h in self.history_set]
        dellist = []
        for h in sinn.inputs:
            if h.name not in histnames:
                dellist.append(h)
        for h in dellist:
            del sinn.inputs[h]

    def apply_updates(self, update_dict):
        """
        Theano functions which produce updates (like scan) naturally will not
        update the history data structures. This method applies those updates
        by replacing the internal _data and _cur_tidx attributes of the history
        with the symbolic expression of the updates, allowing histories to be
        used in subsequent calculations.
        """
        # Update the history data
        for history in self.history_set:
            if history._original_tidx in update_dict:
                assert(history._original_data in update_dict)
                    # If you are changing tidx, then surely you must change _data as well
                history._cur_tidx = update_dict[history._original_tidx]
                history._data = update_dict[history._original_data]
            elif history._original_data in update_dict:
                history._data = update_dict[history._original_data]

        # Update the shim update dictionary
        shim.add_updates(update_dict)

    def eval_updates(self, givens=None):
        """
        Compile and evaluate a function evaluating the `shim` update
        dictionary. Histories' internal _data and _cur_tidx are reset
        to be equal to _original_tidx and _original_data.
        If the updates have symbolic inputs, provide values for them through
        the `givens` argument.
        If there are no updates, no function is compiled, so you can use this
        as a safeguard at the top of a function to ensure there are no
        unapplied updates, without worrying about the cost of repeated calls.
        """
        upds = shim.get_updates()
        if len(upds) > 0:
            f = shim.graph.compile([], [], updates=upds, givens=givens)
            f()
            for h in self.history_set:
                if h._cur_tidx != h._original_tidx:
                    h._cur_tidx = h._original_tidx
                if h._data != h._original_data:
                    h._data = h._original_data

    def get_loglikelihood(self, *args, **kwargs):

        # Sanity check – it's easy to forget to clear histories in an interactive session
        uncleared_histories = []
        # HACK Shouldn't need to combine sinn.inputs
        # TODO Make separate function, so that it can be called within loglikelihood instead
        for hist in self.history_inputs.union(sinn.inputs):
            if ( not hist.locked and ( ( hist.use_theano and hist.compiled_history is not None
                                         and hist.compiled_history._cur_tidx.get_value() >= hist.t0idx )
                                       or (not hist.use_theano and hist._cur_tidx.get_value() >= hist.t0idx) ) ):
                uncleared_histories.append(hist)
        if len(uncleared_histories) > 0:
            raise RuntimeError("You are trying to produce a cost function graph, but have "
                               "uncleared histories. Either lock them (with their .lock() "
                               "method) or clear them (with their individual .clear() method "
                               "or the model's .clear_unlocked_histories() method). The latter "
                               "will delete data.\nUncleared histories: "
                               + str([hist.name for hist in uncleared_histories]))

        if sinn.config.use_theano():
            # TODO Precompile function
            def likelihood_f(model):
                if 'loglikelihood' not in self.compiled:
                    self.theano()
                        # Make clean slate (in particular, clear the list of inputs)
                    logL = model.loglikelihood(*args, **kwargs)
                        # Calling logL sets the sinn.inputs, which we need
                        # before calling get_input_list
                    # DEBUG
                    # with open("logL_graph", 'w') as f:
                    #     theano.printing.debugprint(logL, file=f)
                    input_list, input_vals = self.get_input_list()
                    self.compiled['loglikelihood'] = {
                        'function': theano.function(input_list, logL,
                                                    on_unused_input='warn'),
                        'inputs'  : input_vals }
                    self.theano_reset()

                return self.compiled['loglikelihood']['function'](
                    *self.compiled['loglikelihood']['inputs'] )
                    # * is there to expand the list of inputs
        else:
            def likelihood_f(model):
                return model.loglikelihood(*args, **kwargs)
        return likelihood_f

    def make_binomial_loglikelihood(self, n, N, p, approx=None):
        """
        Parameters
        ----------
        n: History
            Number of successful samples
        N: array of ints:
            Total number of samples.
            Must have n.shape == N.shape
        p: History
            Probability of success; first dimension is time.
            Must have n.shape == p.shape and len(n) == len(p).
        approx: str
            (Optional) If specified, one of:
            - 'low p': The probability is always very low; the Stirling approximation
              is used for the contribution from (1-p) to ensure numerical stability
            - 'high p': The probability is always very high; the Stirling approximation
              is used for the contribution from (p) to ensure numerical stability
            - 'Stirling': Use the Stirling approximation for both the contribution from
              (p) and (1-p) to ensure numerical stability.
            - 'low n' or `None`: Don't use any approximation. Make sure `n` is really low
              (n < 20) to avoid numerical issues.
        """

        def loglikelihood(start=None, stop=None):

            hist_type_msg = ("To compute the loglikelihood, you need to use a NumPy "
                             "history for the {}, or compile the history beforehand.")
            if n.use_theano:
                if n.compiled_history is None:
                    raise RuntimeError(hist_type_msg.format("events"))
                else:
                    nhist = n.compiled_history
            else:
                nhist = n

            phist = p
            # We deliberately use times here (instead of indices) for start/
            # stop so that they remain consistent across different histories
            if start is None:
                start = nhist.t0
            else:
                start = nhist.get_time(start)
            if stop is None:
                stop = nhist.tn
            else:
                stop = nhist.get_time(stop)

            n_arr_floats = nhist[start:stop]
            p_arr = phist[start:stop]

            # FIXME: This would break the Theano graph, no ?
            if shim.isshared(n_arr_floats):
                n_arr_floats = n_arr_floats.get_value()
            if shim.isshared(p_arr):
                p_arr = p_arr.get_value()

            p_arr = sinn.clip_probabilities(p_arr)

            if not shim.is_theano_object(n_arr_floats):
                assert(sinn.ismultiple(n_arr_floats, 1).all())
            n_arr = shim.cast(n_arr_floats, 'int32')

            #loglikelihood: -log n! - log (N-n)! + n log p + (N-n) log (1-p) + cst

            if approx == 'low p':
                # We use the Stirling approximation for the second log
                l = shim.sum( -shim.log(shim.factorial(n_arr, exact=False))
                              -(N-n_arr)*shim.log(N - n_arr) + N-n_arr + n_arr*shim.log(p_arr)
                              + (N-n_arr)*shim.log(1-p_arr) )
                    # with exact=True, factorial is computed only once for whole array
                    # but n_arr must not contain any elements greater than 20, as
                    # 21! > int64 (NumPy is then forced to cast to 'object', which
                    # does not play nice with numerical ops)
            else:
                raise NotImplementedError

            return l

        return loglikelihood

    # ==============================================
    # Model advancing code
    #
    # This code isn't 100% generic yet;
    # look for TODO tags for model-specific hacks
    #
    # Function overview:
    # - advance(self, stop): User-facing function
    # - _advance(self): Returns a function; use as `self._advance(stop)`:
    #   `self._advance` is a property which memoizes the compiled function.
    # - compile_advance_function(self): Function called by `_advance` the first
    #   time to do the compilation. Could conceivably also be used by a user.
    #   Returns a compiled function.
    # - advance_updates(self, stoptidx): Function used by
    #   `compile_advance_function` to retrieve the set of symbolic updates.
    # ==============================================
    def get_state(self, tidx=None):
        """
        Return a State object corresponding to the state at time `tidx`
        If no tidx is given, uses `self.cur_tidx` to return the current state
        TODO: Add support for >1 lags.
        """
        ti = self.cur_tidx
        return self.State(*(h[ti-self.t0idx+h.t0idx] for h in self.statehists))

    def get_state_placeholder(self, name_suffix=' placeholder'):
        """
        Return a State object populated with symbolic placeholder variables.
        TODO: Add support for >1 lags.
        """
        return self.State(*(make_placeholder(h, name_suffix)
                            for h in self.statehists))

    def advance(self, stop):
        """
        Allows advancing (aka integrating) a symbolic model.
        For a non-symbolic model the usual recursion is used – it's the
        same as calling `hist[stop]` on each history in the model.
        For a symbolic model, the function constructs the symbolic update
        function, compiles it, and than evaluates it with `stop` as argument.
        The update function is compiled only once, so subsequent calls to
        `advance` are much faster and benefit from the acceleration of running
        on compiled code.

        Parameters
        ----------
        stop: int, float
            Compute history up to this point (inclusive).
        """

        # TODO: Rename stoptidx -> endidx
        if stop == 'end':
            stoptidx = self.tnidx
        else:
            stoptidx = self.get_tidx(stop)

        # Make sure we don't go beyond given data
        for hist in self.history_set:
            if hist.locked:
                tnidx = hist._original_tidx.get_value()
                if tnidx < stoptidx - self.t0idx + hist.t0idx:
                    logger.warning("Locked history '{}' is only provided "
                                   "up to t={}. Output will be truncated."
                                   .format(hist.name, hist.get_time(tnidx)))
                    stoptidx = tnidx - hist.t0idx + self.t0idx

        if not shim.config.use_theano:
            self._refhist[stoptidx - self.t0idx + self._refhist.t0idx]
            # We want to compute the whole model up to stoptidx, not just what is required for refhist
            for hist in self.statehists:
                hist.compute_up_to(stoptidx - self.t0idx + hist.t0idx)

        else:
            if not shim.graph.is_computable([hist._cur_tidx
                                       for hist in self.statehists]):
                raise TypeError("Advancing models is only implemented for "
                                "histories with a computable current time "
                                "index (i.e. the value of `hist._cur_tidx` "
                                "must only depend on symbolic constants and "
                                "shared vars).")
            # try:
            #     self.eval_updates()
            # except shim.graph.MissingInputError:
            #     raise shim.graph.MissingInputError("There "
            #         "are symbolic inputs to the already present updates:"
            #         "\n{}.\nEither discard them with `theano_reset()` or "
            #         "evaluate them with `eval_updates` (providing values "
            #         "with the `givens` argument) before advancing the model."
            #         .format(shim.graph.inputs(shim.get_updates().values())))
            curtidx = min( shim.graph.eval(hist._cur_tidx, max_cost=50)
                           - hist.t0idx + self.t0idx
                           for hist in self.statehists )
            assert(curtidx >= -1)

            if curtidx < stoptidx:
                self._advance(curtidx, stoptidx+1)
                # _advance applies the updates, so should get rid of them
                self.theano_reset()
    integrate = advance

    @property
    def no_updates(self):
        """
        Return `True` if none of the model's histories have unevaluated
        symbolic updates.
        """
        no_updates = all(h._cur_tidx is h._original_tidx
                         and h._data is h._original_data
                         for h in self.history_set)
        if no_updates and len(shim.get_updates()) > 0:
            raise RuntimeError(
                "Unconsistent state: there are symbolic theano updates "
                " (`shim.get_updates()`) but none of the model's histories "
                "has a symbolic update.")
        elif not no_updates and len(shim.get_updates()) == 0:
            hlist = {h.name: (h._cur_tidx, h._data) for h in self.history_set
                     if h._cur_tidx is not h._original_tidx
                        and h._data is not h._original_data}
            raise RuntimeError(
                "Unconsistent state: some histories have a symbolic update "
                "({}), but there are none in the update dictionary "
                "(`shim.get_updates()`)".forma(hlist))
        return no_updates

    @property
    def _advance(self):
        """
        Attribute which caches the compilation of the advance function.
        """
        if not hasattr(self, '_advance_updates'):
            self._advance_updates = self.get_advance_updates()
            # DEBUG
            # for i, s in enumerate(['base', 'value', 'start', 'stop']):
            #     self._advance_updates[self.V._original_data].owner.inputs[i] = \
            #         shim.print(self._advance_updates[self.V._original_data]
            #                    .owner.inputs[i], s + ' V')
            #     self._advance_updates[self.n._original_data].owner.inputs[i] = \
            #         shim.print(self._advance_updates[self.n._original_data]
            #                    .owner.inputs[i], s + ' n')
        if self.no_updates:
            if not hasattr(self, '_advance_fn'):
                logger.info("Compiling the update function")
                self._advance_fn = self.compile_advance_function(
                    self._advance_updates)
                logger.info("Done.")
            _advance_fn = self._advance_fn
        else:
            # TODO: Find reasonable way of caching these compilations ?
            # We would need to cache the compilation for each different
            # set of symbolic updates.
            advance_updates = OrderedDict(
                (var, shim.graph.clone(upd, replace=shim.get_updates()))
                for var, upd in self._advance_updates.items())

            logger.info("Compiling the update function")
            _advance_fn = self.compile_advance_function(advance_updates)
            logger.info("Done.")

        return _advance_fn

    def get_advance_updates(self):
        """
        Returns a 'blank' update dictionary. Update graphs do not include
        any dependencies from the current state, such as symbolic/transformed
        initial conditions.
        """
        if not hasattr(self, '_curtidx_var'):
            self._curtidx_var = shim.getT().scalar('curtidx (model)',
                                              dtype=self.tidx_dtype)
            self._curtidx_var.tag.test_value = 1
        if not hasattr(self, '_stoptidx_var'):
            self._stoptidx_var = shim.getT().scalar('stoptidx (model)',
                                             dtype=self.tidx_dtype)
            self._stoptidx_var.tag.test_value = 3
                # Allow model to work with compute_test_value != 'ignore'
                # Should be at least 2 more than _curtidx, because scan runs
                # from `_curtidx + 1` to `stoptidx`.
        logger.info("Constructing the update graph.")
        # Stash current symbolic updates
        for h in self.statehists:
            h.stash()  # Stash unfinished symbolic updates
        updates_stash = shim.get_updates()
        shim.reset_updates()

        # Get advance updates
        updates = self.advance_updates(self._curtidx_var, self._stoptidx_var)
        # Reset symbolic updates to their previous state
        self.theano_reset()
        for h in self.statehists:
            h.stash.pop()
        shim.config.theano_updates = updates_stash
        logger.info("Done.")
        return updates

    def compile_advance_function(self, updates):
        self._debug_ag = updates
        fn = self.compile_cache.get([], updates, self.rng)
        if fn is None:
            fn = shim.graph.compile([self._curtidx_var, self._stoptidx_var], [],
                                    updates = updates)
            self.compile_cache.set([], updates, fn, self.rng)
        else:
            logger.info("Compiled advance function loaded from cache.")
        return fn

    def advance_updates(self, curtidx, stoptidx):
        """
        Compute model updates from curtidx to stoptidx.

        Parameters
        ----------
        curtidx: symbolic (int):
            We want to compute the model starting from this point.
        stoptidx: symbolic (int)
            We want to compute the model up to this point.

        Returns
        -------
        Update dictionary:
            Compiling a function and providing this dictionary as 'updates' will return a function
            which fills in the histories up to `stoptidx`.
        """
        self.remove_other_histories()  # HACK
        # self.clear_unlocked_histories()
        # self.theano_reset()
        if not all(np.can_cast(stoptidx.dtype, hist.tidx_dtype)
                   for hist in self.statehists):
            raise TypeError("`stoptidx` cannot be safely cast to a time index. "
                            "This can happen if e.g. a history uses `int32` for "
                            "its time indices while `stoptidx` is `int64`.")

        if len(list(self.unlocked_statehists)) == 0:
            raise NotImplementedError
        # elif len(self.statehists) == 1:
        #     hist = next(iter(self.statehists))
        #     startidx = hist._original_tidx - hist.t0idx + self.t0idx
        # else:
        #     startidx = shim.smallest( *( hist._original_tidx - hist.t0idx + self.t0idx
        #                                 for hist in self.statehists ) )
        try:
            assert( shim.get_test_value(curtidx) >= -1 )
                # Iteration starts at startidx + 1, and will break for indices < 0
        except AttributeError:
            # Unable to find test value; just skip check
            pass

        def onestep(tidx, *args):
            # To help with debugging, assign a name to the symbolic variables
            # created by `scan`
            unlocked_statevar_names = [s + ' (scan)'
                                       for s, h in zip(self.State._fields,
                                                       self.statehists)
                                       if not h.locked]
            for x, name in zip(
                utils.flatten(tidx, *args, terminate=shim.cf._TerminatingTypes),
                utils.flatten('tidx (scan)', unlocked_statevar_names,
                              terminate=shim.cf._TerminatingTypes)):
                if getattr(x, 'name', None) is None:
                    x.name = name
            state_outputs, updates = self.symbolic_update(tidx, *args)
            assert(len(state_outputs) == len(list(self.unlocked_statehists)))
            for i, statehist in enumerate(self.unlocked_statehists):
                state_outputs[i] = shim.cast(state_outputs[i],
                                             statehist.dtype)
            return state_outputs, updates
            #return list(state_outputs.values()), updates

        outputs_info = []
        for hist in self.unlocked_statehists:
            # TODO: Generalize
            maxlag = hist.t0idx
            # maxlag = hist.index_interval(self.params.Δ.get_value())
            # HACK/FIXME: We should query history for its lags
            if maxlag > 1:
                lags = [-maxlag, -1]
            else:
                lags = [-1]
            tidx = curtidx - self.t0idx + hist.t0idx
            assert(maxlag <= hist.t0idx)
                # FIXME Maybe not necessary if built into lag history
            if len(lags) == 1:
                assert(maxlag == 1)
                outputs_info.append( sinn.upcast(hist._data[tidx],
                                                 to_dtype=hist.dtype,
                                                 same_kind=True,
                                                 disable_rounding=True))
            else:
                outputs_info.append(
                    {'initial': sinn.upcast(hist._data[tidx+1-maxlag:tidx+1],
                                            to_dtype=hist.dtype,
                                            same_kind=True,
                                            disable_rounding=True),
                     'taps': lags})


        outputs, upds = shim.scan(onestep,
                                  sequences = shim.arange(curtidx+1, stoptidx),
                                  outputs_info = outputs_info,
                                  return_list = True)
        # Ensure that all updates are of the right type
        # Theano can add updates for variables that don't have a dtype, e.g.
        # a RandomStateType variable, which is why we include the hasattr guard
        upds = OrderedDict([(orig_var,
                             (sinn.upcast(upd, to_dtype=orig_var.dtype,
                                          same_kind=True, disable_rounding=True))
                              if hasattr(orig_var, 'dtype') else upd)
                            for orig_var, upd in upds.items()])
        self.apply_updates(upds)
            # Applying updates ensures we remove the iteration variable
            # scan introduces from the shim updates dictionary
            # FIXME: This sounds pretty hacky, although it seems like a good
            # idea to update the intermediate state of all the histories in
            # case there are subsequent operations.

        # Update the state variables
        # These are stripped from the update dictionary within
        # `_get_symbolic_update` because we want to update them with a slice
        # rather than with a long sequence of nested `IncSubtensor` ops.
        for h in self.history_set:
            h.stash()
        updates_stash = shim.get_updates()
        self.theano_reset()
        for hist, output in zip(self.unlocked_statehists, outputs):
            assert hist._original_data not in upds
            valslice = slice(curtidx - self.t0idx + hist.t0idx + 1,
                             stoptidx  - self.t0idx + hist.t0idx)
            # odata = hist._original_data
            # upd = shim.set_subtensor(hist._data[valslice], output)
            upd = sinn.upcast(output, to_dtype=hist.dtype,
                              same_kind=True, disable_rounding=True)
            hist.update(valslice, upd)
                # `update` applies the update and adds it to shim's update dict
        hist_upds = shim.get_updates()
        for h in self.history_set:
            h.stash.pop()
        shim.config.theano_updates = updates_stash

        # hist_upds = shim.get_updates()
        # # Ensure that all updates are of the right type
        # # Theano can add updates for variables that don't have a dtype, e.g.
        # # a RandomStateType variable, which is why we include the hasattr guard
        # hist_upds = OrderedDict([(orig_var,
        #                           (sinn.upcast(upd, to_dtype=orig_var.dtype,
        #                                       same_kind=True, disable_rounding=True))
        #                            if hasattr(orig_var, 'dtype') else upd)
        #                          for orig_var, upd in hist_upds.items()])

        assert len(set(upds).intersection(hist_upds)) == 0
        upds.update(hist_upds)
        return upds

    def symbolic_update(self, tidx, *statevars):
        """
        Attempts to build a symbolic update automatically. This is work in
        progress, so for the time being will only work on simpler models.
        An error is thrown if the function suspects the output to be wrong.
        For more complicated models you can define the `symbolic_update`
        method yourself in the model's class.
        Creating the graph is quite slow, but the result is cached, so
        subsequent calls don't need to recreate it.

        Parameters
        ----------
        tidx: symbolic int
            The symbolic integer representing the "next" time index.

        *statevars: symbolic expressions
            All subsequent variables should match the shape and type of a
            time slice from each *unlocked* history in `self.statehists`, in
            order. Histories `h` for which `h.locked is True` don't need to
            be updated and should not be passed as arguments.
        """
        # TODO: if module attribute cache is removed, remove the
        # placeholder variable and move the on-disk cache to this
        # function.

        # This function is actually a wrapper which caches the result of
        # `_get_symbolic_update`, to avoid constructing the graph twice.
        # However, this is the function the function that is part of the API
        # and which should be overloaded by a derived class, so this is the
        # one we document.
        l = len(list(self.unlocked_statehists))
        if (len(statevars) > 0
            and not isinstance(statevars[0], shim.cf.GraphTypes)):
            raise TypeError("state variables must be passed separately to "
                            "`symbolic_update`, not as a tuple or list.")
        elif len(statevars) < l:
            raise TypeError("There are {} unlocked state histories, but only "
                            "{} state variables were passed to "
                            "`symbolic_update`.".format(len(statevars), l))
        elif len(statevars) > l:
            raise TypeError("There are {} unlocked state histories, but "
                            "{} state variables were passed to "
                            "`symbolic_update`. Remember that variables should "
                            "not be passed for locked state histories."
                            .format(len(statevars), l))
        return self._get_symbolic_update(tidx, *statevars)
        # if not hasattr(self, '_symbolic_update_graph'):
        # if True:
        #     stateph = self.get_state_placeholder()
        #     symbupd = self._get_symbolic_update(tidx, *stateph)
        # #    self._symbolic_update_graph = (stateph, symbupd)
        # # else:
        # #     stateph, symbupd = self._symbolic_update_graph
        # # symbupd: ([state xt vars], odict(shared var updates))
        # subs = OrderedDict((xph, x) for xph, x in zip(stateph, statevars))
        # outputs = [shim.graph.clone(xt, replace=subs) for xt in symbupd[0]]
        # updates = OrderedDict((var, shim.graph.clone(upd, replace=subs))
        #                       for var, upd in symbupd[1].items())
        # return outputs, updates

    def _get_symbolic_update(self, tidx, *statevars):
        # Stash current symbolic updates
        assert set(self.statehists).issubset(self.history_set)
        for h in self.history_set:
            h.stash()  # Stash unfinished symbolic updates
        updates_stash = shim.get_updates()
        self.theano_reset()

        # It doesn't really matter which time point we use, we just want the
        # t -> t+1 update. But a graph update will be created for every time
        # point between _original_tidx and the chosen _tidx, so making it
        # large can be really costly.
        # Can't just use self._refhist because it could filled while
        # others are empty (e.g. if it is filled with data)
        ush = list(self.unlocked_statehists)
        refhist_idx =  np.argmax([h.cur_tidx - h.t0idx + self._refhist.t0idx
                                  for h in ush])
        refhist = ush[refhist_idx]
        ref_tidx = refhist._original_tidx
        tidcs = [ref_tidx - refhist.t0idx + h.t0idx
                 for h in self.unlocked_statehists]
        # tidxvals = [shim.graph.eval(ti) for ti in tidcs]
        # Get the placeholder current state
        # Get the placeholder new state
        # St = [(h[ti+1], False) for h, ti in zip(self.statehists, tidcs)]
        St = [(h._update_function(ti+1), False)
              for h, ti in zip(self.unlocked_statehists, tidcs)]
            # We exclude locked histories because those shouldn't be modified
        # Bool is flag indicating whether history graph is fully substituted
        # When they are all True, we stop substitutions

        # FIXME: Assumes no dependencies beyond a lag 1 for every one
        # Get S0 after St: don't need to update _data the second time, so it
        # will be the same _data which is indexed for both.
        #S0 = [h[ti] for h, ti in zip(self.statehists, tidcs)]
        # assert(len(S0) == len(St))
        # assert(len(St) == len(statevars))

        # Check if this is in the disk cache
        St_graphs_original = [xt[0] for xt in St]
        updates_original = shim.get_updates()
        St_graphs, updates = self.graph_cache.get(
            St_graphs_original, updates_original,
            other_inputs = statevars + (tidx,), rng = self.rng)

        if St_graphs is not None:
            logger.info("Symbolic update graphs loaded from cache.")
        else:
            # It's not in the cache, so we have to do the substitutions
            # TODO: Move to own function, or combine with `batch_function_decorator`
            for recursion_count in range(5):  # 5: max recursion
                # # ---------------------------
                # # Debugging code
                # # Update variables which still have symbolic inputs
                # odatas = [h._original_data for h in self.statehists]
                # xvars = [(h.name, xt[0]) for h, xt in zip(self.statehists, St)
                #          if any(y in shim.graph.variables([xt[0]])
                #                 for y in odatas + [ref_tidx])]
                # # The unsubstituted symbolic inputs to the above update variables
                # ivars = [[(h.name, h._original_data) for h in self.statehists
                #             if h._original_data in shim.graph.variables([xt])]
                #          for _, xt in xvars]
                # if len(xvars) > 0:
                #     # Locating the first unsubstituted symbolic input in the graph
                #     upd = xvars[0][1]
                #     xin = ivars[0][0][1]
                #     child1 = [v for v in shim.graph.variables([upd])
                #                 if v.owner is not None and xin in v.owner.inputs]
                #     child2 = [v for v in shim.graph.variables([upd])
                #                 if v.owner is not None
                #                 and v.owner.inputs[0].owner is not None
                #                 and xin in v.owner.inputs[0].owner.inputs]
                # import pdb; pdb.set_trace()
                # # ---------------------------
                if all(xt[1] for xt in St):
                    # All placeholders are substituted
                    break
                for i in range(len(St)):
                    # Don't use list comprehension, that way if earlier states
                    # appear in the updates for later states, their substitutions
                    # are already applied. This should save recursion loops.
                    _St = [xt[0] for xt in St]
                    St[i] = self.sub_states_into_graph(
                        St[i][0], self.unlocked_statehists,
                        statevars, _St, tidx, ref_tidx, refhist)

            assert all(xt[1] for xt in St)
                # All update graphs report as successfully substituted

            # Also substitute updates
            # We shouldn't need recursion for these
            updates = shim.get_updates()
            # Remove the updates to histories: those are done by applying
            # the St graphs
            if len(updates) > 0:
                for h in self.statehists:
                    if h._original_data in updates:
                        assert not h.locked
                        del updates[h._original_data]
                    if h._original_tidx in updates:
                        assert not h.locked
                        del updates[h._original_tidx]
                subbed_updates = [self.sub_states_into_graph(
                                    upd, self.unlocked_statehists,
                                    statevars, _St, tidx, ref_tidx, refhist)
                                  for upd in updates.values()]
                updvals, updsuccess = zip(*subbed_updates)
                    # Transpose `subbed_updates`
                assert all(updsuccess)
                    # All updates report as successfully substituted
                for var, upd in zip(updates, updvals):
                    updates[var] = upd
            assert all(u1 is u2 for u1, u2 in zip(updates.values(),
                                                  shim.get_updates().values()))

            St_graphs = [xt[0] for xt in St]
            # Sanity checks
            try:
                assert(shim.graph.is_computable(St_graphs,
                                                with_inputs=statevars+(tidx,)))
                    # If we have >1 time lags, this should catch it
                all_graphs = St_graphs + list(updates.values())
                inputs = shim.graph.inputs(all_graphs)
                # vs = [v for v in shim.graph.variables(inputs, St) if hasattr(v.owner, 'inputs') and any(i.name is not None and 'data' in i.name for i in v.owner.inputs)]  # DEBuG
                assert(ref_tidx not in inputs)
                    # Still test this: if ref_tidx is shared, it's computable
                # assert(not any(x0 in inputs for x0 in S0))
                assert not any(h._original_data in inputs
                               for h in self.unlocked_statehists)
                    # Symbolic update should only depend on `statevars` and `tidx`
            except AssertionError as e:
                raise (AssertionError(failed_build_msg)
                        .with_traceback(e.__traceback__))

            self.graph_cache.set(St_graphs_original,updates_original,
                                 St_graphs, updates, self.rng)

        # Reset symbolic updates to their previous state
        for h in self.history_set:
            h.stash.pop()
        shim.config.theano_updates = updates_stash

        # Return the new state
        return St_graphs, updates

    def nonstate_symbolic_update(self, tidx, hists, curstatevars, newstatevars):
        # TODO: Combine more with _get_symbolic_update ?

        assert all(h not in self.statehists for h in hists)
        assert set(self.statehists).issubset(self.history_set)
            # Basic check that all histories were properly attached to the model
            # This is a necessary but not sufficient condition
        # Stash any current symbolic update
        for h in self.history_set:
            h.stash()
        updates_stash = shim.get_updates()
        self.theano_reset()

        # Can't just use self._refhist because it could filled while
        # others are empty (e.g. if it is filled with data)
        refhist_idx =  np.argmax([h.cur_tidx - h.t0idx + self._refhist.t0idx
                                  for h in hists])
        refhist = hists[refhist_idx]
        ref_tidx = refhist._original_tidx
        tidcs = [ref_tidx - refhist.t0idx + h.t0idx
                  for h in hists]
        # statetidcs = [ref_tidx - self._refhist.t0idx + h.t0idx
        #               for h in self.statehists]
        # statetidxvals = [shim.graph.eval(ti) for ti in tidcs]

        ht = [(h._update_function(ti+1), False)
              for h, ti in zip(hists, tidcs)]
        assert len(ht) == len(hists)

        # Remove the locked histories from the variables we want to substitute:
        # those don't need to be computed (typically they contain they
        # observation data) and so should stay in the graphs.
        curstatevars = tuple(sv for sv, h
                                in zip(curstatevars, self.unlocked_statehists))
        newstatevars = tuple(sv for sv, h
                                in zip(newstatevars, self.unlocked_statehists))
        statehists   = tuple(self.unlocked_statehists)
        assert len(curstatevars) == len(newstatevars) == len(statehists)

        ht = [self.sub_states_into_graph(
                xt[0], statehists, curstatevars, newstatevars,
                tidx, ref_tidx, refhist)
              for xt in ht]
        assert all(xt[1] for xt in ht)
            # All update graphs report as successfully substituted
        graphs = [xt[0] for xt in ht]

        # Drop all updates: this is a side-effect-free calculation, and state
        # variables are taken care of by `scan`'s output variables.
        updates = OrderedDict()
        # # Also substitute updates
        # updates = shim.get_updates()
        # # Ensure that we are not updating histories outside of `hists`
        # if len(updates) > 0:
        #     for h in self.history_set.difference(hists):
        #         assert h._original_data not in updates
        #         assert h._original_tidx not in updates
        #     subbed_updates = [self.sub_states_into_graph(
        #                         upd, statehists, curstatevars, newstatevars, tidx)
        #                       for upd in updates.values()]
        #     updvals, updsuccess = zip(*subbed_updates)  # Transpose `subbed_updates`
        #     assert all(updsuccess)
        #         # All updates report as successfully substituted
        #     for var, upd in zip(updates, updvals):
        #         updates[var] = upd
        # assert all(u1 is u2 for u1, u2 in zip(updates.values(),
        #                                       shim.get_updates().values()))

        # Sanity checks
        try:
            all_graphs = graphs + list(updates.values())
            assert shim.graph.is_computable(
                all_graphs, with_inputs=curstatevars+(tidx,))
                # If we have >1 time lags, this should catch it
            inputs = shim.graph.inputs(all_graphs)
            # vs = [v for v in shim.graph.variables(inputs, St) if hasattr(v.owner, 'inputs') and any(i.name is not None and 'data' in i.name for i in v.owner.inputs)]  # DEBuG
            assert ref_tidx not in inputs
                # Still test this: if ref_tidx is shared, it's computable
            # assert(not any(x0 in inputs for x0 in S0))
            assert not any(h._original_data in inputs for h in statehists)
            assert not any(h._original_data in inputs for h in hists)
                # Symbolic update should only depend on `statevars` and `tidx`
        except AssertionError as e:
            raise (AssertionError(failed_build_msg)
                    .with_traceback(e.__traceback__))

        # Reset symbolic updates to their previous state
        for h in self.history_set:
            h.stash.pop()
        shim.config.theano_updates = updates_stash

        # Return the new values
        return graphs, updates

    def sub_states_into_graph(self, graph, hists, cur_histvars, new_histvars,
                              new_tidx, ref_tidx, refhist):
        """
        Substitute nodes in `graph` where a state variable is indexed
        by the corresponding variable in `statevars`.
        This returns a correctly substituted :param:graph for `t -> t+1`
        calculation where :param:cur_histvars corresponds to `t`and
        :param:new_histvars corresponds to `t+1`.

        Parameters
        ----------

        hists: list of Histories
            The histories for which we want to replace references to their data
            by virtual variables.
        cur_histvars: list of symbolic variables
            The symbolic variables to use for replacing the current state of
            the histories in :param:hists.
        new_statevars: list of symbolic expressions
            The symbolic variables to use for replacing the new / next state
            of the histories in :param:hists.
        new_tidx: symbolic index
            Time index relative to `self.t0idx` corresponding to the new state.
        """
        theano = shim.gettheano()
        # TODO: Allow lists of graphs ?
        # cur_statevars and new_statevars are typically virtual state variables,
        # e.g. symbolics appearing within a scan or which will be replaced
        # by another variable before compilation.

        # TODO: apply varname changes in method's code
        cur_statevars = cur_histvars
        new_statevars = new_histvars

        # If history lists are generators, turn them into lists because
        # we will iterate over them more than once
        if not isinstance(hists, (list, tuple)): hists = list(hists)
        if not isinstance(cur_histvars, (list, tuple)):
            cur_histvars = list(cur_histvars)
        if not isinstance(new_histvars, (list, tuple)):
            new_histvars = list(new_histvars)
        assert len(hists) == len(cur_histvars) == len(new_histvars)

        # # Can't just use self._refhist because it could filled while
        # # others are empty (e.g. if it is filled with data)
        # refhist_idx =  np.argmax([h.cur_tidx - h.t0idx + self._refhist.t0idx
        #                           for h in hists])
        # refhist = hists[refhist_idx]
        # ref_tidx = refhist._original_tidx
        new_tidx = new_tidx - self.t0idx + refhist.t0idx
            # Convert time index to be relative to reference history
        _tidcs = [ref_tidx - refhist.t0idx + h.t0idx
                  for h in hists]
        ref_tidxvals = [shim.graph.eval(ti) for ti in _tidcs]
        inputs = shim.graph.inputs([graph])
        variables = shim.graph.variables([graph])
        odatas = [h._original_data for h in hists]
        statevars = cur_statevars
        St = new_statevars

        # Check if this graph is already fully substituted
        if (not any(odata in inputs for odata in odatas)
            and ref_tidx not in inputs):
            fully_substituted = True
            return graph, fully_substituted

        # Proceed with substitution
        replace = {}
        # Loop over graph nodes, replacing any instance where we index
        # into _original_data by the appropriate virtual state variable
        for y in variables:
            if (shim.graph.symbolic_inputs(y) == [ref_tidx]
                and shim.graph.is_same_graph(y, ref_tidx)):
                # `is_same_graph` is expensive: avoid computing it when possible
                replace[y] = shim.cast(new_tidx - 1, y.dtype)
            elif y.owner is None:
                continue
            elif isinstance(y.owner.op, theano.tensor.Subtensor):
                if (y.owner.inputs[0].owner is not None
                    and isinstance(y.owner.inputs[0].owner.op,
                                   theano.tensor.IncSubtensor)):
                    for xt2, odata, tival in zip(St, odatas, ref_tidxvals):
                        if y.owner.inputs[0].owner.inputs[0] is odata:
                            if tival == 0 and expensive_asserts:
                                iy = shim.graph.eval(y.owner.inputs[1],
                                                    max_cost=50)
                                assert(iy == tival+1)
                                # assert(shim.graph.eval(
                                #   y.owner.inputs[0].owner.inputs[1]-xt2,
                                #   max_cost = 1000,
                                #   givens={new_tidx:ref_tidx+1}) == 0 )
                                #   # FIXME: `givens` assume lag of 1
                            replace[y] = xt2
                            break
                else:
                    for xs, xt2, odata, tival in zip(statevars, St,
                                                     odatas, ref_tidxvals):
                        if y.owner.inputs[0] is odata:
                            # args: data, index
                            i = shim.eval(y.owner.inputs[1])
                            if i == tival:
                                replace[y] = xs
                            elif i == tival+1:
                                # We can land here if the histories'
                                # original tindices aren't synchronized
                                assert(xt2 != graph)
                                replace[y] = xt2
                            break
                    else:
                        # We can land here e.g. when indexing a hist
                        # which is not a state variable (like input),
                        # but is there another way ? Should we throw
                        # a warning ?
                        pass
                # Any situation with different `i` is unsupported
                # and will be caught in the asserts below
                # FIXME: This is where to add support for multiple lags

        new_graph = shim.graph.clone(graph, replace)

        # Check if this graph is fully substituted
        new_inputs = shim.graph.inputs([new_graph])
        if (not any(odata in new_inputs for odata in odatas)
            and ref_tidx not in new_inputs):
            # We replaced all placeholder variables with virtual ones
            fully_substituted = True
        elif (len(replace) == 0):
            # There are placeholder variables in the graph that we are unable
            # to substitute
            raise RuntimeError(failed_build_msg)
        else:
            fully_substituted = False

        return new_graph, fully_substituted

# DEPRECATED ?
# Surrogate models date from a time where I needed to initialize models
# that would never be run (e.g. to build parameters or access member functions)
# I don't need to this anymore, so maybe we could get rid of them ?
def Surrogate(model):
    """
    Execute `Surrogate(MyModel)` to get a class which can serve as a viable
    stand-in for `MyModel`.

    The surrogate model completely hides the model's __init__    method,
    avoiding the instantiation of potentially large data. If there is
    some initialization you do need, you can add attributes after
    instantiation, or subclass Surrogate(MyModel).

    Parameters
    ----------
    model: class
        Class the surrogate should mirror
    """
    if not issubclass(model, Model):
        raise ValueError("Can only create surrogates for subclasses of sinn.models.Model.")

    class SurrogateModel(model):

        def __init__(self, params, t0=0, dt=None):
            """
            Parameters
            ----------
            params:
                Same parameters as would be passed to create the class
            t0: float
                The time corresponding to time index 0. By default this is 0.
            dt: float
                The time step. Required in order to use `get_tidx` and `index_interval`;
                can be omitted if these functions are not used.
            """
            # Since we don't call super().__init__, we need to reproduce
            # the content of ParameterMixin.__init__
            self.set_parameters(params)
            # Set the attributes required for the few provided methods
            self._t0 = t0
            self._dt = dt   # dt is a read-only property of Model: can't just set the value

        @property
        def t0(self):
            return self._t0
        @property
        def dt(self):
            return self._dt

        def get_tidx(self, t, allow_rounding=False):
            if self.dt is None:
                raise AttributeError("You must provide a timestep 'dt' to the surrogate class "
                                    "in order to call 'get_tidx'.")
            if shim.istype(t, 'int'):
                return t
            else:
                try:
                    shim.check( (t * sinn.config.get_rel_tolerance(t) < self.dt).all() )
                except AssertionError:
                    raise ValueError("You've tried to convert a time (float) into an index "
                                    "(int), but the value is too large to ensure the absence "
                                    "of numerical errors. Try using a higher precision type.")
                t_idx = (t - self.t0) / self.dt
                r_t_idx = shim.round(t_idx)
                if (not shim.is_theano_object(r_t_idx) and not allow_rounding
                    and (abs(t_idx - r_t_idx) > config.get_abs_tolerance(t) / self.dt).all() ):
                    logger.error("t: {}, t0: {}, t-t0: {}, t_idx: {}, dt: {}"
                                .format(t, self._tarr[0], t - self._tarr[0], t_idx, self.dt) )
                    raise ValueError("Tried to obtain the time index of t=" +
                                    str(t) + ", but it does not seem to exist.")
                return shim.cast(r_t_idx, dtype = self.cur_tidx.dtype)

        def index_interval(Δt, allow_rounding=False):
            if self.dt is None:
                raise AttributeError("You must provide a timestep 'dt' to the surrogate class "
                                    "in order to call 'index_interval'.")
            if not shim.is_theano_object(Δt) and abs(Δt) < self.dt - config.abs_tolerance:
                if Δt == 0:
                    return 0
                else:
                    raise ValueError("You've asked for the index interval corresponding to "
                                    "Δt = {}, which is smaller than this history's step size "
                                    "({}).".format(Δt, self.dt))
            if shim.istype(Δt, 'int'):
                return Δt
            else:
                try:
                    shim.check( Δt * config.get_rel_tolerance(Δt) < self.dt )
                except AssertionError:
                    raise ValueError("You've tried to convert a time (float) into an index "
                                    "(int), but the value is too large to ensure the absence "
                                    "of numerical errors. Try using a higher precision type.")
                quotient = Δt / self.dt
                rquotient = shim.round(quotient)
                if not allow_rounding:
                    try:
                        shim.check( shim.abs(quotient - rquotient) < config.get_abs_tolerance(Δt) / self.dt )
                    except AssertionError:
                        logger.error("Δt: {}, dt: {}".format(Δt, self.dt) )
                        raise ValueError("Tried to convert t=" + str(Δt) + " to an index interval "
                                        "but its not a multiple of dt.")
                return shim.cast_int16( rquotient )

        def clear_unlocked_histories(self):
            pass

        def theano_reset(self):
            pass

        def advance(self, t):
            pass

        def loglikelihood(self, start, batch_size, data=None):
            pass

    return SurrogateModel

class ModelKernelMixin:
    """
    Kernels within models should include this mixin.
    Adds interoperability with model parameters
    """
    def __init__(self, name, params, shape=None, f=None, memory_time=None, t0=0, **kwargs):
        super().__init__(name,
                         params = self.get_kernel_params(params),
                         shape  = shape,
                         f      = f,
                         memory_time=memory_time,
                         t0     = t0,
                         **kwargs)

    def update_params(self, params):
        super().update_params(self.get_parameter_subset(params))

    def get_parameter_subset(self, params):
        """Given a set of model parameters, return the set which applies
        to this kernel. These will in general not be a strict subset of
        `model_params`, but derived from them using `get_kernel_params`.
        As a special case, if each of the kernel's parameters can
        be found in `params`, then it is assumed that they have already
        been converted, and `get_kernel_params` is not called again.
        """
        if all( field in params._fields for field in self.params._fields ):
            # params already converted for kernel
            return sinn.get_parameter_subset(self, params)
        else:
            # These are model parameters. Convert them for the kernel
            return self.cast_parameters(self.get_kernel_params(params))

    @staticmethod
    def get_kernel_params(model_params):
        raise NotImplementedError("Each of your model's kernels must "
                                  "implement the method `get_kernel_params`.")
