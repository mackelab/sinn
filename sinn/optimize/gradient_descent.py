from collections import OrderedDict, deque, namedtuple, Iterable
from itertools import chain
import itertools
import os.path
import time
import logging
import copy

logger = logging.getLogger('sinn.optimize.gradient_descent')

import numpy as np
import scipy as sp
from parameters import ParameterSet
from tqdm import tqdm

import theano
import theano.tensor as T

import mackelab as ml
import mackelab.iotools
import mackelab.optimizers as optimizers
from mackelab.utils import OrderedEnum
#from mackelab.optimizers import Adam, NPAdam
import theano_shim as shim
import sinn
import sinn.analyze as anlz
import sinn.analyze.heatmap
import sinn.models as models

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    import mackelab.plot
except ImportError:
    logger.warning("Unable to import matplotlib. Plotting functions "
                   "will not work.")

# Gradient descent involves some form of iteration, for which we typically
# want to provide feedback. Often (e.g. in a Jupyter Notebook), we will
# want to refresh the feedback output, to avoid drowning the interface in
# status messages
try:
    __IPYTHON__
except NameError:
    # We are not within an IPython environment
    def clear_output(wait=False):
        return
else:
    # We are within an IPython environment – use its clear_output function
    from IPython.display import clear_output

#######################################
#
# Status constants
#
#######################################

class ConvergeStatus(OrderedEnum):
    # These numbers are saved with the SGD and so should not be changed.
    NOTSTARTED = 0
    NOTCONVERGED = 1
    UNKNOWN = 6  # Used for loading legacy fits that didn't save status
    # Numbers greater or equal to 'STOP' stop SGD iterations
    STOP = 10
    CONVERGED = 11
    ABORTED = 12
    ERROR = 20
    # Specific errors we want to identify can be added with numbers > 20, and
    # a corresponding test added in the `fit()` function.

######################################
#
# General use functions
#
######################################

def standardize_lr(lr, params):
    """
    Return a dictionary of {param name: learning rate} pairs.
    TODO: Move to mackelab.optimizers

    Parameters
    ----------
    lr: float, dict
        If float: same learning rate applied to each parameter.
        If dict: keys correspond to parameter names.
        Dictionary may also contain a 'default' entry; it's value is used
        for any parameter whose name is not in the `lr` dictionary.
    params: list of symbolic variables
        Parameters which will be fit.
    """
    if shim.isscalar(lr):
        lr = {p: lr for p in params}
    elif isinstance(lr, dict):
        new_lr = {}
        default_lr = getattr(lr, 'default', None)
        for p in params:
            if p in lr:
                assert(shim.isshared(p))
                assert(p.name not in lr)
                new_lr[p] = lr[p]
            elif p.name in lr:
                new_lr[p] = lr[p.name]
            else:
                if default_lr is not None:
                    new_lr[p] = default_lr
                else:
                    raise KeyError("No learning rate for variable '{}', "
                                    "and no default learning rate was given."
                                    .format(p.name))
        lr = new_lr

    else:
        raise ValueError("Learning rate must be specified either as a scalar, "
                            "or as a dictionary with a key matching each parameter.")

    return lr

def maxdiff(it, n=10):
    """Return the maximum difference between the last element of it(terable)
    and the n elements before it."""
    return abs( np.array([ it[i]
                           for i in range(len(it) - n-1, len(it)) ])
                - it[-1] ).max()
def meandiff(it, n=10, end=-1):
    """Return the absolute difference between the means of it[-n:] and it[-2n:-n]."""
    assert(end < 0 or end >= n)
    return abs( np.mean([ it[end-i] for i in range(0, n) ])
                - np.mean([ it[end-i] for i in range(n, 2*n)]) )

def difftomean(it, n=10, end=-1):
    """Return the absolute difference between it[end] and the mean of it[end-n : end]."""
    return abs( np.mean([ it[end-i] for i in range(1, n) ]) - it[end] ).max()

def meanabsdev(it, n=10, end=-1):
    """Return the mean absolute deviation of it[-n-1:]"""
    arr = np.array( [it[end-i] for i in range(0, n)] )
    return abs( arr - arr.mean(axis=0) ).mean(axis=0)

def converged(it, r=0.01, p=0.99, n=10, m=10, abs_th=0.001):
    """
    NOTE: if 'it' has no len or is not indexable, it is internally converted to a list.
    r: resolution. Identify a difference in means of r*sigma with probability p
    p: certainty of each hypothesis test
    n: length of each test
    m: number of tests
    abs_th: changes below this threshold are treated as constant, no matter the
            standard deviation (specifically, this amount is added to the std
            to compute the reference deviation)
    """
    try:
        it[-1]  # Check that it is indexable
        length = len(it)
    except TypeError:
        it = list(it)
        length = len(it)

    if length < 2*n+m:
        return False
    else:
        # Get the threshold that ensures p statistical power
        arr = np.array( [it[-i] for i in range(1, n+1)] )
        std = arr.std(ddof=1, axis=0) + abs_th
        if np.any( std > 1):
            # Ensure that a large variance make the test accidentally succeed
            return False
        a = sp.stats.norm.ppf(p, loc=r*std, scale=1)
        s = np.sqrt(n) / std  # rescaling to get normalized Student-t
        # Check if the m last iterations were below the threshold
        return all( (meandiff(it, n=n, end=-i) * s < a).all()
                    for i in range(1, m+1) )

def get_indices(param):
    """Return an iterator for the indices of `param`."""
    if len(param.get_value().shape) == 1:
        for i in range(param.get_value().shape[0]):
            yield (i,)
    elif len(param.get_value().shape) == 2:
        for i in range(param.get_value().shape[0]):
            for j in range(param.get_value().shape[1]):
                yield (i, j)
    else:
        raise NotImplementedError

##########################################
#
# Types of cost
#
##########################################

class Cost:
    """
    'Cost' classes are interface objects that provide conversion methods to
    different cost types.
    These methods are implemented as properties, so they are called without brackets.
    All formats must provide at a minimum be convertible to `cost`, which may be
    anything that can be interpreted as a cost (i.e. that optimization would
    seek to minimize).
    """
    # self conversions (e.g. logL -> logL) are already defined by default
    # conversion functions must work with numpy arrays
    conversions = {
        'cost' : {},
        'logL' : {
            'cost' : lambda x: -x,
            '-logL': lambda x: -x,
            'L'    : lambda x: shim.exp(x)
        },
        '-logL' : {
            'cost' : lambda x: x,
            'logL' : lambda x: -x,
            'L'    : lambda x: shim.exp(-x)
        },
        'L' : {
            'cost' : lambda x: -x,
            'logL' : lambda x: shim.log(x),
            '-logL': lambda x: -shim.log(x)
        }
    }
    def __init__(self, value, format):
        format = str(format)  # normalize input (it may e.g. be a numpy string)
        if format not in self.conversions:
            raise ValueError("Unrecognized format '{}'. Possible values are {}."
                             .format(format, ', '.join(self.conversions.keys())))
        if isinstance(value, Cost):
            self.value = value.to(format)
        else:
            self.value = value
        self.format = format

    def to(self, format):
        if format == self.format:
            return self.value
        else:
            targets = self.conversions[self.format]
            if format in targets:
                return targets[format](self.value)
            else:
                target_names = list(targets.keys())
                target_names.append(self.format)
                raise ValueError("Unrecognized format '{}'. Possible values are {}."
                                 .format(format, ', '.join(target_names)))

    # Redirect attribute access to the stored value while keeping the Cost interface
    def __getitem__(self, key):
        return Cost(self.value[key], self.format)
    def __getattr__(self, attr):
        return Cost(getattr(self.value, attr), self.format)

    @property
    def cost(self):
        return self.to('cost')
    @property
    def logL(self):
        return self.to('logL')
    @property
    def negLogL(self):
        return self.to('-logL')
    @property
    def L(self):
        return self.to('L')

##########################################
#
# Stochastic gradient descent
#
##########################################

class SGDBase:
    # TODO: Spin out GDBase class and inherit
    # TODO: Make an ABC (abstract base class)

    def __init__(self, cost_format, result_choice='take_last'):
        """
        Parameters
        ----------
        cost_format: str
            One of the cost formats listed in Cost.conversions.
        result_choice: str
            One of:
              - 'take_last': result is the last iteration step, regardless of
                cost. This is probably the best option if the cost, like the
                gradient, is evaluated on mini-batches.
              - 'take_best': result is the iteration step with the lowest cost.
                This is most appropriate when we trust the evaluation of the
                cost, e.g. if it is evaluated on the whole trace rather than a
                batch.
        """
        self.cost_format = cost_format
        self.status = ConvergeStatus.NOTSTARTED
        self.result_choice = result_choice

    @property
    def repr_np(self):
        repr = {'version_sgdbase': 2.1,
                'type'   : 'SGDBase'}

        repr['cost_format'] = self.cost_format
        repr['status'] = self.status
        repr['result_choice'] = self.result_choice
        repr['cost_trace'] = self.cost_trace.value  # Property
        repr['cost_trace_stops'] = self.cost_trace_stops  # Property

        return repr

    @classmethod
    def from_repr_np(cls, repr, _instance):
        """
        Parameters
        ----------
        repr: numpy.NpzFile
            Value returned from `repr_np()`.
        _instance: None | class instance
            For internal use.
            If not None, use this class instance instead of creating a new one.
            Designed for calling `from_repr_np()` from derived classes.
            (See `SGDView.from_repr_np` for an example.)
            For abtract base classes this is a required parameter, since we
            should always be calling the method from within a derived class'
            `from_repr_np` method.
        """
        version = (repr['version_sgdbase'] if 'version_sgdbase' in repr
                   else repr['version'])
        if _instance is not None:
            assert(isinstance(_instance, cls))
            o = _instance
        else:
            kwargs = {'cost_format': repr['cost_format']}
            o = cls(**kwargs)

        if version >= 2 or version == 1 and 'status' in repr:
            o.status = repr['status']
        else:
            o.status = ConvergeStatus.UNKNOWN
        if version >= 2.1:
            o.result_choice = repr['result_choice']
        else:
            o.result_choice = 'take_last'
        # Since `cost_trace` and `cost_trace_stops` are properties, they must
        # be set by the derived class' `from_repr_np` method.

        return o

    def Cost(self, cost):
        return Cost(cost, self.cost_format)

    # Abstract definitions
    @property
    def cost_trace(self):
        raise NotImplementedError
    @property
    def cost_trace_stops(self):
        raise NotImplementedError
    @property
    def trace(self):
        raise NotImplementedError
    @property
    def trace_stops(self):
        raise NotImplementedError

    # Result access methods
    # Depending on `self.result_choice`, return either the last iteration or the
    # one with the lowest cost
    @property
    def result_cost_idx(self):
        if self.result_choice == 'take_last':
            return len(self.cost_trace_stops) - 1
        else:
            assert(self.result_choice == 'take_best')
            return np.argmin(self.cost_trace.cost)
    @property
    def result_cost(self):
        return self.cost_trace[self.result_cost_idx]
    @property
    def result_idx(self):
        if self.result_choice == 'take_last':
            return len(self.trace_stops) - 1
        else:
            assert(self.result_choice == 'take_best')
            result_stop = self.cost_trace_stops[self.result_cost_idx]
            return np.searchsorted(self.trace_stops, result_stop)
    @property
    def result(self):
        idx = self.result_idx
        return {name: trace[idx] for name, trace in self.trace.items()}
    @property
    def MLE(self):  # For backwards compatibility
        return self.result

def _dummy_reset_function(**kwargs):
    return None

class SeriesSGD(SGDBase):
    # TODO: Spin out SGD class and inherit

    def __init__(self, cost, start_var, batch_size_var, cost_format,
                 optimize_vars, track_vars,
                 start, datalen, burnin, batch_size, advance,
                 optimizer='adam', optimizer_kwargs=None,
                 reset=None, initialize=None, mode='random', mode_params=None,
                 cost_track_freq=100, var_track_freq=1):
        """

        Parameters
        ----------
        cost: symbolic expression
            Symbolic cost.
        start_var, batch_size_var: symbolic variable
            Variables appearing in the graph of the cost
        cost_format: str
            Indicates how the cost function should be interpreted. This will affect e.g.
            whether the algorithm attempts to minimize or maximize the value. One of:
              + 'logL' : log-likelihood
              + 'negLogL' or '-logL': negative log-likelihood
              + 'L' : likelihood
              + 'cost': An arbitrary cost to minimize. Conversion to other formats will
                   not be possible.
            May also be a subclass of `Cost`. (TODO)
        optimizer: str | class/factory function
            String specifying the optimizer to use. Possible values:
              - 'adam'
            If a class/factory function, should have the signature
              `optimizer(cost, fitparams, **optimizer_kwargs)`
        optimizer_kwargs: dict | None
            Keyword arguments to pass to the optimizer initializer. For the
            default ADAM optimizer, possible keyword arguments are `lr`, `b1`,
            `b2`, `e`, `clip` and `grad_fn`. If `None`, no keyword are passed;
            this requires that the optimizer defines default values for all its
            parameters.
            FIXME: there is a current hard-coded requirement for 'lr' to be in
            optimizer_kwargs.
            TODO: Parameters (e.g. learning rate) which depend on on iteration
            index or convergence status.
        optimize_vars: List of symbolic variables
            Symbolic variable/graph for the cost function we want to optimize.
            All variables should be inputs to the `cost` graph.
            Shared variables are used as-is.
            For non-shared variables, a shared variable is created. To determine the
            value to which to set it, we first call `shim.get_test_value()`. If that
            fails, an array of ones of proper shape and size is used.
        track_vars: dict {'name': symbolic graph}
            Dictionary indicating which variables to track. To track an optimization
            variable, just provide it. To track a transformation of it, provide the
            transformation as a symbolic graph for which the variable is an input.
        start: int
            Start of the data used in the fit.
        datalen: int
            Amount of data on which to evaluate the cost, in the model's time units.
            Cost will be evaluated up to the time point 'start + datalen'.
            burnin: float, int
            Amount of data discarded at the beginning of every batch. In the model's time units (float)
            or in number of bins (int).
        batch_size: int
        advance: function (tidx) -> update dictionary
            Function returning a set of updates which advance the time series
            with the current parameters.
            The update dictionary is compiled into a function which takes as
            single input the current time index.
            If any of the `optimize_vars` are not shared variables, they are
            substituted by the corresponding shared variable before compilation.
        reset: function (**shared vars) -> None
            (Optional) Arbitrary function to run before each batch. This allows
            to reset any state variable between parameter updates.
            If both are defined, this function is executed before `initialize()`.
            The optimization variables are passed as keyword arguments
            `name: shared`, where `name` is the name of variables passed to
            `optimize_vars` and `shared` is the internal shared variable holding
            its current value during the optimization.
        initialize: function  (time index) -> None
            (Optional) Arbitrary function to run before computing the cost. This allows to set any state
            variables required to compute the cost.
            The function is executed either once for every batch ('random' mode) or once per pass
            ('sequential' mode).
        mode: str
            Defines the way in which random samples are drawn from the data.
            Different modes use different default parameters, which can be
            overriden with `mode_params`. Possible values are:
            - 'random': standard SGD, where the starting point of each mini batch is chosen
               randomly within the data. If a function is provided to `initialize`, it is run
               before computing the gradient on every batch. This is the default value.
               Default parameters:
                 + 'burnin_factor': 0.1  The burnin for each batch is randomly
                        chosen between [batch_burnin, batch_burnin + burnin_factor*batch_burnin]
            - 'sequential': Mini-batches are taken sequentially, and loop back to the start
               when we run out of data. This approach is may suffer from the fact that the
               data windows on multiple loops are not independent.
               On the other hand, if the initial burnin is very long, this may provide
               substantial speed improvements.
               To mitigate the effect of correlation between windows, the burnin before each sample
               is randomized by adding by adding a uniform random value between 0–10% of the burnin.
               This prevents the windows on different sequential passes being perfectly aligned.
               The start time of each pass is also randomized with a random value between 0-100% of
               batch size. This ensures that the starting time of each pass is independent
               NOTE: If a function is given to `initialize`, it is run at the beginning of each pass,
               rather than for each batch.
                 + 'burnin_factor': 0.1  The burnin for each batch is randomly
                        chosen between [batch_burnin, batch_burnin + burnin_factor*batch_burnin]
                 + 'start_factor': 1.0  The start time for a run across data is randomly
                        chosen between [start, start+start_factor*batch_size]
        mode_params: dict
            Used to override default parameters associated with a 'mode'.
        cost_track_freq: int
            Frequency at which to record cost, in optimizer steps. Cost is typically more
            expensive to calculate (since it is not done on mini-batches), and usually not
            needed for every iteration.
            TODO: Allow schedule (e.g. higher frequency at beginning).
        var_track_freq: int
            Frequency at which to record tracked variables, in optimizer steps.
        """
        # Member variables defined in this function:
        # self.optimize_vars
        # self.track_vars
        # self.tidx_var
        # self.batch_size_var
        # self.cost  (function)
        # self._step  (function)
        # self.advance  (function)
        # self.initialize_model  (function)
        # self.cost_format
        # self.cost_track_freq
        # self.var_track_freq
        # self.start
        # self.datalen
        # self.burnin
        # self.batch_size
        # self.mode

        super().__init__(cost_format)
        start = shim.cast(start, start_var.dtype)
        batch_size = shim.cast(batch_size, batch_size_var.dtype)
        self.cost_track_freq = cost_track_freq
        self.var_track_freq = var_track_freq
        self.optimizer = optimizer
        self.start = start
        self.datalen = datalen
        self.burnin = burnin
        self.batch_size = batch_size
        self.mode = mode
        if not isinstance(track_vars, OrderedDict):
            self.track_vars = OrderedDict(track_vars)
        else:
            self.track_vars = track_vars
        if reset is None:
            # Use the dummy function
            self.reset_model = _dummy_reset_function
        else:
            self.reset_model = reset
        if initialize is None:
            # Use a dummy function
            self.initialize_model = lambda t: None
        else:
            self.initialize_model = initialize
        # We use 'copy' to make sure we don't modify passed parameters
        mode_params = copy.deepcopy(mode_params)
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        else:
            optimizer_kwargs = copy.deepcopy(optimizer_kwargs)

        # Make fit parameters array-like
        start = np.asarray(start)
        burnin = np.asarray(burnin)
        datalen = np.asarray(datalen)
        batch_size = np.asarray(batch_size)

        # Get time index dtype
        assert(all(np.issubdtype(i.dtype, np.integer)
                   for i in (start, burnin, datalen, batch_size)))
        # tidx_dtype = np.result_type(start, burnin, datalen, batch_size)

        # # Check fit parameters
        # def check_fit_arg(arg, name):
        #     if not np.issubdtype(arg.dtype, int):
        #         raise ValueError("'{}' argument refers to an index and therefore must be an integer"
        #                          .format(name))
        #     if arg.dtype != tidx_dtype:
        #         raise ValueError("All time indices should have the same type, but '{}' "
        #                          "and 'start' differ.".format(name))
        #
        # check_fit_arg(start, 'start')
        # check_fit_arg(burnin, 'burnin')
        # check_fit_arg(datalen, 'datalen')
        # check_fit_arg(batch_size, 'batch_size')

        # Get cost graph
        # self.tidx_var = shim.symbolic.scalar('tidx', dtype=tidx_dtype)
        # self.batch_size_var = shim.symbolic.scalar('batch_size', dtype=tidx_dtype)
        # self.tidx_var.tag.test_value = start  # PyMC3 requires every variable
        # self.batch_size_var.tag.test_value = batch_size  # to have a test value.
        # cost_graph = cost(self.tidx_var, self.batch_size_var)
        self.tidx_var = start_var   # TODO: Necessary ?
        self.batch_size_var = batch_size_var
        cost_graph = cost  # TODO: Just use 'cost'

        # Optimization vars must be shared variables
        self.optimize_vars = []
        self.optimize_vars_access = OrderedDict()  # TODO: Combine w/ optimize_vars
            # Uses the source variable names as keys (before conversion to shared),
            # so that users know them can use them.
        for var in optimize_vars:
            if shim.isshared(var):
                # This variable is already a shared variable; just use it
                self.optimize_vars.append(var)
                self.optimize_vars_access[var.name] = var
            else:
                # Create a shared variable to replace `var`
                try:
                    value = shim.get_test_value(var)
                except AttributeError:
                    # Create a default tensor of 1s
                    # 1s are less likely to cause issues than 0s, e.g. with log()
                    # TODO: How do we get shape ?
                    raise NotImplementedError
                shared_var = shim.shared(value, var.name + '_sgd')
                self.optimize_vars.append(shared_var)
                self.optimize_vars_access[var.name] = shared_var

        # Ensure all optimization variables have unique names
        if len(self.optimize_vars_access) != len(optimize_vars):
            raise ValueError("The optimization variables must have unique names.\n"
                             "Optimization variables: {}"
                            .format([v.name for v in optimize_vars]))

        # Substitute the new shared variables in the computational graphs
        self.var_subs = {orig: new for orig, new in zip(optimize_vars, self.optimize_vars)}
        cost_graph = shim.graph.clone(cost_graph, replace=self.var_subs)
        lr = optimizer_kwargs['lr']
        if isinstance(lr, dict):
            # TODO: Move some of this to `standardize_lr()` ?
            subbed_shared_names = {s.name: s for s in self.var_subs}
            for key, val in lr.items():
                if key in subbed_shared_names:
                    # 'key' is a name
                    new_key = self.var_subs[subbed_shared_names[key]].name
                    assert(new_key not in lr)
                    lr[new_key] = lr[key]
                    del lr[key]
                elif key in self.var_subs:
                    # 'key' is a shared_variable
                    new_key = self.var_subs[key]
                    assert(new_key not in lr)
                    lr[new_name] = lr[key]
                    del lr[key]
        lr = standardize_lr(lr, self.optimize_vars)
        optimizer_kwargs['lr'] = lr

        # Compile cost function
        # cost_graph_subs = shim.clone(cost_graph, {self.tidx_var: start,
        #                                           self.batch_size_var: datalen})
        logger.info("Compiling the cost function.")
        # if 'debugprint' in optimizers.debug_flags:
        #     # TODO: Move `debug_flags` to this module
        #     self.sstart = shim.shared(
        #           np.int32(optimizers.debug_flags['debugprint']), name='ssstart')
        #     self.cost = shim.graph.compile([], cost_graph,
        #                                    givens=[(self.tidx_var, self.sstart),
        #                                            (self.batch_size_var, np.int32(1))])
        if 'nanguard' not in optimizers.debug_flags:
            self.cost = shim.graph.compile([], cost_graph,
                                           givens=[(self.tidx_var, start),
                                                   (self.batch_size_var, datalen)])
        else:
            if optimizers.debug_flags['nanguard'] is True:
                nanguard = {'nan_is_error': True, 'inf_is_error': True, 'big_is_error': False}
            else:
                nanguard = optimizers.debug_flags['nanguard']
                assert('nan_is_error' in nanguard and 'inf_is_error' in nanguard) # Required arguments to NanGuardMode
            from theano.compile.nanguardmode import NanGuardMode
            self.cost = shim.graph.compile([], cost_graph,
                                           givens=[(self.tidx_var, start), (self.batch_size_var, datalen)],
                                           mode=NanGuardMode(**nanguard))
        logger.info("Done compilation.")

        # Get advance updates
        advance_updates = advance(self.tidx_var)
        for var, upd in advance_updates.items():
            upd = shim.graph.clone(upd, replace=self.var_subs)
            advance_updates[var] = shim.cast(upd, var.dtype, same_kind=True)

        # Compile the advance function
        assert(self.tidx_var in shim.graph.symbolic_inputs(advance_updates.values()))
            # theano.function raises a error if inputs are not used to compute outputs
            # Since tidx_var appears in the graphs on the updates, we need to
            # silence this error with `on_unused_input`. The assert above
            # replaces the test.
        self.advance = shim.graph.compile([self.tidx_var], [],
                                          updates=advance_updates)
                                          #on_unused_inputs='ignore')

        # Compile optimizer updates
        # FIXME: changed names in `optimize_vars`
        cost_to_min = self.Cost(cost_graph).cost
            # Ensures we have a cost to minimize (and not, e.g., a likelihood)
        ## Get optimizer updates
        if isinstance(self.optimizer, str):
            if self.optimizer == 'adam':
                logger.info("Calculating Adam optimizer updates.")
                optimizer_updates = optimizers.Adam(cost_to_min, self.optimize_vars, **optimizer_kwargs)
            else:
                raise ValueError("Unrecognized optimizer '{}'.".format(self.optimizer))

        else:
            # Treat optimizer as a factory class or function
            try:
                logger.info("Calculating custom optimizer updates.")
                optimizer_updates = self.optimizer(cost_to_min, self.optimize_vars, **optimizer_kwargs)
            except TypeError as e:
                if 'is not callable' not in str(e):
                    # Some other TypeError was triggered; reraise
                    raise
                else:
                    raise ValueError("'optimizer' parameter should be either a string or a "
                                     "callable which returns an optimizer (such as a class "
                                     "name or a factory function).\nThe original error was:\n"
                                     + str(e))
        ## Compile
        logger.info("Compiling the optimization step function.")
        if 'nanguard' not in optimizers.debug_flags:
            self._step = shim.graph.compile([self.tidx_var, self.batch_size_var], [], updates=optimizer_updates)
        else:
            # TODO: Remove duplicate with above
            if optimizers.debug_flags['nanguard'] is True:
                nanguard = {'nan_is_error': True, 'inf_is_error': True, 'big_is_error': False}
            else:
                nanguard = optimizers.debug_flags['nanguard']
                assert('nan_is_error' in nanguard and 'inf_is_error' in nanguard) # Required arguments to NanGuardMode
            from theano.compile.nanguardmode import NanGuardMode
            self._step = shim.graph.compile([self.tidx_var, self.batch_size_var],
                                            [], updates=optimizer_updates,
                                            mode=NanGuardMode(**nanguard))
        logger.info("Done compilation.")

        # Compile a function to extract tracking variables
        logger.info("Compiling parameter tracking function.")
        self._get_tracked = shim.graph.compile([], list(self.track_vars.values()),
                                               on_unused_input = 'ignore',
                                               givens = self.var_subs.items())

        # Initialize the fitting state variables
        self.initialize_vars()

        # Set the mode parameters
        self.set_mode_params(mode_params)

    @property
    def view(self):
        return SeriesSGDView(self.cost_format, self.trace, self.trace_stops,
                             self.cost_trace, self.cost_trace_stops)

    @property
    def repr_np(self):
        logger.warning("'repr_np' is not yet implemented for SGD; returning "
                       "representation for a view. This is sufficient for "
                       "analyzing the fit, but not for continuing it.")
        return self.view.repr_np

    @classmethod
    def from_repr_np(cls, repr, _instance):
        raise NotImplementedError

    @property
    def trace_stops(self):
        return self._tracked_iterations
    @property
    def cost_trace(self):
        return self.Cost(np.fromiter((val for val in self._cost_trace),
                            dtype=np.float32, count=len(self._cost_trace)))
    @property
    def cost_trace_stops(self):
        return self._tracked_cost_iterations

    @property
    def trace(self):
        # Convert deque to ndarray
        return OrderedDict( ( (varname,
                               np.fromiter(
                                   chain.from_iterable(val.flat for val in trace),
                                   dtype=np.float32,
                                   count=len(trace)*np.prod(trace[0].shape)
                                   ).reshape((len(trace),)+trace[0].shape)
                              )
                              for varname, trace in self._traces.items() ) )

    def set_mode_params(self, mode_params):
        # `defaults` is a dictionary of default values for each parameter
        # `validate` is a dictionary with same keys as `defaults`, and for
        #  each parameter provides a bool function returning False if the value
        #  is invalid.
        if self.mode == 'random':
            defaults = {'burnin_factor': 0.1}
            validate = {'burnin_factor': lambda x: x>=0}
        elif self.mode == 'sequential':
            defaults = {'burnin_factor': 0.1,
                        'start_factor':  1.0}
            validate = {'burnin_factor': lambda x: x>=0,
                        'start_factor': lambda x: x>=0}
        if mode_params is not None:
            defaults.update({key: value for key, value in mode_params.items()
                             if key in defaults})
        # Check that mode_params are valid:
        for key, value in defaults.items():
            if not validate[key](value):
                raise ValueError("Value of {} for mode parameter {} is invalid."
                                 .format(value, key))
        self.mode_params = ParameterSet(defaults)

    def initialize_vars(self, init_vals=None):
        """
        Parameters
        ----------
        init_vals: dict
            Dictionary of 'variable: value' pairs. Key variables must be the
            same as those that were passed as `optimize_vars` to `__init__()`.
            Not all variables need to be initialized: those unspecified keep
            their current value.
        """

        # Member variables defined in this function:
        # self.step_i
        # self.curtidx
        # self._traces
        # self._cost_trace
        # self._tracked_iterations
        # self._tracked_cost_iterations

        if init_vals is None:
            init_vals = {}

        self.step_i = 0
        self.curtidx = 0

        # Since we repeatedly append new values to the traces while fitting,
        # we store them in a 'deque'.
        self._traces = OrderedDict( (varname, deque())
                                    for varname in self.track_vars.keys() )
        self._cost_trace = deque()
        self._tracked_iterations = deque()
        self._tracked_cost_iterations = deque()

        # Set initial values
        for var, value in init_vals.items():
            if var in self.var_subs:
                var = self.var_subs[var]
            if not np.can_cast(value.dtype, var.dtype, 'same_kind'):
                raise TypeError("Trying to initialize variable '{}' (type '{}') "
                                " with value '{}' (type '{}')."
                                .format(var.name, var.dtype, value, value.dtype))
            value = np.asarray(value, dtype=var.dtype)
            var.set_value(value)

        # Initialize traces with current values
        self.record()

    def record(self, cost=True, params=True, last=False):
        """
        Record the current cost and parameter values.
        Recording of either cost or parameters can be prevented by passing
        `cost=False` or `params=False`.
        Passing `last` ensures that cost and params are recorded, no matter
        what the current step is (unless `cost` or `params` is false).
        """
        if params:
            if last or self.step_i % self.var_track_freq == 0:
                state = self._get_tracked()
                assert(len(self.track_vars) == len(state))
                self._tracked_iterations.append(self.step_i)
                for name, value in zip(self.track_vars, state):
                    self._traces[name].append(value)
        if cost:
            # Record the current cost
            if last or self.step_i % self.cost_track_freq == 0:
                self._cost_trace.append(self.cost())
                self._tracked_cost_iterations.append(self.step_i)

    def step(self):
        self.reset_model(**self.optimize_vars_access)
        if self.mode == 'sequential':
            if (self.curtidx < self.start
                or self.curtidx > self.start + self.datalen
                                  - self.batch_size
                                  - (1+self.mode_params.burnin_factor)*self.burnin):
                # We either haven't started or run through the dataset
                # Reset time index to beginning
                if self.mode_params.start_factor == 0:
                    self.curtidx = self.start
                else:
                    self.curtidx = np.random.randint(self.start,
                                                    self.start + self.mode_params.start_factor*self.batch_size)
                self.initialize_model(self.curtidx)
            else:
                # This doesn't seem required anymore
                # [model].clear_other_histories()
                pass

        elif self.mode == 'random':
            self.curtidx = np.random.randint(self.start,
                                             self.start + self.datalen
                                             - self.batch_size
                                             - (1+self.mode_params.burnin_factor)*self.burnin)
            self.initialize_model(self.curtidx)

        else:
            raise ValueError("Unrecognized fit mode '{}'".format(self.mode))

        if self.mode_params.burnin_factor == 0:
            burnin = self.burnin
        else:
            burnin = np.random.randint(self.burnin, (1+self.mode_params.burnin_factor)*self.burnin)
        self.curtidx += burnin
        self.advance(self.curtidx)
        self._step(self.curtidx, self.batch_size)

        self.record()

        self.step_i += 1

        # TODO: Check convergence

        return ConvergeStatus.NOTCONVERGED

    def fit(self, Nmax, threadidx=0):
        """
        Parameters
        ----------
        Nmax: int
            Maximum number of iterations.
        threadidx: int
            (Optional) If you have multiple fitting threads, giving each a
            separate index ensures the progress bars are kept separate. Default
            is zero.
        """

        Nmax = int(Nmax)
        try:
            for i in tqdm(range(Nmax), position=threadidx, dynamic_ncols=True):
                self.status = self.step()
                if self.status >= ConvergeStatus.STOP:
                    break
        except (KeyboardInterrupt, SystemExit):
            self.status = ConvergeStatus.ABORTED
            logger.error("Gradient descent was interrupted by external signal.")
                # This isn't an error per se, but we want to make sure
                # it is printed no matter what the logging level is
        except Exception as e:
            # Catch errors, so we can save fit data before terminating.
            # We may need this data to know why the error occured.
            self.status = ConvergeStatus.ERROR
            logger.error(e)

        if self.status != ConvergeStatus.CONVERGED:
            print("Did not converge.")

        # Make sure the final state is saved
        # TODO: `record()` should check if a state is already recorded
        self.record(last=True)

        # TODO: Print timing statistics ?
        #       Like likelihood evaluation time vs total ?

class SGDView(SGDBase):
    """
    Same interface as SGD, without the internal state allowing to preform
    or continue a fit.
    """

    def __init__(self, cost_format,
                 trace, trace_stops, cost_trace, cost_trace_stops):
        super().__init__(cost_format)
        self._trace = trace
        self._trace_stops = trace_stops
        self._cost_trace = self.Cost(cost_trace)
        self._cost_trace_stops = cost_trace_stops

    @property
    def repr_np(self):
        # TODO: Store enough state information (basically, the optimize vars
        #       and transforms) to continue a fit that was saved ot file.
        repr = super().repr_np
        repr['version_sgdview'] = 2
        repr['type'] = 'SGDView'

        repr['trace'] = self.trace
        repr['trace_stops'] = self.trace_stops

        return repr

    @classmethod
    def from_repr_np(cls, repr, _instance=None):
        init_args = ['cost_format', 'trace', 'trace_stops', 'cost_trace',
                     'cost_trace_stops']
        if _instance is not None:
            assert(isinstance(_instance, cls))
            o = _instance
        else:
            version = (repr['version_sgdview'] if 'version_sgdview' in repr
                       else repr['version'])
            # TODO: Use an SGDBase to extract SGDBase keywords
            kwargs = {'cost_format': repr['cost_format'],
                      'trace': repr['trace'][()],
                      'trace_stops': repr['trace_stops'],
                      'cost_trace': repr['cost_trace'],
                      'cost_trace_stops': repr['cost_trace_stops']}
                # Weird [()] indexing extracts element from 0-dim array
            o = cls(**kwargs)

        super().from_repr_np(repr, o)

        return o

    @property
    def cost_trace(self):
        return self._cost_trace
    @property
    def cost_trace_stops(self):
        return self._cost_trace_stops
    @property
    def trace(self):
        return self._trace
    @property
    def trace_stops(self):
        return self._trace_stops

class SeriesSGDView(SGDView):
    """
    Same interface as SGD, without the internal state allowing to perform
    or continue a fit.
    """

    @property
    def repr_np(self):
        repr = super().repr_np
        repr['version_seriessgdview'] = 1
        repr['type'] = 'SeriesSGDView'
        return repr

# class SGD_old:
#
#     def __init__(self, cost=None, cost_format=None, model=None, optimizer=None,
#                  start=None, datalen=None, burnin=None, mbatch_size=None,
#                  set_params=False, *args, _hollow_initialization=False):
#         """
#         Important that start + datalen not exceed the amount of data
#         The data used to fit corresponds to times [start:start+datalen]
#
#         The 'hollow_initialization' argument should not be used; it is an internal flag
#         used by `from_repr_np` to indicate that the SGD is being loaded from file and
#         thus most initializations should be skipped, as they are done in `from_repr_np`.
#         Users should load from files by calling this function, which internally calls
#         `__init__()`. When 'hollow_initialization' is true, start, datalen, mbatch_size
#         are optional as they are ignored even if specified.
#
#         Parameters
#         ----------
#         cost: Theano graph
#             Theano symbolic variable/graph for the cost function we want to optimize.
#             Must derive from parameters and histories found in `model`
#         cost_format: str
#             Indicates how the cost function should be interpreted. This will affect e.g.
#             whether the algorithm attempts to minimize or maximize the value. One of:
#               + 'logL' : log-likelihood
#               + 'negLogL' or '-logL': negative log-likelihood
#               + 'L' : likelihood
#               + 'cost': An arbitrary cost to minimize. Conversion to other formats will
#                    not be possible.
#             May also be a subclass of `Cost`. (TODO)
#         model: sinn Model instance
#             Provides a handle to the variables appearing in the cost graph.
#         optimizer: str | class/factory function
#             String specifying the optimizer to use. Possible values:
#               - 'adam'
#             If a class/factory function, should have the signature
#               `optimizer(cost, fitparams, **kwargs)`
#             where `fitparams` has the same form as the homonymous parameter to `compile`.
#         start: float, int
#             Start of the data used in the fit.
#         datalen: float, int
#             Amount of data on which to evaluate the cost, in the model's time units.
#             Cost will be evaluated up to the time point 'start + datalen'.
#         burnin: float, int
#             Amount of data discarded at the beginning of every batch. In the model's time units (float)
#             or in number of bins (int).
#         mbatch_size: int, float
#             Size of minibatches, either in the model's time units (float) or time bins (int).
#         set_params: bool
#             If true, call `set_params_to_evols` after loading sgd from file.
#             Default is False.
#
#         _hollow_initialization: bool
#             For internal use. Indicates that the SGD should only be partially initialized.
#
#         Internal treatment of parameter substitution
#         ------------------------------------------
#
#         Transformed variables are stored in the 'substitutions' attribute.
#         This is a dictionary, constructed as 3-element tuples keyed by the
#         Theano variables which are being replaced:
#             self.substitutions = {
#                 variable: (new variable, inverse_transform, transform)
#                 ...
#             }
#         'new variable' is also a Theano shared variable; this is the one on which the
#         gradient descent operates.
#         The transforms 'inverse_transform' and 'transform' are strings defining the transformations;
#         they must be such that if we define:
#             f = self._make_transform(transform)
#             inv_f = self._make_transform(inverse_transform)
#         then the following equalities are always true:
#             variable == inv_f(new variable)
#             new variable == f(variable)
#         """
#         # TODO: Find a way to save cost function and model to file, for completely seamless reload ?
#         #       (I.e. without requiring subsequent calls to .set_model() and .set_cost())
#
#         if cost is None:
#             def dummy_cost(*args, **kwargs):
#                 raise AttributeError("A cost function for this gradient descent was not specified.")
#             self.set_cost(dummy_cost, None)
#         else:
#             if cost_format is None:
#                 raise ValueError("The 'cost_format' argument is mandatory "
#                     "when a cost function is provided.")
#             self.set_cost(cost, cost_format)
#
#         if model is None:
#             self.model = None
#         else:
#             self.set_model(model)
#
#         self.tidx_var = theano.tensor.lscalar('tidx')
#         self.mbatch_var = theano.tensor.lscalar('batch_size')
#
#         self.substitutions = {}
#         self._verified_transforms = {}
#             # For security reasons, users should visually check that transforms defined
#             # with strings are not malicious. We save a flag for each transform, indicating
#             # that it has been verified.
#         self.optimizer = None
#
#         if _hollow_initialization:
#             self.fitparams = OrderedDict()   # dict of param:mask pairs
#             self.param_evol = {}
#
#         else:
#             self.optimizer = optimizer
#                 # FIXME: Raise error when optimizer is None ?
#
#             if self.model is None:
#                 raise TypeError("Unless loading an SGD saved to a file, the `model` argument is required.")
#             if start is None:
#                 raise ValueError("SGD: Unless an sgd file is provided, the parameter 'start' is required.")
#             if datalen is None:
#                 raise ValueError("SGD: Unless an sgd file is provided, the parameter 'datalen' is required.")
#             if mbatch_size is None:
#                 raise ValueError("SGD: Unless an sgd file is provided, the parameter 'mbatch_size' is required")
#             self.start = start
#             self.start_idx = model.get_t_idx(start)
#             self.datalen = datalen
#             self.data_idxlen = model.index_interval(datalen, allow_rounding=True)
#             self.burnin = burnin
#             self.burnin_idxlen = model.index_interval(burnin, allow_rounding=True)
#             self.mbatch_size = model.index_interval(mbatch_size, allow_rounding=True)
#
#             #tidx.tag.test_value = 978
#             #shim.gettheano().config.compute_test_value = 'warn'
#             self.fitparams = None
#             self.trueparams = None
#
#             self.initialize()
#                 # Create the state variables other functions may expect;
#                 # initialize will be called again after 'fitparams' is set.
#
#     # def make_cost(cost):
#     #     if isinstance(cost, Cost):
#     #         # CostType has already been mixed in
#     #         if CostType is not None and not isinstance(cost, CostType):
#     #             raise TypeError("Trying to make cost a {}, but it is already of type {}"
#     #                             .format(CostType.__name__, type(cost).__name__))
#     #     else:
#     #         if CostType is None:
#     #             CostType = self.CostType
#     #         cost.__class__ = type(type(cost).__name__ + "_" + CostType.__name__,
#     #                             (CostType, type(cost)),
#     #                             {})
#     #     return cost
#
#     def Cost(self, cost):
#         return Cost(cost, self.cost_format)
#
#     def set_model(self, model):
#         """
#         Parameters
#         ----------
#         model: Model instance
#         """
#         if not isinstance(model, models.Model):
#             raise ValueError("`model` must be an instance of models.Model.")
#         names = [modelname for modelname in models.registered_models
#                  if models.get_model(modelname) is type(model)]
#         if len(names) == 0:
#             modelname = type(model).__name__
#             models.register_model(type(model))
#         else:
#             assert(len(names) == 1)
#             modelname = names[0]
#             if modelname != type(model).__name__:
#                 logger.warning("The provided model (type '{}') is registered under "
#                                "a different name ('{}')."
#                                .format(type(model).__name__, modelname))
#         self.model = model
#         self.modelname = modelname
#
#     def set_cost(self, cost_function, cost_format):
#         self.cost_fn = cost_function
#         self.cost_format = cost_format
#
#     @property
#     def repr_np(self):
#         return self.raw()
#
#     # TODO: Uncomment once swtiched to variable name indexing and we can fully load
#     #       from a file (without an initialized model)
#     # @classmethod
#     # def from_repr_np(cls, repr_np):
#     #     # TODO: Remove 'trust_transforms' when we've switched to using simpleeval
#     #     return cls.from_raw(repr_np, trust_transforms=True)
#
#     def raw(self, **kwargs):
#         raw = {'version': 3}
#
#         def add_attr(attr, retrieve_fn=none):
#             # doing it this way allows to use keywords to avoid errors triggered by getattr(attr)
#             if retrieve_fn is none:
#                 retrieve_fn = lambda attrname: getattr(self, attrname)
#             raw[attr] = kwargs.pop(attr) if attr in kwargs else retrieve_fn(attr)
#
#         add_attr('start')
#         add_attr('start_idx')
#         add_attr('datalen')
#         add_attr('data_idxlen')
#         add_attr('burnin')
#         add_attr('burnin_idxlen')
#         add_attr('mbatch_size')
#
#         add_attr('curtidx')
#         add_attr('step_i')
#         add_attr('circ_step_i')
#         add_attr('output_width')
#         add_attr('cum_cost')
#         add_attr('cum_step_time')
#         add_attr('tot_time')
#
#         add_attr('step_cost')
#         add_attr('_cost_evol')
#
#         # v3
#         raw['type'] = type(self).__name__
#         add_attr('optimizer')
#         add_attr('cost_format')
#
#         if self.trueparams is not None:
#             raw['true_param_names'] = np.array([p.name for p in self.trueparams])
#             for p, val in self.trueparams.items():
#                 raw['true_param_val_' + p.name] = val
#
#         raw['fit_param_names'] = np.array([p.name for p in self.fitparams])
#         for p, val in self.fitparams.items():
#             raw['mask_' + p.name] = val
#         for p, val in self.param_evol.items():
#             assert(p.name in raw['fit_param_names'])
#             raw['evol_' + p.name] = np.array(val)
#
#         raw['substituted_param_names'] = np.array([p.name for p in self.substitutions])
#         for keyvar, subinfo in self.substitutions.items():
#             raw['subs_' + keyvar.name] = np.array([subinfo[0].name, subinfo[1], subinfo[2]])
#
#         raw.update(kwargs)
#
#         return raw
#
#     @classmethod
#     def from_raw(cls, raw, model, trust_transforms=False):
#         """
#         Don't forget to call `verify_transforms` after this.
#         """
#         # TODO: Remove requirement of a model by indexing traces, etc. by name
#         #       rather than variable.
#         # TODO: Allow loading both repr_np (.npr) and sinn pickles (.sin)
#
#         # Using "{}".format(-) on a NumPy array doesn't work, so we
#         # convert the integer variables to Python variables.
#         if not isinstance(raw, np.lib.npyio.NpzFile):
#             raise TypeError("'raw' data must be a Numpy archive.")
#
#         sgd = cls(_hollow_initialization=True)
#         sgd.set_model(model)  # <<<<<----- Remove this once we have name indexing
#         if 'start' not in raw:
#             # The raw data is not v2 compatible; try v1
#             sgd.start = float(raw['burnin'])
#             sgd.start_idx = float(raw['burnin_idx'])
#             sgd.burnin = 0
#             sgd.burnin_idx = 0
#         else:
#             sgd.start = float(raw['start'])
#             sgd.start_idx = int(raw['start_idx'])
#             sgd.burnin = int(raw['burnin'])
#             sgd.burnin_idxlen = int(raw['burnin_idxlen'])
#         sgd.datalen = float(raw['datalen'])
#         sgd.data_idxlen = int(raw['data_idxlen'])
#         sgd.mbatch_size = int(raw['mbatch_size'])
#
#         sgd.curtidx = int(raw['curtidx'])
#         sgd.step_i = int(raw['step_i'])
#         sgd.circ_step_i = int(raw['circ_step_i'])
#         sgd.output_width = int(raw['output_width'])
#         sgd.cum_cost = float(raw['cum_cost'])
#         sgd.cum_step_time = float(raw['cum_step_time'])
#         sgd.tot_time = float(raw['tot_time'])
#
#         sgd.step_cost = list(raw['step_cost'])
#
#         if 'version' in raw:
#             # version >= 3
#             if not hasattr(sgd, 'optimizer') or sgd.optimizer is None:
#                 sgd.optimizer = str(raw['optimizer'])
#             sgd.cost_format = str(raw['cost_format'])
#             sgd._cost_evol = deque(raw['_cost_evol'])
#         else:
#             sgd.cost_format = 'logL'
#             sgd._cost_evol = deque(raw['cost_evol'])
#
#         # Load parameter transforms
#         for name in raw['substituted_param_names']:
#             for p in sgd.model.params:
#                 if p.name == name:
#                     break
#             # Copied from sgd.transform
#             # New transformed variable value will be set once the transform string is verified
#             newvar = theano.shared(p.get_value(),
#                                     broadcastable = p.broadcastable,
#                                     name = raw['subs_'+name][0])
#             inverse_transform = raw['subs_'+name][1]
#             transform = raw['subs_'+name][2]
#             sgd.substitutions[p] = (newvar, inverse_transform, transform)
#             sgd._verified_transforms[p] = trust_transforms
#
#         # Load true parameters
#         if 'true_param_names' in raw:
#             sgd.trueparams = {}
#             for name in raw['true_param_names']:
#                 p = None
#                 for q in sgd.model.params:
#                     if q.name == name:
#                         p = q
#                         break
#                 # fitparam might also be transformed from a base model parameter
#                 for q, subinfo in sgd.substitutions.items():
#                     if subinfo[0].name == name:
#                         p = subinfo[0]
#                         break
#                 assert(p is not None)
#                 sgd.trueparams[p] = raw['true_param_val_' + name]
#         else:
#             sgd.trueparams = None
#
#         # Load fit parameters
#         fit_param_names = raw['fit_param_names']
#         for name in fit_param_names:
#             p = None
#             for q in sgd.model.params:
#                 if q.name == name:
#                     p = q
#                     break
#             # fitparam might also be transformed from a base model parameter
#             for q, subinfo in sgd.substitutions.items():
#                 if subinfo[0].name == name:
#                     p = subinfo[0]
#                     break
#             assert(p is not None)
#             sgd.fitparams[p] = raw['mask_' + name]
#             sgd.param_evol[p] = deque(raw['evol_' + name])
#
#         # Set the parameters to the last value of their respective evolutions
#         # (We can expect the evolutions to be non-empty, since we are loading from file;
#         # otherwise it just prints a warning.)
#         sgd.set_params_to_evols()
#
#
#         return sgd
#
#     def verify_transforms(self, trust_automatically=False):
#         """
#         Should be called immediately after loading from raw.
#
#         Parameters
#         ----------
#         trust_automatically: bool
#             Bypass the verification. Required to avoid user interaction, but
#             since it beats the purpose of this function, to be used with care.
#         """
#         if len(self.substitutions) == 0 or all(self._verified_transforms.values()):
#             # Nothing to do; don't bother the user
#             return
#
#         if trust_automatically:
#             trusted = True
#         else:
#             trusted = False
#             print("Here are the transforms currently used:")
#             for p, val in self.substitutions.items():
#                 print("{} -> {} – (to) '{}', (from) '{}'"
#                       .format(p.name, val[0].name, val[2], val[1]))
#             res = input("Press y to confirm that these transforms are not malicious.")
#             if res[0].lower() == 'y':
#                 trusted = True
#
#         if trusted:
#             for p, subinfo in self.substitutions.items():
#                 self._verified_transforms[p] = True
#                 # Set the transformed value now that the transform has been verified
#                 subinfo[0].set_value(self._make_transform(p, subinfo[2])(p.get_value()))
#
#     def _make_transform(self, variable, transform_desc):
#         assert variable in self._verified_transforms
#         if not self._verified_transforms[variable]:
#             raise RuntimeError("Because they are interpreted with `eval`, you "
#                                "must verify transforms before using them.")
#
#         comps = transform_desc.split('->')
#
#         try:
#             if len(comps) == 1:
#                 # Specified just a callable, like 'log10'
#                 return eval('lambda x: ' + comps[0] + '(x)')
#             elif len(comps) == 2:
#                 return eval('lambda ' + comps[0] + ': ' + comps[1])
#             else:
#                 raise SyntaxError
#
#         except SyntaxError:
#             raise ValueError("Invalid transform description: \n '{}'"
#                              .format(transform_desc))
#
#     def transform(self, variable, newname, transform, inverse_transform):
#         """
#         Transform a variable, for example replacing it by its logarithm.
#         If called multiple times with the same variable, only the last
#         transform is saved.
#         Transformations must be invertible.
#         LIMITATION: This only really works for element-wise transformations,
#         where `transform(x)[0,0]` only depends on `x[0,0]`. For more complex
#         transformations, the `fitparams` argument to `compile` needs to be
#         defined by directly accessing `substitutions[variable][0]`.
#
#         Parameters
#         ----------
#         variable:  theano variable
#             Must be part of the Theano graph for the log likelihood.
#         newname: string
#             Name to assign the new variable.
#         transform: str
#             TODO: Update docs to string defs for transforms
#             Applied to `variable`, returns the new variable we want to fit.
#             E.g. if we want to fit the log of `variable`, than `transform`
#             could be specified as `lambda x: shim.log10(x)`.
#             Make sure to use `shim` functions, rather directly `numpy` or `theano`
#             to ensure expected behaviour.
#         inverse_transform: str
#             Given the new variable, returns the old one. Continuing with the log
#             example, this would be `lambda x: 10**x`.
#         """
#         # # Check that variable is part of the fit parameters
#         # if variable not in self.fitparams:
#         #     raise ValueError("Variable '{}' is not part of the fit parameters."
#         #                      .format(variable))
#
#         assert(newname != variable.name)
#             # TODO: More extensive test that name doesn't already exist
#
#         # Check that variable is a shared variable
#         if not shim.isshared(variable):
#             raise ValueError("Only shared variables can be transformed.")
#
#         # Check that variable is part of the computational graph
#         self.model.theano_reset()
#         self.model.clear_unlocked_histories()
#         #logL = self.model.loglikelihood(self.tidx_var, self.tidx_var + self.mbatch_size)
#         cost, statevar_upds, shared_upds = self.cost_fn(self.tidx_var, self.mbatch_var)
#         self.model.clear_unlocked_histories()
#         self.model.theano_reset()
#         if variable not in theano.gof.graph.inputs([cost]):
#             raise ValueError("'{}' is not part of the Theano graph for the cost."
#                              .format(variable))
#
#         # Since these transform descriptions are given by the user, they are assumed safe
#         self._verified_transforms[variable] = True
#         _transform = self._make_transform(variable, transform)
#         _inverse_transform = self._make_transform(variable, inverse_transform)
#
#         # Check that the transforms are callable and each other's inverse
#         # Choosing a test value is error prone, since different variables will have
#         # different domains – 0.5 is about as safe a value as we will find
#         testx = 0.5
#         try:
#             _transform(testx)
#         except TypeError as e:
#             if "is not callable" in str(e):
#                 # FIXME This error might be confusing now that we save strings instead of callables
#                 raise ValueError("'transform' argument (current '{}' must be "
#                                  "callable.".format(transform))
#             else:
#                 # Some other TypeError
#                 raise
#         try:
#             _inverse_transform(testx)
#         except TypeError as e:
#             if "is not callable" in str(e):
#                 # FIXME See above
#                 raise ValueError("'inverse_transform' argument (current '{}' must be "
#                                  "callable.".format(inverse_transform))
#             else:
#                 # Some other TypeError
#                 raise
#         if not sinn.isclose(testx, _inverse_transform(_transform(testx))):
#             raise ValueError("The given inverse transform does not actually invert "
#                              "`transform({})`.".format(testx))
#
#         # Create the new transformed variable
#         newvar = theano.shared(_transform(variable.get_value()),
#                                broadcastable = variable.broadcastable,
#                                name = newname)
#
#         # Save the transform
#         # We save the strings rather than callables, as that allows us to save them
#         # along with the optimizer when we save to file
#         self.substitutions[variable] = (newvar, inverse_transform, transform)
#             # From this point on the inverse_transform is more likely to be used,
#             # but in some cases we still need the original transform
#
#         # If the ground truth of `variable` was set, compute that of `newvar`
#         self._augment_ground_truth_with_transforms()
#
#         # Update the fit parameters
#         if variable in self.fitparams:
#             assert(newvar not in self.fitparams)
#             self.fitparams[newvar] = self.fitparams[variable]
#             del self.fitparams[variable]
#
#     @property
#     def _transformed_params(self):
#         return [subinfo[0] for subinfo in self.substitutions.values()]
#
#     def _get_nontransformed_param(self, param):
#         res = None
#         for var, subinfo in self.substitutions.items():
#             if subinfo[0] is param:
#                 assert(res is None)
#                 res = var
#         if res is None:
#             assert(param in self.fitparams)
#             res = param
#         return res
#
#     def set_fitparams(self, fitparams):
#
#         # Ensure fitparams is a dictionary of param : mask pairs.
#         if isinstance(fitparams, dict):
#             fitparamsarg = fitparams
#         else:
#             fitparamsarg = dict(
#                 [ (param[0], param[1]) if isinstance(param, tuple) else (param, True)
#                   for param in fitparams ]
#                 )
#
#         # Create self.fitparams
#         # If any of the fitparams appear in self.substitutions, change those
#         # This assumes that the substitution transforms are element wise
#         if self.fitparams is not None:
#             logger.warning("Replacing 'fitparams'. Previous fitparams are lost.")
#         self.fitparams = OrderedDict()
#         for param, mask in fitparamsarg.items():
#             # Normalize/check the mask shape
#             if isinstance(mask, bool):
#                 # A single boolean indicates to allow or lock the entire parameter
#                 # Convert it to a matrix of same shape as the parameter
#                 mask = np.ones(param.get_value().shape, dtype=int) * mask
#             else:
#                 if mask.shape != param.get_value().shape:
#                     raise ValueError("Provided mask (shape {}) for parameter {} "
#                                      "(shape {}) has a different shape."
#                                      .format(mask.shape, param.name, param.get_value().shape))
#
#             if param in self.substitutions:
#                 self.fitparams[self.substitutions[param][0]] = mask
#             else:
#                 self.fitparams[param] = mask
#
#     def standardize_lr(self, lr, params=None):
#         if params is None:
#             params = self.fitparams.keys()
#         return standardize_lr
#
#     def get_cost_graph(self, **kwargs):
#         # Store the learning rate since it's also used in the convergence test
#         if 'lr' in kwargs:
#             self._compiled_lr = kwargs['lr']
#         else:
#             self._compiled_lr = None
#
#         # Create the Theano `replace` parameter from self.substitutions
#         if len(self.substitutions) > 0:
#             replace = { var: self._make_transform(var, subs[1])(subs[0])
#                         for var, subs in self.substitutions.items() }
#                 # We use the inverse transform here, because transform(var)
#                 # is the variable we want in the graph
#                 # (e.g. to replace τ by log τ, we need a new variable `logτ`
#                 #  and then we would replace in the graph by `10**logτ`.
#
#             for var in self.substitutions:
#                 if var in self.fitparams:
#                     logger.warning("The parameter '{}' has been substituted "
#                                    "but is still among the fit params. This is "
#                                    "likely to cause a 'disconnected graph error."
#                                    .format(var.name))
#         else:
#             replace = None
#
#         self.model.theano_reset()
#         self.model.clear_unlocked_histories()
#
#         logger.info("Producing the cost function theano graph")
#         cost, statevar_upds, shared_upds = self.cost_fn(self.tidx_var, self.mbatch_var)
#         logger.info("Cost function graph complete.")
#         if replace is not None:
#             logger.info("Performing variable substitutions in Theano graph.")
#             cost = theano.clone(cost, replace=replace)
#             logger.info("Substitutions complete.")
#
#         return cost, statevar_upds, shared_upds
#
#     def compile(self, fitparams=None, **kwargs):
#         """
#         Parameters
#         ----------
#         **kwargs
#             Most keyword arguments are passed on to 'SGD.get_cost_graph' and
#             the optimizer constructor. Exceptions are the following, which are
#             captured by 'compile':
#               - 'lr': learning_rate. Format must be of a form accepted by
#                       'sgd.standardize_lr'.
#         """
#         # Compile step function
#
#         if fitparams is not None:
#             self.set_fitparams(fitparams)
#         else:
#             if self.fitparams is None:
#                 raise RuntimeError("You must set 'fitparams' before compiling.")
#
#         lr = self.standardize_lr(kwargs.pop('lr', 0.0002))
#
#         cost, statevar_upds, shared_upds = self.get_cost_graph(lr=lr, **kwargs)
#
#         logger.info("Compiling the minibatch cost function.")
#         # DEBUG (because on mini batches?)
#         self.cost = theano.function([self.tidx_var, self.mbatch_var], cost)#, updates=cost_updates)
#         logger.info("Done compilation.")
#
#         # Function for stepping the model forward, e.g. for burnin
#         #logger.info("Compiling the minibatch advancing function.")
#         #self.cost = theano.function([self.tidx_var], updates=cost_updates)
#         #logger.info("Done compilation.")
#
#         cost_to_min = self.Cost(cost).cost
#             # Ensure we have a cost to minimize (and not, e.g., a likelihood)
#         if isinstance(self.optimizer, str):
#             if self.optimizer == 'adam':
#                 logger.info("Calculating Adam optimizer updates.")
#                 optimizer_updates = optimizers.Adam(cost_to_min, self.fitparams, lr=lr, **kwargs)
#             else:
#                 raise ValueError("Unrecognized optimizer '{}'.".format(self.optimizer))
#
#         else:
#             # Treat optimizer as a factory class or function
#             try:
#                 logger.info("Calculating custom optimizer updates.")
#                 optimizer_updates = self.optimizer(cost_to_min, self.fitparams, lr=lr, **kwargs)
#             except TypeError as e:
#                 if 'is not callable' not in str(e):
#                     # Some other TypeError was triggered; reraise
#                     raise
#                 else:
#                     raise ValueError("'optimizer' parameter should be either a string or a "
#                                      "callable which returns an optimizer (such as a class "
#                                      "name or a factory function).\nThe original error was:\n"
#                                      + str(e))
#
#         logger.info("Done calculating optimizer updates.")
#
#         assert(len(shim.get_updates()) == 0)
#         #shim.add_updates(optimizer_updates)
#
#         logger.info("Compiling the optimization step function.")
#         self._step = theano.function([self.tidx_var, self.mbatch_var], [], updates=optimizer_updates)#shim.get_updates())
#         logger.info("Done compilation.")
#
#         # # Compile likelihood function
#         # self.model.clear_unlocked_histories()
#         # self.model.theano_reset()
#         # #cost = self.cost_fn(self.start, self.start + self.datalen)
#         # cost, cost_updates = self.cost_fn(self.tidx_var, self.tidx_var + self.mbatch_size)
#         # if len(replace) > 0:
#         #     cost = theano.clone(cost, replace)
#
#         # self.cost = theano.function([], cost)
#
#         self.model.clear_unlocked_histories()
#         self.model.theano_reset()  # clean the graph after compilation
#
#         #self.initialize()
#
#     def set_params_to_evols(self):
#         """Set the model parameters to the last ones in param_evols"""
#         for param, evol in self.param_evol.items():
#             if len(evol) == 0:
#                 logger.warning("Unable to set parameter '{}': evol is empty"
#                                .format(param.name))
#             else:
#                 if np.all(evol[-1].shape != shim.eval(param.shape)):
#                     raise TypeError("Trying to set parameter '{}' (shape {}) with "
#                                     "its final value (shape {}).\n"
#                                     "Are you certain the correct model was set ?"
#                                     .format(param.name, shim.eval(param.shape), evol[-1].shape))
#                 param.set_value(evol[-1])
#                 if param in self._transformed_params:
#                     original_param = self._get_nontransformed_param(param)
#                     subinfo = self.substitutions[original_param]
#                     original_param.set_value(self._make_transform(original_param, subinfo[1])(param.get_value()))
#
#
#     def set_ground_truth(self, trueparams):
#         """
#         If the true parameters are specified, they will be indicated in plots.
#
#         Parameters
#         ----------
#         trueparams: Iterable of shared variables or ParameterSet or model.Parameters
#         """
#         if isinstance(trueparams, ParameterSet):
#             self.trueparams = {}
#             for name, val in trueparams.items():
#                 try:
#                     param = self.get_param(name)
#                 except KeyError:
#                     pass
#                 else:
#                     self.trueparams[param] = val
#         elif hasattr(trueparams, '_fields'):
#             # It's a model parameter collection, derived from namedtuple
#             self.trueparams = {}
#             for name, val in zip(trueparams._fields, trueparams):
#                 try:
#                     param = self.get_param(name)
#                 except KeyError:
#                     pass
#                 else:
#                     self.trueparams[param] = val
#         else:
#             self.trueparams = { param: param.get_value() for param in trueparams }
#         self._augment_ground_truth_with_transforms()
#
#     def _augment_ground_truth_with_transforms(self):
#         """Add to the dictionary of ground truth parameters the results of transformed variables."""
#         if self.trueparams is None:
#             return
#         for param, transformedparaminfo in self.substitutions.items():
#             if param in self.trueparams:
#                 transformedparam = transformedparaminfo[0]
#                 transform = self._make_transform(param, transformedparaminfo[2])
#                 self.trueparams[transformedparam] = transform(self.trueparams[param])
#             elif transformedparaminfo[0] in self.trueparams:
#                 transformedparam = transformedparaminfo[0]
#                 inv_transform = self._make_transform(transformedparam, transformedparaminfo[1])
#                 self.trueparams[param] = inv_transform(self.trueparams[transformedparam])
#
#     def get_param(self, name):
#         if shim.isshared(name):
#             # 'name' is actually already a parameter
#             # We just check that it's actually part of the SGD, and return it back
#             if (name in self.fitparams
#                 or name in self.substitutions
#                 or name in (sub[0] for sub in self.substitutions.values())):
#                 return name
#             else:
#                 raise KeyError("The parameter '{}' is not attached to this SGD.")
#         else:
#             for param in self.fitparams:
#                 if param.name == name:
#                     return param
#             for param, subinfo in self.substitutions.items():
#                 if param.name == name:
#                     return param
#                 elif subinfo[0].name == name:
#                     return subinfo[0]
#
#             raise KeyError("No parameter has the name '{}'".format(name))
#
#     def set_param_values(self, new_params, mask=None):
#         """
#         Update parameter values.
#
#         Parameters
#         ----------
#         new_params: dictionary
#             Dictionary where keys are model parameters, and
#             values are the new values they should take. It is possible to
#             specify only a subset of parameters to update.
#         mask: dictionary
#             (Optional) Only makes sense to specify this option along with
#             `new_params`. If given, only variable components corresponding
#             to where the mask is True are updated.
#         """
#         for param, val in new_params.items():
#             # Get the corresponding parameter mask, if specified
#             parammask = None
#             if mask is not None:
#                 if param in mask:
#                     parammask = mask[param]
#                 else:
#                     # Also check the list of substituted parameters
#                     for subp, subinfo in self.substitutions.items():
#                         if subp is param and subinfo[0] in mask:
#                             parammask = mask[subinfo[0]]
#                         elif subinfo[0] is param and subp in mask:
#                             parammask = mask[subp]
#
#                 if isinstance(parammask, bool):
#                     parammask = np.ones(param.get_value().shape) * parammask
#
#             if shim.isshared(val):
#                 val = val.get_value()
#             val = np.array(val, dtype=param.dtype)
#
#             if parammask is not None:
#                 val = np.array([ newval if m else oldval
#                                 for newval, oldval, m in zip(val.flat,
#                                                              param.get_value().flat,
#                                                              parammask.flat) ])
#
#             val = val.reshape(param.get_value().shape)
#             param.set_value(val)
#
#             # TODO: move to separate method, that can also be called e.g. after gradient descent
#             # See if this parameter appears in the substitutions
#             for subp, subinfo in self.substitutions.items():
#                 if param is subp:
#                     # This parameter was substituted by another; update the new parameter
#                     subinfo[0].set_value(self._make_transform(param, subinfo[2])(val))
#                 elif param is subinfo[0]:
#                     # This parameter substitutes another; update the original
#                     subp.set_value(self._make_transform(subp, subinfo[1])(val))
#
#     def initialize(self, new_params=None, mask=None):
#         """
#         Clear the likelihood and parameter histories.
#
#         Parameters
#         ----------
#         new_params: dictionary
#             (Optional) Dictionary where keys are model parameters, and
#             values are the new values they should take. It is possible to
#             specify only a subset of parameters to update.
#         mask: dictionary
#             (Optional) Only makes sense to specify this option along with
#             `new_params`. If given, only variable components corresponding
#             to where the mask is True are updated.
#         """
#         if new_params is not None:
#             self.set_param_values(new_params, mask)
#
#         if self.fitparams is not None:
#             self.param_evol = {param: deque([param.get_value()])
#                                for param in self.fitparams}
#         else:
#             self.param_evol = {}
#
#         self._cost_evol = deque([])
#
#         self.step_i = 0
#         self.circ_step_i = 0
#         self.output_width = 5 # Default width for printing number of steps
#         self.step_cost = []
#         self.curtidx = copy.deepcopy(self.start_idx)
#                 # Copy ensures updating curtidx doesn't also update start_idx
#         self.cum_step_time = 0
#         self.cum_cost = 0
#         self.tot_time = 0
#
#     @staticmethod
#     def converged(it, r, p=0.99, n=10, m=10, abs_th=0.001):
#         """
#         r: resolution. Identify a difference in means of r*sigma with probability p. A smaller r makes the test harder to satisfy; the learning rate is a good starting point for this value.
#         p: certainty of each hypothesis test
#         n: length of each test
#         m: number of tests
#         abs_th: changes below this threshold are treated as constant, no matter the
#                 standard deviation (specifically, this amount is added to the std
#                 to compute the reference deviation)
#         """
#         if len(it) < 2*n+m:
#             return False
#         else:
#             # Get the threshold that ensures p statistical power
#             arr = np.array( [it[-i] for i in range(1, n+1)] )
#             std = arr.std(ddof=1, axis=0) + abs_th
#             if np.any( std > 1):
#                 # Ensure that a large variance does not make the test accidentally succeed
#                 return False
#             a = sp.stats.norm.ppf(p, loc=r*std, scale=1)
#             s = np.sqrt(n) / std  # rescaling to get normalized Student-t
#             # Check if the m last iterations were below the threshold
#             return all( (meandiff(it, end=-i) * s < a).all()
#                         for i in range(1, m+1) )
#
#     def step(self, conv_res, cost_calc='cum', mode='random', **kwargs):
#         """
#         Parameters
#         ----------
#         conv_res: float
#             Convergence resolution. Smaller number means a more stringent test for convergence.
#         cost_calc: str
#             Determines how the cost is computed. Some values allow additional
#             keywords to define behaviour. Default is 'cum'; possible values are:
#               - 'cum': (cumulative) The cost of each step is computed, and added to the
#                 previously computed values. This is effectively the cost seen by the algorithm;
#                 it is cheap to compute, but gives only an approximation of the true cost,
#                 as it aggregates the result of many different values.
#                 After having used up all the data, the sum is saved and the process
#                 repeats for the next round.
#               - 'full': At every step, the full likelihood over the whole data is computed.
#                 This is slower, but gives a better indication of whether the algorithm is
#                 going in the right direction.
#                 *Additional keyword*
#                   + 'cost_period': int. Compute the cost only after this many steps
#         mode: str
#             One of:
#             - 'random': standard SGD, where the starting point of each mini batch is chosen
#                randomly within the data. This is the default value.
#             - 'sequential': Mini-batches are taken sequentially, and loop back to the start
#                when we run out of data. This approach is may suffer from the fact that the
#                data windows on multiple loops are not independent but in fact perfectly aligned.
#                On the other hand, if the initial burnin is very long, this may provide
#                substantial speed improvements.
#         """
#         # Clear notebook of previous iteration's output
#         clear_output(wait=True)
#             #`wait` indicates to wait until something is printed, which avoids flicker
#
#         mode_values = ['random', 'sequential']
#
#         # Check arguments
#         if mode not in mode_values:
#             raise ValueError("Unrecognized mode '{}'; it must be one of {}."
#                              .format(mode, ', '.join(["'{}'".format(mv) for mv in mode_values])))
#
#         # Set the current index
#         if mode == 'sequential':
#             if self.curtidx > self.start_idx + self.data_idxlen - self.mbatch_size - self.burnin_idxlen:
#                 # We've run through the dataset
#                 # Reset time index to beginning
#                 self.curtidx = copy.deepcopy(self.start_idx)
#                     # Copy ensures updating curtidx doesn't also update start_idx
#                 self.model.clear_unlocked_histories()
#                 if cost_calc == 'cum':
#                     self._cost_evol.append(np.array((self.step_i, self.cum_cost)))
#                 #self.model.clear_unlocked_histories()
#                 self.cum_cost = 0
#                 self.circ_step_i = 0
#
#                 # HACK Fill the data corresponding to the start time
#                 #      (hack b/c ugly + repeated in iterate() + does not check indexing)
#                 logger.info("Iteration {:>{}} – Moving current index forward to data start."
#                             .format(self.step_i, self.output_width))
#                 self.model.advance(self.start_idx + self.burnin_idxlen)
#                 #for i in range(0, self.burnin_idx, self.mbatch_size):
#                 #    self._step(i)
#                 logger.info("Done.")
#
#             else:
#                 # TODO: Check what it is that is cleared here, and why we need to do it
#                 self.model.clear_other_histories()
#
#         elif mode == 'random':
#             self.curtidx = np.random.randint(self.start_idx,
#                                              self.start_idx + self.data_idxlen
#                                                - self.mbatch_size - self.burnin_idxlen)
#             self.model.clear_unlocked_histories()
#             #self.model.advance(self.start_idx + self.burnin_idxlen)
#             self.model.initialize(t=self.curtidx)
#                 # TODO: Allow to pass keyword arguments to init_at()
#             #self.model.clear_other_histories()
#
#         else:
#             assert(False) # Should never reach here
#
#         t1 = time.perf_counter()
#         self.curtidx += self.burnin_idxlen
#         self.model.advance(self.curtidx)
#         self._step(self.curtidx, self.mbatch_size)
#         self.cum_step_time += time.perf_counter() - t1
#         for param in self.fitparams:
#             self.param_evol[param].append(param.get_value())
#
#         #if cost_calc == 'cum':
#         if True:
#             if self.circ_step_i >= len(self.step_cost):
#                 # At first, extend step_cost as points are added
#                 assert(self.circ_step_i == len(self.step_cost))
#                 self.step_cost.append(self.cost(self.curtidx, self.mbatch_size))
#             else:
#                 # After having looped through the data, reuse memory for the cumulative cost
#                 self.step_cost[self.circ_step_i] = self.cost(self.curtidx, self.mbatch_size)
#             self.cum_cost += self.step_cost[self.circ_step_i]
#
#         if ( cost_calc == 'full'
#              and self.step_i % kwargs.get('cost_period', 1) == 0 ):
#             self._cost_evol.append(
#                 np.array( (self.step_i, self.cost(self.start_idx, self.data_idxlen)) ) )
#
#         # Increment step counter
#         self.step_i += 1
#         self.circ_step_i += 1
#         self.curtidx += self.mbatch_size
#
#         # TODO: Use a circular iterator for step_cost, so that a) we don't need circ_step_i
#         #       and b) we can test over intervals that straddle a reset of curtidx
#         # Check to see if there have been meaningful changes in the last 10 iterations
#         if ( converged((c[1] for c in self._cost_evol),
#                        r=conv_res, n=4, m=3) and
#              converged(self.step_cost[:self.circ_step_i], r=conv_res, n=100)
#              and all( converged(self.param_evol[p], r=self._compiled_lr[p]) for p in self.fitparams) ):
#             logger.info("Converged. log L = {:.2f}".format(float(self._cost_evol[-1][1])))
#             return ConvergeStatus.CONVERGED
#
#         #Print progress  # TODO: move to top of step
#         logger.info("Iteration {:>{}} – <log L> = {:.2f}"
#                     .format(self.step_i,
#                             self.output_width,
#                             float(sum(self.step_cost[:self.circ_step_i])/(self.curtidx + self.mbatch_size -self.start_idx))))
#         if cost_calc == 'full':
#             logger.info(" "*(13+self.output_width) + "Last evaluated log L: {}".format(self._cost_evol[-1][1]))
#
#         return ConvergeStatus.NOTCONVERGED
#
#     def iterate(self, Nmax=int(5e3), conv_res=0.001, **kwargs):
#         """
#         Parameters
#         ----------
#
#         **kwargs: Additional keyword arguments are passed to `step`.
#         """
#
#         Nmax = int(Nmax)
#         self.output_width = int(np.log10(Nmax))
#
#         # HACK Fill the data corresponding to the burnin time
#         #      (hack b/c ugly + repeated in step() + does not check indexing)
#         # Possible fix: create another "skip burnin" function, with scan ?
#         logger.info("Moving current index forward to data start.")
#         #for i in range(0, self.burnin_idx, self.mbatch_size):
#         #    self._step(i)
#         self.model.advance(self.start_idx)
#         logger.info("Done.")
#
#         t1 = time.perf_counter()
#         try:
#             for i in tqdm(range(Nmax)):
#                 status = self.step(conv_res, **kwargs)
#                 if status in [ConvergeStatus.CONVERGED, ConvergeStatus.ABORT]:
#                     break
#         except KeyboardInterrupt:
#             print("Gradient descent was interrupted.")
#         finally:
#             self.tot_time = time.perf_counter() - t1
#
#         if status != ConvergeStatus.CONVERGED:
#             print("Did not converge.")
#
#         if i > 0:
#             logger.info("Cost/likelihood evaluation : {:.1f}s / {:.1f}s ({:.1f}% total "
#                         "execution time)"
#                         .format(self.cum_step_time,
#                                 self.tot_time,
#                                 self.cum_step_time / (self.tot_time) * 100))
#             logger.info("Time per iteration: {:.3f}ms".format((self.tot_time)/self.step_i*1000))
#
#         #with open("sgd-evol", 'wb') as f:
#         #    pickle.dump((L_evol, param_evol), f)
#
#     def get_evol(self):
#         """
#         DEPRECATED: Use trace() instead.
#         Return a dictionary storing the evolution of the cost and parameters.
#
#         Parameters
#         ----------
#
#         Returns
#         -------
#         dictionary:
#              The log likelihood evolution is associated the string key 'logL'.
#              Parameter evolutions are keyed by the parameter names.
#              Each evolution is stored as an ndarray, with the first dimension
#              corresponding to epochs.
#         """
#         evol = { param.name: np.array([val for val in self.param_evol[param]])
#                  for param in self.fitparams }
#         evol['logL'] = np.array([val for val in self._cost_evol])
#         return evol
#
#     @property
#     def trace(sgdself):
#         class TraceDict:
#             def __getitem__(dictself, param):
#                 pname = sgdself.get_param(param)
#                 # Convert deque to ndarray
#                 # TODO: Allow strides, to avoid allocating ginormous arrays
#                 return np.array([val for val in sgdself.param_evol[pname]])
#             def keys(dictself):
#                 for param in sgdself.param_evol.keys():
#                     yield param.name
#             def values(dictself):
#                 for param in sgdself.param_evol.keys():
#                     yield dictself[param]
#             def items(dictself):
#                 for param in sgdself.param_evol.keys():
#                     yield param.name, dictself[param]
#         return TraceDict()
#
#     @property
#     def trace_stops(self):
#         # Should match the traces, e.g. if we allow them to have strides
#         refevol = next(iter(self.param_evol.values()))
#         return np.arange(len(refevol))
#
#     @property
#     def cost_trace(self):
#         return self.Cost(np.fromiter((val[1] for val in self._cost_evol),
#                                      dtype=float, count=len(self._cost_evol)))
#
#     @property
#     def cost_trace_stops(self):
#         return np.array([val[0] for val in self._cost_evol])
#
#     @property
#     def MLE(self):
#         def invert(name, val):
#             param = self.get_param(name)
#             if param not in self._transformed_params:
#                 return name, val
#             else:
#                 original_param = self._get_nontransformed_param(param)
#                 inverse = self._make_transform(original_param,
#                                                self.substitutions[original_param][1])
#                 return original_param.name, inverse(val)
#         return {origname: val for origname, val
#                 in ( invert(name, trace[-1])
#                      for name, trace in self.trace.items()) }
#         #TODO: Use following once transforms use ml.parameters.Transform
#         #return {name: trace[-1].back for name, trace in self.trace.items()}
#
#     @property
#     def stats(self):
#         return {'number iterations': self.step_i,
#                 'total time (s)' : self.tot_time,
#                 'average step time (ms)': self.cum_step_time / self.step_i * 1000,
#                 'average time per iteration (ms)': self.tot_time / self.step_i * 1000,
#                 'time spent stepping (%)': self.cum_step_time / (self.tot_time) * 100}
#
#     def plot_cost_evol(self, evol=None):
#         if evol is None:
#             evol = self.get_evol()
#         plt.title("Maximization of likelihood")
#         plt.plot(evol['logL'][:,0], evol['logL'][:,1])
#         plt.xlabel("epoch")
#         plt.ylabel("$\log L$")
#
#     def plot_param_evol(self, ncols=3, evol=None):
#
#         nrows = int(np.ceil(len(self.fitparams) / ncols))
#         if evol is None:
#             evol = self.get_evol()
#
#         # if self.trueparams is None:
#         #     trueparams = [None] * len(self.fitparams)
#         # else:
#         #     trueparams = self.trueparams
#         # for i, (name, param, trueparam) in enumerate( zip(self.fitparams._fields,
#         #                                                   self.fitparams,
#         #                                                   trueparams),
#         #                                               start=1):
#
#         for i, param in enumerate(self.fitparams, start=1):
#             plt.subplot(nrows,ncols,i)
#             plt.plot(evol[param.name].reshape(len(evol[param.name]), -1))
#                 # Flatten the parameter values
#             plt.title(param.name)
#             plt.xlabel("iterations")
#             plt.legend(["${{{}}}_{{{}}}$".format(param.name, ', '.join(str(i) for i in idx))
#                         for idx in get_indices(param)])
#                         #for i in range(param.get_value().shape[0])
#                         #for j in range(param.get_value().shape[0])])
#             plt.gca().set_prop_cycle(None)
#             if self.trueparams is not None:
#                 if param in self.trueparams:
#                     plt.plot( [ self.trueparams[param].flatten()
#                                 for i in range(len(evol[param.name])) ],
#                               linestyle='dashed' )
#                 else:
#                     logger.warning("Although ground truth parameters have been set, "
#                                    "the value of '{}' was not.".format(param.name))
#
#     def plot_param_evol_overlay(self, basedata, evol=None, **kwargs):
#         """
#         Parameters
#         ----------
#         basedata: Heatmap or […]
#             The sinn data object on top of which to draw the overlay. Currently
#             only heatmaps are supported.
#
#         evol: dictionary, as returned from `get_evol()`
#             The evolution of parameters, as returned from this instance's `get_evol`
#             method. If not specified, `get_evol` is called to retrieve the latest
#             parameter evolution.
#
#         **kwargs:
#             Additional keyword arguments are passed to `plt.plot`
#         """
#
#         if evol is None:
#             evol = self.get_evol()
#
#         if isinstance(basedata, anlz.heatmap.HeatMap):
#             if len(basedata.axes) == 1:
#                 raise NotImplementedError("Overlaying not implemented on 1D heatmaps.\n"
#                                           "What are you hoping to do with a 1D heatmap "
#                                           "anyway ?")
#
#             elif len(basedata.axes) == 2:
#                 # Construct lists of parameter evolution, one parameter (i.e. coord)
#                 # at a time
#                 plotcoords = []
#                 #fitparamnames = [p.name for p in self.fitparams]
#                 #subsparamnames = [p.name for p in self.substitutions]
#                 for ax in basedata.axes:
#                     found = False
#                     for param in self.fitparams:
#                         if param.name == ax.name:
#                             found = True
#                             # Get the parameter evolution
#                             if shim.isscalar(param):
#                                 plotcoords.append(evol[param.name])
#                             else:
#                                 idx = list(ax.idx)
#                                 # Indexing for the heat map might neglect 1-element dimensions
#                                 if len(idx) < param.ndim:
#                                     shape = param.get_value().shape
#                                     axdim_i = 0
#                                     for i, s in enumerate(shape):
#                                         if s == 1 and len(idx) < param.ndim:
#                                             idx.insert(i, 0)
#                                     assert(len(idx) == param.ndim)
#
#                                 idx.insert(0, slice(None)) # First axis is time: grab all of those
#                                 idx = tuple(idx)           # Indexing won't work with a list
#
#                                 plotcoords.append(evol[param.name][idx])
#
#                     if not found:
#                         for param in self.substitutions:
#                             # As above, but we also invert the variable transformation
#                             if param.name == ax.name:
#                                 found = True
#                                 transformedparam = self.substitutions[param][0]
#                                 inversetransform = self._make_transform(param, self.substitutions[param][1])
#                                 if shim.isscalar(param):
#                                     plotcoords.append(inversetransform( evol[transformedparam.name] ))
#                                 else:
#                                     idx = list(ax.idx)
#                                     # Indexing for the heat map might neglect 1-element dimensions
#                                     if len(idx) < param.ndim:
#                                         shape = param.get_value().shape
#                                         axdim_i = 0
#                                         for i, s in enumerate(shape):
#                                             if s == 1 and len(idx) < param.ndim:
#                                                 idx.insert(i, 0)
#                                         assert(len(idx) == param.ndim)
#
#                                     idx.insert(0, slice(None))  # First axis is time: grab all of those
#                                     idx = tuple(idx)            # Indexing won't work with a list
#
#                                     plotcoords.append(inversetransform( evol[transformedparam.name][idx] ))
#
#                     if not found:
#                         raise ValueError("The base data has a parameter '{}', which "
#                                          "does not match any of the fit parameters."
#                                          .format(ax.name))
#
#                 assert( len(plotcoords) == 2 )
#                 plt.plot(plotcoords[0], plotcoords[1], **kwargs)
#
#             else:
#                 raise NotImplementedError("Overlaying currently implemented "
#                                           "only on 2D heatmaps")
#
#         else:
#             raise ValueError( "Overlays are not currently supported for base data of "
#                               "type {}.".format( str(type(basedata)) ) )

class FitCollection:
    ParamID = namedtuple("ParamID", ['name', 'idx'])
    Fit = namedtuple("Fit", ['parameters', 'data'])

    def __init__(self):
        #self.data_root = data_root
        self.fits = []
        self.reffit = None
            # Reference fit, used for obtaining fit parameters
            # We assume all fits were done on the same model and same parameters
        self._iterator = None
        #self.recordstore = RecordStore(db_records)
        #self.recordstore = recordstore # HACK
        #if data_root is not None:
            #self.load_fits()
        #self.heatmap = heatmap

    # Make collection iterable
    def __iter__(self):
        self._iterator = iter(self.fits)
        return self
    def __next__(self):
        return next(self._iterator).data

    def load(self, fit_list, parameters=None, load=None, **kwargs):
        """
        Parameters
        ----------
        fit_list: iterable of (paths | fits | Sumatra records)
            Each element of iterable should have attributes 'parameters'
            and 'outputpath'.
        parameters: iterable of ParameterSets | ParameterSet | None
            Parameters to associate to each fit. If only one ParameterSet is
            given, associate the same to each fit.
            The default value of `None` is appropriate if we don't need to
            associate parameters to fits, or if we are loading fits from
            Sumatra records (which already provide parameters).
        load: function (loaded data) -> SGDView
            Allows specifying a custom loader. Should take whatever
            `mackelab.iotools.load()` returns, and return a SGDView instance.
            Intended for loading data that wasn't originally exported
            from an SGDView instance, as otherwise the default loader suffices.
        **kwargs:
            Keyword arguments are passed on to mackelab.iotools.load()
        """

        # Invalidate internal caches
        self._nffits = None
        self._ffits = None

        # Load the fits
        try:
            logger.info("Loading {} fits...".format(len(fit_list)))
        except TypeError:
            pass # fit_list may be iterable without having a length

        # TODO: Remove when we don't need to read .sir files
        if 'input_format' in kwargs:
            default_input_format = kwargs.pop('input_format')
        else:
            default_input_format = None

        if (isinstance(fit_list, (str, SGDBase))
            or hasattr(fit_list, 'outputpath')):
            # "fit_list" is actually a single fit. Wrap in in a list
            fit_list = [fit_list]

        if (isinstance(parameters, (str, ParameterSet))
            or not isinstance(parameters, Iterable)):
            parameters = itertools.repeat(parameters)

        for fit, params in zip(fit_list, parameters):
            #params = fit.parameters
            #record.outputpath = os.path.join(record.datastore.root,
                                           #record.output_data[0].path)

            if isinstance(fit, SGDBase):
                data = fit

            else:
                if isinstance(fit, str):
                    fitpath = fit
                elif hasattr(fit, 'outputpath'):
                    # Assume this has a Sumatra record interface
                    fitpath = fit.outputpath
                    # Overwrite passed parameters with record's parameters
                    if params is not None:
                        logger.warning("`parameters` argument is ignored for "
                                       "Sumatra records, since they already "
                                       "provide fit parameter.")
                    params = fit.parameters
                else:
                    raise ValueError("`fit_list` items should be Fit objects, "
                                     "paths to saved fit objects, or Sumatra "
                                     "records pointing to saved fit objects.")
                if default_input_format is not None:
                    input_format = default_input_format
                else:
                    # Get format from file extension
                    input_format = None
                #     # TODO: Remove when we don't need to read .sir files anymore
                #     if os.path.splitext(fit.outputpath)[-1] == '.sir':
                #         # .sir was renamed to .npr
                #         input_format = 'npr'
                #     else:
                #         # Get format from file extension
                #         input_format = None

                data = ml.iotools.load(fitpath, input_format=input_format,
                                       **kwargs)
                if load is not None:
                    data = load(data)
                elif isinstance(data, np.lib.npyio.NpzFile):
                    data = SGDView.from_repr_np(data)
            # TODO: Check if sgd is already loaded ?
            # TODO: Check that all fits correspond to the same posterior/model
            self.fits.append( FitCollection.Fit(params, data) )
            #self.sgds[-1].verify_transforms(trust_automatically=True)
            #self.sgds[-1].set_params_to_evols()
            #sgd.record = record

        if len(self.fits) > 0:
            self.reffit = self.fits[0]
        else:
            logger.warning("No fit files were found.")

    # Result access methods
    @property
    def result(self):
        return self.fits[self.result_index].data.MLE
    @property
    def MLE(self):
        return self.result
    @property
    def result_index(self):
        return np.nanargmin([fit.data.result_cost.cost for fit in self.fits])
    @property
    def result_cost(self):
        return self.fits[self.result_index].data.result_cost

    @property
    def nonfinite_fits(self):
        """Return the fits with non-finite elements."""
        if self._nffits is None:
            self._nffits = []
            for fit in self.fits:
                if any( not np.all(np.isfinite(trace))
                        for trace in fit.data.trace.values() ):
                    self._nffits.append(fit)
        return self._nffits
    @property
    def finite_fits(self):
        """Return the fits with non-finite elements."""
        if self._ffits is None:
            self._ffits = []
            for fit in self.fits:
                if all( np.all(np.isfinite(trace))
                        for trace in fit.data.trace.values() ):
                    self._ffits.append(fit)
        return self._ffits

    def plot_cost(self, only_finite=True, **kwargs):
        """
        Parameters
        ----------
        numpoints: int
            Number of points to plot. Defalut is 150.
        keep_range: float
            Parameter traces who's ultimate loglikelihood is within
            this amount of the maximum logL will be coloured as 'kept'.
        [colors]:
            All colors can be set to `None` to deactivate plotting of their
            associated trace.
          keep_color: matplotlib color
              Color to use for the 'kept' traces.
          discard_color: matplotlib color
              Color to use for the 'discarded' traces.
          true_color: matplotlib color
              Color to use for the horizontal line indicating the true parameter value.
        only_finite: bool
            If `True`, only plot traces which do not contain NaNs or Infs.
        linewidth: size 2 tuple
            Linewidths to use for the plot of the 'kept' and 'discarded'
            fits. Index 0 corresponds to 'kept', index 1 to 'discarded'.
        logscale: bool or None
            True: Use log scale for y-axis
            False: Don't use log scale for y-axis
            None: Use log scale for y-axis only if parameter was transformed.
                  (Default)
        xticks, yticks: float, array or matplotlib.Ticker.Locator instance
            int: Number of ticks. Passed as 'numticks' argument to LinearLocator.
            float: Use this value as a base for MultipleLocator. Ticks will
                 be placed at multiples of these values.
            list/array: Fixed tick locations.
            Locator instance: Will call ax.[xy]axis.set_major_locator()
                with this locator.
        """
        #traces = fitcoll.fits[0].data.cost_trace.logL
        if only_finite:
            traces = [fit.data.cost_trace.logL for fit in self.finite_fits]
            stops = [fit.data.cost_trace_stops for fit in self.finite_fits]
            if len(traces) == 0:
                raise RuntimeError("The list of finite fits is empty")
        else:
            traces = [fit.data.cost_trace.logL for fit in self.fits]
            stops = [fit.data.cost_trace_stops for fit in self.fits]
            if len(traces) == 0:
                raise RuntimeError("The list of fits is empty")

        self._plot(stops, traces, only_finite=only_finite, **kwargs)

    def plot(self, var, idx=None, *args,
             targets=None, true_color='#222222',
             only_finite=True, **kwargs):
        """
        Parameters
        ----------
        var: str
            Tracked variable to plot.
        idx: int, tuple, slice or None
            For multi-valued parameters, the component(s) to plot.
            The default value of 'None' plots all of them.
        targets: list
            True value of the parameters. Should have a shape compatible with `idx`.
            If provided, a horizontal line is drawn at this value. Color of the
            line is determined by `true_color`.
        numpoints: int
            Number of points to plot. Defalut is 150.
        keep_range: float
            Parameter traces who's ultimate loglikelihood is within
            this amount of the maximum logL will be coloured as 'kept'.
        [colors]:
            All colors can be set to `None` to deactivate plotting of their
            associated trace.
          keep_color: matplotlib color
              Color to use for the 'kept' traces.
          discard_color: matplotlib color
              Color to use for the 'discarded' traces.
          true_color: matplotlib color
              Color to use for the horizontal line indicating the true parameter value.
        only_finite: bool
            If `True`, only plot traces which do not contain NaNs or Infs.
        linewidth: size 2 tuple
            Linewidths to use for the plot of the 'kept' and 'discarded'
            fits. Index 0 corresponds to 'kept', index 1 to 'discarded'.
        logscale: bool or None
            True: Use log scale for y-axis
            False: Don't use log scale for y-axis
            None: Use log scale for y-axis only if parameter was transformed.
                  (Default)
        xticks, yticks: float, array or matplotlib.Ticker.Locator instance
            int: Number of ticks. Passed as 'numticks' argument to LinearLocator.
                 Special case: If 0 is given, the axis is removed entirely.
                 If required, it can be added back with
                 `plt.gca().spines['left|right|top|bottom'].set_visible(True)`
            float: Use this value as a base for MultipleLocator. Ticks will
                 be placed at multiples of these values.
            list/array: Fixed tick locations.
            Locator instance: Will call ax.[xy]axis.set_major_locator()
                with this locator.
        """
        # # Get parameter variable matching the parameter name
        # pname = param
        # param = self.reffit.data.get_param(param)
        #
        # # Get the associated traces, inverting the transform if necessary
        # #trace_stops = [ fit.data.trace_stops for fit in self.fits ]
        # #trace_idcs = [ range(len(stops)) for stops in trace_stops ]
        # # TODO: Use lambda to avoid evaluating inverse on non-plotted points
        # if param in self.reffit.data.substitutions:
        #     transformed_param = self.reffit.data.substitutions[param][0]
        #     inverse = self.reffit.data._make_transform(
        #         param,
        #         self.reffit.data.substitutions[param][1])
        #     traces = [ inverse(fit.data.trace[transformed_param.name])
        #                for fit in self.finite_fits ]
        # else:
        #     traces = [ fit.data.trace[param.name]
        #                for fit in self.finite_fits ]

        if only_finite:
            traces = [ fit.data.trace[var] for fit in self.finite_fits ]
            trace_stops = ( fit.data.trace_stops for fit in self.finite_fits )
            if len(traces) == 0:
                raise RuntimeError("The list of finite fits is empty")
        else:
            traces = [ fit.data.trace[var] for fit in self.fits ]
            trace_stops = ( fit.data.trace_stops for fit in self.fits )
            if len(traces) == 0:
                raise RuntimeError("The list of fits is empty")


        # Standardize the `idx` argument
        if idx is None:
            idx = (slice(None),)
        elif isinstance(idx, (int, slice)):
            idx = (idx,)

        # Match the shape of the `idx` and the trace
        reftrace = traces[0]
        traceshape = reftrace.shape
        assert(trace.shape == traceshape for trace in traces)
            # All traces should have the same shape
        Δdim = reftrace.ndim - len(idx)
        if Δdim < 1:
            raise ValueError("Index for variable '{}' has more components than "
                             "the variable itself.".format(var))
        elif Δdim > 1:
            # There are unaccounted for dimensions, possibly due to a reshape parameter
            # If trace has size one dimensions, we can remove those
            # We remove dimensions from left to right, until the shapes of the trace
            # and the index match
            # We don't just use a reshape(idx.shape) here to make sure we don't mix
            # dimensions by accident.
            i = 1
            while i < traces[0].ndim:
                if Δdim == 1:
                    break
                if traces[0].shape[i] == 1:
                    # Remove a dimension
                    newshape = traces[0].shape[:i] + traces[0].shape[i+1:]
                    for j, trace in enumerate(traces):
                        traces[j] = trace.reshape(newshape)
                    truevals = truevals.reshape(newshape[1:])
                else:
                    i += 1
        reftrace = traces[0]
            # Reset reftrace so it has the right shape

        # Create iterable of traces for the right component
        plot_traces = (trace[(slice(None),) + idx] for trace in traces)

        # Set default value for log scale
        logscale = kwargs.get('logscale', None)
        if logscale is None:
            # Use log scale if we tracked a log cost
            logscale = 'log' in self.reffit.data.cost_format
        kwargs['logscale'] = logscale

        # Plot
        self._plot(trace_stops, plot_traces, only_finite=only_finite, **kwargs)

        # Draw the true value line
        if true_color is not None and targets is not None:
            targets = np.array(targets).reshape(traceshape[1:])
            # TODO: Would be nice if shape were more reliable and we didn't
            #       need to do so many reshapes.
            # shape = param.get_value().shape
            # truevals = truevals.reshape(shape)
            for target in targets[idx].flat:
                plt.axhline(target, color=true_color, zorder=0)


    def _plot(self, stops, traces, numpoints=150, only_finite=True,
              keep_range=5,
              keep_color='#BA3A05', discard_color='#BBBBBB',
              linewidth=(2.5, 0.8), logscale=None,
              xticks=1, yticks=3):
        """
        Parameters
        ----------
        numpoints: int
            Number of points to plot. Defalut is 150.
        only_finite: bool
            If `True`, only plot traces which do not contain NaNs or Infs.
        keep_range: float
            Parameter traces who's ultimate loglikelihood is within
            this amount of the maximum logL will be coloured as 'kept'.
        [colors]:
            All colors can be set to `None` to deactivate plotting of their
            associated trace.
          keep_color: matplotlib color
              Color to use for the 'kept' traces.
          discard_color: matplotlib color
              Color to use for the 'discarded' traces.
          true_color: matplotlib color
              Color to use for the horizontal line indicating the true parameter value.
        linewidth: size 2 tuple
            Linewidths to use for the plot of the 'kept' and 'discarded'
            fits. Index 0 corresponds to 'kept', index 1 to 'discarded'.
        logscale: bool or None
            True: Use log scale for y-axis
            False: Don't use log scale for y-axis
            None: Use log scale for y-axis only if parameter was transformed.
                  (Default)
        xticks, yticks: float, array or matplotlib.Ticker.Locator instance
            int: Number of ticks. Passed as 'numticks' argument to LinearLocator.
                 Special case: If 0 is given, the axis is removed entirely.
                 If required, it can be added back with
                 `plt.gca().spines['left|right|top|bottom'].set_visible(True)`
            float: Use this value as a base for MultipleLocator. Ticks will
                 be placed at multiples of these values.
            list/array: Fixed tick locations.
            Locator instance: Will call ax.[xy]axis.set_major_locator()
                with this locator.
        """

        # Definitions
        if only_finite:
            logLs = [fit.data.cost_trace[-1].logL for fit in self.finite_fits]
        else:
            logLs = [fit.data.cost_trace[-1].logL for fit in self.fits]
        maxlogL = max(logLs)

        def get_color(logL):
            # Do the interpolation in hsv space ensures intermediate colors are OK
            keephsv = mpl.colors.rgb_to_hsv(mpl.colors.to_rgb(keep_color))
            discardhsv = (mpl.colors.rgb_to_hsv(mpl.colors.to_rgb(discard_color))
                          if discard_color is not None else np.array([0, 0, 1]))
            r = (maxlogL - logL) / keep_range
            return mpl.colors.hsv_to_rgb(((1 - r) * keephsv + r * discardhsv))

        # Get the stops (x axis), and figure out what stride we need to show the desired
        # no. of points
        trace_stops = list(stops)
        # For each trace, create a range of all indices which we then prune
        trace_idcs = [ range(len(stops)) for stops in trace_stops ]
        tot_stops = max( len(idcs) for idcs in trace_idcs )
        stride = max(int( np.rint( tot_stops // numpoints ) ), 1)
            # We prune by taking a value from trace only every 'stride' points
            # We do this making sure to include the first and last points
        for i, idcs in enumerate(trace_idcs):
            trace_idcs[i] = list(idcs[::-stride][::-1])
                # Stride backwards to keep last point
            if trace_idcs[i][0] != 0:
                # Make sure to include first values
                trace_idcs[i] = [0] + trace_idcs[i]
            trace_idcs[i] = np.array(trace_idcs[i])
            trace_stops[i] = trace_stops[i][trace_idcs[i]]
        plot_traces = [trace[idcs] for trace, idcs in zip(traces, trace_idcs)]

        # Loop over the traces
        for trace, stops, logL in zip(plot_traces, trace_stops, logLs):
            # Set plotting parameters
            if logL > maxlogL - keep_range:
                if keep_color is None:
                    continue
                kwargs = {'color': get_color(logL),
                          'zorder': 1,
                          'linewidth': linewidth[0]}
            else:
                if discard_color is None:
                    continue
                kwargs = {'color': discard_color,
                          'zorder': -1,
                          'linewidth': linewidth[1]}

            # Draw plot
            plt.plot(stops, trace, **kwargs)

        # Set the y scale (log or not)
        if logscale:
            plt.yscale('log')

        # Set the tick frequency
        ax = plt.gca()
        for axis, ticks in zip([ax.xaxis, ax.yaxis], [xticks, yticks]):
            if isinstance(ticks, int):
                if axis is ax.yaxis:
                    vmin = min(trace.min() for trace in plot_traces)
                    vmax = max(trace.max() for trace in plot_traces)
                else:
                    vmin = min(stops.min() for stops in trace_stops)
                    vmax = max(stops.max() for stops in trace_stops)
                if ticks == 0:
                    if axis is ax.yaxis:
                        ax.spines['left'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                    else:
                        ax.spines['bottom'].set_visible(False)
                        ax.spines['top'].set_visible(False)
                elif ticks == 1:
                    axis.set_ticks([vmax])
                else:
                    axis.set_major_locator(ml.plot.LinearTickLocator(
                        vmin, vmax, numticks=ticks))
            elif isinstance(ticks, float):
                axis.set_major_locator(mpl.ticker.MultipleLocator(ticks))
            elif isinstance(ticks, Iterable):
                axis.set_ticks(ticks)
            elif isinstance(ticks, mpl.ticker.Locator):
                axis.set_major_locator(ticks)
            else:
                raise ValueError("Unrecognized tick placement specifier '{}'."
                                 .format(ticks))

################
#
# Type registration with mackelab.iotools
#
################

try:
    import mackelab.iotools
except ImportError:
    pass
else:
    mackelab.iotools.register_datatype(SGDBase)
    mackelab.iotools.register_datatype(SeriesSGD)
    mackelab.iotools.register_datatype(SGDView)
    mackelab.iotools.register_datatype(SeriesSGDView)
    #mackelab.iotools.register_datatype(SGD_old, 'SGD')
