import collections
from collections import OrderedDict, deque, namedtuple, Iterable
from itertools import chain
import itertools
import os.path
import time
import logging
import copy

logger = logging.getLogger(__file__)

import numpy as np
import scipy as sp
from parameters import ParameterSet
from tqdm import tqdm

import theano
import theano.tensor as T

import mackelab_toolbox as mtb
import mackelab_toolbox.iotools
import mackelab_toolbox.optimizers as optimizers
from mackelab.utils import OrderedEnum
from mackelab.theano import CompiledGraphCache
#from mackelab.optimizers import Adam, NPAdam
import theano_shim as shim

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
    params: dict of (name: symbolic variables)
        Parameters which will be fit.
    """
    if shim.isscalar(lr):
        lr = {p: lr for p in params}
    elif isinstance(lr, (dict, collections.abc.Mapping)):
        new_lr = {}
        lr = copy.deepcopy(lr)
        default_lr = lr.get('default', None)
        for pname, p in params.items():
            if p in lr:
                assert(shim.isshared(p))
                assert(pname not in lr)
                assert(p.name not in lr)
                new_lr[p] = lr[p]
            elif pname in lr:
                assert(pname == p.name or p.name not in lr)
                new_lr[p] = lr[pname]
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
        self.compile_cache = CompiledGraphCache('compilecache.' + __name__)

        self.cost_format = cost_format
        self.status = ConvergeStatus.NOTSTARTED
        self.result_choice = result_choice
        self.trackvar_strings = {}  # Used to store fancy variable strings

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

    @property
    def tracked(self):
        return list(self._trace.keys())

    def set_varstrings(self, varstrings, check_names=True):
        """
        Set assign a string to each variable, which may be used in fancy
        output (e.g. figures). Contrary to identifiers, variable strings may
        by any string, so for example may contain '$' characters to indicate
        TeX formatting.
        Variable strings may be retrieved by `get_varstring`, and may be used
        more by SGD functions in the future.
        The default string simply appends the index to the identifier.

        Note: It is not necessary to call this function to use the defaults.

        Parameters
        ----------
        varstrings: dict
            `{name: [string1, string2, …]}` pairs, or `{(name, idx): string}`
            `name` should match the name of one of the tracked variables
            (as given to the `track_vars` argument of the SGD constructor)
            If key is just the name, the associated value should be a list of
            of strings, one for each component.
            If key is a tuple, `idx` is the flat component index, and `string`
            the string to associate to that component.
        check_names: bool
            If `True`, variables names are checked to actually be part of the
            tracked variables.
        """
        def check_name(name):
            if name not in self._trace:
                if check_names:
                    raise ValueError(
                        "`{}` is not a tracked variable. You can disable this "
                        "check by passing `check_names=False`.".format(name))
                else:
                    return False
            return True
        for key, value in varstrings.items():
            if isinstance(key, tuple):
                name, idx = key
                if check_name(name):
                    if not isinstance(varstrings, str):
                        raise ValueError("There should be only a single string "
                                         "associated to the variable `{}`."
                                         ""
                                         .format(key))
                    self.trackvar_strings[(name, idx)] = value
            else:
                name = key
                if check_name(name):
                    for idx, s in enumerate(value):
                        self.trackvar_strings[name, idx] = s

    def get_varstring(self, name, idx=None):
        """
        Retrieve the fancy string associated to a variable. Fancy strings can
        be set with `set_varstrings`.
        A `varidx` is composed of a tracked variable name, and an integer index.
        (all variables are flattened internally; there are no multi-dim indices)
        For convenience the `varidx` can be given as either the to components
        or a tuple; i.e., the following calls will return the same string:
            `get_varstring('w', 0)`, `get_varstring(('w', 0))`.
        If `name` is just a string and no `idx` is given, all strings
        corresponding to that name are returned.

        Parameters
        ----------
        name: tuple | str
            If tuple, (name, index).
        idx: int
            Index, when `varidx` is just a string.
        """
        if isinstance(name, tuple):
            assert(idx is None)
            name, idx = name
        if idx is None:
            return [self.get_varstring(varidx)
                    for varidx in self.get_varidcs(name)]
        else:
            return self.trackvar_strings.get((name, idx), name + str(idx))
    get_varstrings = get_varstring

    def get_varidcs(self, varname):
        if getattr(self, '_varidcs', None) is None:
            self._varidcs = [(name, i) for name, r in self.result.items()
                                       for i in range(len(r))]
        return [varidx for varidx in self._varidcs if varidx[0] == varname]

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

def _dummy_reset_function(**kwargs):
    return None

class SeriesSGD(SGDBase):
    # TODO: Spin out SGD class and inherit
    # TODO: Don't compile advance function. Take a plain function, and let it
    #       produce whatever side-effects it needs.
    #       Caller can use self.var_subs to substitute variables in the advance
    #       function with those that are optimized.

    def __init__(self, cost, start_var, batch_size_var,cost_format,
                 optimize_vars, track_vars,
                 start, datalen, burnin, batch_size, advance,
                 avg_cost=True, optimizer='adam', optimizer_kwargs=None,
                 reset=None, initialize=None, mode='random', mode_params=None,
                 cost_track_freq=100, var_track_freq=1):
        """

        ..TODO: Replace (start_var, start) & (batch_size_var, batch_size) with
        shared variables.

        Parameters
        ----------
        cost: symbolic expression
            Symbolic cost. If `avg_cost` is `True` (the default), make sure the
            cost is scaled to the size of the dataset. E.g. if your cost
            expression is for a mini-batch, scale it by
            `datalen / batch_size_var`.
        start_var, batch_size_var: symbolic variable
            Variables appearing in the graph of the cost
        cost_format: str
            Indicates how the cost function should be interpreted. This will
            affect e.g. whether the algorithm attempts to minimize or maximize
            the value. One of:
              + 'logL' : log-likelihood
              + 'negLogL' or '-logL': negative log-likelihood
              + 'L' : likelihood
              + 'cost': An arbitrary cost to minimize. Conversion to other formats will
                   not be possible.
            May also be a subclass of `Cost`. (TODO)
        avg_cost: bool
            Whether to average the cost (i.e. divide by `datalen`) when fitting.
            This does not affect the reported cost, only the expression for the
            gradient. It's typically better to perform this averaging, as it
            makes fitting parameters (e.g. learning rate) more consistent
            across batch sizes.
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
            TODO: Parameters (e.g. learning rate) which depend on iteration
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
        advance: function (start_tidx, stop_tidx) -> Symbolic update dictionary
            NOTE: Will be changed to function relying on side-effects.
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

        # Get time index dtype
        assert(all(np.issubdtype(np.dtype(np.asarray(i).dtype), np.integer)
                   for i in (start, burnin, datalen, batch_size)))
        # TODO: Upcast to most common type instad of forcing all to have same ?
        assert start_var.dtype == batch_size_var.dtype
        tidx_dtype = start_var.dtype
        #tidx_dtype = np.result_type(start, burnin, datalen, batch_size)

        super().__init__(cost_format)
        self.cost_track_freq = cost_track_freq
        self.var_track_freq = var_track_freq
        self.optimizer = optimizer
        self.start = shim.cast(start, tidx_dtype)
        self.datalen = shim.cast(datalen, tidx_dtype)
        self.burnin = shim.cast(burnin, tidx_dtype)
        self.batch_size = shim.cast(batch_size, tidx_dtype)
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
            optimizer_kwargs = dict(copy.deepcopy(optimizer_kwargs))
                # Use `dict` constructor to ensure `optimizer_kwargs` is mutable

        # Make fit parameters array-like
        # start = shim.asarray(start)
        # burnin = shim.asarray(burnin)
        # datalen = shim.asarray(datalen)
        # batch_size = shim.asarray(batch_size)

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
            # so that users know they can use them.
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
        optimize_vars = list(optimize_vars)
        if len(self.optimize_vars_access) != len(optimize_vars):
            raise ValueError("The optimization variables must have unique names.\n"
                             "Optimization variables: {}"
                            .format([v.name for v in optimize_vars]))

        # Substitute the new shared variables in the computational graphs
        self.var_subs = {orig: new
                        for orig, new in zip(optimize_vars, self.optimize_vars)
                        if orig is not new}
        # self.cost_graph1 = cost_graph # DEBUG
        cost_graph = shim.graph.clone(cost_graph, replace=self.var_subs)
        # self.cost_graph2 = cost_graph # DEBUG
        lr = optimizer_kwargs['lr']
        if isinstance(lr, (dict, collections.abc.Mapping)):
            # Ensure `lr` is mutable
            lr = dict(lr)
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
        lr = standardize_lr(lr, self.optimize_vars_access)
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
            fn = self.compile_cache.get(cost_graph,
                                        other_inputs=(self.start, self.datalen))
            if fn is None:
                self.cost = shim.graph.compile(
                    [], cost_graph,
                    givens=[(self.tidx_var, self.start),
                            (self.batch_size_var, self.datalen)])
            else:
                logger.info("Cost function loaded from cache.")
        else:
            if optimizers.debug_flags['nanguard'] is True:
                nanguard = {'nan_is_error': True, 'inf_is_error': True, 'big_is_error': False}
            else:
                nanguard = optimizers.debug_flags['nanguard']
                assert('nan_is_error' in nanguard and 'inf_is_error' in nanguard) # Required arguments to NanGuardMode
            from theano.compile.nanguardmode import NanGuardMode
            self.cost = shim.graph.compile([], cost_graph,
                                           givens=[(self.tidx_var, self.start), (self.batch_size_var, self.datalen)],
                                           mode=NanGuardMode(**nanguard))
        logger.info("Done compilation.")

        # Get advance updates
        _start = shim.symbolic.scalar('start (advance compilation)',
                                      dtype=tidx_dtype)
        _start.tag.test_value = 1
        _stop  = shim.symbolic.scalar('stop (advance compilation)',
                                      dtype=tidx_dtype)
        _stop.tag.test_value = 3
            # Should be at least 2 more than _start, because `advance`'s scan
            # runs from `_start+1` to `_stop`.

        advance_updates = advance(_start, _stop)
        for var, upd in advance_updates.items():
            upd = shim.graph.clone(upd, replace=self.var_subs)
            if hasattr(var, 'dtype'):
                advance_updates[var] = shim.cast(upd, var.dtype, same_kind=True)
            else:
                advance_updates[var] = upd

        # Compile the advance function
        # TODO: Find a way to use a precompiled function that already has
        #       the right side-effects (i.e. updates). This would allow using
        #       a model's `advance()` method directly.
        # assert(self.tidx_var in shim.graph.pure_symbolic_inputs(advance_updates.values()))
        #     # theano.function raises a error if inputs are not used to compute outputs
        #     # Since tidx_var appears in the graphs on the updates, we need to
        #     # silence this error with `on_unused_input`. The assert above
        #     # replaces the test.
        self.advance = self.compile_cache.get([], advance_updates)
        if self.advance is None:
            self.advance = shim.graph.compile([_start, _stop], [],
                                              updates=advance_updates)
                                              #on_unused_inputs='ignore')
            self.compile_cache.set([], advance_updates, self.advance)
        else:
            logger.info("Compiled advance function loaded from cache.")

        # Compile optimizer updates
        # FIXME: changed names in `optimize_vars`
        cost_to_min = self.Cost(cost_graph).cost
            # Ensures we have a cost to minimize (and not, e.g., a likelihood)
        if avg_cost:
            cost_to_min /= datalen
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
                    raise ValueError(
                        "'optimizer' parameter should be either a string or a "
                        "callable which returning an optimizer (such as a "
                        "class name or a factory function).\nThe original "
                        "error was:\n" + str(e))
        ## Compile
        logger.info("Compiling the optimization step function.")
        step_inputs = [self.tidx_var, self.batch_size_var]
        step_inputs = [i for i in step_inputs
                       if i in shim.graph.symbolic_inputs(
                           optimizer_updates.values())]
            # This line mostly for debugging: allows to have no optimization
            # variables at all, which then have no inputs either
        # TODO: Might not be the best test: updates might still be possible
        if len(step_inputs) > 0:
            if 'nanguard' not in optimizers.debug_flags:
                self._step = self.compile_cache.get([], optimizer_updates)
                if self._step is None:
                    # _step = shim.graph.compile(step_inputs, [],
                    _step = shim.graph.compile([], [],
                                               updates=optimizer_updates)
                    def step(*inputs):
                        for var, val in zip(step_inputs, inputs):
                            var.set_value(val)
                        _step()
                    self._step = step
                    self.compile_cache.set([], optimizer_updates, self._step)
                else:
                    logger.info("Compiled step function loaded from cache.")
            else:
                # TODO: Remove duplicate with above
                if optimizers.debug_flags['nanguard'] is True:
                    nanguard = {'nan_is_error': True, 'inf_is_error': True, 'big_is_error': False}
                else:
                    nanguard = optimizers.debug_flags['nanguard']
                    assert('nan_is_error' in nanguard and 'inf_is_error' in nanguard) # Required arguments to NanGuardMode
                from theano.compile.nanguardmode import NanGuardMode
                _step = shim.graph.compile(step_inputs, [],
                                                updates=optimizer_updates,
                                                mode=NanGuardMode(**nanguard))
                def step(*inputs):
                    for var, val in zip(step_inputs, inputs):
                        var.set_value(val)
                    _step()
                self._step = step
            logger.info("Done compilation.")

            # Compile a function to extract tracking variables
            logger.info("Compiling parameter tracking function.")
            tracked_vars = list(self.track_vars.values())
            self._get_tracked = self.compile_cache.get(tracked_vars)
            if self._get_tracked is None:
                self._get_tracked = shim.graph.compile(
                    [], tracked_vars, on_unused_input='ignore',
                    givens = self.var_subs.items())
                self.compile_cache.set(tracked_vars, self._get_tracked)
            else:
                logger.info("Compiled tracking function loaded from cache.")
            logger.info("Done compilation.")
        else:
            self._step = lambda: None
            self._get_tracked = lambda: []
            logger.info("Skipped step function compilation: nothing to update.")

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
        return OrderedDict(
            ( (varname,
               np.fromiter(chain.from_iterable(val.flat for val in trace),
                           dtype=np.float32,
                           count=len(trace)*int(np.prod(trace[0].shape))
                           ).reshape((len(trace),)+trace[0].shape))
              for varname, trace in self._traces.items() ) )

    def initialize_vars(self, init_vals=None):
        """
        Parameters
        ----------
        init_vals: dict
            Dictionary of 'variable: value' pairs. Key variables must be the
            same as those that were passed as `optimize_vars` to `__init__()`.
            (As a convenience, they may be specified the string of the
            corresponding key in `self.track_vars`.)
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
        self.curtidx = self.start
        self.old_curtidx = self.start

        # Since we repeatedly append new values to the traces while fitting,
        # we store them in a 'deque'.
        self._traces = OrderedDict( (varname, deque())
                                    for varname in self.track_vars.keys() )
        self._cost_trace = deque()
        self._tracked_iterations = deque()
        self._tracked_cost_iterations = deque()

        # Set initial values
        for var, value in init_vals.items():
            # FIXME: Should use `optimize_vars`, not `track_vars`
            # if isinstance(var, str):
            #     if not var in self.track_vars:
            #         raise ValueError("The keys of `init_vals` must refer to "
            #                          "variables in `self.track_vars`")
            #     var = self.track_vars[var]
            # elif not var in self.track_vars.values():
            #     raise ValueError("The keys of `init_vals` must refer to "
            #                      "variables in `self.track_vars`")
            if var in self.var_subs:
                var = self.var_subs[var]
            if not np.can_cast(value.dtype, var.dtype, 'same_kind'):
                raise TypeError(
                    "Trying to initialize variable '{}' (type '{}')  with "
                    "value '{}' (type '{}')."
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
                cost = self.cost()
                if not np.isfinite(cost):
                    logger.error("Non-finite cost: {}.".format(cost))
                self._cost_trace.append(cost)
                self._tracked_cost_iterations.append(self.step_i)

    def step(self):
        self.reset_model(**self.optimize_vars_access)
        end = max(0,
                  (self.start + self.datalen - self.batch_size
                   - (1+self.mode_params.burnin_factor)*self.burnin))
        if self.mode == 'sequential':
            if (self.curtidx < self.start
                or self.curtidx > self.start + self.datalen
                                  - self.batch_size
                                  - (1+self.mode_params.burnin_factor)*self.burnin):
                # We either haven't started or run through the dataset
                # Reset time index to beginning
                self.old_curtidx = 0
                if self.mode_params.start_factor == 0:
                    self.curtidx = self.start
                else:
                    self.curtidx = np.random.randint(
                        self.start,
                        self.start + min(
                            end, self.mode_params.start_factor*self.batch_size))
                self.initialize_model(self.curtidx)
            else:
                # This doesn't seem required anymore
                # [model].clear_other_histories()
                pass

        elif self.mode == 'random':
            self.curtidx = np.random.randint(self.start, end)
            self.initialize_model(self.curtidx)

        else:
            raise ValueError("Unrecognized fit mode '{}'".format(self.mode))

        if self.mode_params.burnin_factor == 0 or self.burnin == 0:
            burnin = self.burnin
        else:
            burnin = np.random.randint(self.burnin, (1+self.mode_params.burnin_factor)*self.burnin)
        self.curtidx += burnin
        assert self.old_curtidx <= self.curtidx
        if self.old_curtidx < self.curtidx:
            self.advance(self.old_curtidx, self.curtidx)
        self.old_curtidx = self.curtidx
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
            logger.exception(
                "Fit terminated abnormally with the following error:")

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

class Fit:
    def __init__(self, parameters, data):
        self.parameters = parameters
        self.data = data
    @property
    def result(self):
        return self.data.result

    @property
    def tracked(self):
        return self.data.tracked

class FitCollection:
    ParamID = namedtuple("ParamID", ['name', 'idx'])
    # Fit = namedtuple("Fit", ['parameters', 'data'])
    Fit = Fit  # Deprecate

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

    def __getitem(self, key):
        return self.fits[key].data

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
        # TODO: If reffit.parameters is None, and a loaded has parameters,
        #       use those ?

        # Invalidate internal caches
        self._nffits = None
        self._ffits = None

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

        # Load the fits
        try:
            logger.info("Loading {} fits...".format(len(fit_list)))
        except TypeError:
            pass # fit_list may be iterable without having a length
        for fit, params in zip(fit_list, parameters):
            #params = fit.parameters
            #record.outputpath = os.path.join(record.datastore.root,
                                           #record.output_data[0].path)

            if isinstance(fit, SGDBase):
                data = [fit]

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

                def load_from_file(fitpath):
                    data = mtb.iotools.load(fitpath, input_format=input_format,
                                           **kwargs)
                    if load is not None:
                        data = load(data)
                    elif isinstance(data, np.lib.npyio.NpzFile):
                        data = SGDView.from_repr_np(data)
                    return data

                if isinstance(fitpath, str):
                    data = [load_from_file(fitpath)]
                else:
                    # Record returns a list of output paths
                    data = [load_from_file(fp) for fp in fitpath]

            # TODO: Check if sgd is already loaded ?
            # TODO: Check that all fits correspond to the same posterior/model
            for d in data:
                self.fits.append( FitCollection.Fit(params, d) )
            #self.sgds[-1].verify_transforms(trust_automatically=True)
            #self.sgds[-1].set_params_to_evols()
            #sgd.record = record

        logger.info("Done.")
        if len(self.fits) > 0:
            self.reffit = self.fits[0]
            # Construct a common denominator parameter set
            # We loop over all fits and remove all parameter names which differ
            # to the reffit in at least one fit. What remains must be equal
            # equal across all fits.
            self.parameters = copy.deepcopy(self.reffit.parameters)
            excluded_keys = set()
            for fit in self.fits[1:]:
                if fit.parameters is None or self.parameters is None:
                    continue
                diff = mtb.parameters._dict_diff(self.parameters,
                                                fit.parameters)
                excluded_keys.update(ParameterSet(diff[0]).flatten().keys())
                excluded_keys.update(ParameterSet(diff[1]).flatten().keys())
            for key in excluded_keys:
                del self.parameters[key]
        else:
            logger.warning("No fit files were found.")

    @property
    def tracked(self):
        return self.reffit.tracked
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

    def set_varstrings(self, varstrings, check_names=True):
        """
        Call `set_varstring()` on all fits.
        """
        for fit in self.fits:
            fit.data.set_varstrings(varstrings, check_names)

    def get_varstring(self, name, idx=None):
        return self.reffit.data.get_varstring(name, idx)
    get_varstrings = get_varstring

    def plot_cost(self, only_finite=True, ax=None, **kwargs):
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
        only_finite: bool
            If `True`, only plot traces which do not contain NaNs or Infs.
        linewidth: size 2 tuple
            Linewidths to use for the plot of the 'kept' and 'discarded'
            fits. Index 0 corresponds to 'kept', index 1 to 'discarded'.
        yscale: str | None
            None: (default): Use log scale for y-axis for transformed parameters
            str: Pass on to `ax.set_scale`, forcing the scaling.
        xticks, yticks: float, array or matplotlib.Ticker.Locator instance
            int: Number of ticks. Passed as 'nbins' argument to MaxNLocator
                 after subtracting one.
            float: Use this value as a base for MultipleLocator. Ticks will
                 be placed at multiples of these values.
            list/array: Fixed tick locations.
            Locator instance: Will call ax.[xy]axis.set_major_locator()
                with this locator.
        """
        # Defaults
        if ax is None:
            ax = plt.gca()

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

        self._plot(ax, stops, traces, only_finite=only_finite, **kwargs)

    def plot(self, var, idx=None, ax=None, *args,
             target=None, target_color='#222222',
             only_finite=True, targetwidth=1, targetkwargs=None, **kwargs):
        """
        Parameters
        ----------
        var: str
            Tracked variable to plot.
        idx: int, tuple, slice or None
            For multi-valued parameters, the component(s) to plot.
            The default value of 'None' plots all of them.
        ax: matplotlib axes
            If omitted, obtained with `plt.gca()`.
        target: list
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
          target_color: matplotlib color
              Color to use for the horizontal line indicating the true parameter value.
        only_finite: bool
            If `True`, only plot traces which do not contain NaNs or Infs.
        targetwidth:
            Linewidth for the horizontal target line.
        targetkwargs:
            Extra keyword arguments passed on to the `axhline` call for plotting
            target values.
        linewidth: 2-tuple | number
            Linewidths to use for the plot of the 'kept' and 'discarded'
            fits. Index 0 corresponds to 'kept', index 1 to 'discarded'.
            line.
            If a single value is provided, it is used for all lines.
        yscale: str | None
            FIXME: 'None' doesn't work as expected because transformations
                 are currently not saved.
            None: (default): Use log scale for y-axis for transformed parameters
            str: Pass on to `ax.set_scale`, forcing the scaling.
        xticks, yticks: float, array or matplotlib.Ticker.Locator instance
            int: Number of ticks. Passed as 'nbins' argument to MaxNLocator
                 after subtracting one.
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

        if ax is None:
            ax = plt.gca()

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

        reftrace = traces[0]

        # Standardize the `idx` argument
        if reftrace.ndim == 1:
            assert idx == () or idx is None
            idx = ()
        elif idx is None:
            idx = (slice(None),)
        elif isinstance(idx, (int, slice)):
            idx = (idx,)

        # Match the shape of the `idx` and the trace
        traceshape = reftrace.shape
        assert(trace.shape == traceshape for trace in traces)
            # All traces should have the same shape
        Δdim = reftrace.ndim - len(idx)
        if Δdim < 1:
            raise ValueError("Index for variable '{}' has more components than "
                             "the variable itself.".format(var))
        elif Δdim > 1:
            # FIXME: Clearly stale code, since `truevals` is not defined
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
        # FIXME: This is stale code. "log cost" ?
        # logscale = kwargs.get('logscale', None)
        # if logscale is None:
        #     # Use log scale if we tracked a log cost
        #     logscale = 'log' in self.reffit.data.cost_format
        # kwargs['logscale'] = logscale

        # Plot
        self._plot(ax, trace_stops, plot_traces, only_finite=only_finite,
                   **kwargs)

        # Draw the target value line(s)
        if target_color is not None and target is not None:
            # First check if target is a data structure; if so, it must
            # have the same keys|attributes as the fits
            if hasattr(target, var):
                _target = getattr(target, var)
            elif isinstance(target, dict) and var in target:
                _target = target[var]
            else:
                _target = target
            # Target may be a list or nested list; convert to array to
            # standardize checks
            _target = np.array(_target)
            if reftrace.ndim == 1:
                ntargets = 1
            else:
                ntargets = reftrace[0, idx].size
                    # HACK ? Not sure how safe this is
            if _target.ndim == 0:
                # A single value was given; just draw a line there
                targets = np.atleast_1d(_target)
            elif _target.size == ntargets:
                # There are just enough targets for each trace:
                # assume target corresponds to one
                targets = _target.flat
            else:
                # Since sizes don't match, assume that the target must be
                # indexed the same way as the slices
                if len(idx) > 1:
                    assert(len(idx) == target.ndim)
                    targets = np.atleast_1d(_target[idx].flat)
                else:
                    targets = np.atleast_1d(_target.flat[idx])
                if len(targets) != ntargets:
                    raise ValueError("Target dimensions (shape {}) don't match "
                                     "the number of traces ({})."
                                     .format(_target.shape, len(traces)))
            if targetkwargs is None: targetkwargs = {}
            zorder = targetkwargs.pop('zorder', -2)
            targetwidth = targetkwargs.pop('linewidth', targetwidth)
            for t in targets:
                ax.axhline(t, color=target_color,
                           zorder=zorder, linewidth=targetwidth,
                           **targetkwargs)

    def _plot(self, ax, stops, traces,
              ylabel=None, ylabelpos='corner', ylabelkwargs=None,
              yscale=None, ylim=None,
              numpoints=150, only_finite=True,
              keep_range=5, keep_color='#BA3A05', discard_color='#BBBBBB',
              linewidth=(1.3, 0.5),
              xticks=1, yticks=3, yticklocator_options=None,
              **plot_kwargs):
        """
        Parameters
        ----------
        stops: 2D iterable
            x-axis values. One list per trace.
        traces: 2D iterable
            y-axis values; list of traces to plot.
            Each inner list must match the dimension of the corresponding list
            in `stops`.
            Number of lists|traces must match number of lists in `stops`.
        ylabel: str
            String to use to label the y-axis.
        ylabelpos: 'corner' | 'middle'
            Where to print the y label. Possible values are:
               - 'corner': Print the y label with `mackelab.plot.corner_ylabel`
                  This places the label in the top left corner, overwriting
                  some of the tick labels.
               - 'middle': Print the y label with the usual `axes.set_ylabel`
                  This centers it vertcally, to the left of the tick labels.
        ylabelkwargs: dict
            Keywords to pass on to the function printing the y label.
        yscale: str | None
            None: (default): Use log scale for y-axis for transformed parameters
            str: Pass on to `ax.set_scale`, forcing the scaling.
        ylim: tuple
            Passed on to `ax.set_ylim()`.
        numpoints: int
            Number of points to plot. Default is 150.
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
        linewidth: size 2 tuple | number
            Linewidths to use for the plot of the 'kept' and 'discarded'
            fits. Index 0 corresponds to 'kept', index 1 to 'discarded'.
            If a single value is provided, it is used for all fits.
        xticks, yticks: float, array or matplotlib.Ticker.Locator instance
            int: Number of ticks. Passed as 'nbins' argument to
                 MaxNLocator after subtracting one.
                 Special case: If 0 is given, the axis is removed entirely.
                 If required, it can be added back with
                 `plt.gca().spines['left|right|top|bottom'].set_visible(True)`
            float: Use this value as a base for MultipleLocator. Ticks will
                 be placed at multiples of these values.
            list/array: Fixed tick locations.
            Locator instance: Will call ax.[xy]axis.set_major_locator()
                with this locator.
        yticklocator_options: dict
            Provided as keyword arguments to the tick locator used for the
            y axis. Which locator is used depends on the value of `yticks`.
        **plot_kwargs:
            Additional keyword arguments passed on to `axes.plot()`.
        """

        # Default values
        if not isinstance(linewidth, Iterable):
            linewidth = (linewidth, linewidth)
        elif len(linewidth) < 2:
            linewidth = (linewidth[0], linewidth[0])

        # Definitions
        if only_finite:
            logLs = [fit.data.cost_trace[-1].logL for fit in self.finite_fits]
        else:
            logLs = [fit.data.cost_trace[-1].logL for fit in self.fits]
        maxlogL = max(logLs)
        σlogLs = np.argsort(logLs)

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
        stride = max( int(np.rint( tot_stops // numpoints )), 1)
            # We prune by taking a value from trace only every 'stride' points
            # We do this making sure to include the first and last points
        # TODO: Stride logarithmically if x-scale is logarithmic
        for i, idcs in enumerate(trace_idcs):
            trace_idcs[i] = list(idcs[::-stride][::-1])
                # Stride backwards to keep last point
            if trace_idcs[i][0] != 0:
                # Make sure to include first values
                trace_idcs[i] = [0] + trace_idcs[i]
            trace_idcs[i] = np.array(trace_idcs[i])
            trace_stops[i] = np.array(trace_stops[i])[trace_idcs[i]]
        plot_traces = [trace[idcs] for trace, idcs in zip(traces, trace_idcs)]

        # Loop over the traces
        ntraces = len(plot_traces)
        #transform = lambda rank: 0.5 - np.tanh(5 * (rank/ntraces - 0.4)) / 2
        transform = lambda rank: 1 - rank/ntraces
        for trace, stops, logL, rank in zip(
            plot_traces, trace_stops, logLs, σlogLs):
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
                          'zorder': -5,
                          'linewidth': linewidth[1]}
                # If there are a lot of discarded traces, vary their opacity
                # according to logL, so that we don't just see a blob
                if ntraces > 100:  # HACK: Hard-coded arbitrary threshold
                    c = mpl.colors.to_rgb(kwargs['color'])
                    α = transform(rank)
                    # Using alpha doesn't always work, so merge with white
                    # background is hard-coded
                    c = 1 + (np.array(c)-1)*α
                    kwargs['color'] = tuple(c)

            # Draw plot
            plot_kwargs.update(kwargs)
            ax.plot(stops, trace, **plot_kwargs)

        # Make the background transparent
        ax.set_facecolor('#FFFFFF00')

        # Set the y scale and limits
        if yscale is not None:
            ax.set_yscale(yscale)
        if ylim is not None:
            ax.set_ylim(ylim)

        # Set the tick frequency
        if yticklocator_options is None: yticklocator_options = {}
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
                    axis.set_major_locator(mtb.plot.MaxNTickLocator(
                        nbins=ticks-1, vmin=vmin, vmax=vmax,
                        **yticklocator_options))
            elif isinstance(ticks, float):
                axis.set_major_locator(mpl.ticker.MultipleLocator(
                    ticks, **yticklocator_options))
            elif isinstance(ticks, Iterable):
                axis.set_ticks(ticks)
            elif isinstance(ticks, mpl.ticker.Locator):
                axis.set_major_locator(ticks)
            else:
                raise ValueError("Unrecognized tick placement specifier '{}'."
                                 .format(ticks))

        # Add the y label
        if ylabel is not None:
            if ylabelkwargs is None: ylabelkwargs = {}
            if ylabelpos == 'middle':
                ax.set_ylabel(ylabel, **ylabelkwargs)
            elif ylabelpos == 'corner':
                mtb.plot.add_corner_ylabel(ax, ylabel, **ylabelkwargs)
            else:
                raise ValueError("Unrecognized argument for `ylabelpos`: {}."
                                 .format(ylabelpos))

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
