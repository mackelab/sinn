from collections import deque
import numpy as np
import pymc3 as pm
import theano_shim as shim
import sinn
import sinn.optimize

# ====================================
# Specialized distributions
# ====================================

# TODO: Find a way to have variable batch sizes ?
def BatchDist(hist, logp, *args, **kwargs):
    """
    Wrapper around PyMC3's
    :class:`~pymc3.distributions.distribution.DensityDist` which retrieves
    the batch dimensions from the model context and the distribution shape &
    dtype from the provided history.
    In the future we should select the distribution based on the history
    type; for example, a history with discrete values should use
    :class:`~.Discrete`.

    We need to specialize PyMC3's `DensityDist` to accomodate the model's
    time dependency. This includes allowing for seqential batches and for
    a burnin-time during which is ignored when computing the cost function.

    Parameters
    ----------
    hist: sinn.History
        History which is observed. It must be locked.
    logp: function `(start, batch_size) -> logp`
        Function taking `start` and `batch_size` arguments and returning
        the log probability for :param:`hist`.
    """
    # Get model
    model = pm.Model.get_context()
    if not isinstance(model, PyMC3Model):
        raise TypeError("To define a `sinn.ObservedRV`, model must be of "
                        "of type `sinn.models.PyMC3Model`.")
    sinn_model = model.sinn_model
    if not hist.locked:
        raise RuntimeError("`sinn.pymc3.ObservedRV` requires a locked "
                           "history.")
    # Get size / shape of data & batch
    shape = (model.batch_size,) + hist.shape
    total_size = (model.data_len,) + hist.shape
    total_size = tuple(np.asscalar(s) if hasattr(s, 'dtype') else s
                       for s in total_size)
        # total_size must have pure Python types, not np.int_
    # Build graph for the logp
    start = sinn_model.batch_start_var
    batch_size = sinn_model.batch_size_var
    batch = hist[start:start+batch_size]
    batch.tag.test_value = np.zeros(shape, dtype=batch.dtype)
    logp_graph = logp(start=start, batch_size=batch_size)
    logp_graph = shim.graph.clone(logp_graph, model.priors.subs)
    def _logp(data):
        # Since logL depends on data before the minibatch, we can't just
        # compute the log-likelihood from `data`. Instead it depends on
        # `start` and `batch_size_var`, which we change ourselves.
        return logp_graph
    # TODO: Define a hinting attribute so histories can define their RV dist ?
    if np.issubdtype(hist.dtype, np.floating):
        var = pm.DensityDist(hist.name + " (dist)",
                             logp=_logp,
                             dtype=hist.dtype, shape=shape, testval=batch,
                             observed=batch, total_size=total_size)
    else:
        raise NotImplementedError

    # Copied from pymc3.models.Var
    # with self:
    #     var = super().__init__(name=name,
    #                            data=batch,
    #                            distribution=dist,
    #                            total_size=total_size, model=self)
    # self.observed_RVs.append(var)
    # assert not var.missing_values
    #
    # self.add_random_variable(var)
    return var

# ================================================
# PyMC3 Model subclass
# ================================================

class PyMC3ModelWrapper:
    """
    Wrapper used on the first call to `model.pymc()` to save a reference to
    the instantiated `PyMC3` model within the `sinn` model.
    """
    # Arguments are stored by the __call__
    def __init__(self, sinn_model):
        self.sinn_model = sinn_model
        self.args = []
        self.kwargs = {'sinn_model': sinn_model}
    def __call__(self, *args, **kwargs):
        self.args.extend(args)
        self.kwargs.update(kwargs)
        self.sinn_model._pymc = PyMC3Model(*self.args, **self.kwargs)
        return self.sinn_model._pymc

class PyMC3Model(pm.Model):
    """
    Extends PyMC3's :class:Model class with a few features for dynamical models.

    Can be attached to a :class:sinn.models.Model instance, which provides
    defaults to simplify the creation of a PyMC3 model from the former:
        - `setup` is set to `self.initialize` if it exists
    All defaults can be overridden by providing a different value to
    :method:__call__().

    Parameters
    ----------
    name, model, theano_config:
        As pymc3.model.Model
    data_start: time index | time
        Time point at which the data starts. May represent an initial burnin
        phase, or just data we want to ignore.
        Defaults to :param:`sinn_model.t0idx`.
        ..Note This is not the batch start. This is the earliest time point
        for *any* batch.
    data_end: time index | time
        Time point at which the data end (inclusive). May represent an initial
        burnin phase, or just data we want to ignore.
        Defaults to :param:`sinn_model``.tnidx`.
        ..Note This is not the batch end. This is the latest time point
        for *any* batch.
    batch_size: index or time interval
        Length of a batch. This must be a plain scalar (not a symbolic).
        Defaults to all of the data (i.e. :param:`sinn_model``.tnidx` -
        :param:`sinn_model``.t0idx`)
    sinn_model: sinn.models.Model
        `sinn` model to which to attach.
        This argument should normally be omitted, since the usual way to
        instantiate :class:`PyMC3Model` is to use the
        :meth:`sinn.model.Models.pymc` property, which already provides it.
    init: callable | None
        Function taking one arguments. Will be called just before evaluating
        any compiled function, passing as argument :param:`data_start`.
        If set to `None`, no initialization is performed.
        If omitted, the attached :class:`Model` is searched for an
        `initialize()` method; if found, this method is used, otherwise
        no initialization is performed.
    """

    def __init__(self, name='', model=None, theano_config=None,
                 data_start=None, data_end=None, batch_size=None,
                 init=sinn._NoValue, sinn_model=None):

        super().__init__(name=name, model=model, theano_config=theano_config)

        if sinn_model is None:
            raise TypeError("Keyword argument `sinn_model` is required.")
        self.sinn_model = sinn_model
        if data_start is None:
            self.data_start = sinn_model.t0idx
        else:
            self.data_start = sinn_model.get_tidx(data_start)
        if data_end is None:
            self.data_end = sinn_model.tnidx
        else:
            self.data_end = sinn_model.get_tidx(data_end)
        if self.data_end < self.data_start:
            raise ValueError("`data_end` must be greater or equal to "
                             "`data_start`.")
        if batch_size is None:
            self.batch_size = self.data_end - self.data_start
        else:
            self.batch_size = sinn_model.index_interval(batch_size)
        if init is sinn._NoValue:
            # Can't use `None` because it signals no init function at all
            self.init = getattr(sinn_model, 'initialize', None)
        else:
            self.init = init

    @property
    def data_len(self):
        return self.data_end - self.data_start + 1
    @property
    def priors(self):
        """
        Return a :class:`PyMC3Prior` object associating each model prior
        to a shared variable of the original model.
        As a convenienc, if both prior and shared var have the same name,
        ' (PyMC)' is appended to the former.
        """
        pymc = deque()
        params = deque()
        for p in self.sinn_model.params:
            prior = getattr(p, 'prior', None)
            if prior is not None:
                if not isinstance(prior,
                                  (pm.model.Factor, pm.model.TransformedRV)):
                    raise TypeError("Prior {} is not a PyMC3 random variable."
                                    .format(prior))
                if prior.name == p.name:
                    prior.name += ' (PyMC)'
                pymc.append(prior)
                params.append(p)
        return PyMC3Prior(pymc, params)

    def makefn(self, outs, mode=None, *args, **kwargs):
        f = super().makefn(outs, mode, *args, **kwargs)
        def makefn_wrapper(*args, **kwargs):
            self.setup(self.data_start)
            return f(*args, **kwargs)
        return makefn_wrapper

    def SeriesSGD(self, burnin=0, cost=None, cost_format=None,
                  optimize_vars=None, track_vars=None,
                  optimizer='adam', optimizer_kwargs=None, mode='sequential',
                  cost_track_freq=5, var_track_freq=1):
        """
        Wrapper around :class:`.optimize.SeriesSGD` requiring fewer arguments,
        since many are taken from the class instance.
        See :class:`sinn.optimize.gradient_descent.SeriesSGD` for a more
        detailed description of the input parameter.

        Parameters
        ----------
        burnin: int | time
            Amount of time to run the model before each batch.
            Default value of 0 is only suitable for models where an observation
            at time index `i` fully determines the internal state at `i+1`
            (modulo any random variables).
        cost: symbolic expr
            Expression must include :attr:`batch_start_var` and
            :attr:`batch_size_var` from :attr:`self.sinn_model` (`sm`).
            Defaults to `self.logpt / sm.batch_size * sm.data_len`.
        cost_format: str
            How to interpret cost. Most importantly, determines whether to
            maximize or minimize.
        optimize_vars: list of variables
            List of variables to optimize. We must be able to differentiate
            the cost function wrt to each of these variables.
            Defaults to `self.priors.transformed`.
        track_vars: dict of `{str: expr}` pairs.
            Dict of symbolic expression associating to each a string label. These are the quantities that the optimizer saves at each step.
            Defaults to `self.priors.labeled`, which tracks the fit variables.
        optimizer: str | class/factory function
            The optimizer to use. Defaults to 'adam'.
        optimizer_kwargs: dict
            Dictionary of keyword values used to initialize the optimizer;
            possible values depend on optimizer.
            FIXME: there is a current hard-coded requirement for 'lr' to be in
            optimizer_kwargs.
            Defaults to `{'lr': 0.0002}`.
        mode: str
            Manner in which the batches are selected. Either 'sequential'
            (default) or 'random'.
        cost_track_freq: int
            How often to compute and record the cost.
            Note that this is currently done on the whole data set, so can be
            computationally costly. Defaults to 5.
        var_track_freq: int
            How often to compute and record the tracked variables.
            Default: 1
        """
        sm = self.sinn_model
        if cost_format is None:
            if cost is not None:
                raise TypeError("Cannot omit `cost_format` if `cost` is given.")
            else:
                cost_format = 'logL'
        if cost is None:
            cost = (self.logpt / sm.batch_size_var * self.data_len)
        if optimize_vars is None:
            optimize_vars = list(self.priors.transformed)
        if track_vars is None:
            track_vars = self.priors.labeled
        if optimizer_kwargs is None:
            optimizer_kwargs = {'lr': 0.0002}
        return sinn.optimize.SeriesSGD(
            start           = self.data_start,
            datalen         = self.data_len,
            batch_size      = self.batch_size,
            start_var       = sm.batch_start_var,
            batch_size_var  = sm.batch_size_var,
            burnin          = sm.index_interval(burnin),
                # Amount of time to run the model before each batch
            cost            = cost,
            cost_format     = cost_format,
            optimize_vars   = optimize_vars,
            track_vars      = track_vars,
            optimizer       = optimizer,
            optimizer_kwargs = optimizer_kwargs,
            cost_track_freq = cost_track_freq,
            var_track_freq  = var_track_freq,
            mode            = mode,
                # The manner in which we draw mini batches, either 'sequential' or 'random'.
            initialize      = self.init,
                # Function to use to initialize the model (e.g. at t=0)
            advance         = sm.advance_updates,
                # Function used to advance (integrate) the model forward in time
        )

class PyMC3Prior:
    """
    Object for storing a collection of PyMC3 random variables (RV)s and their
    associations to plain (shared) model parameters.

    Iterating over an instance yields the PyMC3 RVs. To get the model,
    use `[PyMC3Prior].params`.

    ..Todo: Merge with `mackelab.pymc3.PyMCPrior`
    """
    def __init__(self, pymc_vars=(), param_vars=(), names=None):
        self.pymc_vars = list(pymc_vars)
        self.param_vars = list(param_vars)
        if len(self.pymc_vars) != len(self.param_vars):
            raise ValueError("`pymc_vars` and `param_vars` must have the "
                             "the same length.")
        if names is not None:
            names = list(names)
            if len(names) != len(self.pymc_vars):
                raise ValueError("`pymc_vars` and `names` must have the "
                                 "the same length.")
            self.names = names
        else:
            self.names = [p.name for p in self.param_vars]

    def __len__(self):
        return len(self.pymc_vars)
    def __str__(self):
        return "\n".join(str(pymc) for pymc in self.pymc_vars)
    def __repr__(self):
        return "\n".join(repr(pymc) for pymc in self.pymc_vars)
    def __iter__(self):
        return iter(self.pymc_vars)

    def add_prior(self, pymc_var, param_var, name=None):
        self.pymc_vars.append(pymc_var)
        self.param_vars.append(param_var)
        if name is None:
            self.names.append(param_var.name)
        else:
            self.names.append(name)

    def sample(self, keys='pymc'):
        """
        Returns a dictionary of `{key: value}` pairs where the `value`s are
        sampled from the prior and `key`s are either the variable labels,
        the parameters or the prior symbolic variables. If :param:`keys` is
        `None`, a list is returned instead.

        Parameters
        ----------
        keys : str
            What to use for keys
            - 'names': variable labels
            - 'params': symbolic parameter variables
            - 'priors' | 'pymc': (default) symbolic prior (PyMC3) variables
            - `None`or 'None': No keys: just output a list of values.

        Returns
        -------
        dict of `{key: value}` pairs, or a list of `[value]`.
        """
        samples = [prior.random() for prior in self.pymc_vars]
        if keys is None or keys in ('none', 'None'):
            return samples
        elif keys == 'names':
            return {n: s for n, s in zip(self.names, samples)}
        elif keys == 'params':
            return {p: s for p, s in zip(self.param_vars, samples)}
        elif keys == 'priors' or keys == 'pymc':
            return {p: s for p, s in zip(self.pymc_vars, samples)}
        else:
            raise ValueError("Unrecognized value {} for argument `keys`."
                             .format(keys))

    @property
    def params(self):
        return self.param_vars
    @property
    def transformed(self):
        """
        Return a generator for the PyMC3 variables. For transformed variables,
        returns the transform.
        """
        T = []
        for v in self.pymc_vars:
            if isinstance(v, pm.model.TransformedRV):
                if v.transformed.random is None:
                    def random(v=v):  # Default argument copies current value
                        return v.transformation.forward_val(v.random())
                    v.transformed.random = random
                T.append(v.transformed)
            else:
                T.append(v)
        p = [p if p is t else None
             for p, v, t in zip(self.params, self.pymc_vars, T)]
        return PyMC3Prior(T, p, [t.name for t in T])
    @property
    def subs(self):
        """
        Return a substitution dictionary for replacing pymc with plain vars.
        """
        return {pymc: param
                for pymc, param in zip(self.transformed, self.param_vars)}
    @property
    def labeled(self):
        return {name: prior for name, prior in zip(self.names, self.pymc_vars)}
