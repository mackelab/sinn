"""Collection of stock non-iterative histories, i.e. histories that do not depend on others
and can be batch-computed. This makes them suitable as external inputs.
Since these essentially behave as functions, we refer to them as 'history functions'.

To define a new history function, make a new class which derives from
SeriesFunction and defines two functions:
  - init_params(self, [random_stream], param1, param2...) -> shape
    Takes the function parameters and stores them in the instance. This is to allow them
    being called in the definition of the update function. The way parameters are stored
    is arbitrary.
    This function is also the place to perform sanity checks on the parameters.
  - update_function(self, t)
    A History update_function, which will be past to the history's `set_update_function`
    method. It has access to whatever parameter storage was created in `set_params`.

History functions requiring a random number generator should also set the class attribute
`requires_rng` to True. A random stream object will then be expected to be passed as first
argument to the initializer, which can then be stored for use by `update_function`.
"""

# IMPLEMENTATION NOTE:
# All update functions use an unsafe cast, since we assume there is a reason the
# user specifies a dtype.
# TODO: Move casting into base type so it's not repeated in every derived class

import numpy as np
import inspect
import logging
logger = logging.getLogger("sinn.models.inputs")

import theano_shim as shim
import sinn
from sinn.histories import Series

__all__ = ['Sin', 'GaussianWhiteNoise', 'Constant', 'Step', 'Steps',
           'PiecewiseLinear']


class SeriesFunction(Series):
    """
    Base class for series functions. Not useful on its own.
    """
    requires_rng = False

    def __init__(self, ref_hist=sinn._NoValue, name=None,
                 symbolic=sinn._NoValue, time_array=sinn._NoValue,
                 t0=sinn._NoValue, tn=sinn._NoValue, dt=sinn._NoValue,
                 dtype=sinn._NoValue,
                 **kwargs):

        sig_params = inspect.signature(self.init_params).parameters.keys()
        init_params_kwargs = {key: value for key, value in kwargs.items()
                                         if key in sig_params}
        other_kwargs = {key: value for key, value in kwargs.items()
                                   if key not in sig_params}
        shape = self.init_params(**init_params_kwargs)
        if shape == ():
            logger.warning("History function returned a 0 dimensional shape; "
                           "it must be at least 1D. Resetting shape to `(1,)`.")
            shape = (1,)
        if shape is None:
            raise ValueError("A history function's `init_params` method must return shape.")
        super().__init__(
            hist=ref_hist, name=name, symbolic=symbolic,
            t0=t0, tn=tn, dt=dt, time_array=time_array,
            shape=shape, iterative=False, dtype=dtype,
            **other_kwargs)
        self.set_update_function(self.update_function, inputs=[])

    @property
    def repr_np(self):
        repr = super().repr_np
        repr.update({name: getattr(self, name) for name in self.Parameters})
        return repr

    @classmethod
    def from_repr_np(cls, raw, *args, **kwargs):
        hist = super().from_repr_np(raw, *args, **kwargs)
        for name in cls.Parameters:
            setattr(hist, name, raw[name])
        return hist

class Constant(SeriesFunction):
    Parameters = ['value']

    def init_params(self, value):
        self.value = value  # Can't cast here – self.dtype isn't set yet
        return value.shape

    def update_function(self, t):
        if shim.isscalar(t):
            return shim.cast(self.value, dtype=self.dtype)
        else:
            assert(t.ndim == 1)
            return shim.cast(shim.tile(self.value, (t.shape[0],) + (1,)*self.ndim),
                             dtype=self.dtype, same_kind=False)


class Step(SeriesFunction):
    Parameters = ['baseline', 'baseline_plus_height', 'start', 'stop',
                  'interval']

    def init_params(self, baseline=0, height=None, start=None, stop=None, interval='half-open'):
        """
        Parameters
        ----------
        baseline: float
            Value outside the step; defaults to 0
        height: float
            Height of the step. Added to `baseline`.
        start: float
            Start of the step. Always in time units, never in bin units.
        stop: float
            End of the step. Always in time units, never in bin units.
        interval: str
            Whether or not to include the end points within the step. An included point will
            be at `baseline + height`, an excluded point at `baseline`. Possible values are:
              - 'closed' : Include both start and stop points.
              - 'open'   : Exclude both start and stop points.
              - 'half-open' : (Default) Include start point, exclude stop point.
              - 'half-open-inverted' : Exclude start point, include start point.
        """

        # ensure all required parameters were provided
        for arg in ('height', 'start', 'stop'):
            if vars()[arg] is None:
                raise ValueError("'{}' is a required argument.".format(arg))
        if interval not in ['closed', 'open', 'half-open', 'half-open-inverted']:
            raise ValueError("'interval' argument is '{}'. It must be one of "
                             "'closed', 'open', 'half-open' or 'half-open-inverted'"
                             .format(interval))

        self.baseline = baseline
        self.baseline_plus_height = baseline + height
        self.interval = interval
        self.start = start
        self.stop = stop

        # if not hasattr(self.baseline_plus_height, 'shape'):
        #     raise ValueError("[sinn.history_function.Step] At least one of the "
        #                      "parameters `baseline` or `height` must be "
        #                      "an array, to provide the expected shape. "
        #                      "This can be a size 1 array.")
        shape = getattr(self.baseline_plus_height, 'shape', (1,))
            # `baseline + height` result is already broadcasted to the correct shape
            # If they are both scalars and have no shape, set shape to (1,)
        if shape == (): shape = (1,)
            # This can happen if some parameters are passed as 0-dim arrays
        return shape

    def left_cmp(self, t):
        if self.interval in ('closed', 'half-open'):
            return self.get_time(t) >= self.start
        else:
            return self.get_time(t) > self.start
    def right_cmp(self, t):
        if self.interval in ('closed', 'half-open-inverted'):
            return self.get_time(t) <= self.stop
        else:
            return self.get_time(t) < self.stop

    def update_function(self, t):
        if not shim.isscalar(t):
            ndim = self.ndim
            if ndim == 0:
                ndim = 1
            t = shim.add_axes(t, ndim, 'after')
        return shim.cast( shim.switch( shim.and_(self.left_cmp(t), self.right_cmp(t)),
                                       self.baseline_plus_height, self.baseline ),
                          dtype=self.dtype, same_kind=False )

class Steps(SeriesFunction):
    Parameters = ['init_vals', 'stops', 'values']
    def init_params(self, init_val=0, stops=None, values=None):
        """
        WARNING: UNTESTED

        A series of constant functions changing at times `stops`.
        I[t] = values[i] for stops[i] <= t < stops[i]

        Parameters
        ----------
        init_val: float
            Initial value
        stops: float
            t coordinates of the points we want to go through.
            Always in time units, never in bin units.
        values: float
            x coordinates of the points we want to go through.
        """

        # ensure all required parameters were provided
        for arg in ('stops', 'values'):
            if vars()[arg] is None:
                raise ValueError("'{}' is a required argument.".format(arg))
        self.stops = stops
        self.values = [init_val] + list(values)
        raise NotImplementedError("Must return `shape`")

    def update_function(self, t):
        if not shim.isscalar(t):
            ndim = self.ndim
            if ndim == 0:
                ndim = 1
            t = shim.add_axes(t, ndim, 'after')
        return shim.cast(self.get_values(self.stops, self.values),
                         dtype=self.dtype, same_kind=False)

    def get_values(self, t, stops, values):
        if len(stops) > 1:
            return shim.switch( t <= stops[0],
                                values[0],
                                self.get_values(stops[1:], values[1:]) )
        else:
            assert(len(values) == 2)
            return shim.switch( t <= stops[0],
                                values[0], values[1])

class PiecewiseLinear(SeriesFunction):
    Parameters = ['stops', 'values']

    """
    Linearly interpolate between the points `(stop, value)`.
    Before the first stop and after the last, the function is constant.

    Parameters
    ----------
    [History parameters]
    stops: float
        t coordinates of the points we want to go through.
        Always in time units, never in bin units.
    values: float
        x coordinates of the points we want to go through.
    """
    def init_params(self, stops=None, values=None):
        if stops is None or values is None:
            # Equivalent to Constant
            assert(stops is None and values is None)
            self.stops = []
            values = []

        else:
            # ensure all required parameters were provided
            for arg in ('stops', 'values'):
                if vars()[arg] is None:
                    raise ValueError("'{}' is a required argument.".format(arg))
            assert(len(stops) == len(values))

        val0 = values[0]
        stop0 = stops[0]
        self.stops = stops
        self.values = values  # Only used in `self.repr_np`
        self.functions = [lambda t: val0]
        for stop, val in zip(stops[1:], values[1:]):
            # We can get `stop==stop0` if a stop is repeated.
            # In this case, we just set α to 0; it's value doesn't really
            # matter, because the function will never be evaluated. We just
            # don't want NaNs.
            α = shim.switch( shim.or_(val == val0, stop == stop0),
                             np.zeros(val0.shape),
                             (val-val0)/(stop-stop0) )
            assert np.all(shim.isfinite(α, where=True))
            def f(t, α=α, stop0=stop0, val0=val0):
                # Use optional arguments to tie the current values
                # of α and stop0 to the function
                return α*(t-stop0) + val0
            self.functions.append(f)
            stop0 = stop
            val0 = f(stop)
        self.functions.append(lambda t: val0)

        shape = np.broadcast(*values).shape
        if shape == (): shape = (1,)
            # np.atleast_1d doesn't seem to work with `broadcast`
        return shape

    @classmethod
    def from_repr_np(cls, raw, *args, **kwargs):
        hist = super().from_repr_np(raw, *args, **kwargs)
        hist.init_params(hist.stops, hist.values)

    def update_function(self, t):
        t = self.get_time(t)
        if not shim.isscalar(t):
            ndim = self.ndim
            if ndim == 0:
                ndim = 1
            t = shim.add_axes(t, ndim, 'after')
        return shim.cast(self.apply_functions(t, self.stops, self.functions),
                         dtype=self.dtype, same_kind=False)

    def apply_functions(self, t, stops, functions):
        if len(stops) > 1:
            return shim.switch( t <= stops[0],
                                functions[0](t),
                                self.apply_functions(t, stops[1:], functions[1:]) )
        else:
            assert(len(functions) == 2)
            return shim.switch( t <= stops[0],
                                functions[0](t), functions[1](t) )
class Sin(SeriesFunction):
    # TODO: Allow Theano parameters
    Parameters = ['amplitude', 'phase', 'frequency']

    def init_params(self, baseline=0, amplitude=None, frequency=None, period=None, phase=0, unit='Hz'):
        """
        If 'unit' is 'Hz', frequency gives the number of cycles per unit time,
        while phase is a shift in numbers of cycles. I.e. both frequency and phase
        are multiplied by 2π before being given to sin().
        """

        if amplitude is None:
            raise ValueError("'amplitude' is a required parameter.")

        if frequency is None and period is None:
            raise ValueError("Either frequency or period must be specified to create a sin input")

        self.baseline = baseline

        if (frequency is None) == (period is None):
            raise ValueError("When creating a sin input, exactly one of "
                             "`frequency` and `period` must be specified.")

        if unit.lower() == 'radians':
            self.amplitude = amplitude
            self.phase = phase
            self.frequency = frequency if frequency is not None else 2*np.pi/period
            # if period is not None and frequency != 2*np.pi/period:
            #     raise ValueError("When creating a sin input, don't specify both frequency and period.")
        elif unit.lower() == 'hz':
            self.amplitude = amplitude
            self.phase = phase * 2 * np.pi
            self.frequency = frequency * 2*np.pi if frequency is not None else 2*np.pi/period
            # if period is not None and frequency != 2*np.pi/period:
            #     raise ValueError("When creating a sin input, don't specify both frequency and period.")
        else:
            raise ValueError("When creating a sin input, 'unit' must be either "
                             "'radians' or 'Hz', not '{}'".format(unit))

        try:
            shape = np.broadcast(self.baseline, np.broadcast(
                self.amplitude, np.broadcast(self.frequency, self.phase))).shape
        except ValueError:
            raise ValueError("The arguments 'baseline', 'amplitude', 'frequency' and 'phase' have incompatible "
                             "dimensions: {}, {}, {}, {}."
                             .format(self.baseline.shape, self.amplitude.shape, self.frequency.shape, self.phase.shape))
        return shape

    def update_function(self, t):
        if not shim.isscalar(t):
            ndim = self.ndim
            if ndim == 0:
                ndim = 1
            t = shim.add_axes(t, ndim, 'after')
        t = self.get_time(t)
        return shim.cast(self.baseline
                         + self.amplitude
                           * shim.sin(self.frequency*t + self.phase),
                         dtype = self.dtype, same_kind=False)


class GaussianWhiteNoise(SeriesFunction):
    """
    Values will limited to the range ±clip_limit;
    """
    requires_rng = True
    Parameters = ['rndstream', 'std', 'clip_limit']

    def init_params(self, random_stream, shape, std=1.0, clip_limit=87):
        # exp(88) is the largest value scipy can store in a 32-bit float
        self.rndstream = random_stream
        self.std = shim.cast_floatX(std)
        self.clip_limit = np.int8(clip_limit)
        try:
            np.broadcast(std, np.ones(shape))
        except ValueError:
            raise ValueError("The shape of parameter 'std' is incompatible "
                             "with the requested output shape: "
                             "std: {}, shape: {}.".format(std, shape))

        return shape

    def update_function(self, t):
        if shim.isscalar(t):
            outshape = self.shape
        else:
            assert(t.ndim==1)
            outshape = t.shape + self.shape
        if sinn.config.compat_version < '0.1.dev1':
            std = self.std/np.sqrt(self.dt)  # Bug with which some simulations were done
        else:
            std = self.std*np.sqrt(self.dt)
        return shim.clip(self.rndstream.normal(avg  = 0,
                                               std  = std,
                                               size = np.int32(outshape)),
                                                   # Theano expects array
                         -self.clip_limit,
                         self.clip_limit).astype(self.dtype)
