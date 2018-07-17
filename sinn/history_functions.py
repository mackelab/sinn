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
import logging
logger = logging.getLogger("sinn.models.inputs")

import theano_shim as shim
import sinn
from sinn.histories import Series

__all__ = ['Sin', 'GaussianWhiteNoise', 'Constant', 'Step']


class SeriesFunction(Series):
    """
    Base class for series functions. Not useful on its own.
    """
    requires_rng = False

    def __init__(self, ref_hist=sinn._NoValue, name=None, t0=sinn._NoValue,
                 tn=sinn._NoValue, dt=sinn._NoValue, dtype=sinn._NoValue,
                 **kwargs):

        shape = self.init_params(**kwargs)
        if shape is None:
            raise ValueError("A history function's `init_params` method must return shape.")
        super().__init__(hist=ref_hist, name=name, t0=t0, tn=tn, dt=dt, shape=shape, iterative=False, dtype=dtype)
        self.set_update_function(self.update_function)


class Constant(SeriesFunction):

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
        #self.start = start
        #self.stop = stop

        if interval in ('closed', 'half-open'):
            self.left_cmp = lambda t: self.get_time(t) >= start
        else:
            self.left_cmp = lambda t: self.get_time(t) > start
        if interval in ('closed', 'half-open-inverted'):
            self.right_cmp = lambda t: self.get_time(t) <= stop
        else:
            self.right_cmp = lambda t: self.get_time(t) < stop

        # if not hasattr(self.baseline_plus_height, 'shape'):
        #     raise ValueError("[sinn.history_function.Step] At least one of the "
        #                      "parameters `baseline` or `height` must be "
        #                      "an array, to provide the expected shape. "
        #                      "This can be a size 1 array.")
        return getattr(self.baseline_plus_height, 'shape', (1,))
            # `baseline + height` result is already broadcasted to the correct shape
            # If they are both scalars and have no shape, set shape to (1,)

    def update_function(self, t):
        if not shim.isscalar(t):
            ndim = self.ndim
            if ndim == 0:
                ndim = 1
            t = shim.add_axes(t, ndim, 'after')
        return shim.cast( shim.switch( shim.and_(self.left_cmp(t), self.right_cmp(t)),
                                       self.baseline_plus_height, self.baseline ),
                          dtype=self.dtype, same_kind=False )

class Sin(SeriesFunction):
    # TODO: Allow Theano parameters

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

        if unit.lower() == 'radians':
            self.amplitude = amplitude
            self.phase = phase
            self.frequency = frequency if frequency is not None else 2*np.pi/period
            if period is not None and frequency != 2*np.pi/period:
                raise ValueError("When creating a sin input, don't specify both frequency and period.")
        elif unit.lower() == 'hz':
            self.amplitude = amplitude
            self.phase = phase * 2 * np.pi
            self.frequency = frequency * 2*np.pi if frequency is not None else 2*np.pi/period
            if period is not None and frequency != 2*np.pi/period:
                raise ValueError("When creating a sin input, don't specify both frequency and period.")
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
        return shim.cast(self.baseline
                         + self.amplitude
                           * shim.sin(self.frequency*t + self.phase),
                         dtype = self.dtype, same_kind=False)


class GaussianWhiteNoise(SeriesFunction):
    """
    Values will limited to the range ±clip_limit;
    """
    requires_rng = True

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
