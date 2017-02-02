#TODO: Discretized kernels as a mixin class

class Kernel:
    """Generic Kernel class. All kernels should derive from this."""

    def __init__(self, name, f, memory_time, t0=0):
        """
        Parameters
        ----------
        name: str
            A unique identifier. May be printed to identify this kernel in output.
        f: callable
            Function defining the kernel. Signature: f(t) -> float
        memory_time: float
            Time after which we can truncate the kernel.
        t0: float
            The time corresponding to f(0). Kernel is zero before this time.
        """
        self.name = name
        self.t0 = t0

        self.eval = f
        self.shape = f(0).shape
        self.memory_time = memory_time

    def convolve(self, hist, t):
        return hist.convolve(self, t)


class ExpKernel(Kernel):
    """An exponential kernel, of the form κ(s) = c exp(-(s-t0)/τ).
    """

    def __init__(self, name, height, decay_const, memory_time=None, t0=0):
        """
        Parameters
        ----------
        name: str
            A unique identifier. May be printed to identify this kernel in output.
        height: float, ndarray, Theano var
            Constant multiplying the exponential. c, in the expression above.
        decay_const: float, ndarray, Theano var
            Characteristic time of the exponential. τ, in the expression above.
        memory_time: float
            (Optional) Time after which we can truncate the kernel. If left
            unspecified, calculated automatically.
            Must *not* be a Theano variable.
        t0: float or ndarray
            Time at which the kernel 'starts', i.e. κ(t0) = c,
            and κ(t) = 0 for t< t0.
            Must *not* be a Theano variable.
        """
        self.name = name
        self.height = height
        self.decay_const = decay_const
        self.t0 = t0

        def f(s):
            return self.height * lib.exp(-(s-self.t0) / self.decay_const)
        self.eval = f

        try:
            self.shape = f(0).shape
        except ValueError:
            raise ValueError("The shapes of the parameters 'height', 'decay_const' and 't0' don't seem to match.")

        # Truncating after memory_time should not discard more than a fraction
        # config.truncation_ratio of the total area under the kernel.
        # (Divide ∫_t^∞ by ∫_0^∞ to get this formula.)
        if memory_time is None:
            # We want a numerical value, so we use the test value associated to the variables
            decay_const_val = shim.get_test_value(decay_const)
            self.memory_time = -decay_const_val * np.log(config.truncation_ratio)
        else:
            self.memory_time = memory_time

        self.last_t = None     # Keep track of the last convolution time
        self.last_conv = None  # Keep track of the last convolution result
        self.last_hist = None  # Keep track of the history object used for the last convolution

    def convolve(self, hist, t):

        #TODO: allow t to be a slice
        #TODO: store multiple caches, one per history
        if (self.last_conv is None
            or hist is not self.last_hist
            or t < self.last_t):
            result = hist.convolve(self, t)
        else:
            Δt = t - self.last_t
            result = ( lib.exp(-Δt/self.decay_const) * self.last_conv
                       + hist.convolve(self, t, 0, Δt) )

        self.last_t = t
        self.last_conv = result
        self.last_hist = hist

        return result

# TODO? : Indicator kernel ? Optimizations possible ?
