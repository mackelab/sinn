# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 2017

Author: Alexandre René
"""

import numpy as np
import scipy as sp
import timeit

import sinn.history as history
import sinn.model.common as com

def convolve_test():

    # test convolution: ∫_a^b e^{-s/τ} * sin(t - s) ds
    τ = 0.1
    dt = 0.0001

    def data_fn(t):
        return np.sin(t)
    data = history.Series(0, 4, dt, (), data_fn)

    kern = com.ExpKernel('κ', 1, τ)

    def true_conv(t, a=0, b=np.inf):
        """The analytical solution to the convolution"""
        if b == np.inf:
            return -τ / (τ**2 + 1) * np.exp(-a/τ)*(τ*np.cos(a-t) + np.sin(a-t))
        else:
            return ( τ / (τ**2 + 1) * (np.exp(-b/τ)*(τ*np.cos(b-t) + np.sin(b-t))
                                   - np.exp(-a/τ)*(τ*np.cos(a-t) + np.sin(a-t))) )


    dis_kern = history.Series(0, kern.memory_time,
                              dt, (), kern.eval)
    conv_len = len(dis_kern._tarr)  # The 'valid' convolution will be
                                    # (conv_len-1) bins shorter than data._tarr

    data.pad(kern.memory_time)

    t = 2.0
    tidx = data.get_t_idx(t)

    sinn_conv = kern.convolve(data, t)

    np_conv = np.convolve(data[:], dis_kern[:], 'valid')[tidx - conv_len] * dt

    print(sinn_conv)
    print(np_conv)
    print(true_conv(t))

    print("\nSlice test (slice, without caching, with caching)")
    tslice = slice(tidx, tidx + 5)
    print(data.convolve(kern, tslice))
    print([round(data.convolve(kern, s),8) for s in data._tarr[tslice]]) # witout caching
    print([round(kern.convolve(data, s),8) for s in data._tarr[tslice]]) # with caching

    print("partial kernel")
    print(data.convolve(kern, t, 0.1, 0.3))
    print(true_conv(t, 0.1, 0.3))

    return

def get_timers():

    setup = """
import numpy as np
import sinn.history as history
import sinn.model.common as com
τ = 0.1
dt = 0.0001
def data_fn(t):
    return np.sin(t)
data = history.Series(0, 4, dt, (), data_fn)
kern = com.ExpKernel('κ', 1, τ, 0)
"""

    τ = 0.1
    dt = 0.0001

    def data_fn(t):
        return np.sin(t)
    data = history.Series(0, 4, dt, (), data_fn)


    def convolve_string(t):
        return "data.convolve(kern, {})".format(t)
    def convolve_loop_string(t):
        return ("data._cur_tidx = {}\n".format(t) +
                "for i in range({}+1, {}+6):\n    kern.convolve(data, i)".format(t,t))

    tidx = data.get_t_idx(3.0)
    uncached_timer = timeit.Timer(convolve_string(tidx), setup)

    cached_timer = timeit.Timer(convolve_loop_string(tidx), setup + "\n" +
    convolve_string(tidx))

    return uncached_timer, cached_timer

if __name__ == '__main__':
    convolve_test()

    n=50
    uncached_timer, cached_timer = get_timers()
    uncached_res = uncached_timer.timeit(50)
    cached_res = cached_timer.timeit(50) / 5

    print("\nExecution time, {} uncached convolutions:".format(n))
    print(uncached_res)
    print("\nExecution time, {} cached convolutions:".format(n))
    print(cached_res)
    print("\nSpeedup")
    print(uncached_res / cached_res)
