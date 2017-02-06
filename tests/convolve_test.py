# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 2017

Author: Alexandre René
"""

import numpy as np
import scipy as sp
import time
import timeit

import sinn.config as config
import sinn.theano_shim as shim
import sinn.history as history
import sinn.kernel as kernel
import sinn.model.common as com

def convolve_test():

    # test convolution: ∫_a^b e^{-s/τ} * sin(t - s) ds

    τ = 0.1
    dt = 0.0001
    np.set_printoptions(precision=8)

    def data_fn(t):
        return np.sin(t)
    data = history.Series(0, 4, dt, shape=(1,), f=data_fn)

    params = kernel.ExpKernel.Parameters(
        height = 1,
        decay_const = τ
        )
    kern = kernel.ExpKernel('κ', (1,1), params=params, convolve_shape=(1,1))

    def true_conv(t, a=0, b=np.inf):
        """The analytical solution to the convolution"""
        if b == np.inf:
            return -τ / (τ**2 + 1) * np.exp(-a/τ)*(τ*np.cos(a-t) + np.sin(a-t))
        else:
            return ( τ / (τ**2 + 1) * (np.exp(-b/τ)*(τ*np.cos(b-t) + np.sin(b-t))
                                   - np.exp(-a/τ)*(τ*np.cos(a-t) + np.sin(a-t))) )


    dis_kern = history.Series(0, kern.memory_time,
                              dt, shape=(1,1), f=kern.eval)
    conv_len = len(dis_kern._tarr)  # The 'valid' convolution will be
                                    # (conv_len-1) bins shorter than data._tarr

    data.pad(kern.memory_time)

    t = 2.0
    tidx = data.get_t_idx(t)

    sinn_conv = kern.convolve(data, t)

    np_conv = np.convolve(data[:][:,0], dis_kern[:][:,0,0], 'valid')[tidx - conv_len] * dt

    print("single t, sinn (ExpKernel): ", sinn_conv)
    print("single t, manual numpy:            ", np_conv)
    print("single t, true:             ", true_conv(t))

    print("\nSlice test")
    tslice = slice(tidx, tidx + 5)
    tslice2 = slice(tidx+1, tidx +  5)

#    print("time slice:                        \n", data.convolve(kern, tslice)[:,0,0])
#    print("same slice, uses history cache:    \n", data.convolve(kern, tslice)[:,0,0])
#    print("different slice, still uses cache: \n", data.convolve(kern, tslice2)[:,0,0])
    for txt, slc in zip(["time slice:                        \n",
                         "same slice, uses history cache:    \n",
                         "different slice, still reuses previous calc\n"],
                        [tslice, tslice, tslice2]):
        t1 = time.perf_counter()
        res = data.convolve(kern, slc)[:,0,0]
        t2 = time.perf_counter()
        print(txt, res, "\nCalculation took {}ms\n".format((t2-t1)*1000))

    t1 = time.perf_counter()
    res = np.array([data.convolve(kern, s) for s in data._tarr[tslice]])[:,0,0] # witout caching
    t2 = time.perf_counter()
    print("no slice (5 single t calls):       \n", res, "\nCalculation took {}ms\n".format((t2-t1)*1000))

    t1 = time.perf_counter()
    res = np.array([kern.convolve(data, s) for s in data._tarr[tslice]])[:,0,0] # with caching
    t2 = time.perf_counter()
    print("no slice, with special exponential optimization:  \n", res, "\nCalculation took {}ms\n".format((t2-t1)*1000))

    print("\nPartial kernel\n")
    print("library convolution: ", data.convolve(kern, t, slice(0.1, 0.3)))
    print("true value:          ", true_conv(t, 0.1, 0.3))

    print("\n==========================")
    print("\n2 populations (2x2 kernel)\n")
    def data_fn2(t):
        return np.array((np.sin(t), 2*np.sin(t)))
    data2 = history.Series(0, 4, dt, shape=(2,), f=data_fn2)
    params2 = kernel.ExpKernel.Parameters(
        height = ((1,   0.3),
                  (0.7, 1.3)),
        decay_const = ((τ,   3*τ),
                       (2*τ, 0.3*τ))
        )
    kern2 = kernel.ExpKernel('κ', (2,2), params=params2, convolve_shape=(2,2))


    def true_conv2(t, a=0, b=np.inf):
        """The analytical solution to the convolution"""
        τ2 = kern2.params.decay_const
        h2 = kern2.params.height
        datamp = np.array(((1, 2)))
        if b == np.inf:
            return datamp * h2 * -τ2 / (τ2**2 + 1) * np.exp(-a/τ2)*(τ2*np.cos(a-t) + np.sin(a-t))
        else:
            return datamp * h2 * ( τ2 / (τ2**2 + 1) * (np.exp(-b/τ2)*(τ2*np.cos(b-t) + np.sin(b-t))
                                   - np.exp(-a/τ2)*(τ2*np.cos(a-t) + np.sin(a-t))) )

    config.integration_precision = 1
    dis_kern2 = data2.discretize_kernel(kern2)
    dis_kern2_test = history.Series(0, kern2.memory_time,
                                    dt, shape=(2,2), f=kern2.eval)

    print("\nMaximum difference between discretization method and manual discretization")
    memlen = min(len(dis_kern2), len(dis_kern2_test))
    print(np.max(abs(dis_kern2[:memlen] - dis_kern2_test[:memlen])))


    conv_len2 = len(dis_kern2._tarr)  # The 'valid' convolution will be
                                     # (conv_len-1) bins shorter than data._tarr

    data2.pad(kern2.memory_time)

    t = 2.0
    tidx = data2.get_t_idx(t)

    sinn_conv2 = kern2.convolve(data2, t)
    np_conv2 = np.array(
        [ [ np.convolve(data2[:][:,from_idx],
                        dis_kern2[:][:,to_idx,from_idx], 'valid')[tidx - conv_len2] * dt
            for from_idx in range(2) ]
          for to_idx in range(2) ] )

    print("single t, sinn (ExpKernel): \n", sinn_conv2)
    print("single t, manual numpy:            \n", np_conv2)
    print("single t, true:             \n", true_conv2(t))

    print("\nSlice test")
    print("Printed values show the (1,0) component.")

    for txt, slc in zip(["time slice:        \n",
                         "same slice, reuses previous calc:    \n",
                         "different slice, still reuses previous calc: \n"],
                        [tslice, tslice, tslice2]):
        t1 = time.perf_counter()
        res = data2.convolve(kern2, slc)[:,1,0]
        t2 = time.perf_counter()
        print(txt, res, "\nCalculation took {}ms\n".format((t2-t1)*1000))

    t1 = time.perf_counter()
    res = np.array([data2.convolve(kern2, s) for s in data._tarr[tslice]])[:,1,0] # witout caching
    t2 = time.perf_counter()
    print("no slice (5 single t calls):       \n", res, "\nCalculation took {}ms\n".format((t2-t1)*1000))

    t1 = time.perf_counter()
    res = np.array([kern2.convolve(data2, s) for s in data._tarr[tslice]])[:,1,0] # with caching
    t2 = time.perf_counter()
    print("no slice, with special exponential optimization:  \n", res, "\nCalculation took {}ms\n".format((t2-t1)*1000))

    print("\nPartial kernel\n")
    print("library convolution: \n", data2.convolve(kern2, t, slice(0.1, 0.3)))
    print("true value:          \n", true_conv2(t, 0.1, 0.3))

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
