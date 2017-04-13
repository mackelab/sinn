# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 2017

Author: Alexandre René
"""

import numpy as np
import scipy as sp
import time
import timeit

import theano_shim as shim
import sinn.config as config
import sinn.histories as histories
import sinn.kernels as kernels
import sinn.models.common as com

import sinn.diskcache

sinn.diskcache.set_file("test_cache.db")

def series():

    # test convolution: ∫_a^b e^{-s/τ} * sin(t - s) ds

    τ = 0.1
    dt = 0.001

    def data_fn(t):
        return np.sin(t)
    data = histories.Series(t0=0, tn=4, dt=dt, shape=(1,), f=data_fn)

    params = kernels.ExpKernel.Parameters(
        height = 1,
        decay_const = τ,
        t_offset = 0
        )
    kernel = kernels.ExpKernel('κ', params=params, shape=(1,1))

    def true_conv(t, a=0, b=np.inf):
        """The analytical solution to the convolution"""
        if b == np.inf:
            return -τ / (τ**2 + 1) * np.exp(-a/τ)*(τ*np.cos(a-t) + np.sin(a-t))
        else:
            return ( τ / (τ**2 + 1) * (np.exp(-b/τ)*(τ*np.cos(b-t) + np.sin(b-t))
                                   - np.exp(-a/τ)*(τ*np.cos(a-t) + np.sin(a-t))) )


    convolve_test(data, data_fn, kernel, true_conv)

    print("\n==========================")
    print("\nNon-iterative series (same data, but using scipy.signal.convolve)\n")
    data._iterative = True
    convolve_test(data, data_fn, kernel, true_conv)

    print("\n==========================")
    print("\n2 populations (2x2 kernel)\n")
    def data_fn2(t):
        axis = 0 if shim.isscalar(t) else 1
        return np.stack((np.sin(t), 2*np.sin(t)), axis=axis)
    data2 = histories.Series(t0=0, tn=4, dt=dt, shape=(2,), f=data_fn2)
    params2 = kernels.ExpKernel.Parameters(
        height = ((1,   0.3),
                  (0.7, 1.3)),
        decay_const = ((τ,   3*τ),
                       (2*τ, 0.3*τ)),
        t_offset = ((0, 0),
                    (0, 0))
        )
    kernel2 = kernels.ExpKernel('κ', params=params2, shape=(2,2))


    def true_conv2(t, a=0, b=np.inf):
        """The analytical solution to the convolution"""
        τ2 = kernel2.params.decay_const
        h2 = kernel2.params.height
        datamp = np.array(((1, 2)))
        if b == np.inf:
            return datamp * h2 * -τ2 / (τ2**2 + 1) * np.exp(-a/τ2)*(τ2*np.cos(a-t) + np.sin(a-t))
        else:
            return datamp * h2 * ( τ2 / (τ2**2 + 1) * (np.exp(-b/τ2)*(τ2*np.cos(b-t) + np.sin(b-t))
                                   - np.exp(-a/τ2)*(τ2*np.cos(a-t) + np.sin(a-t))) )

    convolve_test(data2, data_fn2, kernel2, true_conv2)

def spiketimes():

    # test convolution: ∫_a^b e^{-s/τ} * spiketimes ds,
    # where spiketimes is a uniformly sampled set of 50 spiketimes between 0 and 4

    dt = 0.001 # We need dt <= τ/1000 to have at least single digit
                # precision on binned spiketrains, but of course that slows
                # down computations substantially.
    τ = 0.1
    np.set_printoptions(precision=8)

    np.random.seed(314)
    spiketime_list = [np.concatenate(([np.array(-np.inf)],
                                     np.sort(np.unique(np.random.uniform(0, 4, (50,))))))]

    def data_fn(t):
        return np.array([ True if t - (spike_list[np.searchsorted(spike_list, t)-1]) < dt else False
              for spike_list in spiketime_list ])

    data = histories.Spiketimes(t0=0, tn=4, dt=dt, pop_sizes=(1,))
    data.set(spiketime_list)

    params = kernels.ExpKernel.Parameters(
        height = 1,
        decay_const = τ,
        t_offset = 0
        )
    kernel = kernels.ExpKernel('κ', params=params, shape=(1,1))

    def true_conv(t, a=0, b=np.inf):
        """The analytical solution to the convolution"""
        return np.array(
            [ np.sum(kernel.eval(t-s) for s in spike_list if a <= t-s < b)
              for spike_list in spiketime_list ] )

    convolve_test(data, data_fn,
                  kernel, true_conv)

def spiketrain():

    # test convolution: ∫_a^b e^{-s/τ} * spiketimes ds,
    # where spiketimes is a uniformly sampled set of 50 spiketimes between 0 and 4

    dt = 0.001 # We need dt <= τ/1000 to have at least single digit
                # precision on binned spiketrains, but of course that slows
                # down computations substantially.
    τ = 0.1
    np.set_printoptions(precision=8)

    np.random.seed(314)
    data = histories.Spiketrain(t0=0, tn=4, dt=dt, pop_sizes=(1,))
    spiketrain_data = np.random.binomial(1, 25*dt, (len(data),1))
    data.set(spiketrain_data)

    #def data_fn(t):
    #    return np.array([ True if t - (spike_list[np.searchsorted(spike_list, t)-1]) < dt else False
    #                      for spike_list in spiketime_list ])


    params = kernels.ExpKernel.Parameters(
        height = 1,
        decay_const = τ,
        t_offset = 0
        )
    kernel = kernels.ExpKernel('κ', params=params, shape=(1,1))

    def true_conv(t, a=0, b=np.inf):
        """The analytical solution to the convolution"""
        return np.array(
            [ np.sum(kernel.eval(t-s) for i, s in enumerate(data._tarr[data.t0idx:]) if spiketrain_data[i]) ] )

    convolve_test(data, None,#data_fn,
                  kernel, true_conv)


def convolve_test(data, data_fn, kernel, true_conv):

    np.set_printoptions(precision=8)

    config.integration_precision = 1
    dis_kernel = data.discretize_kernel(kernel)
    dis_kernel_test = histories.Series(t0=0, tn=kernel.memory_time,
                                     dt=data.dt, shape=kernel.shape, f=kernel.eval)

    print("\nMaximum difference between discretization method and manual discretization")
    memlen = min(len(dis_kernel), len(dis_kernel_test))
    print(np.max(abs(dis_kernel[:memlen] - dis_kernel_test[:memlen])))

    conv_len = len(dis_kernel._tarr)  # The 'valid' convolution will be
                                      # (conv_len-1) bins shorter than data._tarr

    data.pad(kernel.memory_time)

    t = 2.0
    tidx = data.get_t_idx(t)
    dis_tidx = data.get_t_idx(t)

    sinn_conv = kernel.convolve(data, t)

    #dt = None
    def get_comp(histdata, from_idx):
        #nonlocal dt
        if isinstance(histdata, histories.Spiketimes):
            #dt = data.dt / 10 # We need the higher resolution to get
            #                  # a reasonable estimate
            tidcs = np.asarray( np.fromiter(histdata[:][from_idx], dtype='float') // dis_kernel.dt, dtype='int')
            retval = np.zeros((int((data.tn - data.t0)//dis_kernel.dt),),
                              dtype=int)
            retval[tidcs] = 1/dis_kernel.dt # Dirac deltas
            return retval
        if isinstance(histdata, histories.Spiketrain):
            return histdata[:][:,from_idx] / dis_kernel.dt # Dirac deltas
        else:
            #dt = data.dt
            return histdata[:][:,from_idx]
    np_conv = np.array(
        [ [ np.convolve(get_comp(data, from_idx)[dis_tidx - conv_len:dis_tidx],
                        dis_kernel[:][:,to_idx,from_idx], 'valid')[0] * dis_kernel.dt
            for from_idx in range(kernel.shape[1]) ]
          for to_idx in range(kernel.shape[0]) ] )

    print("single t, sinn (ExpKernel):                \n", sinn_conv)
    print("single t, manual numpy over discretized κ: \n", np_conv)
    print("single t, true:                            \n", true_conv(t))

    print("\nSlice test")
    tslice = slice(tidx, tidx + 5)
    tslice2 = slice(tidx+1, tidx +  5)

#    print("time slice:                        \n", data.convolve(kernel, tslice)[:,0,0])
#    print("same slice, uses history cache:    \n", data.convolve(kernel, tslice)[:,0,0])
#    print("different slice, still uses cache: \n", data.convolve(kernel, tslice2)[:,0,0])
    for txt, slc in zip(["time slice:                        \n",
                         "same slice, uses history cache:    \n",
                         "different slice, still reuses previous calc\n"],
                        [tslice, tslice, tslice2]):
        t1 = time.perf_counter()
        res = data.convolve(kernel, slc)[:,0,0]
        t2 = time.perf_counter()
        print(txt, res, "\nCalculation took {}ms\n".format((t2-t1)*1000))

    t1 = time.perf_counter()
    res = np.array([data.convolve(kernel, s) for s in data._tarr[tslice]])[:,0,0] # witout caching
    t2 = time.perf_counter()
    print("no slice (5 single t calls):       \n", res, "\nCalculation took {}ms\n".format((t2-t1)*1000))

    t1 = time.perf_counter()
    res = np.array([kernel.convolve(data, s) for s in data._tarr[tslice]])[:,0,0] # with caching
    t2 = time.perf_counter()
    print("no slice, with special exponential optimization:  \n", res, "\nCalculation took {}ms\n".format((t2-t1)*1000))

    print("\nPartial kernel\n")
    print("library convolution (history): \n", data.convolve(kernel, t, slice(0.1, 0.3)))
    print("library convolution (kernel): \n", kernel.convolve(data, t, slice(0.1, 0.3)))
    print("true value:          \n", true_conv(t, 0.1, 0.3))


    print("\nCached convolutions\n")
    kernel.convolve(data)
    print("(kernel): \n", kernel.convolve(data, t))
    print("true:     \n", true_conv(t))

    return


    # config.integration_precision = 1
    # dis_kernel2 = data2.discretize_kernel(kernel2)
    # dis_kernel2_test = history.Series(t0=0, tn=kernel2.memory_time,
    #                                   dt=dt, shape=(2,2), f=kernel2.eval)

    # print("\nMaximum difference between discretization method and manual discretization")
    # memlen = min(len(dis_kernel2), len(dis_kernel2_test))
    # print(np.max(abs(dis_kernel2[:memlen] - dis_kernel2_test[:memlen])))


    # conv_len2 = len(dis_kernel2._tarr)  # The 'valid' convolution will be
    #                                  # (conv_len-1) bins shorter than data._tarr

    # data2.pad(kernel2.memory_time)

    # t = 2.0
    # tidx = data2.get_t_idx(t)

    # sinn_conv2 = kernel2.convolve(data2, t)
    # np_conv2 = np.array(
    #     [ [ np.convolve(data2[:][:,from_idx],
    #                     dis_kernel2[:][:,to_idx,from_idx], 'valid')[tidx - conv_len2] * dt
    #         for from_idx in range(2) ]
    #       for to_idx in range(2) ] )

    # print("single t, sinn (ExpKernel): \n", sinn_conv2)
    # print("single t, manual numpy:            \n", np_conv2)
    # print("single t, true:             \n", true_conv2(t))

    # print("\nSlice test")
    # print("Printed values show the (1,0) component.")

    # for txt, slc in zip(["time slice:        \n",
    #                      "same slice, reuses previous calc:    \n",
    #                      "different slice, still reuses previous calc: \n"],
    #                     [tslice, tslice, tslice2]):
    #     t1 = time.perf_counter()
    #     res = data2.convolve(kernel2, slc)[:,1,0]
    #     t2 = time.perf_counter()
    #     print(txt, res, "\nCalculation took {}ms\n".format((t2-t1)*1000))

    # t1 = time.perf_counter()
    # res = np.array([data2.convolve(kernel2, s) for s in data._tarr[tslice]])[:,1,0] # witout caching
    # t2 = time.perf_counter()
    # print("no slice (5 single t calls):       \n", res, "\nCalculation took {}ms\n".format((t2-t1)*1000))

    # t1 = time.perf_counter()
    # res = np.array([kernel2.convolve(data2, s) for s in data._tarr[tslice]])[:,1,0] # with caching
    # t2 = time.perf_counter()
    # print("no slice, with special exponential optimization:  \n", res, "\nCalculation took {}ms\n".format((t2-t1)*1000))

    # print("\nPartial kernel\n")
    # print("library convolution: \n", data2.convolve(kernel2, t, slice(0.1, 0.3)))
    # print("true value:          \n", true_conv2(t, 0.1, 0.3))


    # print("\nCached convolutions\n")
    # kernel2.convolve(data2)
    # print("(kernel): ", kernel2.convolve(data2, t))
    # print("true:     ", true_conv2(t))


def get_timers():

    setup = """
import numpy as np
import sinn.histories as histories
import sinn.models.common as com
τ = 0.1
dt = 0.0001
def data_fn(t):
    return np.sin(t)
data = histories.Series(t0=0, tn=4, dt=dt, shape=(), f=data_fn)
kernel = kernels.ExpKernel('κ', 1, τ, 0)
"""

    τ = 0.1
    dt = 0.0001

    def data_fn(t):
        return np.sin(t)
    data = histories.Series(t0=0, tn=4, dt=dt, shape=(), f=data_fn)


    def convolve_string(t):
        return "data.convolve(kernel, {})".format(t)
    def convolve_loop_string(t):
        return ("data._cur_tidx = {}\n".format(t) +
                "for i in range({}+1, {}+6):\n    kernel.convolve(data, i)".format(t,t))

    tidx = data.get_t_idx(3.0)
    uncached_timer = timeit.Timer(convolve_string(tidx), setup)

    cached_timer = timeit.Timer(convolve_loop_string(tidx), setup + "\n" +
    convolve_string(tidx))

    return uncached_timer, cached_timer

if __name__ == '__main__':
    spiketrain()

    # n=50
    # uncached_timer, cached_timer = get_timers()
    # uncached_res = uncached_timer.timeit(50)
    # cached_res = cached_timer.timeit(50) / 5

    # print("\nExecution time, {} uncached convolutions:".format(n))
    # print(uncached_res)
    # print("\nExecution time, {} cached convolutions:".format(n))
    # print(cached_res)
    # print("\nSpeedup")
    # print(uncached_res / cached_res)
