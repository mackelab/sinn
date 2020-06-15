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

import sinn.diskcache

def test_series_convolution():
    # series(τ=.1, dt=.2, L=20, test_time_point=.2, headless=True)
    series(τ=0.1, dt=0.001, L=4000,
           test_time_point=2.0, headless=True)

def test_spiketrain_convolution():
    spiketrain(τ=0.1, dt=0.001, L=4000,
               test_time_point=2.0, headless=True)

def isclose(a, b, **kwargs):
    return np.all(np.isclose(a, b, **kwargs))

def collapse_conv_result(result, history, kernel):
    if kernel.shape != history.shape:
        # At the moment, the only kernel kernel shape beyond element-wise
        # multiplication is a 2D kernel with from and to axes.
        assert kernel.ndim == 2 and history.ndim == 1
        # TODO: Make these tests work with PopulationHistory
        # assert kernel.shape[0] in (1, history.shape[0])
        # assert kernel.shape[1] in (1, history.shape[0])
        return result.sum(axis=-1)

def series(τ=0.1, dt=0.001, L=4000, test_time_point=2., headless=False):

    # test convolution: ∫_a^b e^{-s/τ} * sin(t - s) ds

    assert 0. < test_time_point < L*dt

    data = histories.Series(t0=0, tn=L*dt, dt=dt, shape=(1,))
    def data_fn(t):
        t = data.get_time(t)
        return np.sin(t).reshape((-1,1))
    data.set_update_function(data_fn, inputs=[])

    params = dict(
        height = 1,
        decay_const = τ,
        t_offset = 0
        )
    kernel = kernels.ExpKernel(name='κ', **params, shape=(1,1))

    def true_conv(t, a=0, b=np.inf):
        """The analytical solution to the convolution
        a, b : Low, high bounds of kernel. Defaults to [0, +inf]

        To recover equation:
        a) Pen and paper
        b)
        >>> import sympy
        >>> sympy.init_session()
        >>> h, τ, A, s, t, a, b = symbols('h τ A s t a b', real=True)
        >>> def κ(t):
                return h * exp(-t/τ)
        >>> def f(t):
                return A*sin(t)
        >>> I = integrate( f(t-s) * κ(s), (s, a, b))
        >>> simplify(I)
        """
        τ = kernel.decay_const
        h = kernel.height
        A = 1.
        if b == np.inf:
            result = A * h * τ / (τ**2 + 1) * -np.exp(-a/τ)*(τ*np.cos(a-t) + np.sin(a-t))
        else:
            result = A * h * τ / (τ**2 + 1) * ( np.exp(-b/τ)*(τ*np.cos(b-t) + np.sin(b-t))
                                          - np.exp(-a/τ)*(τ*np.cos(a-t) + np.sin(a-t)) )
        return collapse_conv_result(result, data, kernel)

    convolve_test(data, data_fn, kernel, true_conv, test_time_point,
                  headless=headless,
                  cache_acceleration_factor = 3.,
                  five_single_t_factor      = 1.,
                  exp_acceleration_factor   = None,
                  exp_rtol                  = 1e-03
                 )

    # # I don't think this tests anything anymore, because sinn can now still
    # # recognize `data` as batch-computable
    # if not headless:
    #     print("\n==========================")
    #     print("\nNon-iterative series (same data, but prevent batch operations)\n")
    # data.unlock()
    # data.clear()
    # data._conv_cache.clear()
    # kernel._conv_cache.clear()
    # data._iterative = True
    # convolve_test(data, data_fn, kernel, true_conv, test_time_point,
    #               headless=headless)

    if not headless:
        print("\n==========================")
        print("\n2 populations (2x2 kernel)\n")
    def data_fn2(t):
        t = data2.get_time(t)
        axis = 0 if shim.isscalar(t) else 1
        return np.stack((np.sin(t), 2*np.sin(t)), axis=axis)
    data2 = histories.Series(t0=0, tn=L*dt, dt=dt, shape=(2,))
    data2.set_update_function(data_fn2, inputs=[])
    params2 = dict(
        height = ((1,   0.3),
                  (0.7, 1.3)),
        decay_const = ((τ,   3*τ),
                       (2*τ, 0.3*τ)),
        t_offset = 0
        )
    kernel2 = kernels.ExpKernel(name='κ', **params2, shape=(2,2))


    def true_conv2(t, a=0, b=np.inf):
        """The analytical solution to the convolution"""
        τ = kernel2.decay_const
        h = kernel2.height
        A = np.array(((1, 2)))
        if b == np.inf:
            result = A * h * τ / (τ**2 + 1) * -np.exp(-a/τ)*(τ*np.cos(a-t) + np.sin(a-t))
        else:
            result = A * h * τ / (τ**2 + 1) * ( np.exp(-b/τ)*(τ*np.cos(b-t) + np.sin(b-t))
                                          - np.exp(-a/τ)*(τ*np.cos(a-t) + np.sin(a-t)) )
        return collapse_conv_result(result, data2, kernel2)

    convolve_test(data2, data_fn2, kernel2, true_conv2, test_time_point,
                  headless=headless,
                  cache_acceleration_factor = None,
                  five_single_t_factor      = None,
                  exp_acceleration_factor   = None,
                  exp_rtol                  = 1e-03
                  )

def spiketimes(τ=0.1, dt=0.001, headless=False):
    """
    test convolution: ∫_a^b e^{-s/τ} * spiketimes ds,
    where spiketimes is a uniformly sampled set of 50 spiketimes
    between 0 and 4

    Remark
    ------
    We need dt <= τ/1000 to have at least single digit
    precision on binned spiketrains, but of course that slows
    down computations substantially.
    """
    """
    test convolution: ∫_a^b e^{-s/τ} * spiketimes ds,
    where spiketimes is a uniformly sampled set of 50 spiketimes
    between 0 and 4
    """
    np.set_printoptions(precision=8)

    np.random.seed(314)
    spiketime_list = [np.concatenate(([np.array(-np.inf)],
                                     np.sort(np.unique(np.random.uniform(0, 4, (50,))))))]

    def data_fn(t):
        return np.array([ True if t - (spike_list[np.searchsorted(spike_list, t)-1]) < dt else False
              for spike_list in spiketime_list ])

    data = histories.Spiketimes(t0=0, tn=4, dt=dt, pop_sizes=(1,))
    data.set(spiketime_list)

    params = dict(
        height = 1,
        decay_const = τ,
        t_offset = 0
        )
    kernel = kernels.ExpKernel(name='κ', **params, shape=(1,1))

    def true_conv(t, a=0, b=np.inf):
        """The analytical solution to the convolution"""
        result = np.array(
            [ np.sum(kernel.eval(t-s) for s in spike_list if a <= t-s < b)
              for spike_list in spiketime_list ] )
        return collapse_conv_result(result, data, kernel)

    convolve_test(data, data_fn,
                  kernel, true_conv, headless=headless,
                  cache_acceleration_factor = None,
                  five_single_t_factor      = None,
                  exp_acceleration_factor   = None)

def spiketrain(τ=0.1, dt=0.001, L=4000,
               test_time_point=2.0, headless=False):
    """
    test convolution: ∫_a^b e^{-s/τ} * spiketimes ds,
    where spiketimes is a uniformly sampled set of 50 spiketimes
    between 0 and 4

    Remark
    ------
    We need dt <= τ/1000 to have at least single digit
    precision on binned spiketrains, but of course that slows
    down computations substantially.
    """

    np.set_printoptions(precision=8)

    np.random.seed(314)
    data = histories.Spiketrain(t0=0, tn=L*dt, dt=dt, pop_sizes=(1,))
    spiketrain_data = np.random.binomial(1, np.clip(25*dt, 0., 1.), (len(data),1))
    data.set(spiketrain_data)
    data.set_connectivity(np.ones((data.npops,)+data.shape))

    #def data_fn(t):
    #    return np.array([ True if t - (spike_list[np.searchsorted(spike_list, t)-1]) < dt else False
    #                      for spike_list in spiketime_list ])


    params = dict(
        height = 1,
        decay_const = τ,
        t_offset = 0
        )
    kernel = kernels.ExpKernel(name='κ', **params, shape=(1,1))

    def true_conv(t, a=0, b=np.inf):
        """The analytical solution to the convolution"""
        result = np.array(
            [ np.sum(kernel.eval(t-s) for i, s in enumerate(data._tarr[data.t0idx:]) if spiketrain_data[i]) ] )
        return collapse_conv_result(result, data, kernel)

    convolve_test(data, None,#data_fn,
                  kernel, true_conv, test_time_point, headless=headless,
                  cache_acceleration_factor = None,
                  five_single_t_factor      = None,
                  exp_acceleration_factor   = None,
                  exp_rtol                  = 1e-02)


def convolve_test(data, data_fn, kernel, true_conv,
                  test_time_point=2.0, headless=False,
                  cache_acceleration_factor = None,
                  five_single_t_factor      = None,
                  exp_acceleration_factor   = None,
                  exp_steps                 = 100,
                  exp_rtol                  = 1e-03,
                  ):
    """Setting factors to `None` disables corresponding tests."""

    sinn.diskcache.set_file("test_cache.db")

    np.set_printoptions(precision=8)

    config.integration_precision = 1
    dis_kernel = kernel.discretize(refhist=data)
    dis_kernel_test = histories.Series(t0=0, tn=kernel.memory_time,
                                       dt=data.dt64, shape=kernel.shape)
    ## Kernel functions can only be defined to take times, so we wrap
    ## the function
    def kernel_func(t):
        return kernel.eval(dis_kernel_test.get_time(t))
    dis_kernel_test.set_update_function(kernel_func, inputs=[])

    memlen = min(len(dis_kernel), len(dis_kernel_test))
    if headless:
        assert memlen > 0
        assert dis_kernel.shape == dis_kernel_test.shape
        assert isclose(dis_kernel[:memlen], dis_kernel_test[:memlen])
    else:
        if memlen > 0:
            print("\nMaximum difference between discretization method and manual discretization")
            print(np.max(abs(dis_kernel[:memlen] - dis_kernel_test[:memlen])))
            print("")
        else:
            print("\nAt least one kernel has zero length.")
            print("len(discretized_kernel):          {}".format(len(dis_kernel)))
            print("len(manually discretized kernel): {}\n".format(len(dis_kernel_test)))

    conv_len = len(dis_kernel._tarr)  # The 'valid' convolution will be
                                      # (conv_len-1) bins shorter than data._tarr

    data.pad(kernel.memory_time)

    t = test_time_point
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
    if conv_len > 0:
        np_conv = np.array(
            [ [ np.convolve(
                    get_comp(data, from_idx)[dis_tidx - conv_len:dis_tidx],
                    dis_kernel[:][:,to_idx,from_idx], 'valid'
                )[0] * dis_kernel.dt
                for from_idx in range(kernel.shape[1]) ]
              for to_idx in range(kernel.shape[0]) ] )
    else:
        np_conv = np.zeros(kernel.shape)

    np_conv = collapse_conv_result(np_conv, data, kernel)

    if headless:
        sinn_conv.shape == np_conv.shape
        assert isclose(sinn_conv, np_conv)
        assert isclose(sinn_conv, true_conv(t), rtol=2e-02)
    else:
        print("single t, sinn (ExpKernel):                \n", sinn_conv)
        print("single t, manual numpy over discretized κ: \n", np_conv)
        print("single t, true:                            \n", true_conv(t))

    #######
    # Caching test
    shapestr = 'x'.join(str(l) for l in kernel.shape)
    print(f"\n\n{shapestr} kernel\n------------")

    # Caching only happens for locked histories
    data._compute_up_to('end')
    data.lock()
    kernel.compute_discretized_kernels()

    # Minimum expected acceleration factor due to caching
    α  = cache_acceleration_factor
    # Expected ratio of the time required (compared to full convolution) to
    # compute five individual convolutions. If the five convolutions are faster,
    # this value should be less than 1.
    # Tested as T(5 convs) < γ*T(full conv)
    γ  = five_single_t_factor

    if not headless:
        print("\nSlice test")
    tslice = slice(tidx, tidx + 5)
    tslice2 = slice(tidx+1, tidx +  5)

#    print("time slice:                        \n", data.convolve(kernel, tslice)[:,0,0])
#    print("same slice, uses history cache:    \n", data.convolve(kernel, tslice)[:,0,0])
#    print("different slice, still uses cache: \n", data.convolve(kernel, tslice2)[:,0,0])
    Δs = []
    for txt, slc in zip(["time slice:                                ",
                         "same slice, uses history cache:            ",
                         "different slice, still reuses previous calc"],
                        [tslice, tslice, tslice2]):
        t1 = time.perf_counter()
        res = data.convolve(kernel, slc)[:,0]
        t2 = time.perf_counter()
        if headless:
            Δs.append(t2-t1)
            print(f"{txt}: {(t2-t1)*1000:.4g}ms")
        else:
            print(f"{txt}\n{res}\nCalculation took {(t2-t1)*1000:.4g}ms\n")
    if headless and α:
        assert Δs[0] > α*Δs[1]
        assert Δs[0] > α*Δs[2]

    t1 = time.perf_counter()
    res = np.array([data.convolve(kernel, s) for s in data._tarr[tslice]])[:,0] # without caching
    t2 = time.perf_counter()
    if headless:
        Δs.append(t2-t1)
        if γ:
            assert Δs[-1] < γ*Δs[0]
        print(f"no slice (5 single t calls)                : {(t2-t1)*1000:.4g}ms")
    else:
        print("no slice (5 single t calls)                :       \n", res, "\nCalculation took {}ms\n".format((t2-t1)*1000))

    # if data.shape == (2,):
    #     for s in data._tarr[tslice]:
    #         import pdb; pdb.set_trace()
    #         print(kernel.convolve(data, s))
    # if data.shape == (2,):
    #     kernel.debug = True

    if isinstance(kernel, kernels.ExpKernel) and exp_steps is not None:
        α_exp = exp_acceleration_factor
            # Expected acceleration factor for 100 steps; still gets better with more steps
        n_steps = exp_steps  # Make enough steps for the optimization to pay off
        Δs_exp = []
        # [kernel.convolve(data, s) for s in data._tarr[tslice]]
        t1 = time.perf_counter()
        res1 = [data.convolve(kernel, s) for s in data._tarr[tidx:tidx+n_steps]]
        t2 = time.perf_counter()
        Δs_exp.append(t2-t1)
        t1 = time.perf_counter()
        res2 = [kernel.convolve(data, s) for s in data._tarr[tidx:tidx+n_steps]]
        t2 = time.perf_counter()
        Δs_exp.append(t2-t1)
        res1 = np.array(res1)[:,0]
        res2 = np.array(res2)[:,0]
        assert res1.shape == res2.shape
        assert isclose(res1, res2, rtol=exp_rtol)  # rtol has to be relaxed (increased) with greater n_steps
        if headless:
            if α_exp:
                assert Δs_exp[0] > α_exp * Δs_exp[1]
            print(f"{n_steps} single evals, no optimization          :  {Δs_exp[0]*1000:.4g}ms")
            print(f"{n_steps} single evals, exp optimization         :  {Δs_exp[1]*1000:.4g}ms")
        else:
            print("no slice, special exponential optimization :  \n", res2, "\nCalculation took {}ms\n".format((t2-t1)*1000))


    # Partial kernel
    dataconv   = data.convolve(kernel, t, slice(0.1, 0.3))
    kernelconv = kernel.convolve(data, t, slice(0.1, 0.3))
    trueconv   = true_conv(t, 0.1, 0.3)
    if headless:
        assert dataconv.shape == trueconv.shape
        assert kernelconv.shape ==  trueconv.shape
        assert isclose(dataconv,   trueconv, rtol=2e-02)
        assert isclose(kernelconv, trueconv, rtol=2e-02)
    else:
        print("\nPartial kernel\n")
        print("library convolution (history): \n", dataconv)
        print("library convolution (kernel): \n", kernelconv)
        print("true value:          \n", trueconv)


    # Cached convolutions
    kernel.convolve(data)
    kernelconv = kernel.convolve(data, t)
    trueconv   = true_conv(t)
    if headless:
        assert kernelconv.shape == trueconv.shape
        assert isclose(kernelconv, trueconv, rtol=2e-02)
    else:
        print("\nCached convolutions\n")
        print("(kernel): \n", kernelconv)
        print("true:     \n", trueconv)

    sinn.diskcache.unset_file()

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


# def get_timers():
#
#     setup = """
# import numpy as np
# import sinn.histories as histories
# import sinn.models.common as com
# τ = 0.1
# dt = 0.0001
# def data_fn(t):
#     return np.sin(t)
# data = histories.Series(t0=0, tn=4, dt=dt, shape=(), f=data_fn)
# kernel = kernels.ExpKernel('κ', 1, τ, 0)
# """
#
#     τ = 0.1
#     dt = 0.0001
#
#     def data_fn(t):
#         return np.sin(t)
#     data = histories.Series(t0=0, tn=4, dt=dt, shape=(), f=data_fn)
#
#
#     def convolve_string(t):
#         return "data.convolve(kernel, {})".format(t)
#     def convolve_loop_string(t):
#         return ("data._sym_tidx = {}\n".format(t) +
#                 "for i in range({}+1, {}+6):\n    kernel.convolve(data, i)".format(t,t))
#
#     tidx = data.get_t_idx(3.0)
#     uncached_timer = timeit.Timer(convolve_string(tidx), setup)
#
#     cached_timer = timeit.Timer(convolve_loop_string(tidx), setup + "\n" +
#     convolve_string(tidx))
#
#     return uncached_timer, cached_timer

if __name__ == '__main__':
    series(dt=4., test_time_point=8.)   # Test with a 0-length kernel
    #series()

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
