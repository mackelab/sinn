import numpy as np
import theano_shim as shim
from parameters import ParameterSet

from sinn.histories import Series, PopulationSeries
from sinn.popterm import PopTermMeso

#from fsGIF.main import init_spiking_model
from fsGIF.core import get_model_params
from mesogif_model_series import GIF

# Use load_theano=False to make debugging easier
shim.load(load_theano=True)

shim.config.compute_test_value = 'warn'


pop_sizes=(9,8)
# n_bins x n, i.e. one time bin per row, one col. per node
spike_trains = np.random.randint(30, size=(1000,sum(pop_sizes))) # 100 time bins, 17 nodes, up to 30 spikes per bin
state_labels_1d = np.random.randint(3, size=(1000,1))  # state labels as state 0, 1, or 2
# states converted to a binary n_bins x n_states matrix
states_binary_2d = np.hstack((state_labels_1d==0, state_labels_1d==1, state_labels_1d==2)) + \
                   np.zeros((state_labels_1d.shape[0], 1))
# states_17d = state_labels_1d * np.ones((1, 17))

print('spike_trains.shape:', spike_trains.shape)
print('states_binary_2d.shape:', states_binary_2d.shape)
# print('states_17d.shape:', states_17d.shape)


param_dt = 4.
tarr = np.arange(1000)*param_dt    # 100 bins, each lasting 4 seconds
spiketrain = PopulationSeries(name='s', time_array=tarr, pop_sizes=pop_sizes)
# spiketrain = Series(name='s', time_array=tarr, shape=(sum(pop_sizes),))
spiketrain.set(source=spike_trains)
# spiketrain.set(source=np.hstack((states_binary_2d, spike_trains)))

# state_hist = Series(t0=param_t0, tn=param_tn, dt=param_dt, shape=(3,))
# state_hist.set(source=states_binary_2d)
state_hist = Series(name='z', time_array=tarr, dt=param_dt, shape=(1,))
state_hist.set(source=state_labels_1d)
# state_hist = Series(t0=param_t0, tn=param_tn, dt=param_dt, shape=(17,))
# state_hist.set(source=states_17d)

# Locking histories identifies them as data
# The model will not modify them, and treats them as known inputs when
# constructing computation graphs.
spiketrain.lock()
state_hist.lock()

model_params = get_model_params(ParameterSet("spike_model_params.ntparameterset"), "GIF_spiking")
# HACK: Casting to PopTerm should be automatic
# model_params = model_params._replace(
#         τ_θ=PopTermMeso(pop_sizes, model_params.τ_θ, ('Meso',)),
#         τ_m=PopTermMeso(pop_sizes, model_params.τ_m, ('Meso',)))

spiking_model = GIF(model_params,
                    spiketrain, state_hist,
                    initializer='silent',
                    set_weights=True)


print("loglikelihood")
# Integrate the model forward to the time point with index 40
spiking_model.advance(40)
print(spiking_model.logp(40, 100))     # Int argument => Interpreted as time index
print(spiking_model.logp(160., 400.))  # Float argument => Interpreted as time in seconds
#gradient_descent(input_filename=None, output_filename="test_output.test",
                 #batch_size=100,
                 #model=spiking_model)


########
## Gradient descent

import sinn.optimize.gradient_descent as gd
gd.raise_on_exception = True
    # Better for testing. For batch jobs best to leave this False

# First create symbolic variables which are used as inputs in GD function
# t = shim.symbolic.scalar('tidx', dtype=spiketrain.tidx_dtype)
# batch_size_var = shim.symbolic.scalar('batch_size', dtype=spiketrain.tidx_dtype)
t = shim.shared(shim.cast(1, spiking_model.tidx_dtype), name='tidx')
batch_size_var = shim.shared(shim.cast(100, spiking_model.tidx_dtype),
                             name='batch_size')
batch_size     = 100
    # Note: instead of `spiketrain`, you can also use `spiking_model.s`
    # It's a bit dumb that we have to specify both a symbolic variable
    # and definite the value; eventually both will be replaced by a shared variable
# t.tag.test_value = 1
# batch_size_var.tag.test_value = 100
    # Always give assign a test_value to symbolic variables

# Next get the computational graph for the log likelihood
logL = spiking_model.logp(t, batch_size=batch_size_var)

# In theory you could compile this graph with `shim.compile([t, batch_size], logL)` and use
# the resulting function it as is.
# The optimizer included in *sinn* has some niceties wrt automatically selecting minibatches,
# so I explain it below.

###
# Instantiating the optimizer

# This requires setting bounds, e.g. in case some initial data segment should be discarded
# Batches will be selected from the interval [start:start+datalen], where burnin & datalen
# are specified in seconds, as specified if the docstring.
# There's lots more info in there, so make sure to type `gd.SeriesSGD?` in a Jupyter
# console or notebook !
# Also: I'm basically picking numbers at random here; you'll have to figure out what makes
# sense for your data.
start = spiking_model.get_tidx(12.)
datalen = len(tarr) - start
burnin = spiking_model.index_interval(4.)
# We also need to specify the burnin before each batch
# Values of start and burnin mostly depend on how quickly the model's internal state
# converges to something sensible.
# For the mesoscopic model this is quite long, but for the spiking model it should be short.

# Normalize logL
logL *= datalen / batch_size

# Select variables we want to optimize
# Others will be fixed to the value in the parameters file
# `original_params` is something I added to mesogif_model_series,
# because `params` is overwritten with expanded parameters
# w = spiking_model.original_params.w
    # To fit w: reduce bin size so that kernels aren't zero
c = spiking_model.original_params.c
R = spiking_model.original_params.R
vars_to_optimize = [c, R]

# Select which variables to track
# You can also track transformations; the one below is just
# a silly example, but this can be useful if you are actually fitting
# a transformation (such as log)
vars_to_track = {
    'c': c,
    'R': R,
    'log(cR)': shim.log(c*R)
}

sgd = gd.SeriesSGD(
    cost           = logL,
    start_var      = t,
    batch_size_var = batch_size_var,
    batch_size     = batch_size,
    cost_format    = 'logL',
    optimizer_kwargs = {'lr': 0.005},
    optimize_vars  = vars_to_optimize,
    track_vars     = vars_to_track,
    start          = start,
    datalen        = datalen,
    burnin         = burnin,
    advance        = spiking_model.advance_updates,
    mode           = 'sequential'  # One of 'sequential', 'random'
    )

sgd.fit(10)
sgd.trace
