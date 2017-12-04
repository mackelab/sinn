import numpy as np
from sinn.popterm import PopTerm, PopTermMicro, PopTermMeso
from sinn.histories import Spiketrain, Series

def popterm_test():
    T = 30
    pop_sizes = (3, 10, 5)
    N = sum(pop_sizes)

    poptermmicro = PopTermMicro(pop_sizes, np.arange(N), ['Micro'])
    assert(np.all(poptermmicro.values == np.arange(N)))
    poptermmeso = PopTermMeso( pop_sizes,
                               [ poptermmicro.values[slc].mean()
                                 for slc in poptermmicro.pop_slices],
                               ['Meso'] )
    #assert(isinstance(poptermmeso.values, np.ndarray))
    means = []
    low = 0
    for s in pop_sizes:
        means.append(low + (s-1)/2)
        low += s
    assert( np.all(poptermmeso.values == np.array(means)))
    assert( np.all(poptermmeso.expand.values[pop_sizes[0]:pop_sizes[0]+pop_sizes[1]] == poptermmicro.values[poptermmicro.pop_slices[1]].mean()) )

    data = np.random.binomial(1, p=0.3, size=(T, N))
    spiketrain = Spiketrain(name='S', time_array=np.arange(T), pop_sizes=pop_sizes)
    spiketrain.set(data)

    target = poptermmeso.expand.values * poptermmicro.values
    assert( all( (( (poptermmeso * poptermmicro.values).values == target).all(),
                  ( (poptermmicro.values * poptermmeso).values == target).all(),
                  ( (poptermmeso * poptermmicro).values        == target).all()) ))
    assert( all( (( (poptermmeso.values * poptermmicro).values == target).all(),
                  ( (poptermmicro * poptermmeso.values).values == target).all(),
                  ( (poptermmeso * poptermmicro).values            == target).all()) ))
    target = poptermmeso.values * poptermmeso.values
    assert( all( (( (poptermmeso * poptermmeso.values).values == target).all(),
                  ( (poptermmeso.values * poptermmeso).values == target).all(),
                  ( (poptermmeso * poptermmeso).values            == target).all()) ))
    target = poptermmicro.values * poptermmicro.values
    assert( all( (( (poptermmicro * poptermmicro.values).values == target).all(),
                  ( (poptermmicro.values * poptermmicro).values == target).all(),
                  ( (poptermmicro * poptermmicro).values        == target).all()) ))

    # Following four fail because Spiketrain does not yet implement operations
    #assert( np.all(poptermmicro * spiketrain == spiketrain * poptermmicro) )
    #assert( np.all(poptermmeso * spiketrain == spiketrain * poptermmeso) )
    #assert( np.all((poptermmicro * spiketrain) == data * np.arange(N).reshape((1, -1))) )
    #assert( np.all((poptermmeso * spiketrain) == data * poptermmeso.expand) )


if __name__ == '__main__':
    popterm_test()
