import numpy as np
import theano_shim as shim
from sinn.popterm import PopTerm, PopTermMicro, PopTermMeso, \
    NumpyPopTerm, SymbolicPopTerm
from sinn.histories import Spiketrain, Series

class Target:
    """Packages dereferencing of shared vars to make tests cleaner."""
    def __init__(self, value):
        if shim.isshared(value):
            self.value = value.get_value()
        elif isinstance(value, np.ndarray):
            self.value = value
        elif shim.graph.is_computable([value]):
            self.value = value.eval()
        else:
            self.value = value
    def __eq__(self, other):
        if isinstance(other, np.ndarray):
            return np.all(self.value == other)
        elif shim.isshared(other):
            return np.all(self.value == other.get_value())
        elif shim.graph.is_computable([other]):
            return np.all(self.value == other.eval())
        else:
            return self.value == other

def check_broadcasting(pop_sizes, data, N, T, means=None):
    M = len(pop_sizes)

    def arr(popterm):
        """Return an array view to the popterm."""
        if isinstance(popterm, np.ndarray):
            return popterm.view(np.ndarray)
        else:
            return popterm.eval()

    ######################
    # Test `expand` method

    # Construct micro and meso terms
    # micro term: 1..18, split into 3 groups of size 3, 10 and 5
    # meso  term: size 3, mean value for each group
    poptermmicro = PopTermMicro(pop_sizes, data, ['Micro'])
    assert Target(data) == arr(poptermmicro)
    if means is None:
        poptermmeso = PopTermMeso( pop_sizes,
                                   [ arr(poptermmicro)[slc].mean()
                                     for slc in poptermmicro.pop_slices],
                                   ['Meso'] )
        assert type(arr(poptermmeso)) is np.ndarray
    else:
        poptermmeso = means

    means = []
    low = 0
    for s in pop_sizes:
        means.append(low + (s-1)/2)
        low += s
    # Validate that meso term was constructed correctly
    assert Target(np.array(means)) == arr(poptermmeso)
    # Test `expand` method
    assert np.all(arr(poptermmeso.expand)[
          pop_sizes[0]:pop_sizes[0]+pop_sizes[1]]
          == arr(poptermmicro)[poptermmicro.pop_slices[1]
        ].mean())

    ######################
    # Test operation broadcasting
    # We test with *; other broadcasted numpy ops  (+,-,/) should work the same

    # Meso-micro ops
    target = Target(arr(poptermmeso.expand) * arr(poptermmicro))
    assert target.value.shape == (N,)
    assert all((
        ( target == (poptermmeso       * poptermmicro     )),
    ))
    if (not isinstance(poptermmeso, shim.config.ShimmedAndGraphTypes)
        and not isinstance(poptermmicro, shim.config.ShimmedAndGraphTypes)):
        # Symbolic popterms don't support inferring the shape from a plain array
        assert all((
            target == (poptermmeso       * arr(poptermmicro)),
            target == (arr(poptermmeso)  * poptermmicro     ),
            target == (arr(poptermmicro) * poptermmeso      ),
            target == (poptermmicro      * arr(poptermmeso) ),
        ))

    # Meso-meso ops
    target = Target(arr(poptermmeso) * arr(poptermmeso))
    assert target.value.shape == (M,)
    assert all((
        ( target == (poptermmeso      * poptermmeso      )),
    ))
    if not isinstance(poptermmeso, shim.config.ShimmedAndGraphTypes):
        assert all((
            target == (poptermmeso      * arr(poptermmeso)),
            target == (arr(poptermmeso) * poptermmeso     ),
        ))

    # Micro-micro ops
    target = Target(arr(poptermmicro) * arr(poptermmicro))
    assert target.value.shape == (N,)
    assert all((
        ( target == (poptermmicro      * poptermmicro      )),
    ))
    if not isinstance(poptermmicro, shim.config.ShimmedAndGraphTypes):
        assert all((
            target == (poptermmicro      * arr(poptermmicro)),
            target == (arr(poptermmicro) * poptermmicro     ),
        ))

    ######################
    # Test operations on a Spiketrain object
    # These fail because Spiketrain does not yet implement operations

    # data = np.random.binomial(1, p=0.3, size=(T, N))
    # spiketrain = Spiketrain(name='S', time_array=np.arange(T), pop_sizes=pop_sizes)
    # spiketrain.set(data)
    #assert( np.all(poptermmicro * spiketrain == spiketrain * poptermmicro) )
    #assert( np.all(poptermmeso * spiketrain == spiketrain * poptermmeso) )
    #assert( np.all((poptermmicro * spiketrain) == data * np.arange(N).reshape((1, -1))) )
    #assert( np.all((poptermmeso * spiketrain) == data * poptermmeso.expand) )

def test_popterm_numpy():
    pop_sizes = (3, 10, 5)
    N         = sum(pop_sizes)
    data      = np.arange(N)
    assert isinstance(PopTermMicro(pop_sizes, data, ['Micro']), NumpyPopTerm)
        # Check casting
    check_broadcasting(
        pop_sizes = pop_sizes,
        data      = data,
        N = N,
        T = 30
        )
def test_popterm_shimmed_shared():
    shim.load(False)
    pop_sizes = (3, 10, 5)
    N         = sum(pop_sizes)
    data      = shim.shared(np.arange(N))
    assert isinstance(PopTermMicro(pop_sizes, data, ['Micro']), SymbolicPopTerm)
        # Check casting
    check_broadcasting(
        pop_sizes = pop_sizes,
        data      = data,
        N = N,
        T = 30
        )

def test_popterm_theano_shared():
    shim.load(True)
    pop_sizes = (3, 10, 5)
    N         = sum(pop_sizes)
    data      = shim.shared(np.arange(N))
    poptermmicro = PopTermMicro(pop_sizes, data, ['Micro'])
    assert isinstance(poptermmicro, SymbolicPopTerm)
        # Check casting
    # Check broadcasting against a Numpy array
    check_broadcasting(
        pop_sizes = pop_sizes,
        data      = data,
        N = N,
        T = 30
        )
    # Check broadcasting against another symbolic array
    means = np.array([ poptermmicro.view(np.ndarray)[slc].mean()
                       for slc in poptermmicro.pop_slices ])
    means = PopTermMeso( pop_sizes, shim.shared(means), ['Meso'] )
    check_broadcasting(
        pop_sizes = pop_sizes,
        data      = data,
        means     = means,
        N = N,
        T = 30
        )


if __name__ == '__main__':
    test_popterm_numpy()
    test_popterm_shimmed_shared()
    test_popterm_theano_shared()
