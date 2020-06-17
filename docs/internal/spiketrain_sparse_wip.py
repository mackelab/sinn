# This is the work sheet where I tested how to manipulate sparse arrays
# by modifying the underlying storage arrays.
# I think it's useful for anyone trying understand the logic within Spiketrain,
# to first work through these examples.

import numpy as np
import scipy as sp
import scipy.sparse
from types import SimpleNamespace

A = np.array(
    [[0, 1, 2],
     [0, 0, 0],
     [1, 0, 0]])

# CSR

Acsr = SimpleNamespace(
    data    = [1, 2, 1],
    indices = [1, 2, 0],     # column indices
    indptr  = [0, 2, 2, 3]
)

Acsr_mat = sp.sparse.csr_matrix(
    (Acsr.data, Acsr.indices, Acsr.indptr), (3,3))

np.all(Acsr_mat.todense() == A)

# Add 3,3 on 3rd row  (index 2)
A = np.array(
    [[0, 1, 2],
     [1, 0, 0],
     [0, 0, 0]])
Acsr = SimpleNamespace(
    data    = [1, 2, 1],
    indices = [1, 2, 0],     # column indices
    indptr  = [0, 2, 3, 3]
)
newindptr = np.array(Acsr.indptr)
newindptr[2+1:] = newindptr[2+1:] + 2
Acsr2 = SimpleNamespace(
    data    = np.concatenate((Acsr.data, [3,3])),
    indices = np.concatenate((Acsr.indices, [0,1])),
    indptr  = newindptr
    )
sp.sparse.csr_matrix(
        (Acsr2.data, Acsr2.indices, Acsr2.indptr), (3,3) ).todense()

# CSC

Acsc = SimpleNamespace(
    data    = [1, 1, 2],
    indices = [2, 0, 0],    # row indices
    indptr  = [0, 1, 2, 3]
)

Acsc_mat = sp.sparse.csc_matrix(
    (Acsc.data, Acsc.indices, Acsc.indptr), (3,3))

np.all(Acsc_mat.todense() == A)

# COO

Acoo = SimpleNamespace(
    data = [1, 2, 1],
    i    = [0, 0, 2],   # row indices
    j    = [0, 1, 2]    # column indices
)

Acoo_mat = sp.sparse.coo_matrix(
    (Acoo.data, (Acoo.i, Acoo.j)), (3,3))


# Conversion functions

# def coo_to_csr(data, i, j):
from theano import sparse as tsparse
import theano_shim as shim
import theano_shim.sparse as sparse
shim.load('theano')

sym_props = tuple(shim.shared(1)*p for p in sparse.csm_properties(Acsc_mat))
sym_Acsc_mat = sparse.CSC(*sym_props)
sym_props

d = shim.shared(np.array((), dtype=float))
i = shim.shared(np.array((), dtype='int16'))
iptr = shim.shared(np.array([0,0,0], dtype='int16'))
csr = shim.sparse.CSR(d, i, iptr, shape=(3,7))

csr.eval()

t_csc = tsparse.csc_from_dense(A)

shim.concatenate((np.arange(4), [2]))

props = sparse.csm_properties(t_csc)
props = sparse.csm_properties(Acsc_mat)
data, indices, indptr, shape = props
data *= 2
data = shim.concatenate((data, [4, 4]))
indices = shim.concatenate((indices, [0, 1]))
indptr = shim.concatenate((indptr, [indptr[-1] + 2]))
shape = shape + shim.asarray([0, 1])
data
indices
indptr
shape

newA = sparse.CSC(data, indices, indptr, shape)
shim.eval(sparse.csm_properties(newA))

x = shim.sparse.csr_matrix('x', (4, 5))

t_csc.eliminate_zeros()
t_csc[:,0:1].eval().todense()
t_csc[:,0:1] = 9


props = sparse.csm_properties(sym_Acsc_mat)
props
data, indices, indptr, shape = props
data

shim.graph.eval((data, indices, indptr, shape), max_cost=30)

A
sparse.dense_from_sparse(t_csc).eval()

sparse.dense_from_sparse(sparse.CSC(*props)).eval()
sparse.dense_from_sparse(sparse.CSR(*props)).eval()

shim.is_symbolic(t_csc)
