Spare array implementation
**************************

Theano-compatible CSR implementation
====================================

Scipy names for CSR arrays: `data`, `indices`, `indptr`

For accumulating updates, we need all data to be stored in shared variables.
So we set `_num_data` to be a tuple of three shared arrays (`data`, `indices`
and `indptr`). `_sym_data` is a corresponding tuple of (possibly symbolic) arrays.

Issue 1:
  While we can pass shared arrays to `theano.sparse.CSR()`, later
  compilation will fail. This is because `theano.sparse` expects the
  arguments to `CSR()` to derive from `CSMProperty` graph node, of the type
  obtained with `theano.sparse.csm_properties(sparse_array)`.

Solution 1:
  Create an empty sparse array on which we call `csm_properties()`;
  this gives us empty arrays for `data` and `indices`, and a vector of 0 for
  `indptr`. We save these as attributes to the history as the tuple
  `_empty_data`, alongside `_num_data` and `_sym_data`.
  Whenever we need an actual symbolic sparse array, we concatenate the empty
  arrays with the values of those of `_sym_data` and call `CSR(*)`.
  (For numeric arrays we can use `_num_data` directly.)

Issue 2:
  When adding n entries to a the sparse array at row _i_, all values
  in `indices` after _i_ need to be incremented by _n_. This is O(n), compared
  to the potentially O(1) cost of appending to the list of indices,
  as we would do with a COO array. On the other hand, incrementing vectors
  is a very well optimized operation, so in the end this may not be an issue.
  Moreover, since NumPy arrays must be contiguous, appending to them is
  potentially also O(n).

Solution 2:
  Since we haven't done a cost/benefit analysis, we use the
  simplest approach: incrementing `indices` when we adding spikes.

Comments
========

Managing the three CSR arrays ourselves not only allows us to get around
limitations in Theano (and eventually other ML libraries), but it is also
potentially more efficient than using a built-in CSR structure because we know
that we are only appending rows and don't need to check whether the data arrays
need to be reordered.
