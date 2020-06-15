*********
Data axes
*********

The `Axis` objects serve as the link between physical and digital dimensions.
They are comparable to the `dims` and `coords` attributes
of a PyData_ :class:`xarray.DataArray`, and similarly form the metadata
frame around a data object. However, in contrast to the PyData objects, they
may or may not be underlined by a NumPy array, and have specialized functions
for index arithmetic.

.. _PyData: http://xarray.pydata.org/en/stable/data-structures.html

Definitions
===========

.. glossary::

Axis
  Physical axis, generally with associated units.

Data index
  *Index*, which is shifted such that its smallest value is always zero.
  This is the value to use to index the underlying data.

Digital dimension
  A dimension consistent with the computer's storage; the basic example is
  an axis of a software array.

Index
  Integer value representing a position along a digital dimension. This may
  differ from the *data index* if an axis includes padding.

Index delta
  Difference between two indices. / Relative index.

Mapping
  Strictly-positive monotone function mapping an integer in digital space to
  a stop in physical space.

Padding
  Extra stops added before (after) the *x:sub:`0`* (*x:sub:`n`*) position of a digital axis.
  These may be used e.g. to initialize a dynamical system, or allow the
  calculation of convolutions at axis bounds.
  Adding left-padding will not affect the *index*, but **will** affect the
  *data index* (by shifting it by an amount equal to the added padding).

Stops
  Physical values on an axis (one may think of ticks on a plot).

x:sub:`0`
  Position corresponding to the start of a discretized axis. If the axis
  corresponds to time, this would usually be the start of the simulation.

x:sub:`n`
  Position corresponding to the end of a discretized axis. If the axis
  corresponds to time, this would usually be the end of the simulation.

.. _sinn-indexing:

Indexing
========

Indexing in `sinn` works a little differently than what you may expect (in fact, that is the reason why in most cases, it *looks* like normal indexing). The first thing to note is that `sinn` does not recognize negative indices (or at least not the way you may expect them to). This is because a core task of `sinn` is to do index arithmetic for us, and it is *way* too easy to accidentally calculate a negative index (for example by trying to retrieve the time point before the origin). After one-too-many times of shooting myself in the foot, I realized that the small convenience of negative indexing (``axis[-i]`` vs. ``axis[axis.xnidx - i]``) was simply more troubled then it was worth and disallowed it entirely.

Later I realized the reason for these issues: that I was using “index” to refer to two related, but conceptually different things: a “data index” and an “axis index”. A *data* index is tied to a structure in computer memory; index ``0`` is always the first element of this structure, and since it has fixed length ``L``, mapping ``-i`` to ``L-i`` is unambiguous. An *axis* index, on the other hand, is simply a position on an arbitrarily defined discretization of that axis. Index `0` is most naturally associated with the origin, *but this need not be the first discretized position*. Correspondingly, a negative index should then be associated with a position left of the origin.

The thing to remember is this: because the structures defined by `sinn` are mathematical structure, *so too is indexing done in axis space*. This includes the interpretation of negative indices as being “more to the left” than 0. The translation into data indices is done internally and should be transparent. As an added bonus, this avoids hard-to-optimize conditionals in the computational graph.

Throughout this library, a “position along a discretized axis” is referred to as a “stop”.

.. Note:: Always use axis indices when passing as arguments between functions or methods; this includes internal functions. Resolve to data indices only when indexing the underlying data. (Remember that axis indices are also monotone and 1-to-1 mapped to data indices, and so just as informational as data indices.)

Operations between indices
==========================

TODO

.. Note::
   Operations with plain integers are allowed, but since it is impossible to know whether a plain integer is an absolute or a relative index, they are necessarily ambiguous. The convention in sinn is to treat integers as absolute when *indexing* and testing *equality*, but as relative (deltas) when performing *operations*. This allows ``hist[5]``, ``hist.cur_tidx == 5`` and ``hist[i+1]`` to work as generally expected.

   If this convention is ill-suited, or to remove any ambiguity, use ``hist.Index`` or ``hist.Index.Delta`` to cast the value to the appropriate::

      hist[hist.Index(5)]
      hist.cur_tidx == hist.Index(5)
      hist[i + hist.Index.Delta(1)]

Main classes
============

Axis types
----------

`Axis`
  Base class for all axis, which represent a *physical* axis. Contains methods
  for converting/checking physical units and converting to a transformed
  space (if the axis was defined with a bijection).
  Does **not** include any indexing functionality; this is a purely
  “physical” concept.

`DiscretizedAxis`  (Virtual)
  Subclass of `Axis` which adds indexing. This is a virtual class because
  it does not specify the structure used for indexing.

`MapAxis`
  A `DiscretizedAxis` where the indexing is provided by `SequenceMapping`.

`RangeAxis`
  A `DiscretizedAxis` where the indexing is provided by `RangeMapping`.

`ArrayAxis`
  A `DiscretizedAxis` where the indexing is provided by `ArrayMapping`.

Mapping types
-------------

`SequenceMapping`
  An object implementing an arbitrary mapping between indices and stops.
  Is responsible for basic index arithmetic, including computing padding.
  A `SequenceMapping` is iterable and supports indexed access.

`RangeMapping`
  A `SequenceMapping` where the indexing is implemented by a memory-efficient
  `range` object and the stops are spaced regularly. Stops are computed with
  index arithmetic, in the same way as `range`.

`ArrayMapping`
  A `SequenceMapping` where the stops are simply stored as an array.

.. Note:: Part of the goal of the *padding* constructs is to make padding as invisible as possible. Consequently, when indexing a `*Mapping` with square brackets, one uses the *unpadded* index. (With regards to the definitions above, brackets use the *index* rather than the *data index*.)

.. Warning:: Negative indices are explicitly disallowed on :py:class:`*Mapping` (or more specifically, they are just treated as "more left" than zero). With large amounts of index arithmetic, I found it too easy to accidentally obtain negative indices, and the hard-to-track bugs they introduce don't justify the minor convenience they bring. As a bonus, negative indices are sometimes useful to add padding stops without changing the position of '0'.

Index types
-----------

`AxisIndex` and `AxisIndexDelta` are very similar, but are treated
somewhat differently in operations. For
example, instances of `AxisIndex` cannot be added together, while instances
of `AxisIndexDelta` can. (c.f. operations table).

One should not normally create these classes directly.

`AxisIndex`
  Absolute index. A new `AxisIndex` *class* is created dynamically every time
  an `*Mapping` object is instantiated. This allows operations between indices
  to check whether they refer to the same `Axis`.
  The index reference is *x:sub:`0`*; to index into the data, use the
  `.data_index` method to correct for padding.
  Technically a subclass of `AxisIndexDelta`.

`AxisIndexDelta`
  Relative index. Created alongside `AxisIndex` when a `*Mapping` object is
  instantiated.
