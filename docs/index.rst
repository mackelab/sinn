.. Sinn documentation master file, created by
   sphinx-quickstart on Mon Apr  9 15:47:56 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Sinn – User Manual
==================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   models
   histories
   axis

A caution regarding indexing
----------------------------

:py:mod:`Sinn` does not recognize negative indices (or at least not the way you may expect them to). This is primarily a protection for the user against hard-to-track bugs, but also helps make the interface more intuitive. (See :ref:`sinn-indexing`.)

Sinn – User API
===============

.. toctree::
  :maxdepth: 2

  user-api/histories-api
  user-api/axis-api

Sinn – Technical references
===========================

.. toctree::
   :maxdepth: 1

   internal/index
   developer-api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
