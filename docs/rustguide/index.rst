The Rust Guide to ``FeOs``
==========================

Welcome to the ``FeOs`` Rust guide.
On the following pages we discuss the structure of the ``FeOs`` project and how to use, compile and extend it.

Getting Started
---------------

.. toctree::
   :maxdepth: 2

   getting_started

``feos-core``
-------------

In this section, we discuss the ``core`` crate.
We will learn how equations of state are abstracted using traits, how generalized (hyper-) dual numbers are utilized and how thermodynamic states and phase equilibria are defined.

.. toctree::
   :maxdepth: 2

   core/index
   core/equation_of_state
   core/state

Thermodynamic States and Properties
-----------------------------------

In this chapter, we will look at the way thermodynamic states are defined and how properties for a given state can be computed.

.. toctree::
   :maxdepth: 2

   state
