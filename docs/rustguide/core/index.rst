Getting Started
---------------

To setup your workspace, `fork <https://docs.github.com/en/get-started/quickstart/fork-a-repo>`_ the ``feos-core`` github repository and from that clone the fork.
To build the Rust library, type:

.. code::

    cargo build

Important crates
----------------

``core`` depends on a number of other important crates. Most notably, we use

* ``quantity``: for scalar and vector valued dimensioned quantities. Those are used in almost all user-facing interfaces.
* ``num-dual``: for generalized (hyper-) dual numbers. These data types are very important because we use them to be able to compute partial, higher-order derivatives of the Helmholtz energy without needing to implement them analytically.
* ``ndarray``: for multidimensional arrays. We use these when mathematical operations are performed on arrays. They are a central data structure for ``feos-dft``.
* ``pyO3``: for the Python interface. All interfaces to Python are written in pure Rust using ``pyO3``. It's awesome.
