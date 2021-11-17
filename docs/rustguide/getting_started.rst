Introduction
------------

``FeOs`` is primarily written and tested on Linux but eventually we want it to run on Linux, macOS and Windows.
We suggest working on Linux if that's an option for you because managing dependencies (of packages such as BLAS and LAPACK) is quite easy.
You need a `Rust compiler <https://www.rust-lang.org/tools/install>`_ (version 1.51+) to compile the code.
``FeOs`` uses some routines from BLAS and LAPACK and for the smoothest development experience, we suggest installing `OpenBLAS <https://www.openblas.net/>`_ on your system which you can then use as ``feature`` when testing ``FeOs``.
This circumvents building a static version and reduces compilation times drastically.
For development, `Visual Studio Code <https://code.visualstudio.com/>`_ with the `rust-analyzer <https://rust-analyzer.github.io/>`_ plugin works pretty well, but you should use what you are comfortable in.

Rust Prerequisites
------------------

If you are unfamiliar with Rust a good place to start is the `Rust Programming Language Book <https://doc.rust-lang.org/book/>`_.
To start following our guide, you should understand the following topics:

* how Rust projects, called *crates*, are structured (the *module* system),
* data types and ``structs``,
* the Rust *ownership model*,
* ``enums`` and *pattern matching*,
* *traits*,

With these foundations you should be able to follow the discussion.
Eventually you'll need to learn and understand

* limitations of *traits* and *trait objects*,
* smart pointers (``Rc`` and ``Box``),
* and the ``pyO3`` crate if you are interested in the Python interface.

Project Structure
-----------------

``FeOs`` is split into multiple crates that build on each other.
The most important ones are

* ``feos-core`` (``core`` for short): defines traits and structs for equations of state and implements thermodynamic states, phase equilibria and critical point routines.
* ``feos-dft`` (``dft`` for short): builds on ``core`` and defines traits and structs for classical density functional theory and implements utilities to work with convolutions, external potentials, etc.

These crates offer abstractions for tasks that are common for all equations of state and Helmholtz energy functionals.
Using ``core`` and ``dft``, the following *implementations* of equations of state and functionals are currently available:

* ``feos-pcsaft``: the `PC-SAFT equation of state <https://pubs.acs.org/doi/abs/10.1021/ie0003887>`_.
* ``feos-gc-pcsaft``: the `hetero-segmented group contribution <https://aip.scitation.org/doi/full/10.1063/1.4945000>`_ method of the PC-SAFT equation of state.
* ``feos-uv-theory``: the equation of state based on UV-Theory.
* ``feos-thol``: the `Thol equation of state <https://aip.scitation.org/doi/full/10.1063/1.4945000>`_ for pure Lennard-Jones fluids.
* ``feos-pets``: the `PeTS equation of state <https://www.tandfonline.com/doi/full/10.1080/00268976.2018.1447153>`_.


Where to Get Help
-----------------

``FeOs`` is openly developed on `github <https://github.com/feos-org>`_. Each crate has it's own github repository where you can use the *discussion* feature or file an *issue*.
