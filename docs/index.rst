Welcome to FeOs
===============

``FeOs`` is a framework for **thermodynamic equations of state** and **classical functional theory**.
It is written in **Rust** with an interface for **Python**.

* You can use ``FeOs`` with :ref:`implemented equations of state <equations_of_state>` to compute

  * thermodynamic properties,
  * phase equilibria,
  * critical points,
  * dynamic properties via entropy scaling, and
  * interfacial properties via classical density function theory.

* ``FeOs`` can use equations of state written as :ref:`Python classes <user_defined_api>` - perfect for quick and easy prototyping.

Getting Started
---------------

Installation
~~~~~~~~~~~~

If you want to use ``FeOs`` in Python, you can install it using ``pip``:

.. code::

    pip install feos

If you want to use ``FeOs`` in Rust, please have a look at the :doc:`Rust Guide </rustguide/index>`.