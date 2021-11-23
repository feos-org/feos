.. _num_dual_api:

.. currentmodule:: feos.user_defined.num_dual

Generalized (hyper-) dual numbers
---------------------------------

`FeOs` uses dual numbers to enable calculating higher order (partial) derivatives of the
Helmholtz energy without actually implementing them. You can learn more about dual numbers in :doc:`this example notebook </examples/DualNumbers>`.
If you are interested in using dual numbers in other projects, you can find the documentation of the `num-dual` python package `here <https://itt-ustutt.github.io/num-dual/>`_.


.. autosummary::
    :toctree: generated/

    derive1
    derive2
    derive3
    Dual64
    Dual2_64
    Dual2Dual64
    Dual3_64
    Dual3Dual64
    HyperDual64
    HyperDualDual64
