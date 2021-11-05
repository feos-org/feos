Equation of state
=================

In FeOs, an equation of state is a struct that implements the `EquationOfState` trait.
Within the trait's definition, a number of additional traits appear, such as `HelmholtzEnergy`, `IdealGasContribution` and `DualNum` which, at a first glance, will possibly lead to a lot of confusion.

In the following sections, we will try to unravel how these traits interact with each other, and what is needed to extend an existing equation of state or to implement a new one.

At the end of this chapter, we will implement a simple cubic equation of state from scratch, applying what we learned.

.. toctree::

    traits
    additional_traits
    ideal_gas