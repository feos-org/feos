.. _peng_robinson_api:

.. currentmodule:: feos.cubic

Peng-Robinson
-------------

.. important::
    This implementation of the Peng-Robinson equation of state is intended to be used
    as simple example when considering implementing an equation of state. It is not
    a sophisticated implementation and should probably not be used to do research.

Parameter and utility
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/

    Identifier
    ChemicalRecord
    JobackRecord
    PengRobinsonRecord
    PureRecord
    BinaryRecord
    PengRobinsonParameters
    Contributions
    Verbosity

Equation of state
~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/

    PengRobinson
    State
    PhaseEquilibrium
    PhaseDiagramPure
    PhaseDiagramBinary
    PhaseDiagramHetero