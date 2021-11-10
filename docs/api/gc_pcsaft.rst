.. _gc_pcsaft_api:

GC PC-SAFT
----------

Implementation of PC-SAFT for hetero-segmented functional groups.

Parameter and utility
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: feos.gc_pcsaft

.. autosummary::
    :toctree: generated/

    Identifier
    ChemicalRecord
    JobackRecord
    PureRecord
    SegmentRecord
    BinaryRecord
    BinarySegmentRecord
    GcPcSaftRecord
    GcPcSaftParameters
    Contributions
    Verbosity


Equation of state
~~~~~~~~~~~~~~~~~

.. currentmodule:: feos.pcsaft.eos

.. autosummary::
    :toctree: generated/

    GcPcSaft
    State
    PhaseEquilibrium
    PhaseDiagramPure
    PhaseDiagramBinary
    PhaseDiagramHetero

Density functional theory
~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: feos.pcsaft.dft

.. autosummary::
    :toctree: generated/

    PcSaftFunctional
    State
    PhaseEquilibrium
    PhaseDiagramPure
    PhaseDiagramBinary
    PhaseDiagramHetero
