.. _pcsaft_api:

PC-SAFT
-------

Implementation of the PC SAFT equation of state.

Parameter and utility
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: feos.pcsaft

.. autosummary::
    :toctree: generated/

    Identifier
    ChemicalRecord
    JobackRecord
    PureRecord
    SegmentRecord
    BinaryRecord
    BinarySegmentRecord
    PcSaftRecord
    PcSaftParameters
    Contributions
    Verbosity


Equation of state
~~~~~~~~~~~~~~~~~

.. currentmodule:: feos.pcsaft.eos

.. autosummary::
    :toctree: generated/

    PcSaft
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

Interfaces
^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    PlanarInterface
    SurfaceTensionDiagram

Adsorption
^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    ExternalPotential
    Geometry
    Pore1D
    Pore3D
    Adsorption1D
    Adsorption3D

Solvation
^^^^^^^^^

.. autosummary::
    :toctree: generated/

    PairCorrelation
    SolvationProfile