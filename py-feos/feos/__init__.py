"""Feos."""

from .feos import (
    version,
    Verbosity,
    Contributions,
    State,
    StateVec,
    PhaseDiagram,
    PhaseDiagramHetero,
    PhaseEquilibrium,
    EquationOfState,
)

__version__ = version

try:
    from .feos import ParameterFit, Model
except:
    pass

try:
    from .feos import HelmholtzEnergyFunctional
except:
    pass
