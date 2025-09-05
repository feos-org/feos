"""Feos."""

from feos.feos import (
    version,
    Verbosity,
    Contributions,
    State,
    StateVec,
    PhaseDiagram,
    PhaseDiagramHetero,
    PhaseEquilibrium,
    EquationOfState,
    Estimator,
    Model,
    __dft__,
)

__version__ = version

__all__ = [
    "parameters",
    # "estimator",
    "Verbosity",
    "Contributions",
    "State",
    "StateVec",
    "PhaseDiagram",
    "PhaseDiagramHetero",
    "PhaseEquilibrium",
    "EquationOfState",
    "Estimator",
    "Model"
]
if __dft__:
    __all__ = ["dft"] + __all__
