# `feos.parameters`
All functionality regarding parameter handling is contained in the `feos.parameters` module.

In FeOs parameters can be read from JSON files or created manually by combining pure-component parameters and binary interaction parameters.


## `Parameters` and `GcParameters`
The core data types for parameter handling in FeOs are `Parameters` for regular component-specific parameters and `GcParameters` for group-contribution models.

```{eval-rst}
.. currentmodule:: feos.parameters

.. autosummary::
    :toctree: generated/

    Parameters
    GcParameters
```

### Methods to build `Parameters`
```{eval-rst}
.. currentmodule:: feos.parameters

.. autosummary::
    Parameters.from_records
    Parameters.new_pure
    Parameters.new_binary
    Parameters.from_json
    Parameters.from_multiple_json
```

### Methods to build `GcParameters`
```{eval-rst}
.. currentmodule:: feos.parameters

.. autosummary::
    GcParameters.from_segments
    GcParameters.from_json_segments
    GcParameters.from_smiles
    GcParameters.from_json_smiles
```

## Other data types

```{eval-rst}
.. currentmodule:: feos.parameters

.. autosummary::
    :toctree: generated/

    Identifier
    IdentifierOption
    PureRecord
    SegmentRecord
    BinaryRecord
    BinarySegmentRecord
    ChemicalRecord
    SmartsRecord
```