# Parameter Loading

This example shows different ways to load and manage parameters in FeOs.

## Loading Parameters from JSON

FeOs can load parameters from JSON files, which is convenient for working with parameter databases:

```python
from feos.parameters import Parameters, IdentifierOption

# Load pure component parameters from JSON file
# (Note: You'll need actual JSON parameter files for this to work)
try:
    parameters = Parameters.from_json(
        substances=["methane", "ethane"],
        pure_path="path/to/pure_parameters.json",
        identifier_option=IdentifierOption.Name
    )
    print("Parameters loaded successfully")
except FileNotFoundError:
    print("Parameter file not found - creating parameters manually")
    # Fall back to manual parameter creation
    # ... (manual creation code)
```

## Creating Parameters Manually

When you don't have JSON files, create parameters directly:

```python
from feos.parameters import PureRecord, BinaryRecord, Identifier, Parameters

# Create pure component records
methane = PureRecord(
    identifier=Identifier(name="methane", cas="74-82-8"),
    molarweight=16.04,
    m=1.0,
    sigma=3.7039,
    epsilon_k=150.03,
)

ethane = PureRecord(
    identifier=Identifier(name="ethane", cas="74-84-0"),
    molarweight=30.07,
    m=1.6069,
    sigma=3.5206,
    epsilon_k=191.42,
)

# Create binary interaction parameter
binary = BinaryRecord(
    id1=Identifier(name="methane"),
    id2=Identifier(name="ethane"),
    k_ij=0.0,  # No binary interaction
)

# Combine into Parameters object
parameters = Parameters.from_records(
    pure_records=[methane, ethane],
    binary_records=[binary]
)

print(f"Created parameters for {len(parameters.pure_records)} components")
```

## Working with Identifiers

FeOs supports multiple identifier types for flexibility:

```python
from feos.parameters import Identifier, IdentifierOption

# Different ways to create identifiers
id_by_name = Identifier(name="water")
id_by_cas = Identifier(cas="7732-18-5")
id_by_smiles = Identifier(smiles="O")

# Complete identifier with multiple fields
complete_id = Identifier(
    name="water",
    cas="7732-18-5",
    iupac_name="oxidane",
    smiles="O",
    formula="H2O"
)

print(f"Identifier: {complete_id}")

# When loading from JSON, specify which identifier to match on
parameters_by_name = Parameters.from_json(
    substances=["water"],
    pure_path="parameters.json",
    identifier_option=IdentifierOption.Name
)

parameters_by_cas = Parameters.from_json(
    substances=["7732-18-5"],
    pure_path="parameters.json",
    identifier_option=IdentifierOption.Cas
)
```

## Parameter Inspection

Examine loaded parameters:

```python
# Inspect pure records
for record in parameters.pure_records:
    print(f"Component: {record.identifier.name}")
    print(f"  Molar weight: {record.molarweight} g/mol")
    print(f"  Model parameters: {record.model_record}")
    print()

# Inspect binary records
if hasattr(parameters, 'binary_records') and parameters.binary_records:
    for binary in parameters.binary_records:
        print(f"Binary interaction: {binary.id1.name} - {binary.id2.name}")
        print(f"  Parameters: {binary.model_record}")
```

## JSON Export

Export parameters to JSON format:

```python
# Convert parameters to JSON strings
pure_json, binary_json = parameters.to_json_str(pretty=True)

print("Pure component parameters (JSON):")
print(pure_json[:200] + "..." if len(pure_json) > 200 else pure_json)

if binary_json:
    print("\nBinary parameters (JSON):")
    print(binary_json[:200] + "..." if len(binary_json) > 200 else binary_json)

# Save to files
with open("pure_params.json", "w") as f:
    f.write(pure_json)

if binary_json:
    with open("binary_params.json", "w") as f:
        f.write(binary_json)
```

## Different Parameter Sets

Different equation of state models require different parameters:

```python
# PC-SAFT parameters (associating component)
water_pcsaft = PureRecord(
    identifier=Identifier(name="water"),
    molarweight=18.015,
    m=1.2047,      # segment number
    sigma=2.7927,  # segment diameter [Å]
    epsilon_k=353.944,  # dispersion energy [K]
    kappa_ab=0.0451,    # association volume
    epsilon_k_ab=2500.7, # association energy [K]
    na=1,          # association sites A
    nb=1,          # association sites B
)

# PETS parameters (simpler model)
argon_pets = PureRecord(
    identifier=Identifier(name="argon"),
    molarweight=39.948,
    sigma=3.4,     # hard sphere diameter [Å]
    epsilon_k=120, # dispersion energy [K]
)

# Create different parameter sets
pcsaft_params = Parameters.new_pure(water_pcsaft)
pets_params = Parameters.new_pure(argon_pets)

print("Created PC-SAFT parameters for water")
print("Created PETS parameters for argon")
```

## Group Contribution Parameters

For group contribution methods like GC-PC-SAFT:

```python
from feos.parameters import (
    GcParameters, ChemicalRecord, SegmentRecord, 
    BinarySegmentRecord, SmartsRecord
)

# Define segments (functional groups)
ch3_segment = SegmentRecord(
    identifier="CH3",
    molarweight=15.035,
    m=0.7745,
    sigma=3.7729,
    epsilon_k=229.0,
)

ch2_segment = SegmentRecord(
    identifier="CH2", 
    molarweight=14.027,
    m=0.6744,
    sigma=3.8395,
    epsilon_k=247.0,
)

# Define chemical structure (e.g., propane = CH3-CH2-CH3)
propane_structure = ChemicalRecord(
    identifier=Identifier(name="propane"),
    segments=["CH3", "CH2", "CH3"],
    bonds=[[0, 1], [1, 2]]  # Connect segments
)

# Binary segment interaction
ch3_ch2_binary = BinarySegmentRecord(
    id1="CH3",
    id2="CH2", 
    model_record=0.0  # No interaction
)

# Create GC parameters
gc_params = GcParameters.from_segments(
    chemical_records=[propane_structure],
    segment_records=[ch3_segment, ch2_segment],
    binary_segment_records=[ch3_ch2_binary]
)

print("Created group contribution parameters for propane")
```

## SMILES-based Parameter Generation

Generate parameters from SMILES strings (requires rdkit):

```python
try:
    # This requires rdkit to be installed
    from feos.parameters import SmartsRecord
    
    # Define SMARTS patterns for fragmentation
    smarts = [
        SmartsRecord(group="CH3", smarts="[CH3]"),
        SmartsRecord(group="CH2", smarts="[CH2]"),
        # ... more patterns
    ]
    
    # Generate parameters from SMILES
    gc_params_smiles = GcParameters.from_smiles(
        identifier="CCC",  # SMILES for propane
        smarts_records=smarts,
        segment_records=[ch3_segment, ch2_segment],
        binary_segment_records=[ch3_ch2_binary]
    )
    
    print("Generated parameters from SMILES")
    
except ImportError:
    print("rdkit not available - cannot generate from SMILES")
except Exception as e:
    print(f"SMILES generation failed: {e}")
```

## Parameter Database Management

Best practices for managing parameter databases:

```python
import json
from pathlib import Path

# Create a parameter database structure
def create_parameter_database():
    """Create a simple parameter database"""
    
    # Pure component database
    pure_db = [
        {
            "identifier": {"name": "methane", "cas": "74-82-8"},
            "molarweight": 16.04,
            "model_record": {
                "m": 1.0,
                "sigma": 3.7039,
                "epsilon_k": 150.03
            }
        },
        {
            "identifier": {"name": "ethane", "cas": "74-84-0"},
            "molarweight": 30.07,
            "model_record": {
                "m": 1.6069,
                "sigma": 3.5206,
                "epsilon_k": 191.42
            }
        }
    ]
    
    # Binary interaction database
    binary_db = [
        {
            "id1": {"name": "methane"},
            "id2": {"name": "ethane"},
            "model_record": {"k_ij": 0.0}
        }
    ]
    
    # Save to files
    Path("pure_parameters.json").write_text(
        json.dumps(pure_db, indent=2)
    )
    Path("binary_parameters.json").write_text(
        json.dumps(binary_db, indent=2)
    )
    
    return "pure_parameters.json", "binary_parameters.json"

# Create and use database
pure_file, binary_file = create_parameter_database()

# Load parameters from the database
params = Parameters.from_json(
    substances=["methane", "ethane"],
    pure_path=pure_file,
    binary_path=binary_file,
    identifier_option=IdentifierOption.Name
)

print(f"Loaded parameters from database: {len(params.pure_records)} components")
```

## Error Handling

Handle common parameter loading errors:

```python
try:
    # Try to load non-existent component
    params = Parameters.from_json(
        substances=["nonexistent_component"],
        pure_path="pure_parameters.json",
        identifier_option=IdentifierOption.Name
    )
except Exception as e:
    print(f"Failed to load parameters: {e}")

try:
    # Invalid parameter values
    invalid_record = PureRecord(
        identifier=Identifier(name="invalid"),
        molarweight=-10,  # Invalid negative mass
        m=1.0,
        sigma=3.0,
        epsilon_k=100,
    )
except ValueError as e:
    print(f"Invalid parameter: {e}")
```

This example covers the various ways to work with parameters in FeOs. Next, explore [binary mixtures](binary-mixtures.md) to see how to use these parameters for mixture calculations.