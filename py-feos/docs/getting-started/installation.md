# Installation

This guide covers different ways to install the FeOs Python package.

## Requirements

- **Python**: 3.12 or higher
- **Operating System**: Windows, macOS, or Linux

## Install from PyPI

The easiest way to install FeOs is from PyPI using pip:

```bash
pip install feos
```

This will install the latest stable release with all equation of state models and DFT functionality included.

## Install with uv (Recommended for Development)

If you're using [uv](https://docs.astral.sh/uv/) for Python package management:

```bash
uv add feos
```

## Install from Source

### Prerequisites for Building from Source

To build from source, you need:

- **Rust toolchain**: Install from [rustup.rs](https://rustup.rs/)
- **maturin**: Python package for building Rust extensions

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin
pip install maturin
```

### Build and Install

Clone the repository and build the package:

```bash
git clone https://github.com/feos-org/feos.git
cd feos/py-feos

# Install in development mode
maturin develop --release

# Or build a wheel
maturin build --release
```

### Building with Specific Features

You can build with only specific equation of state models:

```bash
# Build with only PC-SAFT
maturin develop --release --features "python pcsaft"

# Build with PC-SAFT and DFT
maturin develop --release --features "python pcsaft dft"
```

Available features:
- `pcsaft` - PC-SAFT equation of state
- `epcsaft` - Electrolyte PC-SAFT  
- `pets` - PETS equation of state
- `saftvrmie` - SAFT-VR Mie
- `saftvrqmie` - SAFT-VR quantum Mie
- `uvtheory` - UV Theory
- `dft` - Density functional theory
- `all_models` - All available models (default)

## Verify Installation

Test your installation by running:

```python
import feos
print(f"FeOs version: {feos.version}")

# Test basic functionality
from feos import EquationOfState
from feos.parameters import PureRecord, Identifier, Parameters

# This should run without errors
record = PureRecord(
    Identifier(name="methane"),
    molarweight=16.04,
    m=1.0,
    sigma=3.7039,
    epsilon_k=150.03,
)
parameters = Parameters.new_pure(record)
eos = EquationOfState.pcsaft(parameters)
print("✓ FeOs installed successfully!")
```

## Optional Dependencies

For enhanced functionality, you may want to install additional packages:

```bash
# For Jupyter notebook support
pip install jupyter ipykernel

# For plotting (used in some examples)
pip install matplotlib

# For RDKit integration (SMILES support)
pip install rdkit
```

## Troubleshooting

### Import Errors

If you get import errors, ensure that:

1. You're using the correct Python environment
2. The package was installed correctly (`pip list | grep feos`)
3. Your Python version is 3.12 or higher

### Build Errors

If building from source fails:

1. **Check Rust installation**: `rustc --version`
2. **Update Rust**: `rustup update`
3. **Check maturin**: `maturin --version`
4. **Clean build**: Remove `target/` directory and rebuild

### Performance Issues

For optimal performance:

1. **Use release builds**: Always include `--release` flag when building
2. **Enable LTO**: Use `maturin build --profile=release-lto` for maximum optimization
3. **Check CPU features**: The package is optimized for modern CPUs

## Next Steps

Once installed, continue with the [Quick Start](quickstart.md) guide to learn the basics of using FeOs.