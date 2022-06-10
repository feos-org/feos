# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Added getters for the fields of `Pore1D` in Python. [#30](https://github.com/feos-org/feos-dft/pull/30)

### Changed
- Made FMT functional more flexible w.r.t. the shape of the weight functions. [#31](https://github.com/feos-org/feos-dft/pull/31)
- Changed interface of `PairCorrelationFunction` to facilitate the calculation of pair correlation functions in mixtures. [#29](https://github.com/feos-org/feos-dft/pull/29)

## [0.2.0] - 2022-04-12
### Added
- Added `grand_potential_density` getter for DFT profiles in Python. [#22](https://github.com/feos-org/feos-dft/pull/22)

### Changed
- Renamed `AxisGeometry` to `Geometry`. [#19](https://github.com/feos-org/feos-dft/pull/19)
- Removed `PyGeometry` and `PyFMTVersion` in favor of a simpler implementation using `PyO3`'s new `#[pyclass]` for fieldless enums feature. [#19](https://github.com/feos-org/feos-dft/pull/19)
- `DFTSolver` now uses `Verbosity` instead of a `bool` to control its output. [#19](https://github.com/feos-org/feos-dft/pull/19)
- `SurfaceTensionDiagram` now uses the new `StateVec` struct to access properties of the bulk phases. [#19](https://github.com/feos-org/feos-dft/pull/19)
- `Pore1D::initialize` and `Pore3D::initialize` now accept initial values for the density profiles as optional arguments. [#24](https://github.com/feos-org/feos-dft/pull/24)
- Internally restructured the `DFT` structure to avoid redundant data. [#24](https://github.com/feos-org/feos-dft/pull/24)
- Removed the `m` function in `FluidParameters`, it is instead inferred from `HelmholtzEnergyFunctional` which is now a supertrait of `FluidParameters`. [#24](https://github.com/feos-org/feos-dft/pull/24)
- Added optional field `cutoff_radius` to `ExternalPotential::FreeEnergyAveraged`. [#25](https://github.com/feos-org/feos-dft/pull/25)

### Packaging
- Updated `pyo3` and `numpy` dependencies to 0.16.
- Updated `quantity` dependency to 0.5.
- Updated `num-dual` dependency to 0.5.
- Updated `feos-core` dependency to 0.2.
- Updated `ang` dependency to 0.6.
- Removed `log` dependency.

## [0.1.3] - 2022-02-17
### Fixed
- The pore volume for `Pore3D` is now also accesible from Python. [#16](https://github.com/feos-org/feos-dft/pull/16)

## [0.1.2] - 2022-02-16
### Added
- The pore volume using Helium at 298 K as reference is now directly accesible from `Pore1D` and `Pore3D`. [#13](https://github.com/feos-org/feos-dft/pull/13)

### Changed
- Removed the `unsendable` tag from python classes wherever possible. [#14](https://github.com/feos-org/feos-dft/pull/14)

## [0.1.1] - 2022-02-14
### Added
- `HelmholtzEnergyFunctional`s can now overwrite the `ideal_gas` method to provide a non-default ideal gas contribution that is accounted for in the calculation of the entropy, the internal energy and other properties. [#10](https://github.com/feos-org/feos-dft/pull/10)

### Changed
- Removed the `functional` field in `Pore1D` and `Pore3D`. [#9](https://github.com/feos-org/feos-dft/pull/9)

### Fixed
- Fixed the units of default values for adsorption isotherms. [#8](https://github.com/feos-org/feos-dft/pull/8)

### Packaging
- Updated `rustdct` dependency to 0.7.

## [0.1.0] - 2021-12-22
### Added
- Initial release
