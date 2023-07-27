# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Added `IdealGasModel` enum that collects all implementors of the `IdealGas` trait. [#158](https://github.com/feos-org/feos/pull/158)
- Added `feos.ideal_gas` module in Python from which (currently) `Joback` and `JobackParameters` are available. [#158](https://github.com/feos-org/feos/pull/158)
- Added binary association parameters to PC-SAFT. [#167](https://github.com/feos-org/feos/pull/167)
- Added derive for `EntropyScaling` for SAFT-VRQ Mie to `ResidualModel` and adjusted parameter initialization. [#179](https://github.com/feos-org/feos/pull/179)

### Changed
- Changed the internal implementation of the association contribution to accomodate more general association schemes. [#150](https://github.com/feos-org/feos/pull/150)
- To comply with the new association implementation, the default values of `na` and `nb` are now `0` rather than `1`. Parameter files have been adapted accordingly. [#150](https://github.com/feos-org/feos/pull/150)
- Added the possibility to specify a pure component correction parameter `phi` for the heterosegmented gc PC-SAFT equation of state. [#157](https://github.com/feos-org/feos/pull/157)
- Adjusted all models' implementation of the `Parameter` trait which now requires `Result`s in some methods. [#161](https://github.com/feos-org/feos/pull/161)
- Renamed `EosVariant` to `ResidualModel`. [#158](https://github.com/feos-org/feos/pull/158)
- Added methods to add an ideal gas contribution to an initialized equation of state object in Python.  [#158](https://github.com/feos-org/feos/pull/158)
- Moved `molar_weight` impls to `Residual` due to removal of `MolarWeight` trait. [#177](https://github.com/feos-org/feos/pull/158)

### Packaging
- Updated `num-dual` dependency to 0.7. [#137](https://github.com/feos-org/feos/pull/137)

## [0.4.3] - 2023-03-20
- Python only: Release the changes introduced in `feos-core` 0.4.2.

## [0.4.2] - 2023-03-20
- Python only: Release the changes introduced in `feos-core` 0.4.1 and `feos-dft` 0.4.1.

## [0.4.1] - 2023-01-28
### Changed
- Replaced some slow array operations to make calculations with multiple associating molecules significantly faster. [#129](https://github.com/feos-org/feos/pull/129)

### Fixed
- Fixed a regression introduced in [#108](https://github.com/feos-org/feos/pull/108) that lead to incorrect results for the 3B association scheme. [#129](https://github.com/feos-org/feos/pull/129)

## [0.4.0] - 2023-01-27
### Added
- Added SAFT-VRQ Mie equation of state and Helmholtz energy functional for first order Feynman-Hibbs corrected Mie fluids. [#79](https://github.com/feos-org/feos/pull/79)
- Added `estimator` module to documentation. [#86](https://github.com/feos-org/feos/pull/86)
- Added benchmarks for the evaluation of the Helmholtz energy and some properties of the `State` object for PC-SAFT. [#89](https://github.com/feos-org/feos/pull/89)
- The Python class `StateVec` is exposed in both the `feos.eos` and `feos.dft` module. [#113](https://github.com/feos-org/feos/pull/113)
- Added uv-B3-theory and additional optional argument `virial_order` to uvtheory constructor to enable uv-B3. [#98](https://github.com/feos-org/feos/pull/98)

### Changed
- Export `EosVariant` and `FunctionalVariant` directly in the crate root instead of their own modules. [#62](https://github.com/feos-org/feos/pull/62)
- Changed constructors `VaporPressure::new` and `DataSet.vapor_pressure` (Python) to take a new optional argument `critical_temperature`. [#86](https://github.com/feos-org/feos/pull/86)
- The limitations of the homo gc method for PC-SAFT are enforced more strictly. [#88](https://github.com/feos-org/feos/pull/88)
- Removed generics for units in all structs and traits in favor of static SI units. [#115](https://github.com/feos-org/feos/pull/115)

### Packaging
- Updated `pyo3` and `numpy` dependencies to 0.18. [#119](https://github.com/feos-org/feos/pull/119)
- Updated `quantity` dependency to 0.6. [#119](https://github.com/feos-org/feos/pull/119)
- Updated `num-dual` dependency to 0.6. [#119](https://github.com/feos-org/feos/pull/119)


### Fixed
- Fixed incorrect indexing that lead to panics in the polar contribution of gc-PC-SAFT. [#104](https://github.com/feos-org/feos/pull/104)
- `VaporPressure` now returns an empty array instead of crashing. [#124](https://github.com/feos-org/feos/pull/124)

## [0.3.0] - 2022-09-14
- Major restructuring of the entire `feos` project. All individual models are reunited in the `feos` crate. `feos-core` and `feos-dft` still live as individual crates within the `feos` workspace.

## [0.2.1] - 2022-05-13
### Fixed
- Fixed a bug due to which the default ideal gas contribution was used for every equation of state. [#17](https://github.com/feos-org/feos/pull/17)

## [0.2.0] - 2022-05-10
### Added
- Added [gc-PC-SAFT](https://github.com/feos-org/feos-gc-pcsaft) equation of state and Helmholtz energy functional.
- Added [PeTS](https://github.com/feos-org/feos-pets) equation of state and Helmholtz energy functional.
- Added [UV-Theory](https://github.com/feos-org/feos-uvtheory) equation of state for Mie fluids.

### Changed
- Combined all equations of state into a single Python class `EquationOfState` and all Helmholtz energy functionals into the Python class `HelmholtzEnergyFunctional`. [#11](https://github.com/feos-org/feos/pull/11)

### Packaging
- Updated [`quantity`](https://github.com/itt-ustutt/quantity/blob/master/CHANGELOG.md) to 0.5.0.
- Updated [`feos-core`](https://github.com/feos-org/feos-core/blob/main/CHANGELOG.md) to 0.2.0.
- Updated [`feos-dft`](https://github.com/feos-org/feos-dft/blob/main/CHANGELOG.md) to 0.2.0.
- Updated [`feos-pcsaft`](https://github.com/feos-org/feos-pcsaft/blob/main/CHANGELOG.md) to 0.2.0.

## [0.1.1] - 2022-02-23
### Added
- Added `pyproject.toml`. [#8](https://github.com/feos-org/feos/pull/8)
- Fixed modules of `SI` classes to make them pickleable. [#8](https://github.com/feos-org/feos/pull/8)

### Packaging
- Updated [`feos-core`](https://github.com/feos-org/feos-core/blob/main/CHANGELOG.md) to 0.1.5.
- Updated [`feos-dft`](https://github.com/feos-org/feos-dft/blob/main/CHANGELOG.md) to 0.1.3.

## [0.1.0] - 2022-01-14
### Added
- Initial release
