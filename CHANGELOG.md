# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Breaking]
### Added
- Extended tp-flash algorithm to static numbers of components and enabled automatic differentiation for binary systems. [#336](https://github.com/feos-org/feos/pull/336)
- Rewrote `PhaseEquilibrium::pure_p` to mirror `pure_t` and enable automatic differentiation. [#337](https://github.com/feos-org/feos/pull/337)
- Added `boiling_temperature` to the list of properties for parallel evaluations of gradients. [#337](https://github.com/feos-org/feos/pull/337)

### Packaging
- Updated `quantity` dependency to 0.13 and removed the `typenum` dependency. [#323](https://github.com/feos-org/feos/pull/323)

## [Unreleased]
### Added
- Added `py-feos` tests to GitHub Actions and moved `pyo3/extension-module` feature to `pyproject.toml`. [#334](https://github.com/feos-org/feos/pull/334)

### Fixed
- Fixed `None` transformation when binary parameters are provided to `PyParameters` to properly raise errors. [#334](https://github.com/feos-org/feos/pull/334)
- Fixed the calculation of critical points and tp-flashes when one or more of the mixture components have zero composition. [#331](https://github.com/feos-org/feos/pull/331)

## [0.9.2] - 2025-12-11
### Fixed
- Fixed calculation of enthalpies of adsorption for mixtures. [#329](https://github.com/feos-org/feos/pull/329)
- Updated to `ndarray` 0.17 and `num-dual`0.13 to fix a broken dependency resolution. [#327](https://github.com/feos-org/feos/pull/327)
- Fixed calculation of parameter combination in entropy scaling for mixtures in `viscosity_correlation`, `diffusion_correlation`, `thermal_conductivity_correlation`. [#323](https://github.com/feos-org/feos/pull/323)

## [0.9.1] - 2025-11-24
### Fixed
- Fixed a wrong Jacobian entry in `heteroazeotrope_t`. (Thanks to @ImagineBaggins for reporting the issue) [#320](https://github.com/feos-org/feos/pull/320)

## [0.9.0] - 2025-11-08
### Added
- Integrated the functionalities of [`feos-ad`](https://github.com/feos-org/feos-ad). [#289](https://github.com/feos-org/feos/pull/289)
    - In Rust: Full access to arbitrary derivatives of properties and phase equilibria with respect to model parameters. See, e.g., [`feos-campd`](https://github.com/feos-org/feos-campd) for an application to molecular design.
    - In Python: Specialized functions for the parallel evaluation of relevant properties (vapor pressure, liquid density, bubble/dew point pressure) including the gradients with respect to model parameters for parameter estimations or the inclusion in backpropagation frameworks.
- Implement pure-component multiparameter equations of state from CoolProp. [#301](https://github.com/feos-org/feos/pull/301)

### Changed
- :warning: Changed the format of parameter files. The contents of the old `model_record` field are now flattened into the `PureRecord`/`SegmentRecord`. [#233](https://github.com/feos-org/feos/pull/233)
- Generalized the implementation of association to allow for arbitrarily many association sites per molecule or group and full control over each interaction. [#233](https://github.com/feos-org/feos/pull/233) [#290](https://github.com/feos-org/feos/pull/290)
- Reimplemented the Python interface to avoid the necessity of having multiple classes with the same name.
    - `feos.eos.State` and `feos.dft.State` (and analogous classes) are combined into `feos.State`. [#274](https://github.com/feos-org/feos/pull/274)
    - All `feos.<model>.PureRecord` (and similar classes) are combined into `feos.PureRecord`. [#271](https://github.com/feos-org/feos/pull/271)
- All Python classes are exported at the package root. [#309](https://github.com/feos-org/feos/pull/309)
- Add initial density as optional argument to critical point algorithms. [#300](https://github.com/feos-org/feos/pull/300)

### Packaging
- Updated `quantity` dependency to 0.12.
- Updated `num-dual` dependency to 0.12.
- Updated `numpy`, `PyO3` and `pythonize` dependencies to 0.27.
- Updated `nalgebra` dependency to 0.34.

## [0.8.0] - 2024-12-28
### Fixed
- Fixed the handling of association records in combination with induced association in PC-SAFT [#264](https://github.com/feos-org/feos/pull/264)

### Packaging
- Updated `quantity` dependency to 0.10. [#262](https://github.com/feos-org/feos/pull/262)
- Updated `num-dual` dependency to 0.11. [#262](https://github.com/feos-org/feos/pull/262)
- Updated `numpy` and `PyO3` dependencies to 0.23. [#262](https://github.com/feos-org/feos/pull/262)

## [0.7.0] - 2024-05-21
### Added
- Added SAFT-VR Mie equation of state. [#237](https://github.com/feos-org/feos/pull/237)
- Added ePC-SAFT equation of state. [#229](https://github.com/feos-org/feos/pull/229)

### Changed
- Updated model implementations to account for the removal of trait objects for Helmholtz energy contributions and the de Broglie in `feos-core`. [#226](https://github.com/feos-org/feos/pull/226)
- Changed Helmholtz energy functions in `PcSaft` contributions so that the temperature-dependent diameter is re-used across different contributions. [#226](https://github.com/feos-org/feos/pull/226)
- Renamed structs in `uvtheory` module in accordance with names in other models (`UV...` to `UVTheory...`). [#226](https://github.com/feos-org/feos/pull/226)
- Restructured `uvtheory` module: added modules for BH and WCA. [#226](https://github.com/feos-org/feos/pull/226)
- Updated github action versions for CI/CD. [#226](https://github.com/feos-org/feos/pull/226)
- Added `codegen-units = 1` to `release-lto` profile. [#226](https://github.com/feos-org/feos/pull/226)

### Removed
- Removed `VirialOrder` from `uvtheory` module. Orders are now variants of the existing `Perturbation` enum. [#226](https://github.com/feos-org/feos/pull/226)

### Packaging
- Updated `quantity` dependency to 0.8. [#238](https://github.com/feos-org/feos/pull/238)
- Updated `num-dual` dependency to 0.9. [#238](https://github.com/feos-org/feos/pull/238)
- Updated `numpy` and `PyO3` dependencies to 0.21. [#238](https://github.com/feos-org/feos/pull/238)

## [0.6.1] - 2024-01-11
- Python only: Release the changes introduced in `feos-core` 0.6.1.

## [0.6.0] - 2023-12-19
### Added
- Added `EquationOfState.ideal_gas()` to initialize an equation of state that only consists of an ideal gas contribution. [#204](https://github.com/feos-org/feos/pull/204)
- Added `PureRecord`, `SegmentRecord`, `Identifier`, and `IdentifierOption` to `feos.ideal_gas`. [#205](https://github.com/feos-org/feos/pull/205)
- Added implementation of the Joback ideal gas model that was previously part of `feos-core`. [#204](https://github.com/feos-org/feos/pull/204)
- Added an implementation of the ideal gas heat capacity based on DIPPR equations. [#204](https://github.com/feos-org/feos/pull/204)
- Added re-exports for the members of `feos-core` and `feos-dft` in the new modules `feos::core` and `feos::dft`. [#212](https://github.com/feos-org/feos/pull/212)

### Changed
- Split `feos.ideal_gas` into `feos.joback` and `feos.dippr`. [#204](https://github.com/feos-org/feos/pull/204)

## [0.5.1] - 2023-11-23
- Python only: Release the changes introduced in `feos-core` 0.5.1.

## [0.5.0] - 2023-10-20
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
- Updated `quantity` dependency to 0.7.
- Updated `num-dual` dependency to 0.8. [#137](https://github.com/feos-org/feos/pull/137)
- Updated `numpy` and `PyO3` dependencies to 0.20.

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
