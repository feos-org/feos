# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Implemented `HelmholtzEnergyFunctional` for `EquationOfState` to be able to use functionals as equations of state. [#158](https://github.com/feos-org/feos/pull/158)
- Added the possibility to specify the angles of non-orthorombic unit cells. Currently, the external potential must be specified if non-orthorombic unit cells are calculated. [#176](https://github.com/feos-org/feos/pull/176)

### Changed
- `HelmholtzEnergyFunctional`: added `Components` trait as trait bound and removed `ideal_gas` method. [#158](https://github.com/feos-org/feos/pull/158)
- `DFT<F>` now implements `Residual` and furthermore `IdealGas` if `F` implements `IdealGas`. [#158](https://github.com/feos-org/feos/pull/158)
- What properties (and contributions) of `DFTProfile` are available now depends on whether an ideal gas model is provided or not. [#158](https://github.com/feos-org/feos/pull/158)
- Pore volumes and other integrated properties (loadings, energies, ...) are now always reported in units of volume. For translationally symmetric pores this is achieved by multiplying with the reference length (1 Å) in every direction with symmetries. [#181](https://github.com/feos-org/feos/pull/181)

### Removed
- Removed `DefaultIdealGasContribution` [#158](https://github.com/feos-org/feos/pull/158)
- Removed getters for `chemical_potential` (for profiles) and `molar_gibbs_energy` (for `Adsorption1D` and `Adsorption3D`) from Python interface. [#158](https://github.com/feos-org/feos/pull/158)

### Fixed
- Added an error, if the unit cells in 3D DFT are too small and violate the minimum image convention. [#176](https://github.com/feos-org/feos/pull/176)

### Packaging
- Updated `num-dual` dependency to 0.7. [#137](https://github.com/feos-org/feos/pull/137)

## [0.4.1] - 2023-03-20
### Added
- Added new methods `drho_dmu`, `drho_dp` and `drho_dt` that calculate partial derivatives of density profiles to every DFT profile. Also includes direct access to the integrated derivatives `dn_dmu`, `dn_dp` and `dn_dt`. [#134](https://github.com/feos-org/feos/pull/134)
- Added `enthalpy_of_adsoprtion` and `partial_molar_enthalpy_of_adsorption` getters to pores and adsorption isotherms. [#135](https://github.com/feos-org/feos/pull/135)

## [0.4.0] - 2023-01-27
### Added
- `PlanarInterface` now has methods for the calculation of the 90-10 interface thickness (`PlanarInterface::interfacial_thickness`), interfacial enrichtment (`PlanarInterface::interfacial_enrichment`), and relative adsorption (`PlanarInterface::relative_adsorption`). [#64](https://github.com/feos-org/feos/pull/64)
- Added a matrix-free Newton DFT solver that uses GMRES for the linear subproblem. [#75](https://github.com/feos-org/feos/pull/75)
- Solver metrics like execution time and the residuals are automatically logged during every computation. The results can be accessed via the `solver_log` field of `DFTProfile`. [#75](https://github.com/feos-org/feos/pull/75)

### Changed
- Added `Send` and `Sync` as supertraits to `HelmholtzEnergyFunctional` and all related traits. [#57](https://github.com/feos-org/feos/pull/57)
- Renamed the `3d_dft` feature to `rayon` in accordance to the other feos crates. [#62](https://github.com/feos-org/feos/pull/62)
- internally rewrote the implementation of the Euler-Lagrange equation to use a bulk density instead of the chemical potential as variable. [#60](https://github.com/feos-org/feos/pull/60)
- Renamed the parameter `beta` of the Picard iteration and Anderson mixing solvers to `damping_coefficient`. [#75](https://github.com/feos-org/feos/pull/75)
- Removed generics for units in all structs and traits in favor of static SI units. [#115](https://github.com/feos-org/feos/pull/115)

### Fixed
- Fixed the sign of vector weighted densities in Cartesian and cylindrical geometries. The wrong sign had no effects on the Helmholtz energy and thus on density profiles. [#120](https://github.com/feos-org/feos/pull/120)

### Packaging
- Updated `pyo3` and `numpy` dependencies to 0.18. [#119](https://github.com/feos-org/feos/pull/119)
- Updated `quantity` dependency to 0.6. [#119](https://github.com/feos-org/feos/pull/119)
- Updated `num-dual` dependency to 0.6. [#119](https://github.com/feos-org/feos/pull/119)

## [0.3.2] - 2022-10-13
### Changed
- The 3D DFT functionalities (3D pores, solvation free energy, free-energy-averaged potentials) are hidden behind the new `3d_dft` feature. For the Python package, the feature is always enabled. [#51](https://github.com/feos-org/feos/pull/51)

## [0.3.1] - 2022-08-26
### Changed
- `Adsorption::equilibrium_isotherm`, `Adsorption::adsorption_isotherm`, and `Adsorption::desorption_isotherm` only accept `SIArray1`s as input for the pressure discretization. [#50](https://github.com/feos-org/feos/pull/50)
- For the calculation of desorption isotherms and the filled branches of equilibrium isotherms, the density profiles are initialized using a liquid bulk density. [#50](https://github.com/feos-org/feos/pull/50)

## [0.3.0] - 2022-07-13
### Added
- Added getters for the fields of `Pore1D` in Python. [#30](https://github.com/feos-org/feos-dft/pull/30)
- Added Steele potential with custom combining rules. [#29](https://github.com/feos-org/feos/pull/29)

### Changed
- Made FMT functional more flexible w.r.t. the shape of the weight functions. [#31](https://github.com/feos-org/feos-dft/pull/31)
- Changed interface of `PairCorrelationFunction` to facilitate the calculation of pair correlation functions in mixtures. [#29](https://github.com/feos-org/feos-dft/pull/29)

### Removed
- Moved the implementation of fundamental measure theory to the `feos` crate. [#27](https://github.com/feos-org/feos/pull/27)

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
