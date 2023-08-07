# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased
### Added
- Added new functions `isenthalpic_compressibility`, `thermal_expansivity` and `grueneisen_parameter` to `State`. [#154](https://github.com/feos-org/feos/pull/154)
- Readded `PhaseEquilibrium::new_npt` to the public interface in Rust and Python.  [#164](https://github.com/feos-org/feos/pull/164)
- Added `Components`, `Residual`, `IdealGas` and `DeBroglieWavelength` traits to decouple ideal gas models from residual models. [#158](https://github.com/feos-org/feos/pull/158)
- Added `JobackParameters` struct that implements `Parameters` including Python bindings. [#158](https://github.com/feos-org/feos/pull/158)
- Added `Parameter::from_model_records` as a simpler interface to generate parameters.  [#169](https://github.com/feos-org/feos/pull/169)
- Added optional `Phase` argument for constructors of dynamic properties of `DataSet`s.  [#174](https://github.com/feos-org/feos/pull/174)
- Added `molar_weight` method to `Residual` trait. [#177](https://github.com/feos-org/feos/pull/177)
- Added molar versions for entropy, enthalpy, etc. for residual properties. [#177](https://github.com/feos-org/feos/pull/177)

### Changed
- Changed constructors of `Parameter` trait to return `Result`s. [#161](https://github.com/feos-org/feos/pull/161)
- Changed `EquationOfState` from a trait to a `struct` that is generic over `Residual` and `IdealGas` and implements all necessary traits to be used as equation of state including the ideal gas contribution. [#158](https://github.com/feos-org/feos/pull/158)
- The `Parameter` trait no longer has an associated type `IdealGas`. [#158](https://github.com/feos-org/feos/pull/158)
- Split properties of `State` into those that require the `Residual` trait (`residual_properties.rs`) and those that require both `Residual + IdealGas` (`properties.rs`). [#158](https://github.com/feos-org/feos/pull/158)
- State creation routines are split into those that can be used with `Residual` and those that require `Residual + IdealGas`. [#158](https://github.com/feos-org/feos/pull/158)
- `Contributions` enum no longer includes the `ResidualNpt` variant. `ResidualNvt` variant is renamed to `Residual`. [#158](https://github.com/feos-org/feos/pull/158)
- Moved `Verbosity` and `SolverOption` from `phase_equilibria` module to `lib.rs`. [#158](https://github.com/feos-org/feos/pull/158)
- Moved `StateVec` into own file and module. [#158](https://github.com/feos-org/feos/pull/158)
- Ideal gas and residual Helmholtz energy models can now be separately implemented in Python via the `PyIdealGas` and `PyResidual` structs. [#158](https://github.com/feos-org/feos/pull/158)
- Bubble and dew point iterations will not attempt a second iteration if no solution is found for the given initial pressure. [#166](https://github.com/feos-org/feos/pull/166)
- Made the binary records in the constructions and getters of the `Parameter` trait optional. [#169](https://github.com/feos-org/feos/pull/169)
- Changed the second argument of `new_binary` in Python from a `BinaryRecord` to the corresponding binary model record (analogous to the Rust implementation). [#169](https://github.com/feos-org/feos/pull/169)
- Renamed `c_v` and `c_p` to `isochoric_heat_capacity` and `isobaric_heat_capacity`, respectively, and added prefixes for molar and specific properties. [#177](https://github.com/feos-org/feos/pull/177)
- `State.helmholtz_energy_contributions` in Python now accepts optional `Contributions`. [#177](https://github.com/feos-org/feos/pull/177)
- Changed `StateVec` Python getters for entropy and enthalpy to functions that accept optional `Contributions`. [#177](https://github.com/feos-org/feos/pull/177)
- `PhaseDiagram.to_dict` in Python now accepts optional `Contributions` and includes mass specific properties. [#177](https://github.com/feos-org/feos/pull/177)
- Replaced the run-time unit checks from the `quantity` crate with compile-time unit checks with custom implementations in the new `feos_core.si` module. [#181](https://github.com/feos-org/feos/pull/181)

### Removed
- Removed `EquationOfState` trait. [#158](https://github.com/feos-org/feos/pull/158)
- Removed ideal gas dependencies from `PureRecord` and `SegmentRecord`. [#158](https://github.com/feos-org/feos/pull/158)
- Removed Python getter and setter functions and optional arguments for ideal gas records in macros. [#158](https://github.com/feos-org/feos/pull/158)
- Removed `MolarWeight` trait. [#177](https://github.com/feos-org/feos/pull/177)

### Fixed
- The vapor and liquid states in a bubble or dew point iteration are assigned correctly according to the inputs, rather than based on the mole density which can be incorrect for mixtures with large differences in molar weights. [#166](https://github.com/feos-org/feos/pull/166)

### Packaging
- Updated `num-dual` dependency to 0.7. [#137](https://github.com/feos-org/feos/pull/137)

## [0.4.2] - 2023-04-03
### Fixed
- Fixed a wrong reference state in the implementation of the Peng-Robinson equation of state. [#151](https://github.com/feos-org/feos/pull/151)

## [0.4.1] - 2023-03-20
### Fixed
- Fixed a bug that caused the bubble and dew point solvers to ignore the initial values for the opposing phase given by the user if no initial value for temperature/pressure was also given. [#138](https://github.com/feos-org/feos/pull/138)

## [0.4.0] - 2023-01-27
### Added
- Added `PhaseDiagram::par_pure` that uses rayon to calculate phase diagrams in parallel. [#57](https://github.com/feos-org/feos/pull/57)
- Added `StateVec::moles` getter. [#113](https://github.com/feos-org/feos/pull/113)
- Added public constructors `PhaseDiagram::new` and `StateVec::new` that allow the creation of the respective structs from a list of `PhaseEquilibrium`s or `State`s in Rust and Python. [#113](https://github.com/feos-org/feos/pull/113)
- Added new variant `EosError::Error` which allows dispatching generic errors that are not covered by the existing variants. [#98](https://github.com/feos-org/feos/pull/98)
- Added `binary_records` getter for parameter classes in Python. [#104](https://github.com/feos-org/feos/pull/104)
- Added `BinaryRecord::from_json` and `BinarySegmentRecord::from_json` that read a list of records from a file. [#104](https://github.com/feos-org/feos/pull/104)

### Changed
- Added `Sync` and `Send` as supertraits to `EquationOfState`. [#57](https://github.com/feos-org/feos/pull/57)
- Added `Dual2_64` dual number to `HelmholtzEnergy` trait to facilitate efficient non-mixed second order derivatives. [#94](https://github.com/feos-org/feos/pull/94)
- Renamed `PartialDerivative::Second` to `PartialDerivative::SecondMixed`. [#94](https://github.com/feos-org/feos/pull/94)
- Added `PartialDerivative::Second` to enable non-mixed second order partial derivatives. [#94](https://github.com/feos-org/feos/pull/94)
- Changed `dp_dv_` and `ds_dt_` to use `Dual2_64` instead of `HyperDual64`. [#94](https://github.com/feos-org/feos/pull/94)
- Added `get_or_insert_with_d2_64` to `Cache`. [#94](https://github.com/feos-org/feos/pull/94)
- The critical point algorithm now uses vector dual numbers to reduce the number of model evaluations and computation times. [#96](https://github.com/feos-org/feos/pull/96)
- Renamed `State::molar_volume` to `State::partial_molar_volume` and `State::ln_phi_pure` to `State::ln_phi_pure_liquid`. [#107](https://github.com/feos-org/feos/pull/107)
- Added a const generic parameter to `PhaseDiagram` that accounts for the number of phases analogously to `PhaseEquilibrium`. [#113](https://github.com/feos-org/feos/pull/113)
- Removed generics for units in all structs and traits in favor of static SI units. [#115](https://github.com/feos-org/feos/pull/115)

### Fixed
- Automatically generate all required data types to calculate higher order derivatives for equations of state implemented in Python. [#114](https://github.com/feos-org/feos/pull/114)

### Packaging
- Updated `pyo3` and `numpy` dependencies to 0.18. [#119](https://github.com/feos-org/feos/pull/119)
- Updated `quantity` dependency to 0.6. [#119](https://github.com/feos-org/feos/pull/119)
- Updated `num-dual` dependency to 0.6. [#119](https://github.com/feos-org/feos/pull/119)

### Fixed
- `Parameter::from_segments` only calculates binary interaction parameters for unlike molecules. [#104](https://github.com/feos-org/feos/pull/104)

## [0.3.1] - 2022-08-25
### Added
- Added `State::spinodal` that calculates both spinodal points for a given temperature and composition using the same objective function that is also used in the critical point calculation. [#23](https://github.com/feos-org/feos/pull/23)
- Added `PhaseDiagram::bubble_point_line`, `PhaseDiagram::dew_point_line`, and `PhaseDiagram::spinodal` to calculate phase envelopes for mixtures with fixed compositions. [#23](https://github.com/feos-org/feos/pull/23)

### Changed
- Made binary parameters in `from_records` Python routine an `Option`. [#35](https://github.com/feos-org/feos/pull/35)
- Added panic with message when parsing missing Identifiers variants. [#35](https://github.com/feos-org/feos/pull/35)
- Generalized the initialization routines for pure component VLEs at given temperature to multicomponent systems. [#23](https://github.com/feos-org/feos/pull/23)
- Increased the default number of maximum iterations for binary critical point calculations from 50 to 200. [#48](https://github.com/feos-org/feos/pull/48)

### Removed
- Removed the (internal) `SpinodalPoint` struct that was used within density iterations in favor of a simpler interface. [#23](https://github.com/feos-org/feos/pull/23)

### Fixed
- Avoid panicking when calculating `ResidualNpt` properties of states with negative pressures. [#42](https://github.com/feos-org/feos/pull/42)

## [0.3.0] - 2022-06-13
### Added
- Added `pure_records` getter in the `impl_parameter` macro. [#54](https://github.com/feos-org/feos-core/pull/54)
- Implemented `Deref` and `IntoIterator` for `StateVec` for additional vector functionalities of the `StateVec`. [#55](https://github.com/feos-org/feos-core/pull/55) 
- Added `StateVec.__len__` and `StateVec.__getitem__` to allow indexing and iterating over `StateVec`s in Python. [#55](https://github.com/feos-org/feos-core/pull/55)
- Added `SegmentCount` trait that allows the construction of parameter sets from arbitrary chemical records. [#56](https://github.com/feos-org/feos-core/pull/56)
- Added `ParameterHetero` trait to generically provide utility functions for parameter sets of heterosegmented Helmholtz energy models. [#56](https://github.com/feos-org/feos-core/pull/56)

### Changed
- Changed datatype for binary parameters in interfaces of the `from_records` and `new_binary` methods for parameters to take either numpy arrays of `f64` or a list of `BinaryRecord` as input. [#54](https://github.com/feos-org/feos-core/pull/54)
- Modified `PhaseDiagram.to_dict` function in Python to account for pure components and mixtures. [#55](https://github.com/feos-org/feos-core/pull/55)
- Changed `StateVec` to a tuple struct. [#55](https://github.com/feos-org/feos-core/pull/55)
- Made `cas` field of `Identifier` optional. [#56](https://github.com/feos-org/feos-core/pull/56)
- Added type parameter to `FromSegments` and made its `from_segments` function fallible for more control over model limitations. [#56](https://github.com/feos-org/feos-core/pull/56)
- Reverted `ChemicalRecord` back to a struct that only contains the structural information (and not segment and bond counts). [#56](https://github.com/feos-org/feos-core/pull/56)
- Made `IdentifierOption` directly usable in Python using `PyO3`'s new `#[pyclass]` for fieldless enums feature. [#58](https://github.com/feos-org/feos-core/pull/58)

## [0.2.0] - 2022-04-12
### Added
- Added conversions between `ParameterError` and `EosError` to improve the error messages in some cases. [#40](https://github.com/feos-org/feos-core/pull/40)
- Added new struct `StateVec`, that gives easy access to properties of lists of states, e.g. in phase diagrams. [#48](https://github.com/feos-org/feos-core/pull/48)
- Added `ln_symmetric_activity_coefficient` and `ln_phi_pure` to the list of state properties that can be calculated. [#50](https://github.com/feos-org/feos-core/pull/50)

### Changed
- Removed `State` from `EntropyScaling` trait and adjusted associated methods to use temperature, volume and moles instead of state. [#36](https://github.com/feos-org/feos-core/pull/36)
- Replaced the outer loop iterations for the critical point of binary systems with dedicated algorithms. [#34](https://github.com/feos-org/feos-core/pull/34)
- Renamed `VLEOptions` to `SolverOptions`. [#38](https://github.com/feos-org/feos-core/pull/38)
- Renamed methods of `StateBuilder` and the parameters in the `State` constructor in python to `molar_enthalpy`, `molar_entropy`, and `molar_internal_energy`.  [#35](https://github.com/feos-org/feos-core/pull/35)
- Removed `PyContributions` and `PyVerbosity` in favor of a simpler implementation using `PyO3`'s new `#[pyclass]` for fieldless enums feature. [#41](https://github.com/feos-org/feos-core/pull/41)
- Renamed `Contributions::Residual` to `Contributions::ResidualNvt` and `Contributions::ResidualP` to `Contributions::ResidualNpt`. [#43](https://github.com/feos-org/feos-core/pull/43)
- Renamed macro `impl_vle_state!` to `impl_phase_equilibrium!`. [#48](https://github.com/feos-org/feos-core/pull/48)
- Removed `_t` and `_p` functions in favor of simpler interfaces. The kind of specification (temperature or pressure) is determined from the unit of the argument. [#48](https://github.com/feos-org/feos-core/pull/48)
  - `PhaseEquilibrium::pure_t`, `PhaseEquilibrium::pure_p` -> `PhaseEquilibrium::pure`
  - `PhaseEquilibrium::vle_pure_comps_t`, `PhaseEquilibrium::vle_pure_comps_p` -> `PhaseEquilibrium::vle_pure_comps`\
  The `PhaseEquilibria` returned by this function now have the same number of components as the (mixture) eos, that it is called with.
  - `PhaseEquilibrium::bubble_point_tx`, `PhaseEquilibrium::bubble_point_px` -> `PhaseEquilibrium::bubble_point`
  - `PhaseEquilibrium::dew_point_tx`, `PhaseEquilibrium::dew_point_px` -> `PhaseEquilibrium::dew_point`
  - `PhaseEquilibrium::heteroazeotrope_t`, `PhaseEquilibrium::heteroazeotrope_p` -> `PhaseEquilibrium::heteroazeotrope`
  - `State::critical_point_binary_t`, `State::critical_point_binary_p` -> `State::crititcal_point_binary`
- Combined `PhaseDiagramPure` and `PhaseDiagramBinary` into a single struct `PhaseDiagram` and renamed its constructors.  Properties of the phase diagram are available from the `vapor` and `liquid` getters, that return `StateVec`s. [#48](https://github.com/feos-org/feos-core/pull/48)
  - `PhaseDiagramPure::new` -> `PhaseDiagram::pure`
  - `PhaseDiagramBinary::new_txy`, `PhaseDiagramBinary::new_pxy` -> `PhaseDiagram::binary_vle`
  - `PhaseDiagramBinary::new_txy_lle`, `PhaseDiagramBinary::new_pxy_lle` -> `PhaseDiagram::lle`
  - `PhaseDiagramHetero::new_txy`, `PhaseDiagramHetero::new_pxy` -> `PhaseDiagram::binary_vlle`\
  which still returns an instance of `PhaseDiagramHetero`
- Changed the internal implementation of the Peng-Robinson equation of state to use contributions like the more complex equations of state and removed the suggestion to overwrite the `evaluate_residual` function of `EquationOfState`. [#51](https://github.com/feos-org/feos-core/pull/51)
- Moved the creation of the python module to the `build_wheel` auxilliary crate, so that only the relevant structs and macros are available for the dependents. [#47](https://github.com/feos-org/feos-core/pull/47)

### Removed
- Removed the `utils` module containing `DataSet` and `Estimator` in favor of a separate crate. [#47](https://github.com/feos-org/feos-core/pull/47)

### Packaging
- Updated `pyo3` and `numpy` dependencies to 0.16.
- Updated `num-dual` dependency to 0.5.
- Updated `quantity` dependency to 0.5.

## [0.1.5] - 2022-02-21
### Fixed
- Fixed bug in `predict` of `Estimator`. [#30](https://github.com/feos-org/feos-core/pull/30)

### Added
- Add `pyproject.toml`. [#29](https://github.com/feos-org/feos-core/pull/29)

## [0.1.4] - 2022-02-18
### Fixed
- Fix state constructor for `T`, `p`, `V`, `x_i` specification. [#26](https://github.com/feos-org/feos-core/pull/26)

### Added
- Added method `predict` to `Estimator`. [#27](https://github.com/feos-org/feos-core/pull/27)

### Changed
- Changed method for vapor pressure in `DataSet` to `vapor_pressure` (was `pressure` of VLE liquid phase). [#27](https://github.com/feos-org/feos-core/pull/27)

## [0.1.3] - 2022-01-21
### Added
- Added the following properties to `State`: [#21](https://github.com/feos-org/feos-core/pull/21)
  - `dp_drho` partial derivative of pressure w.r.t. density
  - `d2p_drho2` second partial derivative of pressure w.r.t. density
  - `isothermal_compressibility` the isothermal compressibility
- Read a list of segment records directly from a JSON file. [#22](https://github.com/feos-org/feos-core/pull/22)


## [0.1.2] - 2022-01-10
### Changed
- Changed `ChemicalRecord` to an enum that can hold either the full structural information of a molecule or only segment and bond counts and added an `Identifier`. [#19](https://github.com/feos-org/feos-core/pull/19)
- Removed the `chemical_record` field from `PureRecord` and made `model_record` non-optional. [#19](https://github.com/feos-org/feos-core/pull/19)

## [0.1.1] - 2021-12-22
### Added
- Added `from_multiple_json` function to `Parameter` trait that is able to read parameters from separate JSON files. [#15](https://github.com/feos-org/feos-core/pull/15)

### Packaging
- Updated `pyo3` and `numpy` dependencies to 0.15.
- Updated `quantity` dependency to 0.4.
- Updated `num-dual` dependency to 0.4.
- Removed `ndarray-linalg` and `ndarray-stats` dependencies.
- Removed obsolete features for the selection of the BLAS/LAPACK library.

## [0.1.0] - 2021-12-02
### Added
- Initial release
