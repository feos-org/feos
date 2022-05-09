# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2022-05-??
### Added
- Added [gc-PC-SAFT](https://github.com/feos-org/feos-gc-pcsaft) equation of state and Helmholtz energy functional.
- Added [PeTS](https://github.com/feos-org/feos-pets) equation of state and Helmholtz energy functional.

### Changed
- Combined all equations of state into a single Python class `EquationOfState` and all Helmholtz energy functionals into the Python class `HelmholtzEnergyFunctional`. [#11](https://github.com/feos-org/feos/pull/11)


### Packaging
- Updated [`quantity`](https://github.com/itt-ustutt/quantity/blob/master/CHANGELOG.md) to 0.5.
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
