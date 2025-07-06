# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.3] - 2025-05-28
### Changed
- Simplify the `Eigen` trait for better readable trait bounds. [#5](https://github.com/feos-org/feos-ad/pull/5)

## [0.2.2] - 2025-05-28
### Fixed
- Export `Eigen` to be able to calculate critical points for pure components and binary mixtures generically. [#4](https://github.com/feos-org/feos-ad/pull/4)

## [0.2.1] - 2025-04-14
### Added
- Added `StateAD::molar_isochoric_heat_capacity` and `StateAD::molar_isobaric_heat_capacity`. [#3](https://github.com/feos-org/feos-ad/pull/3)

## [0.2.0] - 2025-01-27
### Changed
- Made `PcSaftBinary` generic for associating/non-associating systems. [#2](https://github.com/feos-org/feos-ad/pull/2)

### Removed
- Removed `ChemicalRecord`. [#2](https://github.com/feos-org/feos-ad/pull/2)

## [0.1.1] - 2025-01-21
### Added
- Implemented PC-SAFT for a binary mixture. [#1](https://github.com/feos-org/feos-ad/pull/1)

## [0.1.0] - 2025-01-08
### Added
- Initial release
