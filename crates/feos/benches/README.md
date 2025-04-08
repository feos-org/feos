# Benchmarks

This directory contains different benchmarks.
For best performance, use the `release-lto` profile.

For example, to run the benchmarks in `dual_numbers`, which uses PC-SAFT, use

```
cargo bench --profile=release-lto --bench=dual_numbers
``` 

|Name|Description|
|--|--|
|`dual_numbers`|Helmholtz energy function evaluated using `StateHD` with different dual number types using the PC-SAFT equation of state.|
|`dual_numbers_saftvrmie`|Helmholtz energy function evaluated using `StateHD` with different dual number types using the SAFT-VR-Mie equation of state.|
|`state_properties`|Properties of `State`. Including state creation using the natural variables of the Helmholtz energy (no density iteration).|
|`state_creation`|Different constructors of `State` and `PhaseEquilibrium` including critical point calculations. For pure substances and mixtures.|
|`contributions`|Helmholtz energy evaluated for various binary mixtures with different Helmholtz energy contributions. |
|`dft_pore`|Calculation of density profiles in pores using different functionals and bulk conditions. For pure substances, mixtures and heterosegmented chains.|