# Benchmarks

This directory contains different benchmarks.
For best performance, use the `release-lto` profile.
Depending on the benchmark, you might have to consider different Cargo `features` as denoted in the table below.

For example, to run the benchmarks in `dual_numbers`, which uses PC-SAFT, use

```
cargo bench --profile=release-lto --features=pcsaft --bench=dual_numbers
``` 

|Name|Description|Cargo features|
|--|--|--|
|`dual_numbers`|Helmholtz energy function evaluated using `StateHD` with different dual number types.|`pcsaft`|
|`state_properties`|Properties of `State`. Including state creation using the natural variables of the Helmholtz energy (no density iteration).|`pcsaft`|