# Parameters

This directory contains files with parameters for SAFT-VRQ Mie equation of state.
The files named according to the pattern `NameYear.json` correspond to published parameters. The corresponding publication is provided in the [`literature.bib`](literature.bib) file.

- Following [Aasen et al. (2019)](https://aip.scitation.org/doi/full/10.1063/1.5111364) we provide separate parameters for the specific orders of the Feynman-Hibbs correction. The compatible order is denoted in the "model" column.

## Pure Substance Parameters
|file|model|publication|
|-|-|:-:|
[`hammer2023.json`](hammer2023.json) | first-order Feynman-Hibbs correction parameters used for (but not adjusted to) surface tensions. | TBD

## Binary Parameters

|file|model|publication|
|-|-|:-:|
[`hammer2023_binary.json`](hammer2023_binary.json) | first-order Feynman-Hibbs correction binary interaction parameters used for (but not adjusted to) surface tensions. | TBD
