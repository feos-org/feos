# Parameters

This directory contains files with parameters for SAFT-VRQ Mie equation of state.
The files named according to the pattern `NameYear.json` correspond to published parameters. The corresponding publication is provided in the [`literature.bib`](literature.bib) file.

## Notes

- Para-hydrogen, ortho-hydrogen, and normal hydrogen (3:1 mixture of ortho- and para-hydrogen) have different parameters. **Use the `name` field of the identifier to distinguish between the different versions**. Using other identifier fields (e.g. `formula` or `smiles`) may lead to the wrong set of parameters being used. 
- Following [Aasen et al. (2019)](https://aip.scitation.org/doi/full/10.1063/1.5111364) we provide separate parameters for the **specific orders** of the Feynman-Hibbs correction. The compatible order is denoted in the "model" column.


## Pure Substance Parameters
|file|model|publication|
|-|-|:-:|
[`aasen2019.json`](aasen2019.json) | first-order Feynman-Hibbs correction parameters. | [&#128279;](https://doi.org/10.1063/1.5111364)
[`hammer2023.json`](hammer2023.json) | first-order Feynman-Hibbs correction parameters used for (but not adjusted to) surface tensions. | [&#128279;](https://doi.org/10.1063/5.0137226)

## Binary Parameters

|file|model|publication|
|-|-|:-:|
[`aasen2020_binary.json`](aasen2020_binary.json) | first-order Feynman-Hibbs correction binary interaction parameters. Compatible with [`aasen2019.json`](aasen2019.json) and [`hammer2023.json`](hammer2023.json) | [&#128279;](https://doi.org/10.1063/1.5136079)
