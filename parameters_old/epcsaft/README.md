# ePC-SAFT Parameters

This directory contains files with parameters for ePC-SAFT equation of state.
The files named according to the pattern `NameYear.json` correspond to published parameters. The corresponding publication is provided in the [`literature.bib`](literature.bib) file.

## Notes

- Experimental data for the permittivity has not been part of the original publication and has been added. 
- Ion permittivity has been set to 8.0 as in [Bülow et al. (2021)](https://www.sciencedirect.com/science/article/pii/S0378381221000297).
- Contains only univalent cations and anions from [Held et al. (2014)](https://www.sciencedirect.com/science/article/pii/S0263876214002469) and carbonate. Multivalent ions need to be added.
- Correlation for `k_ij` of water/Na+ and water/K+ has not been adapted from [Held et al. (2014)](https://www.sciencedirect.com/science/article/pii/S0263876214002469). Instead, a constant value at 298.15 K is assumed.

## Pure Substance Parameters
| file                                                                       | model                                    |                       publication                        |
| -------------------------------------------------------------------------- | ---------------------------------------- | :------------------------------------------------------: |
| [`held2014_w_permittivity_added.json`](held2014_w_permittivity_added.json) | parameters for univalent ions and water. | [&#128279;](https://doi.org/10.1016/j.cherd.2014.05.017) |

## Binary Parameters

| file                                           | model                                           |                       publication                        |
| ---------------------------------------------- | ----------------------------------------------- | :------------------------------------------------------: |
| [`held2014_binary.json`](held2014_binary.json) | binary parameters for univalent ions and water. | [&#128279;](https://doi.org/10.1016/j.cherd.2014.05.017) |
