# Parameters

This directory contains files with parameters for PC-SAFT.
The files named according to the pattern `NameYear.json` correspond to published parameters. The corresponding publication is provided in the [`literature.bib`](literature.bib) file.

- We provide *regular* PC-SAFT parameters, i.e. parameters for substances that are *not* described via group contribution (GC) methods.
- Substances that can be described via GC approaches are given in `gc_substances.json` alongside their segment and bond information.
- Segment SAFT parameters are given in files denoted as `NameYear_homo.json` or `NameYear_hetero.json` for homo-segmented and hetero-segmented GC methods, respectively.

## List of Substances with Segment Information

|file||
|-|-|
[`gc_substances.json`](gc_substances.json) | Chemical structure of substances to be used in group contribution methods |


## Regular Parameters

|file|model|publication|
|-|-|:-:|
[`gross2001.json`](gross2001.json) | non-associating and non-polar substances| [&#128279;](https://doi.org/10.1021/ie0003887)
[`gross2002.json`](gross2002.json) | associating substances | [&#128279;](https://doi.org/10.1021/ie010954d)
[`gross2005_fit.json`](gross2005_fit.json) | quadrupolar substances, quadrupole adjusted in regression | [&#128279;](https://doi.org/10.1002/aic.10502)
[`gross2005_literature.json`](gross2005_literature.json) | quadrupolar substances, quadrupole moment taken from literature | [&#128279;](https://doi.org/10.1002/aic.10502)
[`gross2006.json`](gross2006.json) | dipolar substances | [&#128279;](https://doi.org/10.1002/aic.10683)
[`rehner2020.json`](rehner2020.json) | water and alcohols with surface tension data included in the regression | [&#128279;](https://doi.org/10.1021/acs.jced.0c00684)

## Group-Contribution Parameters

|file|model|publication|
|-|-|:-:|
[`sauer2014_homo.json`](sauer2014_homo.json) | GC segment parameters for homo segmented PC-SAFT | [&#128279;](https://doi.org/10.1021/ie502203w) |
[`sauer2014_hetero.json`](sauer2014_hetero.json) | GC segment parameters for hetero segmented PC-SAFT | [&#128279;](https://doi.org/10.1021/ie502203w)

