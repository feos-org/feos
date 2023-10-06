# Parameters

This directory contains files with parameters for PC-SAFT (including gc-PC-SAFT).
The files named according to the pattern `NameYear.json` correspond to published parameters. The corresponding publication is provided in the [`literature.bib`](literature.bib) file.

- We provide *regular* PC-SAFT parameters, i.e. parameters for substances that are *not* described via group contribution (GC) methods.
- Substances that can be described via GC approaches are given in `gc_substances.json` alongside their segment and bond information.
- Segment SAFT parameters are given in files denoted as `NameYear_homo.json` or `NameYear_hetero.json` for homo-segmented and hetero-segmented GC methods, respectively.

## List of Substances with Segment Information

|file||
|-|-|
[`gc_substances.json`](gc_substances.json) | Chemical structure of substances to be used in group contribution methods |


## Regular Parameters

|file|description|publication|
|-|-|:-:|
[`gross2001.json`](gross2001.json) | non-associating and non-polar substances| [&#128279;](https://doi.org/10.1021/ie0003887)
[`gross2002.json`](gross2002.json) | associating substances | [&#128279;](https://doi.org/10.1021/ie010954d)
[`gross2005_fit.json`](gross2005_fit.json) | quadrupolar substances, quadrupole adjusted in regression | [&#128279;](https://doi.org/10.1002/aic.10502)
[`gross2005_literature.json`](gross2005_literature.json) | quadrupolar substances, quadrupole moment taken from literature | [&#128279;](https://doi.org/10.1002/aic.10502)
[`gross2006.json`](gross2006.json) | dipolar substances | [&#128279;](https://doi.org/10.1002/aic.10683)
[`loetgeringlin2018.json`](loetgeringlin2018.json) | 146 components including viscosity parameters | [&#128279;](https://doi.org/10.1021/acs.iecr.7b04871)
[`rehner2020.json`](rehner2020.json) | water and alcohols with surface tension data included in the regression | [&#128279;](https://doi.org/10.1021/acs.jced.0c00684)
[`eller2022.json`](eller2022.json) | hydrogen used in subsurface storage | [&#128279;](https://doi.org/10.1029/2021WR030885)
[`esper2023.json`](esper2023.json) | 1842 non-associating, associating and polar substances | [&#128279;](https://doi.org/10.1021/acs.iecr.3c02255)

## Group-Contribution Parameters

|file|description|publication(s)|
|-|-|:-:|
[`sauer2014_homo.json`](sauer2014_homo.json) | GC segment parameters for homosegmented PC-SAFT | [&#128279;](https://doi.org/10.1021/ie502203w) |
[`sauer2014_homo_joback.json`](sauer2014_homo.json) | GC segment parameters for homosegmented PC-SAFT including ideal gas parameters | [&#128279;](https://doi.org/10.1021/ie502203w) [&#128279;](https://doi.org/10.1080/00986448708960487)|
[`sauer2014_hetero.json`](sauer2014_hetero.json) | GC segment parameters for heterosegmented PC-SAFT | [&#128279;](https://doi.org/10.1021/ie502203w)
[`sauer2014_hetero_joback.json`](sauer2014_hetero.json) | GC segment parameters for heterosegmented PC-SAFT including ideal gas parameters | [&#128279;](https://doi.org/10.1021/ie502203w) [&#128279;](https://doi.org/10.1080/00986448708960487)
[`loetgeringlin2015_homo.json`](loetgeringlin2018.json) | GC segment parameters for homosegmented PC-SAFT including viscosity parameter | [&#128279;](https://doi.org/10.1021/acs.iecr.5b01698)

