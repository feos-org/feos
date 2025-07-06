# PC(P)-SAFT Parameters

This directory contains files with parameters for PC(P)-SAFT (including gc-PC-SAFT).
The files named according to the pattern `NameYear.json` correspond to published parameters. The corresponding publication is provided in the [`literature.bib`](literature.bib) file.

For most applications, we recommend using the recent parametrization by [Esper et al.](https://doi.org/10.1021/acs.iecr.3c02255) together with binary interaction parameters from [Rehner et al.](https://doi.org/10.1007/s10765-023-03290-3) In Python reading parameters from the two JSON files is done by

```python
from feos.pcsaft import PcSaftParameters

params = PcSaftParameters.from_json(
    ["acetone", "hexane"], 
    "esper2023.json", 
    "rehner2023_binary.json"
)
```
For some applications (e.g., aqueous systems) more specialized parametrizations are required. See below for a full list of PC(P)-SAFT pure-component parameter sets available in the repository.

## Individual Parameters

|file|file binary|description|publication(s)|
|-|-|-|:-:|
[`gross2001.json`](gross2001.json) | | non-associating and non-polar substances| [&#128279;](https://doi.org/10.1021/ie0003887)
[`gross2002.json`](gross2002.json) | | associating substances | [&#128279;](https://doi.org/10.1021/ie010954d)
[`gross2005_fit.json`](gross2005_fit.json) | | quadrupolar substances, quadrupole moment adjusted in regression | [&#128279;](https://doi.org/10.1002/aic.10502)
[`gross2005_literature.json`](gross2005_literature.json) | | quadrupolar substances, quadrupole moment taken from literature | [&#128279;](https://doi.org/10.1002/aic.10502)
[`gross2006.json`](gross2006.json) | | dipolar substances | [&#128279;](https://doi.org/10.1002/aic.10683)
[`loetgeringlin2018.json`](loetgeringlin2018.json) | | 146 components including viscosity parameters | [&#128279;](https://doi.org/10.1021/acs.iecr.7b04871)
[`rehner2020.json`](rehner2020.json) | | water and alcohols with surface tension data included in the regression | [&#128279;](https://doi.org/10.1021/acs.jced.0c00684)
[`eller2022.json`](eller2022.json) | | hydrogen used in subsurface storage | [&#128279;](https://doi.org/10.1029/2021WR030885)
[`esper2023.json`](esper2023.json) | [`rehner2023_binary.json`](rehner2023_binary.json) | 1842 non-associating, associating and polar substances | [&#128279;](https://doi.org/10.1021/acs.iecr.3c02255)[&#128279;](https://doi.org/10.1007/s10765-023-03290-3)

## Group-Contribution (GC) Methods
Parameters can also be constructed from group-contribution methods. In Python only and if you have [`rdkit`](https://pypi.org/project/rdkit/) installed in your environment, you can generate parameters directly from a SMILES code:
```Python
PcSaftParameters.from_json_smiles(
    ["CCC(C)=O"], 
    "sauer2014_smarts.json", 
    "sauer2014_homo.json"
)
```
or
```Python
PcSaftParameters.from_json_smiles(
    [Identifier(name="2-butanone", smiles="CCC(C)=O")], 
    "sauer2014_smarts.json", 
    "sauer2014_homo.json"
)
```
The rules that are applied in the determination of the group counts from SMILES are defined by SMARTS. All GC models that are implemented currently are compatible with the SMARTS defined in  [`sauer2014_smarts.json`](sauer2014_smarts.json).

For a more detailed description of parameter handling in `FeOs`, check out the [example notebook](https://github.com/feos-org/feos/blob/binary_interaction_parameter_files/examples/pcsaft_working_with_parameters.ipynb).

### Parameters for the homosegmented GC method

|file|file binary|description|publication(s)|
|-|-|-|:-:|
[`sauer2014_homo.json`](sauer2014_homo.json) | | group parameters for homosegmented PC-SAFT | [&#128279;](https://doi.org/10.1021/ie502203w) |
[`loetgeringlin2015_homo.json`](loetgeringlin2015_homo.json) | | Sauer et al. plus viscosity parameter | [&#128279;](https://doi.org/10.1021/acs.iecr.5b01698)
[`rehner2023_homo.json`](rehner2023_homo.json) | [`rehner2023_homo_binary.json`](rehner2023_homo_binary.json) | Sauer et al. plus induced association | [&#128279;](https://doi.org/10.1007/s10765-023-03290-3) |

### Parameters for the heterosegmented GC method (gc-PC-SAFT)

|file|file binary|description|publication(s)|
|-|-|-|:-:|
[`sauer2014_hetero.json`](sauer2014_hetero.json) | | group parameters for gc-PC-SAFT | [&#128279;](https://doi.org/10.1021/ie502203w) |
[`rehner2023_hetero.json`](rehner2023_hetero.json) | [`rehner2023_hetero_binary.json`](rehner2023_hetero_binary.json) | Sauer et al. plus induced association | [&#128279;](https://doi.org/10.1007/s10765-023-03290-3) |
