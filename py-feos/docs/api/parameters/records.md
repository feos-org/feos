# Records

Record classes represent parameter sets for individual components, segments, and their interactions.

::: feos.PureRecord
    options:
      members:
        - from_json_str
        - to_json_str
        - to_dict
      summary: 
        attributes: true
        functions: true

::: feos.SegmentRecord
    options:
      members:
        - from_json_str
        - to_json_str
        - to_dict
        - from_json
      summary: 
        attributes: true
        functions: true

::: feos.BinaryRecord
    options:
      members:
        - from_json_str
        - to_json_str
        - to_dict
      summary: 
        attributes: true
        functions: true

::: feos.BinarySegmentRecord
    options:
      members:
        - from_json_str
        - to_json_str
      summary: 
        attributes: true
        functions: true

::: feos.ChemicalRecord
    options:
      members:
        - from_smiles
      summary: 
        attributes: true
        functions: true

::: feos.SmartsRecord
    options:
      members:
        - from_json_str
        - to_json_str
        - from_json
      summary: 
        attributes: true
        functions: true
