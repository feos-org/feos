# State

The `State` class represents a thermodynamic state at given conditions and provides access to all thermodynamic properties.

```python
# Todo example
```

::: feos.State
    options:
      members:
        - __init__
        - critical_point
        - critical_point_pure  
        - critical_point_binary
        - spinodal
        - dp_dv
        - dp_drho
        - dp_dt
        - dp_dni
        - d2p_dv2
        - d2p_drho2
        - partial_molar_volume
        - chemical_potential
        - chemical_potential_contributions
        - dmu_dt
        - dmu_dni
        - ln_phi
        - ln_phi_pure_liquid
        - ln_symmetric_activity_coefficient
        - dln_phi_dt
        - dln_phi_dp
        - dln_phi_dnj
        - thermodynamic_factor
        - henrys_law_constant
        - henrys_law_constant_binary
        - molar_isochoric_heat_capacity
        - dc_v_dt
        - molar_isobaric_heat_capacity
        - entropy
        - ds_dt
        - molar_entropy
        - partial_molar_entropy
        - enthalpy
        - molar_enthalpy
        - partial_molar_enthalpy
        - helmholtz_energy
        - molar_helmholtz_energy
        - residual_helmholtz_energy_contributions
        - gibbs_energy
        - molar_gibbs_energy
        - internal_energy
        - molar_internal_energy
        - joule_thomson
        - isentropic_compressibility
        - isothermal_compressibility
        - isenthalpic_compressibility
        - thermal_expansivity
        - grueneisen_parameter
        - structure_factor
        - speed_of_sound
        - mass
        - total_mass
        - mass_density
        - massfracs
        - specific_helmholtz_energy
        - specific_entropy
        - specific_internal_energy
        - specific_gibbs_energy
        - specific_enthalpy
        - specific_isochoric_heat_capacity
        - specific_isobaric_heat_capacity
        - stability_analysis
        - is_stable
      summary: 
        attributes: true
        functions: true
---

::: feos.StateVec
    options:
      members:
        - __init__
        - __len__
        - __getitem__
        - molar_entropy
        - specific_entropy
        - molar_enthalpy
        - specific_enthalpy
        - to_dict
        - temperature
        - pressure
        - compressibility
        - density
        - moles
        - molefracs
        - mass_density
        - massfracs
      summary: 
        attributes: true
        functions: true

