# Properties

(Bulk) equilibrium properties can be calculated as derivatives of a thermodynamic potential. In the case of equations of state, this thermodynamic potential is the Helmholtz energy $A$ as a function of its characteristic variables temperature $T$, volume $V$, and number of particles $N_i$. Examples for common (measurable) properties that are calculated from an equation of state are the pressure

$$p=-\left(\frac{\partial A}{\partial V}\right)_{T,N_i}$$

and the isochoric heat capacity

$$c_V=\frac{1}{N}\left(\frac{\partial U}{\partial T}\right)_{V,N_i}=\frac{T}{N}\left(\frac{\partial S}{\partial T}\right)_{V,N_i}=-\frac{T}{N}\left(\frac{\partial^2 A}{\partial T^2}\right)_{V,N_i}$$

with the total number of particles $N=\sum_i N_i$, the internal energy $U$, and the entropy $S$.

## Residual properties

In most cases, the total value of a property is required because it is the one which can actually be measured. However, for some applications, e.g. phase equilibria or entropy scaling, residual properties are useful. Residual properties are based on the separation of the Helmholtz energy in an ideal gas ($\mathrm{ig}$) contribution and a residual ($\mathrm{res}$) contribution:

$$A=A^\mathrm{ig}+A^\mathrm{res}$$

The ideal gas contribution contains only kinetic and intramolecular energies and can be derived from statistical mechanics as

$$A^\mathrm{ig}(T,V,N_i)=RT\sum_iN_i\left(\ln\left(\frac{N_i\Lambda_i^3}{V}\right)-1\right)$$(eqn:a_ig)

with the thermal de Broglie wavelength $\Lambda_i$ that only depends on the temperature. Residual properties depend on the reference state. The two commonly used reference states are at constant volume or at constant pressure, i.e.,

$$A=A^\mathrm{ig,V}(T,V,N_i)+A^\mathrm{res,V}(T,V,N_i)=A^\mathrm{ig,p}(T,p,N_i)+A^\mathrm{res,p}(T,p,N_i)$$(eqn:a_res)

Because the Helmholtz energy is expressed in $T$, $V$ and $N_i$, residual properties at constant volume can be evaluated straightforwardly. If a property $X$ is calculated from the Helmholtz energy via the differential operator $\mathcal{D}$, i.e., $X=\mathcal{D}\left(A\right)$, then the residual contributions is calculated from

$$X^\mathrm{res,V}=\mathcal{D}\left(A\right)-\mathcal{D}\left(A^\mathrm{ig,V}\right)$$

In cases where the operator $\mathcal{D}$ is linear, the expression can be simplified to

$$X^\mathrm{res,V}=\mathcal{D}\left(A-A^\mathrm{ig,V}\right)=\mathcal{D}\left(A^\mathrm{res,V}\right)$$

For residual properties at constant pressure, the reference state has to be evaluated for an ideal gas volume that corresponds to the constant pressure

$$A^\mathrm{ig,p}(T,p,N_i)=A^\mathrm{ig,V}\left(T,V^\mathrm{ig}(T,p,N_i),N_i\right)=A^\mathrm{ig,V}\left(T,\frac{NRT}{p},N_i\right)$$

Then the residual contribution for $X$ can be evaluated as

$$X^\mathrm{res,p}=\mathcal{D}\left(A\right)-\mathcal{D}\left(A^\mathrm{ig,p}\right)$$

For linear operators $\mathcal{D}$ eqs. {eq}`eqn:a_ig` and {eq}`eqn:a_res` can be used to simplify the expression

$$X^\mathrm{res,p}=\mathcal{D}\left(A-A^\mathrm{ig,p}\right)=\mathcal{D}\left(A\right)-\mathcal{D}\left(NRT\ln Z\right)$$

with the compressiblity factor $Z=\frac{pV}{NRT}$.

## List of properties available in $\text{FeO}_\text{s}$

The table below lists all properties that are available in $\text{FeO}_\text{s}$, their definition, and whether they can be evaluated as residual contributions as well.

| Name | definition | residual? |
|-|:-:|-|
| Pressure $p$ | $-\left(\frac{\partial A}{\partial V}\right)_{T,N_i}$ | yes |
| Compressibility factor $Z$ | $\frac{pV}{NRT}$ | yes |
| Partial derivative of pressure w.r.t. volume | $\left(\frac{\partial p}{\partial V}\right)_{T,N_i}$ | yes |
| Partial derivative of pressure w.r.t. density | $\left(\frac{\partial p}{\partial \rho}\right)_{T,N_i}$ | yes |
| Partial derivative of pressure w.r.t. temperature | $\left(\frac{\partial p}{\partial T}\right)_{V,N_i}$ | yes |
| Partial derivative of pressure w.r.t. moles | $\left(\frac{\partial p}{\partial N_i}\right)_{T,V,N_j}$ | yes |
| Second partial derivative of pressure w.r.t. volume | $\left(\frac{\partial^2 p}{\partial V^2}\right)_{T,N_i}$ | yes |
| Second partial derivative of pressure w.r.t. density | $\left(\frac{\partial^2 p}{\partial \rho^2}\right)_{T,N_i}$ | yes |
| Partial molar volume $v_i$ | $\left(\frac{\partial V}{\partial N_i}\right)_{T,p,N_j}$ | yes |
| Chemical potential $\mu_i$ | $\left(\frac{\partial A}{\partial N_i}\right)_{T,V,N_j}$ | yes |
| Partial derivative of chemical potential w.r.t. temperature | $\left(\frac{\partial\mu_i}{\partial T}\right)_{V,N_i}$ | yes |
| Partial derivative of chemical potential w.r.t. moles | $\left(\frac{\partial\mu_i}{\partial N_j}\right)_{V,N_k}$ | yes |
| Logarithmic fugacity coefficient $\ln\varphi_i$ | $\beta\mu_i^\mathrm{res}\left(T,p,\lbrace N_i\rbrace\right)$ | no |
| Pure component logarithmic fugacity coefficient $\ln\varphi_i^\mathrm{pure}$ | $\lim_{x_i\to 1}\ln\varphi_i$ | no |
| Logarithmic (symmetric) activity coefficient $\ln\gamma_i$ | $\ln\left(\frac{\varphi_i}{\varphi_i^\mathrm{pure}}\right)$ | no |
| Partial derivative of the logarithmic fugacity coefficient w.r.t. temperature | $\left(\frac{\partial\ln\varphi_i}{\partial T}\right)_{p,N_i}$ | no |
| Partial derivative of the logarithmic fugacity coefficient w.r.t. pressure | $\left(\frac{\partial\ln\varphi_i}{\partial p}\right)_{T,N_i}=\frac{v_i^\mathrm{res,p}}{RT}$ | no |
| Partial derivative of the logarithmic fugacity coefficient w.r.t. moles |  $\left(\frac{\partial\ln\varphi_i}{\partial N_j}\right)_{T,p,N_k}$ | no |
| Thermodynamic factor $\Gamma_{ij}$ | $\delta_{ij}+x_i\left(\frac{\partial\ln\varphi_i}{\partial x_j}\right)_{T,p,\Sigma}$ | no |
| Molar isochoric heat capacity $c_v$ | $\left(\frac{\partial u}{\partial T}\right)_{V,N_i}$ | yes |
| Partial derivative of the molar isochoric heat capacity w.r.t. temperature | $\left(\frac{\partial c_V}{\partial T}\right)_{V,N_i}$ | yes |
| Molar isobaric heat capacity $c_p$ | $\left(\frac{\partial h}{\partial T}\right)_{p,N_i}$ | yes |
| Entropy $S$ | $-\left(\frac{\partial A}{\partial T}\right)_{V,N_i}$ | yes |
| Partial derivative of the entropy w.r.t. temperature | $\left(\frac{\partial S}{\partial T}\right)_{V,N_i}$ | yes |
| Molar entropy $s$ | $\frac{S}{N}$ | yes |
| Enthalpy $H$ | $A+TS+pV$ | yes |
| Molar enthalpy $h$ | $\frac{H}{N}$ | yes |
| Helmholtz energy $A$ | | yes |
| Molar Helmholtz energy $a$ | $\frac{A}{N}$ | yes |
| Internal energy $U$ | $A+TS$ | yes |
| Molar internal energy $u$ | $\frac{U}{N}$ | yes |
| Gibbs energy $G$ | $A+pV$ | yes |
| Molar Gibbs energy $g$ | $\frac{G}{N}$ | yes |
| Partial molar entropy $s_i$ | $\left(\frac{\partial S}{\partial N_i}\right)_{T,p,N_j}$ | yes |
| Partial molar enthalpy $h_i$ | $\left(\frac{\partial H}{\partial N_i}\right)_{T,p,N_j}$ | yes |
| Joule Thomson coefficient $\mu_\mathrm{JT}$ | $\left(\frac{\partial T}{\partial p}\right)_{H,N_i}$ | no |
| Isentropic copmressibility $\kappa_s$ | $-\frac{1}{V}\left(\frac{\partial V}{\partial p}\right)_{S,N_i}$ | no |
| Isothermal copmressibility $\kappa_T$ | $-\frac{1}{V}\left(\frac{\partial V}{\partial p}\right)_{T,N_i}$ | no |
| (Static) structure factor $S(0)$ | $S(0)=RT\left(\frac{\partial\rho}{\partial p}\right)_{T,N_i}$ | no |

## Additional properties for fluids with known molar weights

If the Helmholtz energy model includes information about the molar weigt $MW_i$ of each species, additional properties are available in $\text{FeO}_\text{s}$

| Name | definition | residual? |
|-|:-:|-|
| Total molar weight $MW$ | $\sum_ix_iMW_i$ | no |
| Mass of each component $m_i$ | $n_iMW_i$ | no |
| Total mass $m$ | $\sum_im_i=nMW$ | no |
| Mass density $\rho^{(m)}$ | $\frac{m}{V}$ | no |
| Mass fractions $w_i$ | $\frac{m_i}{m}$ | no |
| Specific entropy $s^{(m)}$ | $\frac{S}{m}$ | yes |
| Specific enthalpy $h^{(m)}$ | $\frac{H}{m}$ | yes |
| Specific Helmholtz energy $a^{(m)}$ | $\frac{A}{m}$ | yes |
| Specific internal energy $u^{(m)}$ | $\frac{U}{m}$ | yes |
| Specific Gibbs energy $g^{(m)}$ | $\frac{G}{m}$ | yes |
| Speed of sound $c$ | $\sqrt{\left(\frac{\partial p}{\partial\rho^{(m)}}\right)_{S,N_i}}$ | no |