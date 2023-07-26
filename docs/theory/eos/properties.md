# Properties

(Bulk) equilibrium properties can be calculated as derivatives of a thermodynamic potential. In the case of equations of state, this thermodynamic potential is the Helmholtz energy $A$ as a function of its characteristic variables temperature $T$, volume $V$, and amount of substance of each component $n_i$. Examples for common (measurable) properties that are calculated from an equation of state are the pressure

$$p=-\left(\frac{\partial A}{\partial V}\right)_{T,n_i}$$

and the isochoric heat capacity

$$c_V=\frac{1}{n}\left(\frac{\partial U}{\partial T}\right)_{V,n_i}=\frac{T}{n}\left(\frac{\partial S}{\partial T}\right)_{V,n_i}=-\frac{T}{n}\left(\frac{\partial^2 A}{\partial T^2}\right)_{V,n_i}$$

with the total number of particles $n=\sum_i n_i$, the internal energy $U$, and the entropy $S$.

## Residual properties

In most cases, the total value of a property is required because it is the one which can actually be measured. However, for some applications, e.g. phase equilibria or entropy scaling, residual properties are useful. Residual properties are based on the separation of the Helmholtz energy in an ideal gas ($\mathrm{ig}$) contribution and a residual ($\mathrm{res}$) contribution:

$$A=A^\mathrm{ig}+A^\mathrm{res}$$

The ideal gas contribution contains only kinetic and intramolecular energies and can be derived from statistical mechanics as

$$A^\mathrm{ig}(T,V,n_i)=RT\sum_in_i\left(\ln\left(\frac{n_i\Lambda_i^3}{V}\right)-1\right)$$(eqn:a_ig)

with the thermal de Broglie wavelength $\Lambda_i$ that only depends on the temperature. Residual properties depend on the reference state. The two commonly used reference states are at constant volume or at constant pressure, i.e.,

$$A=A^\mathrm{ig,V}(T,V,n_i)+A^\mathrm{res,V}(T,V,n_i)=A^\mathrm{ig,p}(T,p,n_i)+A^\mathrm{res,p}(T,p,n_i)$$(eqn:a_res)

Because the Helmholtz energy is expressed in $T$, $V$ and $n_i$, residual properties at constant volume can be evaluated straightforwardly. If a property $X$ is calculated from the Helmholtz energy via the differential operator $\mathcal{D}$, i.e., $X=\mathcal{D}\left(A\right)$, then the residual contributions is calculated from

$$X^\mathrm{res,V}=\mathcal{D}\left(A\right)-\mathcal{D}\left(A^\mathrm{ig,V}\right)$$

In cases where the operator $\mathcal{D}$ is linear, the expression can be simplified to

$$X^\mathrm{res,V}=\mathcal{D}\left(A-A^\mathrm{ig,V}\right)=\mathcal{D}\left(A^\mathrm{res,V}\right)$$

For residual properties at constant pressure, the reference state has to be evaluated for an ideal gas volume that corresponds to the constant pressure

$$A^\mathrm{ig,p}(T,p,n_i)=A^\mathrm{ig,V}\left(T,V^\mathrm{ig}(T,p,n_i),n_i\right)=A^\mathrm{ig,V}\left(T,\frac{nRT}{p},n_i\right)$$

Then the residual contribution for $X$ can be evaluated as

$$X^\mathrm{res,p}=\mathcal{D}\left(A\right)-\mathcal{D}\left(A^\mathrm{ig,p}\right)$$

For linear operators $\mathcal{D}$ eqs. {eq}`eqn:a_ig` and {eq}`eqn:a_res` can be used to simplify the expression

$$X^\mathrm{res,p}=\mathcal{D}\left(A-A^\mathrm{ig,p}\right)=\mathcal{D}\left(A^\mathrm{ig,V}\right)-\mathcal{D}\left(nRT\ln Z\right)$$

with the compressiblity factor $Z=\frac{pV}{nRT}$.

For details on how the evaluation of properties from Helmholtz energy models is implemented in $\text{FeO}_\text{s}$ check out the [Rust guide](../../rustguide/core/state.rst).

## List of properties available in $\text{FeO}_\text{s}$

The table below lists all properties that are available in $\text{FeO}_\text{s}$, their definition, and whether they can be evaluated as residual contributions as well.

In general, the evaluation of (total) Helmholtz energies and their derivatives requires a model for the residual Helmholtz energy and a model for the ideal gas contribution, specifically for the temperature dependence of the thermal de Broglie wavelength $\Lambda_i$. However, for many properties like the pressure including its derivatives and fugacity coefficients, the de Broglie wavelength cancels out.

Due to different language paradigms, $\text{FeO}_\text{s}$ handles the ideal gas term slightly different in Rust and Python.
- In **Rust**, if no ideal gas model is provided, users can only evaluate properties for which no ideal gas model is required because the de Broglie wavelength cancels. For those properties that require an ideal gas model but the table below indicates that they can be evaluated as residual, extra functions are provided.
- In **Python**, no additional functions are required, instead the property evaluation will throw an exception if an ideal gas contribution is required but not provided. 

| Name | definition | ideal gas model required? | residual? |
|-|:-:|-|-|
| Total molar weight $MW$ | $\sum_ix_iMW_i$ | no | no |
| Mass of each component $m_i$ | $n_iMW_i$ | no | no |
| Total mass $m$ | $\sum_im_i=nMW$ | no | no |
| Mass density $\rho^{(m)}$ | $\frac{m}{V}$ | no | no |
| Mass fractions $w_i$ | $\frac{m_i}{m}$ | no | no |
| Pressure $p$ | $-\left(\frac{\partial A}{\partial V}\right)_{T,n_i}$ |  no | yes |
| Compressibility factor $Z$ | $\frac{pV}{nRT}$ |  no | yes |
| Partial derivative of pressure w.r.t. volume | $\left(\frac{\partial p}{\partial V}\right)_{T,n_i}$ |  no | yes |
| Partial derivative of pressure w.r.t. density | $\left(\frac{\partial p}{\partial \rho}\right)_{T,n_i}$ |  no | yes |
| Partial derivative of pressure w.r.t. temperature | $\left(\frac{\partial p}{\partial T}\right)_{V,n_i}$ |  no | yes |
| Partial derivative of pressure w.r.t. moles | $\left(\frac{\partial p}{\partial n_i}\right)_{T,V,n_j}$ |  no | yes |
| Second partial derivative of pressure w.r.t. volume | $\left(\frac{\partial^2 p}{\partial V^2}\right)_{T,n_i}$ |  no | yes |
| Second partial derivative of pressure w.r.t. density | $\left(\frac{\partial^2 p}{\partial \rho^2}\right)_{T,n_i}$ |  no | yes |
| Partial molar volume $v_i$ | $\left(\frac{\partial V}{\partial n_i}\right)_{T,p,n_j}$ |  no | no |
| Chemical potential $\mu_i$ | $\left(\frac{\partial A}{\partial n_i}\right)_{T,V,n_j}$ |  yes | yes |
| Partial derivative of chemical potential w.r.t. temperature | $\left(\frac{\partial\mu_i}{\partial T}\right)_{V,n_i}$ |  yes | yes |
| Partial derivative of chemical potential w.r.t. moles | $\left(\frac{\partial\mu_i}{\partial n_j}\right)_{V,n_k}$ |  no | yes |
| Logarithmic fugacity coefficient $\ln\varphi_i$ | $\beta\mu_i^\mathrm{res}\left(T,p,\lbrace n_i\rbrace\right)$ | no | no |
| Pure component logarithmic fugacity coefficient $\ln\varphi_i^\mathrm{pure}$ | $\lim_{x_i\to 1}\ln\varphi_i$ | no | no |
| Logarithmic (symmetric) activity coefficient $\ln\gamma_i$ | $\ln\left(\frac{\varphi_i}{\varphi_i^\mathrm{pure}}\right)$ | no | no |
| Partial derivative of the logarithmic fugacity coefficient w.r.t. temperature | $\left(\frac{\partial\ln\varphi_i}{\partial T}\right)_{p,n_i}$ | no | no |
| Partial derivative of the logarithmic fugacity coefficient w.r.t. pressure | $\left(\frac{\partial\ln\varphi_i}{\partial p}\right)_{T,n_i}=\frac{v_i^\mathrm{res,p}}{RT}$ | no | no |
| Partial derivative of the logarithmic fugacity coefficient w.r.t. moles |  $\left(\frac{\partial\ln\varphi_i}{\partial n_j}\right)_{T,p,n_k}$ | no | no |
| Thermodynamic factor $\Gamma_{ij}$ | $\delta_{ij}+x_i\left(\frac{\partial\ln\varphi_i}{\partial x_j}\right)_{T,p,\Sigma}$ | no | no |
| Molar isochoric heat capacity $c_v$ | $\left(\frac{\partial u}{\partial T}\right)_{V,n_i}$ |  yes | yes |
| Partial derivative of the molar isochoric heat capacity w.r.t. temperature | $\left(\frac{\partial c_V}{\partial T}\right)_{V,n_i}$ |  yes | yes |
| Molar isobaric heat capacity $c_p$ | $\left(\frac{\partial h}{\partial T}\right)_{p,n_i}$ |  yes | yes |
| Entropy $S$ | $-\left(\frac{\partial A}{\partial T}\right)_{V,n_i}$ |  yes | yes |
| Partial derivative of the entropy w.r.t. temperature | $\left(\frac{\partial S}{\partial T}\right)_{V,n_i}$ |  yes | yes |
| Second partial derivative of the entropy w.r.t. temperature | $\left(\frac{\partial^2 S}{\partial T^2}\right)_{V,n_i}$ |  yes | yes |
| Molar entropy $s$ | $\frac{S}{n}$ | yes | yes |
| Specific entropy $s^{(m)}$ | $\frac{S}{m}$ | yes | yes |
| Enthalpy $H$ | $A+TS+pV$ |  yes | yes |
| Molar enthalpy $h$ | $\frac{H}{n}$ |  yes | yes |
| Specific enthalpy $h^{(m)}$ | $\frac{H}{m}$ | yes | yes |
| Helmholtz energy $A$ | |  yes | yes |
| Molar Helmholtz energy $a$ | $\frac{A}{n}$ |  yes | yes |
| Specific Helmholtz energy $a^{(m)}$ | $\frac{A}{m}$ | yes | yes |
| Internal energy $U$ | $A+TS$ |  yes | yes |
| Molar internal energy $u$ | $\frac{U}{n}$ |  yes | yes |
| Specific internal energy $u^{(m)}$ | $\frac{U}{m}$ | yes | yes |
| Gibbs energy $G$ | $A+pV$ |  yes | yes |
| Molar Gibbs energy $g$ | $\frac{G}{n}$ |  yes | yes |
| Specific Gibbs energy $g^{(m)}$ | $\frac{G}{m}$ | yes | yes |
| Partial molar entropy $s_i$ | $\left(\frac{\partial S}{\partial n_i}\right)_{T,p,n_j}$ | yes | no |
| Partial molar enthalpy $h_i$ | $\left(\frac{\partial H}{\partial n_i}\right)_{T,p,n_j}$ | yes | no |
| Joule Thomson coefficient $\mu_\mathrm{JT}$ | $\left(\frac{\partial T}{\partial p}\right)_{H,n_i}$ | yes | no |
| Isentropic compressibility $\kappa_s$ | $-\frac{1}{V}\left(\frac{\partial V}{\partial p}\right)_{S,n_i}$ | yes | no |
| Isothermal compressibility $\kappa_T$ | $-\frac{1}{V}\left(\frac{\partial V}{\partial p}\right)_{T,n_i}$ | no | no |
| Isenthalpic compressibility $\kappa_h$ | $-\frac{1}{V}\left(\frac{\partial V}{\partial p}\right)_{H,n_i}$ | yes | no |
| Thermal expansivity $\alpha_p$ | $-\frac{1}{V}\left(\frac{\partial V}{\partial T}\right)_{p,n_i}$ | yes | no |
| Grüneisen parameter $\phi$ | $V\left(\frac{\partial p}{\partial U}\right)_{V,n_i}=\frac{v}{c_v}\left(\frac{\partial p}{\partial T}\right)_{v,n_i}=\frac{\rho}{T}\left(\frac{\partial T}{\partial \rho}\right)_{s, n_i}$ | yes | no |
| (Static) structure factor $S(0)$ | $RT\left(\frac{\partial\rho}{\partial p}\right)_{T,n_i}$ | no | no |
| Speed of sound $c$ | $\sqrt{\left(\frac{\partial p}{\partial\rho^{(m)}}\right)_{S,n_i}}$ | yes | no |
