# Enthalpy of adsorption and the Clausius-Clapeyron relation

## Enthalpy of adsorption
The energy balance of a simple adsorption process can be written as

$$\frac{\mathrm{d}U}{\mathrm{d}t}=\sum_i\dot{n}_ih_i^\mathrm{b}+\dot{Q}$$

Here the balance is chosen to only include the fluid in the porous medium. The partial molar enthalpy $h_i^\mathrm{b}$ is the enthalpy of the fluid at the point where it enters or leaves the adsorber at which point it can be considered in a bulk state. The component balance is simply

$$\frac{\mathrm{d}N_i}{\mathrm{d}t}=\dot{n}_i$$

and can be used in the energy balance to yield

$$\dot{Q}=\frac{\mathrm{d}U}{\mathrm{d}t}-\sum_ih_i^\mathrm{b}\frac{\mathrm{d}N_i}{\mathrm{d}t}$$

or in differential form

$$\delta Q=\mathrm{d}U-\sum_ih_i^\mathrm{b}\mathrm{d}N_i$$

The expression can be rewritten using the total differential of the internal energy (the volume of the adsorber is fixed and therefore not considered as a variable)

$$\delta Q=\left(\frac{\partial U}{\partial T}\right)_{N_k}\mathrm{d}T-\sum_i\left(h_i^\mathrm{b}-\left(\frac{\partial U}{\partial N_i}\right)_T\right)\mathrm{d}N_i$$

The heat of adsorption $\delta Q$ can thus be split into a sensible part that depends on the change in temperature, and a latent part that depends on the change in loading. The expression can be simplified by using the definitions of the isochoric heat capacity $C_v=\left(\frac{\partial U}{\partial T}\right)_{N_k}$ and the **partial molar enthalpy of adsorption**

$$\Delta h_i^\mathrm{ads}=h_i^\mathrm{b}-\left(\frac{\partial U}{\partial N_i}\right)_T$$

yielding

$$\delta Q=C_v\mathrm{d}T-\sum_i\Delta h_i^\mathrm{ads}\mathrm{d}N_i$$

If the composition of the bulk phase is fixed, which can be a fair assumption for an adsorption process but is in general not the case for a desorption process, the heat of adsorption simplifies to

$$\delta Q=C_v\mathrm{d}T-\Delta h^\mathrm{ads}\mathrm{d}N$$

with the **enthalpy of adsorption**

$$\Delta h^\mathrm{ads}=\sum_ix_i\Delta h_i^\mathrm{ads}=h^\mathrm{b}-\sum_ix_i\left(\frac{\partial U}{\partial N_i}\right)_T$$

## Clausius-Clapeyron relation for porous media
The Clausius-Clapeyron relation relates the $p-T$ slope of a pure component phase transition line to the corresponding enthalpy of phase change. For a vapor-liquid phase transition, the exact relation is

$$\frac{\mathrm{d}p^\mathrm{sat}}{\mathrm{d}T}=\frac{s^\mathrm{V}-s^\mathrm{L}}{v^\mathrm{V}-v^\mathrm{L}}=\frac{h^\mathrm{V}-h^\mathrm{L}}{T\left(v^\mathrm{V}-v^\mathrm{L}\right)}$$ (eqn:temp_dep_press)

In this expression, the enthalpy of vaporization $\Delta h^\mathrm{vap}=h^\mathrm{V}-h^\mathrm{L}$ can be identified. If the molar volume of the liquid phase $v^\mathrm{L}$ is assumed to be negligible compared to the molar volume of the vapor phase $v^\mathrm{V}$ and the gas phase is assumed to be ideal, the relation simplifies to

$$\frac{\mathrm{d}p^\mathrm{sat}}{\mathrm{d}T}=\frac{p}{RT^2}\Delta h^\mathrm{vap}$$

which can be compactly written as

$$\frac{\mathrm{d}\ln p^\mathrm{sat}}{\mathrm{d}\frac{1}{RT}}=-\Delta h^\mathrm{vap}$$ (eqn:Clausius_Clapeyron)

Without assuming neither ideal gas nor neglecting the volume of the liquid, eq. {eq}`eqn:temp_dep_press` can be rewritten using the compressibility factor $Z$ to

$$\frac{\mathrm{d}p^\mathrm{sat}}{\mathrm{d}T}=\frac{h^\mathrm{V}-h^\mathrm{L}}{T\left(v^\mathrm{V}-v^\mathrm{L}\right)}=\frac{p}{R T^2}\frac{\Delta h^\mathrm{vap}}{Z^\mathrm{V}-Z^\mathrm{L}}$$

Neglecting the compressibility of the liquid phase ($Z^\mathrm{L}=0$) and assuming ideal gas for the vapor phase ($Z^\mathrm{V}=1$) leads to eq. {eq}`eqn:Clausius_Clapeyron`. 


A similar relation can be derived for fluids adsorbed in a porous medium that is in equilibrium with a bulk phase. At this point it is important to clarify which variables describe the system
- The adsorbed fluid and the bulk phase are in equilibrium. Therefore, the temperature $T$ and chemical potentials $\mu_i$ are the same for both phases.
- The density profiles and hence the number of particles $N_i$ in the porous medium is determined by $T$ and $\mu_i$. The volume of the porous medium is not considered as a thermodynamic variable but rather as a (constant) property of the adsorbent.
- All intensive properties of the bulk phase are fully determined by $T$ and $\mu_i$. In practice it can be useful to relate these properties to measurable properties like the pressure $p$ and the composition $x_i$.

To find an expression of the slope of an isostere (constant $N_i$), the pressure, which is only defined for the bulk phase, has to be related to properties of the adsorbed fluid.

$$\frac{\mathrm{d}\ln p}{\mathrm{d}\frac{1}{RT}}=-\frac{RT^2}{p}\frac{\mathrm{d}p}{\mathrm{d}T}$$

First, the pressure can be replaced using the Gibbs-Duhem relation for the bulk phase (index $\mathrm{b}$)

$$\frac{\mathrm{d}\ln p}{\mathrm{d}\frac{1}{RT}}=-\frac{RT^2}{pv^\mathrm{b}}\left(s^\mathrm{b}+\sum_ix_i\left(\frac{\partial\mu_i}{\partial T}\right)_{N_k}\right)$$ (eqn:clausius_clapeyron_intermediate)

Here the directional derivative $\frac{\mathrm{d}\mu_i}{\mathrm{d}T}$ could be replaced with a partial derivative amongst the variables describing the adsorbed fluid. The partial derivative can then be replaced using a Maxwell relation based on the Helmholtz energy $F$ as follows

$$\left(\frac{\partial\mu_i}{\partial T}\right)_{N_k}=\left(\frac{\partial^2 F}{\partial T\partial N_i}\right)=-\left(\frac{\partial S}{\partial N_i}\right)_T$$

Using the Maxwell relation together with the compressibility factor of the bulk phase $Z^\mathrm{b}=\frac{pv^\mathrm{b}}{RT}$ in eq. {eq}`eqn:clausius_clapeyron_intermediate` results in

$$\frac{\mathrm{d}\ln p}{\mathrm{d}\frac{1}{RT}}=-\frac{T}{Z^\mathrm{b}}\left(s^\mathrm{b}-\sum_ix_i\left(\frac{\partial S}{\partial N_i}\right)_T\right)$$

Finally, using $h^\mathrm{b}=Ts^\mathrm{b}+\sum_ix_i\mu_i$ and $\mathrm{d}U=T\mathrm{d}S+\sum_i\mu_i\mathrm{d}N_i$ leads to

$$\frac{\mathrm{d}\ln p}{\mathrm{d}\frac{1}{RT}}=-\frac{1}{Z^\mathrm{b}}\left(h^\mathrm{b}-\sum_ix_i\left(\frac{\partial U}{\partial N_i}\right)_T\right)=-\frac{\Delta h^\mathrm{ads}}{Z^\mathrm{b}}$$ (eqn:deriv_relation_hads)

The relation is exact and valid for an arbitrary number of components in the fluid phase. 


## Calculation of the enthalpy of adsorption from classical DFT
In a DFT context, the introduction of entropies and internal energies are just unnecessary complications. The most useful definition of the (partial molar) enthalpy of adsorption is

$$\Delta h_i^\mathrm{ads}=T\left(s_i^\mathrm{b}+\left(\frac{\partial\mu_i}{\partial T}\right)_{N_k}\right)$$

The derivative at constant number of particles is problematic and has to be replaced. This is done starting from the total differential of the number of particles

$$\mathrm{d}N_i=\sum_j\left(\frac{\partial N_i}{\partial\mu_j}\right)_T\mathrm{d}\mu_j+\left(\frac{\partial N_i}{\partial T}\right)_{\mu_k}\mathrm{d}T$$ (eqn:dn)

Calculating the derivative with respect to $T$ at constant $N_i$ leads to

$$0=\sum_j\left(\frac{\partial N_i}{\partial\mu_j}\right)_T\left(\frac{\partial\mu_j}{\partial T}\right)_{N_k}+\left(\frac{\partial N_i}{\partial T}\right)_{\mu_k}$$ (eqn:dndt_1)

from which the unknown derivative $\left(\frac{\partial\mu_i}{\partial T}\right)_{N_k}$ can be calculated. In practice the expression has the disadvantage that $\left(\frac{\partial N_i}{\partial T}\right)_{\mu_k}$ depends on the (sometimes unknown) thermal de Broglie wavelength which cancels later with $s_i^\mathrm{b}$. This can be remedied by first calculating the derivative of eq. {eq}`eqn:dn` with respect to $T$ at constant (bulk) pressure and composition.

$$\left(\frac{\partial N_i}{\partial T}\right)_{p,x_k}=\sum_j\left(\frac{\partial N_i}{\partial\mu_j}\right)_T\left(\frac{\partial\mu_j}{\partial T}\right)_{p,x_k}+\left(\frac{\partial N_i}{\partial T}\right)_{\mu_k}$$ (eqn:dndt_2)

From classical bulk thermodynamics we know $\left(\frac{\partial\mu_j}{\partial T}\right)_{p,x_k}=-s_j^\mathrm{b}$ and therefore, eq. {eq}`eqn:dndt_2` can be used in eq. {eq}`eqn:dndt_1` to give

$$0=\sum_j\left(\frac{\partial N_i}{\partial\mu_j}\right)_T\left(s_j^\mathrm{b}+\left(\frac{\partial\mu_j}{\partial T}\right)_{N_k}\right)+\left(\frac{\partial N_i}{\partial T}\right)_{p,x_k}$$

After multiplying with $T$, the following elegant expression remains

$$0=\sum_j\left(\frac{\partial N_i}{\partial\mu_j}\right)_T\Delta h_j^\mathrm{ads}+T\left(\frac{\partial N_i}{\partial T}\right)_{p,x_k}$$

which is a symmetric linear system of equations due to $\left(\frac{\partial N_i}{\partial\mu_j}\right)_T=-\left(\frac{\partial^2\Omega}{\partial\mu_i\partial\mu_j}\right)_T$. The derivatives of the particle numbers are obtained by integrating over the respective derivatives of the density profiles which were discussed [previously](derivatives.md).