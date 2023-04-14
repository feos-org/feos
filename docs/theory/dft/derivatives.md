# Derivatives of density profiles
For converged density properties equilibrium properties can be calculated as partial derivatives of thermodynamic potentials analogous to classical (bulk) thermodynamics. The difference is that the derivatives have to be along a path of valid density profiles (solutions of the [Euler-Lagrange equation](euler_lagrange_equation.md)).

The density profiles are calculated implicitly from the Euler-Lagrange equation, which can be written simplified as

$$\Omega_{\rho_i}(T,\lbrace\mu_k\rbrace,[\lbrace\rho_k(\mathbf{r})\rbrace])=F_{\rho_i}(T,[\lbrace\rho_k(\mathbf{r})\rbrace])-\mu_i+V_i^\mathrm{ext}(\mathbf{r})=0$$ (eqn:euler_lagrange)

Incorporating bond integrals can be done similar to the section on the [Newton solver](solver.md) but will not be discussed in this section. The derivatives of the density profiles can then be calculated from the total differential of eq. {eq}`eqn:euler_lagrange`, leading to

$$\mathrm{d}\Omega_{\rho_i}(\mathbf{r})=\left(\frac{\partial\Omega_{\rho_i}(\mathbf{r})}{\partial T}\right)_{\mu_k,\rho_k}\mathrm{d}T+\sum_j\left(\frac{\partial\Omega_{\rho_i}(\mathbf{r})}{\partial\mu_j}\right)_{T,\mu_k,\rho_k}\mathrm{d}\mu_j+\int\sum_j\left(\frac{\delta\Omega_{\rho_i}(\mathbf{r})}{\delta\rho_j(\mathbf{r}')}\right)_{T,\mu_k,\rho_k}\delta\rho_j(\mathbf{r}')\mathrm{d}\mathbf{r}'=0$$

Using eq. {eq}`eqn:euler_lagrange` and the shortened notation for derivatives of functionals in their natural variables, e.g., $F_T=\left(\frac{\partial F}{\partial T}\right)_{\rho_k}$, the expression can be simplified to

$$F_{T\rho_i}(\mathbf{r})\mathrm{d}T-\mathrm{d}\mu_i+\int\sum_j F_{\rho_i\rho_j}(\mathbf{r},\mathbf{r}')\delta\rho_j(\mathbf{r}')\mathrm{d}\mathbf{r}'=0$$ (eqn:gibbs_duhem)

Similar to the Gibbs-Duhem relation for bulk phases, eq. {eq}`eqn:gibbs_duhem` shows how temperature, chemical potentials and the density profiles in an inhomogeneous system cannot be varied independently. The derivatives of the density profiles with respect to the intensive variables can be directly identified as

$$\int\sum_j F_{\rho_i\rho_j}(\mathbf{r},\mathbf{r}')\left(\frac{\partial\rho_j(\mathbf{r}')}{\partial T}\right)_{\mu_k}\mathrm{d}\mathbf{r}'=-F_{T\rho_i}(\mathbf{r})$$

and

$$\int\sum_j F_{\rho_i\rho_j}(\mathbf{r},\mathbf{r}')\left(\frac{\partial\rho_j(\mathbf{r}')}{\partial\mu_k}\right)_{T}\mathrm{d}\mathbf{r}'=\delta_{ik}$$ (eqn:drho_dmu)

Both of these expressions are implicit (linear) equations for the derivatives. They can be solved rapidly analogously to the implicit expression appearing in the [Newton solver](solver.md). In practice, it is useful to explicitly cancel out the (often unknown) thermal de Broglie wavelength $\Lambda_i$ from the expression where it has no influence. This is done by splitting the intrinsic Helmholtz energy into an ideal gas and a residual part.

$$F=k_\mathrm{B}T\int\sum_im_i\rho_i(\mathbf{r})\left(\ln\left(\rho_i(\mathbf{r})\Lambda_i^3\right)-1\right)\mathrm{d}\mathbf{r}+\mathcal{\hat F}^\mathrm{res}$$

Then $F_{\rho_i\rho_j}(\mathbf{r},\mathbf{r}')=m_i\frac{k_\mathrm{B}T}{\rho_i(\mathbf{r})}\delta_{ij}\delta(\mathbf{r}-\mathbf{r}')+\mathcal{\hat F}_{\rho_i\rho_j}^\mathrm{res}(\mathbf{r},\mathbf{r}')$ and eq. {eq}`eqn:drho_dmu` can be rewritten as

$$m_i\frac{k_\mathrm{B}T}{\rho_i(\mathbf{r})}\left(\frac{\partial\rho_i(\mathbf{r})}{\partial\mu_k}\right)_T+\int\sum_j\mathcal{\hat F}_{\rho_i\rho_j}^\mathrm{res}(\mathbf{r},\mathbf{r}')\left(\frac{\partial\rho_j(\mathbf{r}')}{\partial\mu_k}\right)_{T}\mathrm{d}\mathbf{r}'=\delta_{ik}$$

In practice, the division by the density should be avoided for numerical reasons and the energetic properties are reduced with the factor $\beta=\frac{1}{k_\mathrm{B}T}$. The final expression is

$$m_i\left(\frac{\partial\rho_i(\mathbf{r})}{\partial\beta\mu_k}\right)_T+\rho_i(\mathbf{r})\int\sum_j\beta\mathcal{\hat F}_{\rho_i\rho_j}^\mathrm{res}(\mathbf{r},\mathbf{r}')\left(\frac{\partial\rho_j(\mathbf{r}')}{\partial\beta\mu_k}\right)_{T}\mathrm{d}\mathbf{r}'=\rho_i(\mathbf{r})\delta_{ik}$$

For the temperature derivative, it is more convenient to express eq. {eq}`eqn:gibbs_duhem` in terms of the pressure of a bulk phase that is in equilibrium with the inhomogeneous system. In the following, only paths along **constant bulk composition** are considered. With this constraint, the total differential of the chemical potential simplifies to

$$\mathrm{d}\mu_i=-s_i\mathrm{d}T+v_i\mathrm{d}p$$

which can be used in eq. {eq}`eqn:gibbs_duhem` to give

$$\left(F_{T\rho_i}(\mathbf{r})+s_i\right)\mathrm{d}T+\int\sum_j F_{\rho_i\rho_j}(\mathbf{r},\mathbf{r}')\delta\rho_j(\mathbf{r}')\mathrm{d}\mathbf{r}'=v_i\mathrm{d}p$$

Even though $s_i$ is readily available in $\text{FeO}_\text{s}$ it is useful at this point to rewrite the partial molar entropy as

$$s_i=v_i\left(\frac{\partial p}{\partial T}\right)_{V,N_k}-F_{T\rho_i}^\mathrm{b}$$

Then, the intrinsic Helmholtz energy can be split into an ideal gas and a residual part again, and the de Broglie wavelength cancels.

$$\begin{align*}
&\left(m_ik_\mathrm{B}\ln\left(\frac{\rho_i(\mathbf{r})}{\rho_i^\mathrm{b}}\right)+F_{T\rho_i}^\mathrm{res}(\mathbf{r})-F_{T\rho_i}^\mathrm{b,res}+v_i\left(\frac{\partial p}{\partial T}\right)_{V,N_k}\right)\mathrm{d}T\\
&~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+m_i\frac{k_\mathrm{B}T}{\rho_i(\mathbf{r})}\delta\rho_i(\mathbf{r})+\int\sum_j\mathcal{\hat F}_{\rho_i\rho_j}^\mathrm{res}(\mathbf{r},\mathbf{r}')\delta\rho_j(\mathbf{r}')\mathrm{d}\mathbf{r}'=v_i\mathrm{d}p
\end{align*}$$

Finally the expressions for the derivatives with respect to pressure

$$m_i\left(\frac{\partial\rho_i(\mathbf{r})}{\partial\beta p}\right)_{T,x_k}+\rho_i(\mathbf{r})\int\sum_j\beta\mathcal{\hat F}_{\rho_i\rho_j}^\mathrm{res}(\mathbf{r},\mathbf{r}')\left(\frac{\partial\rho_j(\mathbf{r}')}{\partial\beta p}\right)_{T,x_k}\mathrm{d}\mathbf{r}'=v_i\rho_i(\mathbf{r})$$

and temperature

$$\begin{align*}
&m_i\left(\frac{\partial\rho_i(\mathbf{r})}{\partial T}\right)_{p,x_k}+\rho_i(\mathbf{r})\int\sum_j\beta\mathcal{\hat F}_{\rho_i\rho_j}^\mathrm{res}(\mathbf{r},\mathbf{r}')\left(\frac{\partial\rho_j(\mathbf{r}')}{\partial T}\right)_{p,x_k}\mathrm{d}\mathbf{r}'\\
&~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~=-\frac{\rho_i(\mathbf{r})}{k_\mathrm{B}T}\left(m_ik_\mathrm{B}\ln\left(\frac{\rho_i(\mathbf{r})}{\rho_i^\mathrm{b}}\right)+F_{T\rho_i}^\mathrm{res}(\mathbf{r})-F_{T\rho_i}^\mathrm{b,res}+v_i\left(\frac{\partial p}{\partial T}\right)_{V,N_k}\right)
\end{align*}$$

follow. All derivatives $x_i$ shown here can be calculated from the same linear equation

$$m_ix_i+\rho_i(\mathbf{r})\int\sum_j\beta\mathcal{\hat F}_{\rho_i\rho_j}^\mathrm{res}(\mathbf{r},\mathbf{r}')x_i\mathrm{d}\mathbf{r}'=y_i$$

by just replacing the right hand side $y_i$.

|derivative|right hand side|
|-|-|
|$\left(\frac{\partial\rho_i(\mathbf{r})}{\partial\beta\mu_k}\right)_T$|$\rho_i(\mathbf{r})\delta_{ik}$|
|$\left(\frac{\partial\rho_i(\mathbf{r})}{\partial\beta p}\right)_{T,x_k}$|$\rho_i(\mathbf{r})v_i$|
|$\left(\frac{\partial\rho_i(\mathbf{r})}{\partial T}\right)_{p,x_k}$|$-\frac{\rho_i(\mathbf{r})}{k_\mathrm{B}T}\left(m_ik_\mathrm{B}\ln\left(\frac{\rho_i(\mathbf{r})}{\rho_i^\mathrm{b}}\right)+F_{T\rho_i}^\mathrm{res}(\mathbf{r})-F_{T\rho_i}^\mathrm{b,res}+v_i\left(\frac{\partial p}{\partial T}\right)_{V,N_k}\right)$|