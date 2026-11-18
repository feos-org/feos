# Euler-Lagrange equation
The fundamental expression in classical density functional theory is the relation between the grand potential functional $\Omega$ and the intrinsic Helmholtz energy functional $F$.

$$\Omega(T,\mu,[\rho(\mathbf{r})])=F(T,[\rho(\mathbf{r})])-\sum_i\int\rho_i(\mathbf{r})\left(\mu_i-V_i^\mathrm{ext}(\mathbf{r})\right)\mathrm{d}\mathbf{r}$$

What makes this expression so appealing is that the intrinsic Helmholtz energy functional only depends on the temperature $T$ and the density profiles $\rho_i(\mathbf{r})$ of the system and not on the external potentials $V_i^\mathrm{ext}(\mathbf{r})$.

For a given temperature $T$, chemical potentials $\mu_i$ and external potentials $V_i^\mathrm{ext}(\mathbf{r})$ the grand potential reaches a minimum at equilibrium. Mathematically this condition can be written as

$$
\left.\frac{\delta\Omega}{\delta\rho_i(\mathbf{r})}\right|_{T,\mu}=F_{\rho_i}(\mathbf{r})-\mu_i+V_i^{\mathrm{ext}}(\mathbf{r})=0
$$ (eqn:euler_lagrange_mu)

where $F_{\rho_i}(\mathbf{r})=\left.\frac{\delta F}{\delta\rho_i(\mathbf{r})}\right|_T$ is short for the functional derivative of the intrinsic Helmholtz energy. In this context, eq. {eq}`eqn:euler_lagrange_mu` is commonly referred to as the Euler-Lagrange equation, an implicit nonlinear integral equation which needs to be solved for the equilibrium density profiles of the system.

For a homogeneous (bulk) system, $V_i^\mathrm{ext}=0$ and we get

$$
F_{\rho_i}^\mathrm{b}-\mu_i=0
$$ (eqn:euler_lagrange_bulk)

The functional derivative of the Helmholtz energy of a bulk system $F_{\rho_i}^\mathrm{b}$ is a function of the temperature $T$ and bulk densities $\rho^\mathrm{b}$. At this point, it can be advantageous to relate the grand potential of an inhomogeneous system to the densities of a bulk system that is in equilibrium with the inhomogeneous system. This approach has several advantages:
- The thermal de Broglie wavelength $\Lambda$ cancels out.
- If the chemical potential of the system is not known, all variables are the same quantity (densities).
- The bulk system is described by a Helmholtz energy which is explicit in the density, so there are no internal iterations required.

Using eq. {eq}`eqn:euler_lagrange_bulk` in eq. {eq}`eqn:euler_lagrange_mu` leads to the Euler-Lagrange equation

$$
\left.\frac{\delta\Omega}{\delta\rho_i(\mathbf{r})}\right|_{T,\rho^\mathrm{b}}=F_{\rho_i}(\mathbf{r})-F_{\rho_i}^\mathrm{b}+V_i^{\mathrm{ext}}(\mathbf{r})=0
$$ (eqn:euler_lagrange_rho)

## Spherical molecules
In the simplest case, the molecules under consideration can be described as spherical. Then the Helmholtz energy can be split into an ideal and a residual part:

$$\beta F=\sum_i\int\rho_i(\mathbf{r})\left(\ln\left(\rho_i(\mathbf{r})\Lambda_i^3\right)-1\right)\mathrm{d}\mathbf{r}+\beta F^\mathrm{res}$$

with the thermal de Broglie wavelength $\Lambda_i$. The functional derivatives for an inhomogeneous and a bulk system follow as

$$\beta F_{\rho_i}(\mathbf{r})=\ln\left(\rho_i(\mathbf{r})\Lambda_i^3\right)+\beta F_{\rho_i}^\mathrm{res}(\mathbf{r})$$

$$\beta F_{\rho_i}^\mathrm{b}=\ln\left(\rho_i^\mathrm{b}\Lambda_i^3\right)+\beta F_{\rho_i}^\mathrm{b,res}$$

Using these expressions in eq. {eq}`eqn:euler_lagrange_rho` results in

$$\left.\frac{\delta\beta\Omega}{\delta\rho_i(\mathbf{r})}\right|_{T,\rho^\mathrm{b}}=\ln\left(\frac{\rho_i(\mathbf{r})}{\rho_i^\mathrm{b}}\right)+\beta\left(F_{\rho_i}^\mathrm{res}(\mathbf{r})-F_{\rho_i}^\mathrm{b,res}+V_i^{\mathrm{ext}}(\mathbf{r})\right)=0$$

The Euler-Lagrange equation can be recast as

$$\rho_i(\mathbf{r})=\rho_i^\mathrm{b}e^{\beta\left(F_{\rho_i}^\mathrm{b,res}-F_{\rho_i}^\mathrm{res}(\mathbf{r})-V_i^\mathrm{ext}(\mathbf{r})\right)}$$

which is convenient because it leads directly to a recurrence relation known as Picard iteration. The expression can be further simplified by defining the fugacity $f_i=\rho_i^\mathrm{b}e^{\beta F_{\rho_i}^\mathrm{b,res}}$ which leads to

$$\rho_i(\mathbf{r})=f_ie^{-\beta\left(F_{\rho_i}^\mathrm{res}(\mathbf{r})+V_i^\mathrm{ext}(\mathbf{r})\right)}$$

## Homosegmented chains
For chain molecules that do not resolve individual segments (essentially the PC-SAFT Helmholtz energy functional) a chain contribution is introduced as

$$\beta F^\mathrm{chain}=-\sum_i\int\rho_i(\mathbf{r})\left(m_i-1\right)\ln\left(\frac{y_{ii}\lambda_i(\mathbf{r})}{\rho_i(\mathbf{r})}\right)\mathrm{d}\mathbf{r}$$

where $m_i$ is the number of segments (i.e., the PC-SAFT chain length parameter), $y_{ii}$ is the cavity correlation function at contact in the reference fluid, and $\lambda_i$ is a weighted density.
The presence of $\rho_i(\mathbf{r})$ in the logarithm poses numerical problems. Therefore, it is convenient to rearrange the expression as

$$\begin{aligned}
\beta F^\mathrm{chain}=&\sum_i\int\rho_i(\mathbf{r})\left(m_i-1\right)\left(\ln\left(\rho_i(\mathbf{r})\Lambda_i^3\right)-1\right)\mathrm{d}\mathbf{r}\\
&\underbrace{-\sum_i\int\rho_i(\mathbf{r})\left(m_i-1\right)\left(\ln\left(y_{ii}\lambda_i(\mathbf{r})\Lambda_i^3\right)-1\right)\mathrm{d}\mathbf{r}}_{\beta\hat{F}^\mathrm{chain}}
\end{aligned}$$

Then the total Helmholtz energy

$$\beta F=\sum_i\int\rho_i(\mathbf{r})\left(\ln\left(\rho_i(\mathbf{r})\Lambda_i^3\right)-1\right)\mathrm{d}\mathbf{r}+\beta F^\mathrm{chain}+\beta F^\mathrm{res}$$

can be rearranged to

$$\beta F=\sum_i\int\rho_i(\mathbf{r})m_i\left(\ln\left(\rho_i(\mathbf{r})\Lambda_i^3\right)-1\right)\mathrm{d}\mathbf{r}+\underbrace{\beta\hat{F}^\mathrm{chain}+\beta F^\mathrm{res}}_{\beta\hat{F}^\mathrm{res}}$$

The functional derivatives are then similar to the spherical case

$$\beta F_{\rho_i}(\mathbf{r})=m_i\ln\left(\rho_i(\mathbf{r})\Lambda_i^3\right)+\beta\hat{F}_{\rho_i}^\mathrm{res}(\mathbf{r})$$

$$\beta F_{\rho_i}^\mathrm{b}=m_i\ln\left(\rho_i^\mathrm{b}\Lambda_i^3\right)+\beta\hat{F}_{\rho_i}^\mathrm{b,res}$$

and lead to a slightly modified Euler-Lagrange equation

$$\rho_i(\mathbf{r})=\rho_i^\mathrm{b}e^{\frac{\beta}{m_i}\left(\hat F_{\rho_i}^\mathrm{b,res}-\hat F_{\rho_i}^\mathrm{res}(\mathbf{r})-V_i^\mathrm{ext}(\mathbf{r})\right)}$$

or with the modified fugacity $\hat f_i=\rho_i^\mathrm{b}e^{\frac{\beta}{m_i}\hat F_{\rho_i}^\mathrm{b,res}}$

$$\rho_i(\mathbf{r})=\hat f_ie^{-\frac{\beta}{m_i}\left(\hat F_{\rho_i}^\mathrm{res}(\mathbf{r})+V_i^\mathrm{ext}(\mathbf{r})\right)}$$

## Heterosegmented chains
The expressions are more complex for models in which density profiles of individual segments are considered. A derivation is given in the appendix of [Rehner et al. (2022)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.105.034110). The resulting Euler-Lagrange equation is given as

$$\rho_\alpha(\mathbf{r})=f_\alpha e^{-\beta\left(\hat F_{\rho_\alpha}^\mathrm{res}(\mathbf{r})+V_\alpha^\mathrm{ext}(\mathbf{r})\right)}\prod_{\alpha'}I_{\alpha\alpha'}(\mathbf{r})$$

with the bond integrals $I_{\alpha\alpha'}(\mathbf{r})$ that are calculated recursively from

$$I_{\alpha\alpha'}(\mathbf{r})=\int e^{-\beta\left(\hat{F}_{\rho_{\alpha'}}^\mathrm{res}(\mathbf{r}')+V_{\alpha'}^\mathrm{ext}(\mathbf{r}')\right)}\left(\prod_{\alpha''\neq\alpha}I_{\alpha'\alpha''}(\mathbf{r}')\right)\omega_\mathrm{chain}^{\alpha\alpha'}(\mathbf{r}-\mathbf{r}')\mathrm{d}\mathbf{r}'$$

Here, $\alpha'$ refers to all segments bonded to segment $\alpha$ and $\alpha''$ to all *other* segments bonded to $\alpha'$ (excluding $\alpha$). 
For bulk systems the expressions simplify to

$$\rho_\alpha^\mathrm{b}=f_\alpha e^{-\beta\sum_\gamma\hat F_{\rho_\gamma}^\mathrm{b,res}}$$

with $\gamma$ referring to all segments on the same molecule as $\alpha$. Solving for the fugacity leads to

$$f_\alpha=\rho_\alpha^\mathrm{b}e^{\beta\sum_\gamma\hat F_{\rho_\gamma}^\mathrm{b,res}}$$

which is by construction equal for every segment $\alpha$ on the same molecule.

## Associating molecules
Association is described with Wertheim's theory, which introduces the fractions of non-bonded association sites $X_A$. In the bulk, they are determined from the mass action equations, which are usually solved for given densities before the Helmholtz energy is evaluated. In inhomogeneous systems, we treat the fractions of non-bonded sites $X_A(\mathbf{r})$ as additional variables of the density profile. Following [Michelsen and Hendriks (2001)](https://doi.org/10.1016/S0378-3812(01)00344-2), we define the functional

$$Q[X,\rho]=\int\sum_A\tilde\rho_A(\mathbf{r})\left(\ln X_A(\mathbf{r})-X_A(\mathbf{r})+1\right)\mathrm{d}\mathbf{r}-\frac{1}{2}\iint\sum_{AB}\tilde\rho_A(\mathbf{r})X_A(\mathbf{r})\tilde\rho_B(\mathbf{r}')X_B(\mathbf{r}')\Delta^{AB}(\mathbf{r},\mathbf{r}')\mathrm{d}\mathbf{r}\mathrm{d}\mathbf{r}'$$

of the density profiles and the fractions of non-bonded sites. Here, $\tilde\rho_A$ is the density of site $A$, i.e., the multiplicity of the site times the (possibly weighted) density of the segment that carries the site, and $\Delta^{AB}(\mathbf{r},\mathbf{r}')$ is the association strength between sites $A$ and $B$ at positions $\mathbf{r}$ and $\mathbf{r}'$. For local approaches, $\Delta^{AB}(\mathbf{r},\mathbf{r}')=\Delta^{AB}(\mathbf{r})\delta(\mathbf{r}-\mathbf{r}')$, whereas the non-local approach restricts $\mathbf{r}'$ to a shell at contact around $\mathbf{r}$. The specific expressions are given in the section on the [association functionals](../models/association.md).

The association contribution to the Helmholtz energy follows as the stationary point of $Q$ with respect to the fractions of non-bonded sites

$$\beta F^\mathrm{assoc}=\underset{X}{\mathrm{stat}}~Q[X,\rho]$$

The stationarity condition $\frac{\delta Q}{\delta X_A(\mathbf{r})}=0$ yields the mass action equations

$$
X_A(\mathbf{r})=\frac{1}{1+S_A(\mathbf{r})}
$$ (eqn:mass_action)

with the association sums

$$S_A(\mathbf{r})=\sum_B\int\tilde\rho_B(\mathbf{r}')X_B(\mathbf{r}')\Delta^{AB}(\mathbf{r},\mathbf{r}')\mathrm{d}\mathbf{r}'$$

Inserting the mass action equations into $Q$ recovers the well-known form of the association contribution

$$\beta F^\mathrm{assoc}=\int\sum_A\tilde\rho_A(\mathbf{r})\left(\ln X_A(\mathbf{r})-\frac{X_A(\mathbf{r})}{2}+\frac{1}{2}\right)\mathrm{d}\mathbf{r}$$

The stationarity also simplifies the functional derivative with respect to the density profiles because the implicit dependence of $X_A(\mathbf{r})$ on the density profiles does not contribute. The functional derivative is therefore evaluated at fixed $X_A(\mathbf{r})$

$$\beta F_{\rho_\alpha}^\mathrm{assoc}(\mathbf{r})=\int\sum_A\frac{\delta\tilde\rho_A(\mathbf{r}')}{\delta\rho_\alpha(\mathbf{r})}\left(\ln X_A(\mathbf{r}')-X_A(\mathbf{r}')+1-X_A(\mathbf{r}')S_A(\mathbf{r}')\right)\mathrm{d}\mathbf{r}'-\frac{1}{2}\iint\sum_{AB}\tilde\rho_A(\mathbf{r}')X_A(\mathbf{r}')\tilde\rho_B(\mathbf{r}'')X_B(\mathbf{r}'')\frac{\delta\Delta^{AB}(\mathbf{r}',\mathbf{r}'')}{\delta\rho_\alpha(\mathbf{r})}\mathrm{d}\mathbf{r}'\mathrm{d}\mathbf{r}''$$

The association contribution is added to $\hat F_{\rho_\alpha}^\mathrm{res}$, so the Euler-Lagrange equation keeps its form and is solved together with eq. {eq}`eqn:mass_action` for $\rho_\alpha(\mathbf{r})$ and $X_A(\mathbf{r})$. In the bulk phase, the fractions of non-bonded sites $X_A^\mathrm{b}$ follow from the mass action equations at the bulk densities, and the association contribution to $\hat F_{\rho_\alpha}^\mathrm{b,res}$ is evaluated at $X_A^\mathrm{b}$.

## Full expression
To avoid having multiple implementations of the central part of the DFT code, the different descriptions of molecules are combined in a single set of equations:

$$\rho_\alpha(\mathbf{r})=f_\alpha e^{-\frac{\beta}{m_\alpha}\left(\hat F_{\rho_\alpha}^\mathrm{res}(\mathbf{r})+V_\alpha^\mathrm{ext}(\mathbf{r})\right)}\prod_{\alpha'}I_{\alpha\alpha'}(\mathbf{r})$$

$$I_{\alpha\alpha'}(\mathbf{r})=\int e^{-\frac{\beta}{m_{\alpha'}}\left(\hat F_{\rho_{\alpha'}}^\mathrm{res}(\mathbf{r}')+V_{\alpha'}^\mathrm{ext}(\mathbf{r}')\right)}\left(\prod_{\alpha''\neq\alpha}I_{\alpha'\alpha''}(\mathbf{r}')\right)\omega_\mathrm{chain}^{\alpha\alpha'}(\mathbf{r}-\mathbf{r}')\mathrm{d}\mathbf{r}'$$

$$X_A(\mathbf{r})=\frac{1}{1+S_A(\mathbf{r})}$$

$$f_\alpha=\rho_\alpha^\mathrm{b}e^{\frac{\beta}{m_\alpha}\sum_\gamma\hat F_{\rho_\gamma}^\mathrm{b,res}}$$

Here, $\hat F_{\rho_\alpha}^\mathrm{res}$ contains the association contribution at fixed $X_A(\mathbf{r})$, and $\hat F_{\rho_\gamma}^\mathrm{b,res}$ the association contribution at the fractions of non-bonded sites of the bulk phase. If molecules consist of single (possibly non-spherical) segments, the Euler-Lagrange equation simplifies to that of the homosegmented chains shown above. For heterosegmented chains, the correct expression is obtained by setting $m_\alpha=1$. Without association, the mass action equations drop out.
