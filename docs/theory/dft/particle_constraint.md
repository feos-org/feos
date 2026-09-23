# DFT with constrained number of particles

In a grand-canonical DFT calculation, the chemical potentials are specified and the equilibrium density profiles determine the number of particles in the system. In some applications (e.g., transition states), it is desirable to specify the particle number in the system instead. This can be handled by solving a constrained optimization using Lagrange multipliers or equivalently a Michelsen $Q$ function, i.e.,

$$
Q=\Omega+\sum_i\mu_iN_i^\mathrm{spec}
$$

Using the definition of the grand potential from the previous section on the [Euler-Lagrange equation](euler_lagrange_equation.md) (spherical particles for now) leads to

$$
Q=F+\sum_i\int\rho_i(r)V_i^\mathrm{ext}(r)dr+\sum_i\mu_i\left(N_i^\mathrm{spec}-\int\rho_i(r)\mathrm{d}r\right)
$$

which makes clear that the unconstrained optimization of $Q$ with the density profiles and chemical potentials as degrees of freedom is equivalent to a constrained optimization of the total Helmholtz energy (including the external potential contribution) with the chemical potentials as Lagrange multipliers. The stationary points of $Q$ fulfill

$$
\frac{\delta Q}{\delta\rho_i(r)}=F_{\rho_i}(r)+V_i^\mathrm{ext}-\mu_i=0
$$

$$
\frac{\delta Q}{\delta\mu_i}=N_i^\mathrm{spec}-\int\rho_i(r)\mathrm{d}r=0
$$

which can be identified as the Euler-Lagrange equation and the particle number constraint. The two equations can be solved together by using the Euler-Lagranging (now using the [combined expression](euler_lagrange_equation.md))

$$
N_\alpha^\mathrm{spec}=f_\alpha\int e^{-\frac{\beta}{m_\alpha}\left(\hat F_{\rho_\alpha}^\mathrm{res}(r)+V_\alpha^\mathrm{ext}(r)\right)}\prod_{\alpha'}I_{\alpha\alpha'}(r)\mathrm{d}r\equiv f_\alpha z_\alpha
$$

which introduced $z_\alpha$ as the full configurational integral which is by constrution identical for every segment on the same molecule. The fugacity is thus determined from

$$
f_\alpha=\frac{N_\alpha^\mathrm{spec}}{z_\alpha}
$$

In some cases, it can be of interest to specify the total number of particles $N^\mathrm{spec}$ and keep the ratio of fugacities fixed. This is mostly relevant for numerical stability, because it can avoid saddle points in the energy surface of systems that are translationally invariant in one dimension (e.g., a planar interface). The corresponding expression for the fugacity is

$$
f_\alpha=\frac{N^\mathrm{spec}f_\alpha^0}{\sum_\beta f_\beta^0z_\beta}
$$

where $f_\alpha^0$ is the initial fugacity from which the constant ratios of fugacities follows.

In summary, we can extend the Euler-Lagrange equation to other specifications by adding different expressions for the fugacity $f_\alpha$. The full set of equation becomes

$$\rho_\alpha(r)=f_\alpha e^{-\frac{\beta}{m_\alpha}\left(\hat F_{\rho_\alpha}^\mathrm{res}(r)+V_\alpha^\mathrm{ext}(r)\right)}\prod_{\alpha'}I_{\alpha\alpha'}(r)$$

$$I_{\alpha\alpha'}(r)=\int e^{-\frac{\beta}{m_{\alpha'}}\left(\hat F_{\rho_{\alpha'}}^\mathrm{res}(r')+V_{\alpha'}^\mathrm{ext}(r')\right)}\left(\prod_{\alpha''\neq\alpha}I_{\alpha'\alpha''}(r')\right)\omega_\mathrm{chain}^{\alpha\alpha'}(r-r')\mathrm{d}r'$$

|specification|$f_\alpha$|
|:-:|:-:|
|fixed bulk densities $\rho_\alpha^\mathrm{b}$|$\rho_\alpha^\mathrm{b}e^{-\frac{\beta}{m_\alpha}\sum_\gamma\hat F_{\rho_\gamma}^\mathrm{b,res}}$|
|fixed particle numbers $N_\alpha^\mathrm{spec}$|$\frac{N_\alpha^\mathrm{spec}}{z_\alpha}$|
|fixed total particle number $N^\mathrm{spec}$|$\frac{N^\mathrm{spec}f_\alpha^0}{\sum_\beta f_\beta^0z_\beta}$|