# Predictive density gradient theory

Predictive density gradient theory (pDGT)  is an efficient approach for the prediction of surface tensions, which is derived from non-local DFT, see [Rehner et al. (2018)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.98.063312). A gradient expansion is applied to the weighted densities of the Helmholtz energy functional to second order as well as to the Helmholtz energy density to first order. 

Weighted densities (in non-local DFT) are determined from

$$ n_\alpha(\mathbf{r})=\sum_in_\alpha^i(\mathbf{r})=\sum_i\int\rho_i(\mathbf{r}- \mathbf{r}')\omega_\alpha^i(\mathbf{r}')\mathrm{d}\mathbf{r}'.$$

These convolutions are time-consuming calculations. Therefore, these equations are simplified by using a Taylor expansion around $\mathbf{r}$ for the density of each component $\rho_i$ as

$$\rho_i(\mathbf{r}-\mathbf{r}')=\rho_i(\mathbf{r})-\nabla\rho_i(\mathbf{r})\cdot \mathbf{r}'+\frac{1}{2}\nabla\nabla\rho(\mathbf{r}):\mathbf{r}'\mathbf{r}'+\ldots$$

In the convolution integrals, the integration over angles can now be performed analytically for the spherically symmetric weight functions $\omega_\alpha^i(\mathbf{r})=\omega_\alpha^i(r)$
which provides

$$ n_\alpha^i(\mathbf{r})=\rho_i(\mathbf{r})\underbrace{4\pi\int_0^\infty \omega_\alpha^i(r)r^2\mathrm{d} r}_{\omega_\alpha^{i0}}
	+\nabla^2\rho_i(\mathbf{r})\underbrace{\frac{2}{3}\pi\int_0^\infty\omega_\alpha^i(r)r^4\mathrm{d} r}_{\omega_\alpha^{i2}}+\ldots$$

with the weight constants $\omega_\alpha^{i0}$ and $\omega_\alpha^{i2}$. 

The resulting weighted densities can be split into a local part $n_\alpha^0(\mathbf{r})$ and an excess part $\Delta n_\alpha(\mathbf{r})$ as

$$n_\alpha(\mathbf{r})=\underbrace{\sum_i\rho_i(\mathbf{r}) \omega_\alpha^{i0}}_{n_\alpha^0} +\underbrace{\sum_i\nabla^2\rho_i(\mathbf{r})\omega_\alpha^{i2}+\ldots}_{\Delta n_\alpha}.$$


The second simplification is the expansion of the reduced residual
Helmholtz energy density $\Phi(\{ n_\alpha\})$ around the local density approximation truncated after the second term

$$ \Phi(\lbrace n_\alpha\rbrace)
	=\Phi(\lbrace n_\alpha^0\rbrace)
	+\sum_i\sum_\alpha\frac{\partial\Phi}{\partial n_\alpha}\omega_\alpha^{i2}\nabla^2\rho_i + \ldots $$

The Helmholtz energy functional (which was introduced in the section about the [Euler-Lagrange equation](euler_lagrange_equation.md)) then reads

$$	F[\mathbf{\rho}(\mathbf{r})]=\int\left(f(\mathbf{\rho})+\sum_{ij}\frac{c_{ij}(\mathbf{\rho})}{2}\nabla\rho_i\cdot\nabla\rho_j\right)\mathrm{d}\mathbf{r}$$

with the density dependent influence parameter

$$	\beta c_{ij}(\mathbf{\rho})=-\sum_{\alpha\beta}\frac{\partial^2\Phi}{\partial n_\alpha\partial n_\beta}\left(\omega_\alpha^{i2}\omega_\beta^{j0}+ \omega_\alpha^{i0}\omega_\beta^{j2}\right).$$

and the local Helmholtz energy density $f(\mathbf{\rho})$.



For pure components, as derived in the original publication, the surface tension can be calculated from the surface excess grand potential per area according to

$$	\gamma=\frac{F-\mu N+pV}{A}=\int_{\rho^\mathrm{V}}^{\rho^\mathrm{L}}   \sqrt{2c \left(f(\rho)-\rho\mu+p\right) } d\rho $$


Thus, no iterative solver is necessary to calculate the surface tension of pure components, which is a major advantage of pDGT. Finally, the density profile can be calculated from

$$ z(\rho)=\int_{\rho^\mathrm{V}}^{\rho}   \sqrt{\frac{c/2}{ f(\rho)-\rho\mu+p} } d\rho $$

