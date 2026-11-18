# Derivatives of density profiles
For converged density profiles equilibrium properties can be calculated as partial derivatives of thermodynamic potentials analogous to classical (bulk) thermodynamics. The difference is that the derivatives have to be along a path of valid density profiles (solutions of the [Euler-Lagrange equation](euler_lagrange_equation.md)).

## Implicit differentiation
The density profiles and the fractions of non-bonded sites are defined implicitly as the root of the residual $\mathcal{R}\left(\mathbf{r};\left[x(\mathbf{r})\right]\right)$ that is introduced in the section on the [Newton solver](solver.md), with the unknowns $x(\mathbf{r})=\left(\ln\rho_\alpha(\mathbf{r}),\ln X_A(\mathbf{r})\right)$. If the residual depends explicitly on a parameter $\theta$ (e.g., a chemical potential, the pressure, or the temperature), the solution $x(\mathbf{r};\theta)$ fulfills

$$\mathcal{R}\left(\mathbf{r};\left[x(\mathbf{r};\theta)\right],\theta\right)=0$$

for every $\theta$. Then, the total derivative with respect to $\theta$ has to vanish as well, which leads to

$$
\frac{\mathrm{d}\mathcal{R}\left(\mathbf{r};\left[x(\mathbf{r})\right]\right)}{\mathrm{d}\theta}=\int\frac{\delta\mathcal{R}\left(\mathbf{r};\left[x(\mathbf{r})\right]\right)}{\delta x(\mathbf{r}')}\frac{\partial x(\mathbf{r}')}{\partial\theta}\mathrm{d}\mathbf{r}'+\frac{\partial\mathcal{R}\left(\mathbf{r};\left[x(\mathbf{r})\right]\right)}{\partial\theta}=0
$$ (eqn:implicit_derivative)

Eq. {eq}`eqn:implicit_derivative` is a linear integral equation for the derivatives $\frac{\partial x(\mathbf{r})}{\partial\theta}$ with the same Jacobian as the Newton step. It is solved with GMRES and the linearization of the residual, exactly as described for the [Newton solver](solver.md), only with a different right-hand side. The derivatives of the logarithms of the fractions of non-bonded sites are obtained along the way, and the derivatives of the density profiles follow as

$$\frac{\partial\rho_\alpha(\mathbf{r})}{\partial\theta}=\rho_\alpha(\mathbf{r})\frac{\partial\ln\rho_\alpha(\mathbf{r})}{\partial\theta}$$

## Specified chemical potentials
Written out with the [full expression](euler_lagrange_equation.md) of the Euler-Lagrange equation, the residual is

$$
\mathcal{R}\left(\mathbf{r};\left[x(\mathbf{r})\right]\right)=\begin{pmatrix}\ln f_\alpha-\frac{\beta}{m_\alpha}\left(\hat F_{\rho_\alpha}^\mathrm{res}(\mathbf{r})+V_\alpha^\mathrm{ext}(\mathbf{r})\right)+\sum_{\alpha'}\ln I_{\alpha\alpha'}(\mathbf{r})-\ln\rho_\alpha(\mathbf{r})\\-\ln\left(1+S_A(\mathbf{r})\right)-\ln X_A(\mathbf{r})\end{pmatrix}
$$ (eqn:residual)

with the density rows $\mathcal{R}_\alpha$ and the rows of the mass action equations $\mathcal{R}_A$. The fugacities only enter the density rows, and they do so linearly. If the chemical potentials are specified, the fugacities $f_\alpha$ are constant and do not contribute to the Jacobian. They are related to the chemical potentials via $m_\alpha\ln f_\alpha=\beta\mu_i-\ln\Lambda_i^3$, where $i$ is the component that segment $\alpha$ belongs to. The partial derivative of the residual in eq. {eq}`eqn:implicit_derivative` for the derivatives with respect to the chemical potentials, which yield $\left(\frac{\partial\rho_\alpha(\mathbf{r})}{\partial\mu_k}\right)_{T,\mu_{j\neq k}}$, is therefore simply

$$\left(\frac{\partial\mathcal{R}}{\partial\beta\mu_k}\right)_{T,\mu_{j\neq k}}=\begin{pmatrix}\left(\frac{\partial\ln f_\alpha}{\partial\beta\mu_k}\right)_{T,\mu_{j\neq k}}\\0\end{pmatrix}=\begin{pmatrix}\frac{\delta_{ik}}{m_\alpha}\\0\end{pmatrix}$$

Instead of the chemical potentials, the fugacities can also be expressed by the state of the bulk phase that is in equilibrium with the inhomogeneous system

$$
m_\alpha\ln f_\alpha=\ln\left(\frac{x_i^\mathrm{b}p\varphi_i}{k_\mathrm{B}T}\right)
$$ (eqn:fugacity_bulk)

with the mole fraction $x_i^\mathrm{b}$ and the fugacity coefficient $\varphi_i$ of component $i$ in the bulk phase. With eq. {eq}`eqn:fugacity_bulk`, the derivatives with respect to the pressure and the temperature of the bulk phase at constant bulk composition can be calculated in the same way. For the pressure, the partial derivative of the residual

$$\left(\frac{\partial\mathcal{R}}{\partial\beta p}\right)_{T,x_k^\mathrm{b}}=\begin{pmatrix}\left(\frac{\partial\ln f_\alpha}{\partial\beta p}\right)_{T,x_k^\mathrm{b}}\\0\end{pmatrix}=\begin{pmatrix}\frac{v_i}{m_\alpha}\\0\end{pmatrix}$$

with the partial molar volume $v_i$ of the bulk phase yields $\left(\frac{\partial\rho_\alpha(\mathbf{r})}{\partial p}\right)_{T,x_k^\mathrm{b}}$. For the temperature, the residual depends on the temperature both through the fugacities and explicitly, and the partial derivative of the residual for $\left(\frac{\partial\rho_\alpha(\mathbf{r})}{\partial T}\right)_{p,x_k^\mathrm{b}}$ is

$$\left(\frac{\partial\mathcal{R}}{\partial T}\right)_{p,x_k^\mathrm{b}}=\begin{pmatrix}\left(\frac{\partial\mathcal{R}_\alpha}{\partial T}\right)_{f_k}+\left(\frac{\partial\ln f_\alpha}{\partial T}\right)_{p,x_k^\mathrm{b}}\\\left(\frac{\partial\mathcal{R}_A}{\partial T}\right)_{f_k}\end{pmatrix}=\left(\frac{\partial\mathcal{R}}{\partial T}\right)_{f_k}+\begin{pmatrix}\frac{1}{m_\alpha}\left(\left(\frac{\partial\ln\varphi_i}{\partial T}\right)_{p,x_k^\mathrm{b}}-\frac{1}{T}\right)\\0\end{pmatrix}$$

Here, the derivative $\left(\frac{\partial\mathcal{R}}{\partial T}\right)_{f_k}$ is obtained using automatic differentiation of the residual with the bulk chemical potential (fugacity) specification.

## Specified number of particles
Derivatives at constant numbers of particles are calculated by replacing the fugacities in eq. {eq}`eqn:residual` with

$$\ln f_\alpha=\ln N_i-\ln z_\alpha$$

which is exactly what the specification of the number of particles does in the solver (see the section on [constrained DFT](particle_constraint.md)). A converged density profile is a solution for its chemical potentials and for its numbers of particles at the same time. Therefore, both kinds of derivatives can be calculated for every density profile, independent of the specification that was used to solve it. Because the configurational integrals $z_\alpha$ depend on the density profiles, the Jacobian differs from the one for specified chemical potentials, which is included automatically by the linearization of the residual. The partial derivative of the residual for the derivatives with respect to the numbers of particles, which yield $\left(\frac{\partial\rho_\alpha(\mathbf{r})}{\partial N_k}\right)_{T,N_{j\neq k}}$, is

$$\left(\frac{\partial\mathcal{R}}{\partial N_k}\right)_{T,N_{j\neq k}}=\begin{pmatrix}\left(\frac{\partial\ln f_\alpha}{\partial N_k}\right)_{T,N_{j\neq k}}\\0\end{pmatrix}=\begin{pmatrix}\frac{\delta_{ik}}{N_i}\\0\end{pmatrix}$$

For the temperature, the partial derivative of the residual for $\left(\frac{\partial\rho_\alpha(\mathbf{r})}{\partial T}\right)_{N_k}$ is $\left(\frac{\partial\mathcal{R}}{\partial T}\right)_{N_k}$, which is obtained using automatic differentiation of the residual with the specification of the number of particles. It contains the temperature dependence of the configurational integrals $z_\alpha$, and no bulk phase is required.

## Derivatives of the number of particles
The derivatives of the number of particles follow by integrating the derivatives of the density profiles

$$\frac{\partial N_i}{\partial\theta}=\int\frac{\partial\rho_\alpha(\mathbf{r})}{\partial\theta}\mathrm{d}\mathbf{r}$$

with any segment $\alpha$ of component $i$ (by construction, the integral is the same for all segments of a molecule).

## Summary
All derivatives are obtained from eq. {eq}`eqn:implicit_derivative` with the Jacobian for the respective specification and the following partial derivatives of the residual.

|derivative|specification|$\frac{\partial\mathcal{R}_\alpha}{\partial\theta}$|$\frac{\partial\mathcal{R}_A}{\partial\theta}$|
|:-:|:-:|:-:|:-:|
|$\left(\frac{\partial\rho_\alpha(\mathbf{r})}{\partial\beta\mu_k}\right)_{T,\mu_{j\neq k}}$|chemical potential|$\frac{\delta_{ik}}{m_\alpha}$|$0$|
|$\left(\frac{\partial\rho_\alpha(\mathbf{r})}{\partial\beta p}\right)_{T,x_k^\mathrm{b}}$|chemical potential|$\frac{v_i}{m_\alpha}$|$0$|
|$\left(\frac{\partial\rho_\alpha(\mathbf{r})}{\partial T}\right)_{p,x_k^\mathrm{b}}$|chemical potential|$\left(\frac{\partial\mathcal{R}_\alpha}{\partial T}\right)_{f_k}+\frac{1}{m_\alpha}\left(\left(\frac{\partial\ln\varphi_i}{\partial T}\right)_{p,x_k^\mathrm{b}}-\frac{1}{T}\right)$|$\left(\frac{\partial\mathcal{R}_A}{\partial T}\right)_{f_k}$|
|$\left(\frac{\partial\rho_\alpha(\mathbf{r})}{\partial N_k}\right)_{T,N_{j\neq k}}$|number of particles|$\frac{\delta_{ik}}{N_i}$|$0$|
|$\left(\frac{\partial\rho_\alpha(\mathbf{r})}{\partial T}\right)_{N_k}$|number of particles|$\left(\frac{\partial\mathcal{R}_\alpha}{\partial T}\right)_{N_k}$|$\left(\frac{\partial\mathcal{R}_A}{\partial T}\right)_{N_k}$|
