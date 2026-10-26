# DFT solvers
Different solvers can be used to calculate the density profiles from the Euler-Lagrange equation introduced previously. The solvers differ in their stability, the rate of convergence, and the execution time. Unfortunately, the optimal solver and solver parameters depend on the studied system.

## Picard iteration
The form of the Euler-Lagrange equation

$$\rho_\alpha(\mathbf{r})=\underbrace{f_\alpha e^{-\frac{\beta}{m_\alpha}\left(\hat F_{\rho_\alpha}^\mathrm{res}(\mathbf{r})+V_\alpha^\mathrm{ext}(\mathbf{r})\right)}\prod_{\alpha'}I_{\alpha\alpha'}(\mathbf{r})}_{\equiv \mathcal{P}_\alpha(\mathbf{r};[\rho(\mathbf{r})])}$$

suggests using a simple fixed point iteration

$$\rho_\alpha^{(k+1)}(\mathbf{r})=\mathcal{P}_\alpha\left(\mathbf{r};\left[\rho^{(k)}(\mathbf{r})\right]\right)$$

Except for some systems – typically at low densities – this iteration is unstable. Instead the new solution is obtained as combination of the old solution and the projected solution $\mathcal{P}$

$$\rho_\alpha^{(k+1)}(\mathbf{r})=(1-\nu)\rho_\alpha^{(k)}(\mathbf{r})+\nu\mathcal{P}_\alpha\left(\mathbf{r};\left[\rho^{(k)}(\mathbf{r})\right]\right)$$

The weighting between the old and projected solution is specified by the damping coefficient $\nu$. The expression can be rewritten as 

$$\rho_\alpha^{(k+1)}(\mathbf{r})=\rho_\alpha^{(k)}(\mathbf{r})+\nu\Delta\rho_\alpha^{(k)}(\mathbf{r})$$

with the search direction $\Delta\rho_\alpha(\mathbf{r})$ which is identical to the residual $\mathcal{R}_\alpha\left(\mathbf{r};\left[\rho(\mathbf{r})\right]\right)$

$$\Delta\rho_\alpha(\mathbf{r})=\mathcal{R}_\alpha\left(\mathbf{r};\left[\rho(\mathbf{r})\right]\right)\equiv\mathcal{P}_\alpha\left(\mathbf{r};\left[\rho(\mathbf{r})\right]\right)-\rho_\alpha(\mathbf{r})$$

The Euler-Lagrange equation can be reformulated as the "logarithmic" version

$$\ln\rho_\alpha(\mathbf{r})=\ln\mathcal{P}_\alpha\left(\mathbf{r};\left[\rho(\mathbf{r})\right]\right)$$

Then repeating the same steps as above leads to the "logarithmic" Picard iteration

$$\ln\rho_\alpha^{(k+1)}(\mathbf{r})=\ln\rho_\alpha^{(k)}(\mathbf{r})+\nu\Delta\ln\rho_\alpha^{(k)}(\mathbf{r})$$

or

$$\rho_\alpha^{(k+1)}(\mathbf{r})=\rho_\alpha^{(k)}(\mathbf{r})e^{\nu\Delta\ln\rho_\alpha^{(k)}(\mathbf{r})}$$

with

$$\Delta\ln\rho_\alpha(\mathbf{r})=\hat{\mathcal{R}}_\alpha\left(\mathbf{r};\left[\rho(\mathbf{r})\right]\right)\equiv\ln\mathcal{P}_\alpha\left(\mathbf{r};\left[\rho(\mathbf{r})\right]\right)-\ln\rho_\alpha(\mathbf{r})$$

For associating molecules, the projected density $\mathcal{P}_\alpha\left(\mathbf{r};\left[\rho(\mathbf{r}),X(\mathbf{r})\right]\right)$ also depends on the fractions of non-bonded sites. These are iterated using the mass action equations (eq. {eq}`eqn:mass_action`). In every iteration, the fractions of non-bonded sites are updated with a damped successive substitution

$$\ln X_A^{(k+1)}(\mathbf{r})=\frac{1}{2}\left(\ln X_A^{(k)}(\mathbf{r})-\ln\left(1+S_A^{(k)}(\mathbf{r})\right)\right)$$

before the density profiles are updated as described above. The iteration is converged if both the density residual and the residual of the mass action equations vanish.

## Newton algorithm
A Newton iteration is a more refined approach to calculate the roots of the residual. In $\text{FeO}_\text{s}$, the Newton solver uses the logarithmic form of both the Euler-Lagrange equation and the mass action equations. The unknowns are combined into $x(\mathbf{r})=\left(\ln\rho_\alpha(\mathbf{r}),\ln X_A(\mathbf{r})\right)$ and the residual is

$$
\mathcal{R}\left(\mathbf{r};\left[x(\mathbf{r})\right]\right)=\begin{pmatrix}\ln\mathcal{P}_\alpha\left(\mathbf{r};\left[\rho(\mathbf{r}),X(\mathbf{r})\right]\right)-\ln\rho_\alpha(\mathbf{r})\\-\ln\left(1+S_A(\mathbf{r})\right)-\ln X_A(\mathbf{r})\end{pmatrix}
$$

From a Taylor expansion of the residual

$$\mathcal{R}\left(\mathbf{r};\left[x(\mathbf{r})+\Delta x(\mathbf{r})\right]\right)=\mathcal{R}\left(\mathbf{r};\left[x(\mathbf{r})\right]\right)+\int\frac{\delta\mathcal{R}\left(\mathbf{r};\left[x(\mathbf{r})\right]\right)}{\delta x(\mathbf{r}')}\Delta x(\mathbf{r}')\mathrm{d}\mathbf{r}'+\ldots$$

the Newton step is derived by setting the updated residual to 0 and neglecting higher order terms.

$$
\int\frac{\delta\mathcal{R}\left(\mathbf{r};\left[x(\mathbf{r})\right]\right)}{\delta x(\mathbf{r}')}\Delta x(\mathbf{r}')\mathrm{d}\mathbf{r}'=-\mathcal{R}\left(\mathbf{r};\left[x(\mathbf{r})\right]\right)
$$ (eqn:newton)

The linear integral equation has to be solved for the step $\Delta x(\mathbf{r})$. In the following, the left-hand side is abbreviated with the Jacobian $J$, an integral operator whose action on a profile $v(\mathbf{r})$ with the same structure as $\Delta x(\mathbf{r})$ is

$$\left(Jv\right)(\mathbf{r})\equiv\int\frac{\delta\mathcal{R}\left(\mathbf{r};\left[x(\mathbf{r})\right]\right)}{\delta x(\mathbf{r}')}v(\mathbf{r}')\mathrm{d}\mathbf{r}'$$

Explicitly evaluating the functional derivatives of the residual is not feasible due to their high dimensionality. Instead, we apply the Newton-Krylov method that uses the matrix-free linear solver GMRES, which only requires the action of the Jacobian on a profile $v(\mathbf{r})$, i.e., the Jacobian-vector product (JVP) $Jv$. The JVP is calculated exactly by linearizing the residual. All operations that are required to evaluate the residual are either linear (convolutions, integrals, sums) or local nonlinear functions (e.g., exponentials, logarithms, products, and the Helmholtz energy densities and contact values as functions of the weighted densities). When the residual is evaluated at the beginning of a Newton step, the derivatives of all local functions are stored. The JVP is then obtained by repeating the same sequence of operations with $v$, where every linear operation is applied as before and every local function is replaced by its stored derivative. This only requires convolutions and local products and no additional evaluation of the model. The bond integrals and the fugacities for the different specifications (see the section on [constrained DFT](particle_constraint.md)) are included in the same way. Details of GMRES, including the pseudocode that the implementation in $\text{FeO}_\text{s}$ is based on, are given on [Wikipedia](https://de.wikipedia.org/wiki/GMRES-Verfahren) (German).

For the density rows, the functional derivatives of the residual with respect to the logarithmic densities are

$$\frac{\delta\mathcal{R}_\alpha\left(\mathbf{r};\left[x(\mathbf{r})\right]\right)}{\delta\ln\rho_\beta(\mathbf{r}')}=-\delta_{\alpha\beta}\delta(\mathbf{r}-\mathbf{r}')+\frac{\delta\ln\mathcal{P}_\alpha\left(\mathbf{r};\left[\rho(\mathbf{r}),X(\mathbf{r})\right]\right)}{\delta\ln\rho_\beta(\mathbf{r}')}$$

Neglecting the second term (and the dependence on $X_A$) leads to $\Delta\ln\rho_\alpha(\mathbf{r})=\hat{\mathcal{R}}_\alpha\left(\mathbf{r};\left[\rho(\mathbf{r})\right]\right)$, which is the search direction of the logarithmic Picard iteration. This observation implies that the Picard iteration is an approximation of the Newton solver that neglects the residual contribution to the Jacobian. Only using the ideal gas contribution to the Jacobian is a reasonable approximation for low densities and therefore, the Picard iteration converges quickly (with a large damping coefficient $\nu$) for low densities.

The Newton solver converges exceptionally fast compared to a simple Picard iteration. The faster convergence comes at the cost of requiring multiple steps for solving the linear subsystem. To keep this cost low, the linear system is only solved approximately (inexact Newton method), i.e., GMRES stops as soon as

$$\left\|J\Delta x^{(k)}+\mathcal{R}\left[x^{(k)}\right]\right\|_2\leq\max\left(10^{-3}\left\|\mathcal{R}\left[x^{(k)}\right]\right\|_2,10^{-2}\,\mathrm{tol}\right)$$

where $\mathcal{R}\left[x^{(k)}\right]$ is short for the residual of iteration $k$ and $\mathrm{tol}$ is the tolerance of the Newton iteration. Far from the solution, the linear system is solved only roughly, close to the solution, it is solved more accurately. In addition, the step size is controlled. The new solution is

$$x^{(k+1)}(\mathbf{r})=x^{(k)}(\mathbf{r})+\lambda\Delta x^{(k)}(\mathbf{r})$$

where the step size starts from

$$\lambda=\min\left(1,\frac{10}{\max_\mathbf{r}\left|\Delta x^{(k)}(\mathbf{r})\right|}\right)$$

so that no logarithm changes by more than 10, and is halved (backtracking line search) until the norm of the residual decreases sufficiently (Armijo condition)

$$\left\|\mathcal{R}\left[x^{(k+1)}\right]\right\|<\left(1-10^{-4}\lambda\right)\left\|\mathcal{R}\left[x^{(k)}\right]\right\|$$

## Anderson mixing
For historic reasons, $\text{FeO}_\text{s}$ contains an implementation of the Anderson mixing method that uses a history of previous solutions to approximate the system's Jacobian. While it might be a fast method for some systems, in general, the algorithm is significantly less stable than the default Newton solver.