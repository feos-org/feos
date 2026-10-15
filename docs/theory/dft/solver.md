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


## Newton algorithm
A Newton iteration is a more refined approach to calculate the roots of the residual $\mathcal{R}$. From a Taylor expansion of the residual

$$\mathcal{R}_\alpha\left(\mathbf{r};\left[\rho(\mathbf{r})+\Delta\rho(\mathbf{r})\right]\right)=\mathcal{R}_\alpha\left(\mathbf{r};\left[\rho(\mathbf{r})\right]\right)+\int\sum_\beta\frac{\delta\mathcal{R}_\alpha\left(\mathbf{r};\left[\rho(\mathbf{r})\right]\right)}{\delta\rho_\beta(\mathbf{r}')}\Delta\rho_\beta(\mathbf{r}')\mathrm{d}\mathbf{r}'+\ldots$$

the Newton step is derived by setting the updated residual $\mathcal{R}_\alpha[\rho(\mathbf{r})+\Delta\rho(\mathbf{r})]$ to 0 and neglecting higher order terms.

$$
\mathcal{R}_\alpha\left(\mathbf{r};\left[\rho(\mathbf{r})\right]\right)=-\int\sum_\beta\frac{\delta\mathcal{R}_\alpha\left(\mathbf{r};\left[\rho(\mathbf{r})\right]\right)}{\delta\rho_\beta(\mathbf{r}')}\Delta\rho_\beta(\mathbf{r}')\mathrm{d}\mathbf{r}'
$$ (eqn:newton)

The linear integral equation has to be solved for the step $\Delta\rho(\mathbf{r})$. Explicitly evaluating the functional derivatives of the residuals is not feasible due to their high dimensionality. Instead, a matrix-free linear solver like GMRES can be used. For GMRES only the action of the linear system on the variable is required (an evaluation of the right-hand side in the equation above for a given $\Delta\rho$). This action can be approximated numerically via

$$\int\sum_\beta\frac{\delta\mathcal{R}_\alpha\left(\mathbf{r};\left[\rho(\mathbf{r})\right]\right)}{\delta\rho_\beta(\mathbf{r}')}\Delta\rho_\beta(\mathbf{r}')\mathrm{d}\mathbf{r}'\approx\frac{\mathcal{R}_\alpha\left(\mathbf{r};\left[\rho(\mathbf{r})+s\Delta\rho(\mathbf{r})\right]\right)-\mathcal{R}_\alpha\left(\mathbf{r};\left[\rho(\mathbf{r})\right]\right)}{s}$$

However this approach requires the choice of an appropriate step size $s$ (something that we want to avoid in $\text{FeO}_\text{s}$) and also an evaluation of the full residual in every step of the linear solver. The solver can be sped up by doing parts of the functional derivative analytically beforehand. Using the definition of the residual in the rhs of eq. {eq}`eqn:newton` and, for now, assuming a constant fugacity $f_\alpha$ (i.e., fixed bulk densities) leads to

$$
\begin{aligned}
q_\alpha(\mathbf{r})&\equiv-\int\sum_\beta\frac{\delta\mathcal{R}_\alpha\left(\mathbf{r};\left[\rho(\mathbf{r})\right]\right)}{\delta\rho_\beta(\mathbf{r}')}\Delta\rho_\beta(\mathbf{r}')\mathrm{d}\mathbf{r}'\\
&=\int\sum_\beta\frac{\delta}{\delta\rho_\beta(\mathbf{r}')}\left(\rho_\alpha(\mathbf{r})-f_\alpha e^{-\frac{\beta}{m_\alpha}\left(\hat F_{\rho_\alpha}^\mathrm{res}(\mathbf{r})+V_\alpha^\mathrm{ext}(\mathbf{r})\right)}\prod_{\alpha'}I_{\alpha\alpha'}(\mathbf{r})\right)\Delta\rho_\beta(\mathbf{r}')\mathrm{d}\mathbf{r}'
\end{aligned}
$$

The functional derivative can be simplified using $\hat F_{\rho_\alpha\rho_\beta}^\mathrm{res}(\mathbf{r},\mathbf{r}')=\frac{\delta \hat F_{\rho_\alpha}^\mathrm{res}(\mathbf{r})}{\delta\rho_\beta(\mathbf{r}')}=\frac{\delta^2\hat F^\mathrm{res}}{\delta\rho_\alpha(\mathbf{r})\delta\rho_\beta(\mathbf{r}')}$

$$
\begin{aligned}
q_\alpha(\mathbf{r})&=\int\sum_\beta\left(\delta_{\alpha\beta}\delta(\mathbf{r}-\mathbf{r}')+\left(\frac{\beta}{m_\alpha}\hat F_{\rho_\alpha\rho_\beta}^\mathrm{res}(\mathbf{r},\mathbf{r}')-\sum_{\alpha'}\frac{1}{I_{\alpha\alpha'}(\mathbf{r})}\frac{\delta I_{\alpha\alpha'}(\mathbf{r})}{\delta\rho_\beta(\mathbf{r}')}\right)\right.\\
&~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\left.\times f_\alpha e^{-\frac{\beta}{m_\alpha}\left(\hat F_{\rho_\alpha}^\mathrm{res}(\mathbf{r})+V_\alpha^\mathrm{ext}(\mathbf{r})\right)}\prod_{\alpha'}I_{\alpha\alpha'}(\mathbf{r})\right)\Delta\rho_\beta(\mathbf{r}')\mathrm{d}\mathbf{r}'\\
&=\Delta\rho_\alpha(\mathbf{r})+\left(\frac{\beta}{m_\alpha}\underbrace{\int\sum_\beta\hat F_{\rho_\alpha\rho_\beta}^\mathrm{res}(\mathbf{r},\mathbf{r}')\Delta\rho_\beta(\mathbf{r}')\mathrm{d}\mathbf{r}'}_{\Delta\hat F_{\rho_\alpha}^\mathrm{res}(\mathbf{r})}-\sum_{\alpha'}\frac{1}{I_{\alpha\alpha'}(\mathbf{r})}\underbrace{\int\sum_\beta\frac{\delta I_{\alpha\alpha'}(\mathbf{r})}{\delta\rho_\beta(\mathbf{r}')}\Delta\rho_\beta(\mathbf{r}')\mathrm{d}\mathbf{r}'}_{\Delta I_{\alpha\alpha'}(\mathbf{r})}\right)\\
&~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\times f_\alpha e^{-\frac{\beta}{m_\alpha}\left(\hat F_{\rho_\alpha}^\mathrm{res}(\mathbf{r})+V_\alpha^\mathrm{ext}(\mathbf{r})\right)}\prod_{\alpha'}I_{\alpha\alpha'}(\mathbf{r})
\end{aligned}
$$

and finally

$$
q_\alpha(\mathbf{r})=\Delta\rho_\alpha(\mathbf{r})+\left(\frac{\beta}{m_\alpha}\Delta\hat F_{\rho_\alpha}^\mathrm{res}(\mathbf{r})-\sum_{\alpha'}\frac{\Delta I_{\alpha\alpha'}(\mathbf{r})}{I_{\alpha\alpha'}(\mathbf{r})}\right)\mathcal{P}_\alpha\left(\mathbf{r};\left[\rho(\mathbf{r})\right]\right)
$$ (eqn:newton_rhs)

Neglecting the second term in eq. {eq}`eqn:newton_rhs` leads to $\Delta\rho_\alpha(\mathbf{r})=\mathcal{R}_\alpha\left(\mathbf{r};\left[\rho(\mathbf{r})\right]\right)$ which is the search direction of the Picard iteration. This observation implies that the Picard iteration is an approximation of the Newton solver that neglects the residual contribution to the Jacobian. Only using the ideal gas contribution to the Jacobian is a reasonable approximation for low densities and therefore, the Picard iteration converges quickly (with a large damping coefficient $\nu$) for low densities.

The second functional derivative of the residual Helmholtz energy can be rewritten in terms of the weight functions.

$$\hat F_{\rho_\alpha\rho_\beta}^\mathrm{res}(\mathbf{r},\mathbf{r}')=\int\frac{\delta\hat f^\mathrm{res}(\mathbf{r}'')}{\delta\rho_\alpha(\mathbf{r})\delta\rho_\beta(\mathbf{r}')}\mathrm{d}\mathbf{r}''=\int\sum_{\alpha\beta}\hat f^\mathrm{res}_{\alpha\beta}(\mathbf{r}'')\frac{\delta n_\alpha(\mathbf{r}'')}{\delta\rho_\alpha(\mathbf{r})}\frac{\delta n_\beta(\mathbf{r}'')}{\delta\rho_\beta(\mathbf{r}')}\mathrm{d}\mathbf{r}''$$

Here $\hat f^\mathrm{res}_{\alpha\beta}=\frac{\partial^2\hat f^\mathrm{res}}{\partial n_\alpha\partial n_\beta}$ is the second partial derivative of the reduced Helmholtz energy density with respect to the weighted densities $n_\alpha$ and $n_\beta$. The definition of the weighted densities $n_\alpha(\mathbf{r})=\sum_i\int\rho_i(\mathbf{r}')\omega_\alpha^i(\mathbf{r}-\mathbf{r}')\mathrm{d}\mathbf{r}'$ is used to simplify the expression further.

$$\hat F_{\rho_\alpha\rho_\beta}^\mathrm{res}(\mathbf{r},\mathbf{r}')=\int\sum_{\alpha\beta}\hat f^\mathrm{res}_{\alpha\beta}(\mathbf{r}'')\omega_\alpha^i(\mathbf{r}''-\mathbf{r})\omega_\beta^j(\mathbf{r}''-\mathbf{r}')\mathrm{d}\mathbf{r}''$$

With the weighted-density variation $\Delta n_\beta(\mathbf{r}'')=\sum_j\int\Delta\rho_j(\mathbf{r}')\omega_\beta^j(\mathbf{r}''-\mathbf{r}')\mathrm{d}\mathbf{r}'$, $\Delta\hat F_{\rho_\alpha}^\mathrm{res}(\mathbf{r})$ can be rewritten as

$$
\begin{aligned}
\Delta\hat F_{\rho_\alpha}^\mathrm{res}(\mathbf{r})&=\int\sum_{\alpha,\beta}\hat f^\mathrm{res}_{\alpha\beta}(\mathbf{r}'')\omega_\alpha^i(\mathbf{r}''-\mathbf{r})\Delta n_\beta(\mathbf{r}'')\mathrm{d}\mathbf{r}''\\
&=\int\sum_\alpha\sum_\beta \hat f^\mathrm{res}_{\alpha\beta}(\mathbf{r}'')\Delta n_\beta(\mathbf{r}'')\omega_\alpha^i(\mathbf{r}''-\mathbf{r})\mathrm{d}\mathbf{r}''\\
&=\int\sum_\alpha \Delta\hat f^\mathrm{res}_\alpha(\mathbf{r}'')\omega_\alpha^i(\mathbf{r}''-\mathbf{r})\mathrm{d}\mathbf{r}''
\end{aligned}
$$ (eqn:newton_F)

To simplify the expression for $\Delta I_{\alpha\alpha'}(\mathbf{r})$, the recursive definition of the bond integrals is used.

$$
\begin{aligned}
\Delta I_{\alpha\alpha'}(\mathbf{r})&=\iint\sum_\beta\frac{\delta}{\delta\rho_\beta(\mathbf{r}'')}\left(e^{-\frac{\beta}{m_{\alpha'}}\left(\hat F_{\rho_{\alpha'}}^\mathrm{res}(\mathbf{r}')+V_{\alpha'}^\mathrm{ext}(\mathbf{r}')\right)}\left(\prod_{\alpha''\neq\alpha}I_{\alpha'\alpha''}(\mathbf{r}')\right)\omega_\mathrm{chain}^{\alpha\alpha'}(\mathbf{r}-\mathbf{r}')\right)\\
&~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\times\Delta\rho_\beta(\mathbf{r}'')\mathrm{d}\mathbf{r}'\mathrm{d}\mathbf{r}''\\
&=\iint\sum_\beta\left(-\frac{\beta}{m_{\alpha'}}\hat F_{\rho_{\alpha'}\rho_\beta}^\mathrm{res}(\mathbf{r}',\mathbf{r}'')+\sum_{\alpha''\neq\alpha}\frac{1}{I_{\alpha'\alpha''}(\mathbf{r}')}\frac{\delta I_{\alpha'\alpha''}(\mathbf{r}')}{\delta\rho_\beta(\mathbf{r}'')}\right)e^{-\frac{\beta}{m_{\alpha'}}\left(\hat F_{\rho_{\alpha'}}^\mathrm{res}(\mathbf{r}')+V_{\alpha'}^\mathrm{ext}(\mathbf{r}')\right)}\\
&~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\times\left(\prod_{\alpha''\neq\alpha}I_{\alpha'\alpha''}(\mathbf{r}')\right)\omega_\mathrm{chain}^{\alpha\alpha'}(\mathbf{r}-\mathbf{r}')\Delta\rho_\beta(\mathbf{r}'')\mathrm{d}\mathbf{r}'\mathrm{d}\mathbf{r}''
\end{aligned}
$$

Here, the definition of $\Delta\hat F_{\rho_\alpha}^\mathrm{res}(\mathbf{r})$ and $\Delta I_{\alpha\alpha'}(\mathbf{r})$ can be inserted leading to a recursive calculation of $\Delta I_{\alpha\alpha'}(\mathbf{r})$ similar to the original bond integrals.

$$
\begin{aligned}
\Delta I_{\alpha\alpha'}(\mathbf{r})&=\int\left(-\frac{\beta}{m_{\alpha'}}\Delta\hat F_{\rho_{\alpha'}}^\mathrm{res}(\mathbf{r}')+\sum_{\alpha''\neq\alpha}\frac{\Delta I_{\alpha'\alpha''}(\mathbf{r}')}{I_{\alpha'\alpha''}(\mathbf{r}')}\right)\\
&\qquad\times e^{-\frac{\beta}{m_{\alpha'}}\left(\hat F_{\rho_{\alpha'}}^\mathrm{res}(\mathbf{r}')+V_{\alpha'}^\mathrm{ext}(\mathbf{r}')\right)}\left(\prod_{\alpha''\neq\alpha}I_{\alpha'\alpha''}(\mathbf{r}')\right)\omega_\mathrm{chain}^{\alpha\alpha'}(\mathbf{r}-\mathbf{r}')\mathrm{d}\mathbf{r}'
\end{aligned}
$$ (eqn:newton_I)

If the number of particles is specified instead of the bulk densities (see the section on [constrained DFT](particle_constraint.md)), the fugacities depend on the density profiles through the configurational integrals $z_\alpha$. Then, the functional derivative of the fugacity has to be included in eq. {eq}`eqn:newton_rhs`, leading to

$$
q_\alpha(\mathbf{r})=\Delta\rho_\alpha(\mathbf{r})+\left(\frac{\beta}{m_\alpha}\Delta\hat F_{\rho_\alpha}^\mathrm{res}(\mathbf{r})-\sum_{\alpha'}\frac{\Delta I_{\alpha\alpha'}(\mathbf{r})}{I_{\alpha\alpha'}(\mathbf{r})}-\frac{\Delta f_\alpha}{f_\alpha}\right)\mathcal{P}_\alpha\left(\mathbf{r};\left[\rho(\mathbf{r})\right]\right)
$$ (eqn:newton_rhs_f)

with $\Delta f_\alpha=\int\sum_\beta\frac{\delta f_\alpha}{\delta\rho_\beta(\mathbf{r}')}\Delta\rho_\beta(\mathbf{r}')\mathrm{d}\mathbf{r}'$. The expression for $\Delta f_\alpha$ follows from the variation of the configurational integrals

$$
\Delta z_\alpha=-\int\left(\frac{\beta}{m_\alpha}\Delta\hat F_{\rho_\alpha}^\mathrm{res}(\mathbf{r})-\sum_{\alpha'}\frac{\Delta I_{\alpha\alpha'}(\mathbf{r})}{I_{\alpha\alpha'}(\mathbf{r})}\right)e^{-\frac{\beta}{m_\alpha}\left(\hat F_{\rho_\alpha}^\mathrm{res}(\mathbf{r})+V_\alpha^\mathrm{ext}(\mathbf{r})\right)}\prod_{\alpha'}I_{\alpha\alpha'}(\mathbf{r})\mathrm{d}\mathbf{r}
$$ (eqn:newton_z)

and depends on the specification.

|specification|$f_\alpha$|$\frac{\Delta f_\alpha}{f_\alpha}$|
|:-:|:-:|:-:|
|fixed bulk densities $\rho_\alpha^\mathrm{b}$|$\rho_\alpha^\mathrm{b}e^{\frac{\beta}{m_\alpha}\sum_\gamma\hat F_{\rho_\gamma}^\mathrm{b,res}}$|$0$|
|fixed particle numbers $N_\alpha^\mathrm{spec}$|$\frac{N_\alpha^\mathrm{spec}}{z_\alpha}$|$-\frac{\Delta z_\alpha}{z_\alpha}$|
|fixed total particle number $N^\mathrm{spec}$|$\frac{N^\mathrm{spec}f_\alpha^0}{\sum_\beta f_\beta^0z_\beta}$|$-\frac{\sum_\beta f_\beta^0\Delta z_\beta}{\sum_\beta f_\beta^0z_\beta}$|

In every iteration of GMRES, $q(\mathbf{r})$ needs to be evaluated from eqs. {eq}`eqn:newton_rhs_f`, {eq}`eqn:newton_F`, {eq}`eqn:newton_I` and {eq}`eqn:newton_z`. The operations required for that are analogous to the calculation of weighted densities and the functional derivative in the Euler-Lagrange equation itself. Details of GMRES, including the pseudocode that the implementation in $\text{FeO}_\text{s}$ is based on, are given on [Wikipedia](https://de.wikipedia.org/wiki/GMRES-Verfahren) (German).

The Newton solver converges exceptionally fast compared to a simple Picard iteration. The faster convergence comes at the cost of requiring multiple steps for solving the linear subsystem. With the algorithm outlined here, the evaluation of the second partial derivatives of the Helmholtz energy density is only required once for every Newton step. The GMRES algorithm only uses the very efficient convolution integrals and no additional evaluation of the model.

## Anderson mixing
