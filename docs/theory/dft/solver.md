# DFT solvers
Different solvers can be used to calculate the density profiles from the Euler-Lagrange equation introduced previously. The solvers differ in their stability, the rate of convergence, and the execution time. Unfortunately, the optimal solver and solver parameters depend on the studied system.

## Picard iteration
The form of the Euler-Lagrange equation

$$\rho_\alpha(r)=\underbrace{\rho_\alpha^\mathrm{b}e^{\frac{\beta}{m_\alpha}\left(\hat F_{\rho_\alpha}^\mathrm{b,res}-\hat F_{\rho_\alpha}^\mathrm{res}(r)-V_\alpha^\mathrm{ext}(r)\right)}\prod_{\alpha'}I_{\alpha\alpha'}(r)}_{\equiv \mathcal{P}_\alpha(r;[\rho(r)])}$$

suggests using a simple fixed point iteration

$$\rho_\alpha^{(k+1)}(r)=\mathcal{P}_\alpha\left(r;\left[\rho^{(k)}(r)\right]\right)$$

Except for some systems – typically at low densities – this iteration is unstable. Instead the new solution is obtained as combination of the old solution and the projected solution $\mathcal{P}$

$$\rho_\alpha^{(k+1)}(r)=(1-\nu)\rho_\alpha^{(k)}(r)+\nu\mathcal{P}_\alpha\left(r;\left[\rho^{(k)}(r)\right]\right)$$

The weighting between the old and projected solution is specified by the damping coefficient $\nu$. The expression can be rewritten as 

$$\rho_\alpha^{(k+1)}(r)=\rho_\alpha^{(k)}(r)+\nu\Delta\rho_\alpha^{(k)}(r)$$

with the search direction $\Delta\rho_\alpha(r)$ which is identical to the residual $\mathcal{F}_\alpha\left(r;\left[\rho(r)\right]\right)$

$$\Delta\rho_\alpha(r)=\mathcal{F}_\alpha\left(r;\left[\rho(r)\right]\right)\equiv\mathcal{P}_\alpha\left(r;\left[\rho(r)\right]\right)-\rho_\alpha(r)$$

The Euler-Lagrange equation can be reformulated as the "logarithmic" version

$$\ln\rho_\alpha(r)=\ln\mathcal{P}_\alpha\left(r;\left[\rho(r)\right]\right)$$

Then repeating the same steps as above leads to the "logarithmic" Picard iteration

$$\ln\rho_\alpha^{(k+1)}(r)=\ln\rho_\alpha^{(k)}(r)+\nu\Delta\ln\rho_\alpha^{(k)}(r)$$

or

$$\rho_\alpha^{(k+1)}(r)=\rho_\alpha^{(k)}(r)e^{\nu\Delta\ln\rho_\alpha^{(k)}(r)}$$

with

$$\Delta\ln\rho_\alpha(r)=\mathcal{\hat F}_\alpha\left(r;\left[\rho(r)\right]\right)\equiv\ln\mathcal{P}_\alpha\left(r;\left[\rho(r)\right]\right)-\ln\rho_\alpha(r)$$


## Newton algorithm
A Newton iteration is a more refined approach to calculate the roots of the residual $\mathcal{F}$. From a Taylor expansion of the residual

$$\mathcal{F}_\alpha\left(r;\left[\rho(r)+\Delta\rho(r)\right]\right)=\mathcal{F}_\alpha\left(r;\left[\rho(r)\right]\right)+\int\sum_\beta\frac{\delta\mathcal{F}_\alpha\left(r;\left[\rho(r)\right]\right)}{\delta\rho_\beta(r')}\Delta\rho_\beta(r')\mathrm{d}r'+\ldots$$

the Newton step is derived by setting the updated residual $\mathcal{F}_\alpha[\rho(r)+\Delta\rho(r)]$ to 0 and neglecting higher order terms.

$$\mathcal{F}_\alpha\left(r;\left[\rho(r)\right]\right)=-\int\sum_\beta\frac{\delta\mathcal{F}_\alpha\left(r;\left[\rho(r)\right]\right)}{\delta\rho_\beta(r')}\Delta\rho_\beta(r')\mathrm{d}r'$$ (eqn:newton)

The linear integral equation has to be solved for the step $\Delta\rho(r)$. Explicitly evaluating the functional derivatives of the residuals is not feasible due to their high dimensionality. Instead, a matrix-free linear solver like GMRES can be used. For GMRES only the action of the linear system on the variable is required (an evaluation of the right-hand side in the equation above for a given $\Delta\rho$). This action can be approximated numerically via

$$\int\sum_\beta\frac{\delta\mathcal{F}_\alpha\left(r;\left[\rho(r)\right]\right)}{\delta\rho_\beta(r')}\Delta\rho_\beta(r')\mathrm{d}r'\approx\frac{\mathcal{F}_\alpha\left(r;\left[\rho(r)+s\Delta\rho(r)\right]\right)-\mathcal{F}_\alpha\left(r;\left[\rho(r)\right]\right)}{s}$$

However this approach requires the choice of an appropriate step size $s$ (something that we want to avoid in $\text{FeO}_\text{s}$) and also an evaluation of the full residual in every step of the linear solver. The solver can be sped up by doing parts of the functional derivative analytically beforehand. Using the definition of the residual in the rhs of eq. {eq}`eqn:newton` leads to

$$\begin{align*}
q_\alpha(r)&\equiv-\int\sum_\beta\frac{\delta\mathcal{F}_\alpha\left(r;\left[\rho(r)\right]\right)}{\delta\rho_\beta(r')}\Delta\rho_\beta(r')\mathrm{d}r'\\
&=\int\sum_\beta\frac{\delta}{\delta\rho_\beta(r')}\left(\rho_\alpha(r)-\rho_\alpha^\mathrm{b}e^{\frac{\beta}{m_\alpha}\left(\hat F_{\rho_\alpha}^\mathrm{b,res}-\hat F_{\rho_\alpha}^\mathrm{res}(r)-V_\alpha^\mathrm{ext}(r)\right)}\prod_{\alpha'}I_{\alpha\alpha'}(r)\right)\Delta\rho_\beta(r')\mathrm{d}r'
\end{align*}$$

The functional derivative can be simplified using $\hat F_{\rho_\alpha\rho_\beta}^\mathrm{res}(r,r')=\frac{\delta \hat F_{\rho_\alpha}^\mathrm{b,res}(r)}{\delta\rho_\beta(r')}=\frac{\delta^2\hat F^\mathrm{b,res}}{\delta\rho_\alpha(r)\delta\rho_\beta(r')}$

$$\begin{align*}
q_\alpha(r)&=\int\sum_\beta\left(\delta_{\alpha\beta}\delta(r-r')+\left(\frac{\beta}{m_\alpha}\hat F_{\rho_\alpha\rho_\beta}^\mathrm{res}(r,r')-\sum_{\alpha'}\frac{1}{I_{\alpha\alpha'}(r)}\frac{\delta I_{\alpha\alpha'}(r)}{\delta\rho_\beta(r')}\right)\right.\\
&~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\left.\times\rho_\alpha^\mathrm{b}e^{\frac{\beta}{m_\alpha}\left(\hat F_{\rho_\alpha}^\mathrm{b,res}-\hat F_{\rho_\alpha}^\mathrm{res}(r)-V_\alpha^\mathrm{ext}(r)\right)}\prod_{\alpha'}I_{\alpha\alpha'}(r)\right)\Delta\rho_\beta(r')\mathrm{d}r'\\
&=\Delta\rho_\alpha(r)+\left(\frac{\beta}{m_\alpha}\underbrace{\int\sum_\beta\hat F_{\rho_\alpha\rho_\beta}^\mathrm{res}(r,r')\Delta\rho_\beta(r')\mathrm{d}r'}_{\Delta\hat F_{\rho_\alpha}^\mathrm{res}(r)}-\sum_{\alpha'}\frac{1}{I_{\alpha\alpha'}(r)}\underbrace{\int\sum_\beta\frac{\delta I_{\alpha\alpha'}(r)}{\delta\rho_\beta(r')}\Delta\rho_\beta(r')\mathrm{d}r'}_{\Delta I_{\alpha\alpha'}(r)}\right)\\
&~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\times\rho_\alpha^\mathrm{b}e^{\frac{\beta}{m_\alpha}\left(\hat F_{\rho_\alpha}^\mathrm{b,res}-\hat F_{\rho_\alpha}^\mathrm{res}(r)-V_\alpha^\mathrm{ext}(r)\right)}\prod_{\alpha'}I_{\alpha\alpha'}(r)
\end{align*}$$

and finally

$$q_\alpha(r)=\Delta\rho_\alpha(r)+\left(\frac{\beta}{m_\alpha}\Delta\hat F_{\rho_\alpha}^\mathrm{res}(r)-\sum_{\alpha'}\frac{\Delta I_{\alpha\alpha'}(r)}{I_{\alpha\alpha'}(r)}\right)\mathcal{P}_\alpha\left(r;\left[\rho(r)\right]\right)$$ (eqn:newton_rhs)

Neglecting the second term in eq. {eq}`eqn:newton_rhs` leads to $\Delta_\alpha(r)=\mathcal{F}_\alpha\left(r;\left[\rho(r)\right]\right)$ which is the search direction of the Picard iteration. This observation implies that the Picard iteration is an approximation of the Newton solver that neglects the residual contribution to the Jacobian. Only using the ideal gas contribution to the Jacobian is a reasonable approximation for low densities and therefore, the Picard iteration converges quickly (with a large damping coefficient $\nu$) for low densities.

The second functional derivative of the residual Helmholtz energy can be rewritten in terms of the weight functions.

$$\hat F_{\rho_\alpha\rho_\beta}^\mathrm{res}(r,r')=\int\frac{\delta\hat f^\mathrm{res}(r'')}{\delta\rho_\alpha(r)\delta\rho_\beta(r')}\mathrm{d}r''=\int\sum_{\alpha\beta}\hat f^\mathrm{res}_{\alpha\beta}(r'')\frac{\delta n_\alpha(r'')}{\delta\rho_\alpha(r)}\frac{\delta n_\beta(r'')}{\delta\rho_\beta(r')}\mathrm{d}r''$$

Here $\hat f^\mathrm{res}_{\alpha\beta}=\frac{\partial^2\hat f^\mathrm{res}}{\partial n_\alpha\partial n_\beta}$ is the second partial derivative of the reduced Helmholtz energy density with respect to the weighted densities $n_\alpha$ and $n_\beta$. The definition of the weighted densities $n_\alpha(r)=\sum_\alpha\int\rho_\alpha(r')\omega_\alpha^i(r-r')\mathrm{d}r'$ is used to simplify the expression further.

$$\hat F_{\rho_\alpha\rho_\beta}^\mathrm{res}(r,r')=\int\sum_{\alpha\beta}\hat f^\mathrm{res}_{\alpha\beta}(r'')\omega_\alpha^i(r''-r)\omega_\beta^j(r''-r')\mathrm{d}r''$$

With that, $\Delta\hat F_{\rho_\alpha}^\mathrm{res}(r)$ can be rewritten as

$$\begin{align*}
\Delta\hat F_{\rho_\alpha}^\mathrm{res}(r)&=\int\sum_{\alpha\beta}\hat f^\mathrm{res}_{\alpha\beta}(r'')\omega_\alpha^i(r''-r)\underbrace{\sum_\beta\int\omega_\beta^j(r''-r')\Delta\rho_\beta(r')\mathrm{d}r'}_{\Delta n_\beta(r'')}\mathrm{d}r''\\
&=\int\sum_\alpha\sum_\beta \hat f^\mathrm{res}_{\alpha\beta}(r'')\Delta n_\beta(r'')\omega_\alpha^i(r''-r)\mathrm{d}r''
\end{align*}$$ (eqn:newton_F)

To simplify the expression for $\Delta I_{\alpha\alpha'}(r)$, the recursive definition of the bond integrals is used.

$$\begin{align*}
\Delta I_{\alpha\alpha'}(r)&=\iint\sum_\beta\frac{\delta}{\delta\rho_\beta(r'')}\left(e^{\frac{\beta}{m_{\alpha'}}\left(\hat F_{\rho_{\alpha'}}^\mathrm{b,res}-\hat F_{\rho_{\alpha'}}^\mathrm{res}(r')-V_{\alpha'}^\mathrm{ext}(r')\right)}\left(\prod_{\alpha''\neq\alpha}I_{\alpha'\alpha''}(r')\right)\omega_\mathrm{chain}^{\alpha\alpha'}(r-r')\right)\\
&~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\times\Delta\rho_\beta(r'')\mathrm{d}r'\mathrm{d}r''\\
&=\iint\sum_\beta\left(-\frac{\beta}{m_{\alpha'}}\hat F_{\rho_{\alpha'}\rho_\beta}^\mathrm{res}(r',r'')+\sum_{\alpha''\neq\alpha}\frac{1}{I_{\alpha'\alpha''}(r')}\frac{\delta I_{\alpha'\alpha''}(r')}{\delta\rho_\beta(r'')}\right)e^{\frac{\beta}{m_{\alpha'}}\left(\hat F_{\rho_{\alpha'}}^\mathrm{b,res}-\hat F_{\rho_{\alpha'}}^\mathrm{res}(r')-V_{\alpha'}^\mathrm{ext}(r')\right)}\\
&~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\times\left(\prod_{\alpha''\neq\alpha}I_{\alpha'\alpha''}(r')\right)\omega_\mathrm{chain}^{\alpha\alpha'}(r-r')\Delta\rho_\beta(r'')\mathrm{d}r'\mathrm{d}r''
\end{align*}$$

Here, the definition of $\Delta\hat F_{\rho_\alpha}^\mathrm{res}(r)$ and $\Delta I_{\alpha\alpha'}(r)$ can be inserted leading to a recursive calculation of $\Delta I_{\alpha\alpha'}(r)$ similar to the original bond integrals.

$$\begin{align*}
\Delta I_{\alpha\alpha'}(r)&=\int\left(-\frac{\beta}{m_{\alpha'}}\Delta\hat F_{\rho_{\alpha'}}^\mathrm{res}(r')+\sum_{\alpha''\neq\alpha}\frac{\Delta I_{\alpha'\alpha''}(r')}{I_{\alpha'\alpha''}(r')}\right)\\
&~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\times e^{\frac{\beta}{m_{\alpha'}}\left(\hat F_{\rho_{\alpha'}}^\mathrm{b,res}-\hat F_{\rho_{\alpha'}}^\mathrm{res}(r')-V_{\alpha'}^\mathrm{ext}(r')\right)}\left(\prod_{\alpha''\neq\alpha}I_{\alpha'\alpha''}(r')\right)\omega_\mathrm{chain}^{\alpha\alpha'}(r-r')\mathrm{d}r'
\end{align*}$$ (eqn:newton_I)

In every iteration of GMRES, $q(r)$ needs to be evaluated from eqs. {eq}`eqn:newton_rhs`, {eq}`eqn:newton_F` and {eq}`eqn:newton_I`. The operations required for that are analogous to the calculation of weighted densities and the functional derivative in the Euler-Lagrange equation itself. Details of GMRES, including the pseudocode that the implementation in $\text{FeO}_\text{s}$ is based on, are given on [Wikipedia](https://de.wikipedia.org/wiki/GMRES-Verfahren) (German).

The Newton solver converges exceptionally fast compared to a simple Picard iteration. The faster convergence comes at the cost of requiring multiple steps for solving the linear subsystem. With the algorithm outlined here, the evaluation of the second partial derivatives of the Helmholtz energy density is only required once for every Newton step. The GMRES algorithm only uses the very efficient convolution integrals and no additional evaluation of the model.

## Anderson mixing
