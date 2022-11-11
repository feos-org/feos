## DFT solvers
Different solvers can be used to calculate the density profiles from the Euler-Lagrange equation introduced previously.

### Picard iteration

### Newton algorithm
The residual function for which the roots are calculated is obtained from rearranging the Euler-Lagrange equation

$$\mathcal{F}_\alpha[\rho(r)]=\rho_\alpha(r)-\rho_\alpha^\mathrm{b}e^{\frac{\beta}{m_\alpha}\left(\hat F_{\rho_\alpha}^\mathrm{b,res}-\hat F_{\rho_\alpha}^\mathrm{res}(r)-V_\alpha^\mathrm{ext}(r)\right)}\prod_{\alpha'}I_{\alpha\alpha'}(r)$$

From a Taylor expansion of the residual

$$\mathcal{F}_\alpha[\rho(r)+\Delta\rho(r)]=\mathcal{F}_\alpha[\rho(r)]+\int\sum_\beta\frac{\delta\mathcal{F}_\alpha(r)}{\delta\rho_\beta(r')}\Delta\rho_\beta(r')dr'+\ldots$$

the Newton step is derived by setting the updated residual $\mathcal{F}_\alpha[\rho(r)+\Delta\rho(r)]$ to 0 and neglecting higher order terms.

$$0=\rho_\alpha(r)-\rho_\alpha^\mathrm{b}e^{\frac{\beta}{m_\alpha}\left(\hat F_{\rho_\alpha}^\mathrm{b,res}-\hat F_{\rho_\alpha}^\mathrm{res}(r)-V_\alpha^\mathrm{ext}(r)\right)}\prod_{\alpha'}I_{\alpha\alpha'}(r)+\int\sum_\beta\frac{\delta}{\delta\rho_\beta(r')}\left(\rho_\alpha(r)-\rho_\alpha^\mathrm{b}e^{\frac{\beta}{m_\alpha}\left(\hat F_{\rho_\alpha}^\mathrm{b,res}-\hat F_{\rho_\alpha}^\mathrm{res}(r)-V_\alpha^\mathrm{ext}(r)\right)}\prod_{\alpha'}I_{\alpha\alpha'}(r)\right)\Delta\rho_\beta(r')dr'$$

The functional derivative can be simplified using $\hat F_{\rho_\alpha\rho_\beta}^\mathrm{res}(r,r')=\frac{\delta \hat F_{\rho_\alpha}^\mathrm{b,res}(r)}{\delta\rho_\beta(r')}=\frac{\delta^2\hat F^\mathrm{b,res}}{\delta\rho_\alpha(r)\delta\rho_\beta(r')}$

$$0=\rho_\alpha(r)-\rho_\alpha^\mathrm{b}e^{\frac{\beta}{m_\alpha}\left(\hat F_{\rho_\alpha}^\mathrm{b,res}-\hat F_{\rho_\alpha}^\mathrm{res}(r)-V_\alpha^\mathrm{ext}(r)\right)}\prod_{\alpha'}I_{\alpha\alpha'}(r)+\int\sum_\beta\left(\delta_{ij}\delta(r-r')+\left(\frac{\beta}{m_\alpha}\hat F_{\rho_\alpha\rho_\beta}^\mathrm{res}(r,r')-\sum_{\alpha'}\frac{1}{I_{\alpha\alpha'}(r)}\frac{\delta I_{\alpha\alpha'}(r)}{\delta\rho_\beta(r')}\right)\rho_\alpha^\mathrm{b}e^{\frac{\beta}{m_\alpha}\left(\hat F_{\rho_\alpha}^\mathrm{b,res}-\hat F_{\rho_\alpha}^\mathrm{res}(r)-V_\alpha^\mathrm{ext}(r)\right)}\prod_{\alpha'}I_{\alpha\alpha'}(r)\right)\Delta\rho_\beta(r')dr'$$

$$0=\rho_\alpha(r)-\rho_\alpha^\mathrm{b}e^{\frac{\beta}{m_\alpha}\left(\hat F_{\rho_\alpha}^\mathrm{b,res}-\hat F_{\rho_\alpha}^\mathrm{res}(r)-V_\alpha^\mathrm{ext}(r)\right)}\prod_{\alpha'}I_{\alpha\alpha'}(r)+\Delta\rho_\alpha(r)+\left(\frac{\beta}{m_\alpha}\underbrace{\int\sum_\beta\hat F_{\rho_\alpha\rho_\beta}^\mathrm{res}(r,r')\Delta\rho_\beta(r')dr'}_{\Delta\hat F_{\rho_\alpha}^\mathrm{res}(r)}-\sum_{\alpha'}\frac{1}{I_{\alpha\alpha'}(r)}\underbrace{\int\sum_\beta\frac{\delta I_{\alpha\alpha'}(r)}{\delta\rho_\beta(r')}\Delta\rho_\beta(r')dr'}_{\Delta I_{\alpha\alpha'}(r)}\right)\rho_\alpha^\mathrm{b}e^{\frac{\beta}{m_\alpha}\left(\hat F_{\rho_\alpha}^\mathrm{b,res}-\hat F_{\rho_\alpha}^\mathrm{res}(r)-V_\alpha^\mathrm{ext}(r)\right)}\prod_{\alpha'}I_{\alpha\alpha'}(r)$$

The second functional derivative of the residual Helmholtz energy can be rewritten in terms of the weight functions.

$$\hat F_{\rho_\alpha\rho_\beta}^\mathrm{res}(r,r')=\int\frac{\delta\hat f^\mathrm{res}(r'')}{\delta\rho_\alpha(r)\delta\rho_\beta(r')}dr''=\int\sum_{\alpha\beta}\hat f^\mathrm{res}_{\alpha\beta}(r'')\frac{\delta n_\alpha(r'')}{\delta\rho_\alpha(r)}\frac{\delta n_\beta(r'')}{\delta\rho_\beta(r')}dr''$$

Here $\hat f^\mathrm{res}_{\alpha\beta}=\frac{\partial^2\hat f^\mathrm{res}}{\partial n_\alpha\partial n_\beta}$ is the second partial derivative of the reduced Helmholtz energy density with respect to the weighted densities $n_\alpha$ and $n_\beta$. The definition of the weighted densities $n_\alpha(r)=\sum_\alpha\int\rho_\alpha(r')\omega_\alpha^i(r-r')dr'$ is used to simplify the expression further.

$$\hat F_{\rho_\alpha\rho_\beta}^\mathrm{res}(r,r')=\int\sum_{\alpha\beta}\hat f^\mathrm{res}_{\alpha\beta}(r'')\omega_\alpha^i(r''-r)\omega_\beta^j(r''-r')dr''$$

With that, the $\Delta\hat F_{\rho_\alpha}^\mathrm{res}(r)$ can be rewritten as

$$\Delta\hat F_{\rho_\alpha}^\mathrm{res}(r)=\int\sum_{\alpha\beta}\hat f^\mathrm{res}_{\alpha\beta}(r'')\omega_\alpha^i(r''-r)\underbrace{\sum_\beta\int\omega_\beta^j(r''-r')\Delta\rho_\beta(r')dr'}_{\Delta n_\beta(r'')}dr''=\int\sum_\alpha\sum_\beta \hat f^\mathrm{res}_{\alpha\beta}(r'')\Delta n_\beta(r'')\omega_\alpha^i(r''-r)dr''$$

To simplify the expression for $\Delta I_{\alpha\alpha'}(r)$, the definition of the bond integrals is used.

$$\Delta I_{\alpha\alpha'}(r)=\iint\sum_\beta\frac{\delta}{\delta\rho_\beta(r'')}\left(e^{\frac{\beta}{m_{\alpha'}}\left(\hat F_{\rho_{\alpha'}}^\mathrm{b,res}-\hat F_{\rho_{\alpha'}}^\mathrm{res}(r')-V_{\alpha'}^\mathrm{ext}(r')\right)}\left(\prod_{\alpha''\neq\alpha}I_{\alpha'\alpha''}(r')\right)\omega_\mathrm{chain}^{\alpha\alpha'}(r-r')\right)\Delta\rho_\beta(r'')dr'dr''$$

$$\Delta I_{\alpha\alpha'}(r)=\iint\sum_\beta\left(-\frac{\beta}{m_{\alpha'}}\hat F_{\rho_{\alpha'}\rho_\beta}^\mathrm{res}(r',r'')+\sum_{\alpha''\neq\alpha}\frac{1}{I_{\alpha'\alpha''}(r')}\frac{\delta I_{\alpha'\alpha''}(r')}{\delta\rho_\beta(r'')}\right)e^{\frac{\beta}{m_{\alpha'}}\left(\hat F_{\rho_{\alpha'}}^\mathrm{b,res}-\hat F_{\rho_{\alpha'}}^\mathrm{res}(r')-V_{\alpha'}^\mathrm{ext}(r')\right)}\left(\prod_{\alpha''\neq\alpha}I_{\alpha'\alpha''}(r')\right)\omega_\mathrm{chain}^{\alpha\alpha'}(r-r')\Delta\rho_\beta(r'')dr'dr''$$

Here, the definition of $\Delta\hat F_{\rho_\alpha}^\mathrm{res}(r)$ and $\Delta I_{\alpha\alpha'}(r)$ can be identified leading to a recursive calculation of $\Delta I_{\alpha\alpha'}(r)$ similar to the original bond integrals.

$$\Delta I_{\alpha\alpha'}(r)=\int\left(-\frac{\beta}{m_{\alpha'}}\Delta\hat F_{\rho_{\alpha'}}^\mathrm{res}(r')+\sum_{\alpha''\neq\alpha}\frac{\Delta I_{\alpha'\alpha''}(r')}{I_{\alpha'\alpha''}(r')}\right)e^{\frac{\beta}{m_{\alpha'}}\left(\hat F_{\rho_{\alpha'}}^\mathrm{b,res}-\hat F_{\rho_{\alpha'}}^\mathrm{res}(r')-V_{\alpha'}^\mathrm{ext}(r')\right)}\left(\prod_{\alpha''\neq\alpha}I_{\alpha'\alpha''}(r')\right)\omega_\mathrm{chain}^{\alpha\alpha'}(r-r')dr'$$

The calculation of the linear system in the Newton algorithm is done by using operations analogous to the calculation of weighted densities and the functional derivative. The calculation of the full Jacobian is unfeasible due to its size. Instead a matrix-free solution algorithm (GMRES) is used. The properties and a pseudocode of GMRES are given on [Wikipedia](https://de.wikipedia.org/wiki/GMRES-Verfahren) (German).

The Newton solver converges exceptionally fast compared to a simple Picard iteration. The faster convergence comes at the cost of requiring multiple steps for solving the linear subsystem. With the algorithm outlined here, the evaluation of the second partial derivatives of the Helmholtz energy density is only required once for every Newton step. The GMRES algorithm only uses the very efficient convolution integrals and no additional evaluation of the model.

### Anderson mixing