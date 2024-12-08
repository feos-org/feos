# Ideal gas properties
Classical DFT can be used to rapidly determine the ideal gas limit of fluids in porous media. In an ideal gas, there are no interactions between the fluid molecules and therefore the residual Helmholtz energy $F^\mathrm{res}$ and its derivatives vanish. Note that this is only the case for spherical or heterosegmented molecules ($m_\alpha=1$), as the chain contribution in the homosegmented model contains intramolecular interactions. The ideal gas density profile can then be obtained directly from the [Euler-Lagrange equation](euler_lagrange_equation.md):

$$\rho_\alpha^\mathrm{ig}(\mathbf{r})=\rho_\alpha^\mathrm{b}e^{-\beta V_\alpha^\mathrm{ext}(\mathbf{r})}\prod_{\alpha'}I^\mathrm{ig}_{\alpha\alpha'}(\mathbf{r})$$ (eqn:rho_ideal_gas)

$$I^\mathrm{ig}_{\alpha\alpha'}(\mathbf{r})=\int e^{-\beta V_{\alpha'}^\mathrm{ext}(\mathbf{r'})}\left(\prod_{\alpha''\neq\alpha}I^\mathrm{ig}_{\alpha'\alpha''}(\mathbf{r'})\right)\omega_\mathrm{chain}^{\alpha\alpha'}(\mathbf{r}-\mathbf{r'})\mathrm{d}\mathbf{r'}$$

 Either from the derivatives presented [previously](derivatives.md), or from directly calculating derivatives of eq. {eq}`eqn:euler_lagrange_mu`, the **Henry coefficient** $H_\alpha$ can be calculated, as

 $$H_\alpha(T)=\left(\frac{\partial N_\alpha^\mathrm{ig}}{\partial p_\alpha}\right)_{T,x_k}=\int\left(\frac{\partial\rho_\alpha^\mathrm{ig}(\mathbf{r})}{\partial p_\alpha}\right)_{T,x_k}\mathrm{d}\mathbf{r}=\beta\int e^{-\beta V_\alpha^\mathrm{ext}(\mathbf{r})}\prod_{\alpha'}I^\mathrm{ig}_{\alpha\alpha'}(\mathbf{r})\mathrm{d}\mathbf{r}$$

By construction the Henry coefficients for all segments $\alpha$ of a molecule $i$ are identical. Therefore, the number of adsorbed molecules in an ideal gas state (the Henry regime) can be calculated from

$$N_i^\mathrm{ig}=k_\mathrm{B}T\rho_i^\mathrm{b}H_i(T)$$

The expression can be used in the general equation for the **enthalpy of adsorption** (see [here](enthalpy_of_adsorption.md))

$$0=\sum_j\left(\frac{\partial N_i^\mathrm{ig}}{\partial\mu_j}\right)_T\Delta h_j^\mathrm{ads,ig}+T\left(\frac{\partial N_i^\mathrm{ig}}{\partial T}\right)_{p,x_k}$$

to simplify to

$$0=\rho_i^\mathrm{b}H_i(T)\Delta h_i^\mathrm{ads,ig}+k_\mathrm{B}T^2\rho_i^\mathrm{b}H_i'(T)$$

and finally

$$\Delta h_i^\mathrm{ads,ig}=-k_\mathrm{B}T^2\frac{H_i'(T)}{H_i(T)}$$

For a spherical molecule without bond integrals, the derivative can be evaluated straightforwardly to yield

$$\Delta h_i^\mathrm{ads,ig}=\frac{\int\left(k_\mathrm{B}T-V_i^\mathrm{ext}(\mathbf{r})\right)e^{-\beta V_i^\mathrm{ext}(\mathbf{r})}\mathrm{d}\mathbf{r}}{\int e^{-\beta V_i^\mathrm{ext}(\mathbf{r})}\mathrm{d}\mathbf{r}}$$

Analytical derivatives of the bond integrals can be determined, however, in $\text{FeO}_\text{s}$ automatic differentiation with dual numbers is used to obtain correct derivatives with barely any implementation overhead.