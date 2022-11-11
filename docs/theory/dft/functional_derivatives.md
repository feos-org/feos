## Functional derivatives

In the last section the functional derivative

$$\hat F_{\rho_\alpha}^\mathrm{res}(r)=\left(\frac{\delta\hat F^\mathrm{res}}{\delta\rho_\alpha(r)}\right)_{T,\rho_{\alpha'\neq\alpha}}$$

was introduced as part of the Euler-Lagrange equation. The implementation of these functional derivatives can be a major difficulty during the development of a new Helmholtz energy model. In $\text{FeO}_\text{s}$ it is fully automated. The core assumption is that the residual Helmholtz energy functional $\hat F^\mathrm{res}$ can be written as a sum of contributions that each can be written in the following way:

$$F=\int f[\rho(r)]dr=\int f(\lbrace n_\gamma\rbrace)dr$$

The Helmholtz energy density $f$ which would in general be a functional of the density itself can be expressed as a *function* of weighted densities $n_\gamma$ which are obtained by convolving the density profiles with weight functions $\omega_\gamma^\alpha$

$$n_\gamma(r)=\sum_\alpha\int\rho_\alpha(r')\omega_\gamma^\alpha(r-r')dr'\tag{1}$$

In practice the weight functions tend to have simple shapes like step functions (i.e. the weighted density is an average over a sphere) or Dirac distributions (i.e. the weighted density is an average over the surface of a sphere).

For Helmholtz energy functionals that can be written in this form, the calculation of the functional derivative can be automated. In general the functional derivative can be written as

$$F_{\rho_\alpha}(r)=\int\frac{\delta f(r')}{\delta\rho_\alpha(r)}dr'=\int\sum_\gamma f_{n_\gamma}(r')\frac{\delta n_\gamma(r')}{\delta\rho_\alpha(r)}dr'$$

with $f_{n_\gamma}$ as abbreviation for the *partial* derivative $\frac{\partial f}{\partial n_\gamma}$. Using the definition of the weighted densities (1), the expression can be rewritten as

$$\begin{align}
F_{\rho_\alpha}(r)&=\int\sum_\gamma f_{n_\gamma}(r')\frac{\delta n_\gamma(r')}{\delta\rho_\alpha(r)}dr'=\int\sum_\gamma f_{n_\gamma}(r')\sum_{\alpha'}\int\underbrace{\frac{\delta\rho_{\alpha'}(r'')}{\delta\rho_\alpha(r)}}_{\delta(r-r'')\delta_{\alpha\alpha'}}\omega_\gamma^{\alpha'}(r'-r'')dr''dr'\\
&=\sum_\gamma\int f_{n_\gamma}(r')\omega_\gamma^\alpha(r'-r)dr
\end{align}$$

At this point the parity of the weight functions has to be taken into account. By construction scalar and spherically symmetric weight functions (the standard case) are even functions, i.e., $\omega(-r)=\omega(r)$. In contrast, vector valued weight functions, as they appear in fundamental measure theory, have odd parity, i.e., $\omega(-r)=-\omega(r)$. Therefore, the sum over the weight functions needs to be split into two contributions, as

$$F_{\rho_\alpha}(r)=\sum_\gamma^\mathrm{scal}\int f_{n_\gamma}(r')\omega_\gamma^\alpha(r-r')dr-\sum_\gamma^\mathrm{vec}\int f_{n_\gamma}(r')\omega_\gamma^\alpha(r-r')dr\tag{2}$$

With this distinction, the calculation of the functional derivative is split into three steps

1. Calculate the weighted densities $n_\gamma$ from eq. (1)
2. Evaluate the partial derivatives $f_{n_\gamma}$
3. Use eq. (2) to obtain the functional derivative $F_{\rho_\alpha}$

A fast method to calculate the convolution integrals required in steps 1 and 3 is shown in the next section.

The implementation of partial derivatives by hand can be cumbersome and error prone. $\text{FeO}_\text{s}$ uses automatic differentiation with dual numbers to facilitate this step. For details about dual numbers and their generalization, we refer to [our publication](https://www.frontiersin.org/articles/10.3389/fceng.2021.758090/full). The essential relation is that the Helmholtz energy density evaluated with a dual number as input for one of the weighted densities evaluates to a dual number with the function value as real part and the partial derivative as dual part.

$$f(n_\gamma+\varepsilon,\lbrace n_{\gamma'\neq\gamma}\rbrace)=f+f_{n_\gamma}\varepsilon$$