# Association

The Helmholtz contribution due to short range attractive interaction ("association") in SAFT models can be conveniently expressed as

$$\frac{A^\mathrm{assoc}}{k_\mathrm{B}TV}=\sum_\alpha\rho_\alpha\left(\ln X_\alpha-\frac{X_\alpha}{2}+\frac{1}{2}\right)$$

Here, $\alpha$ is the index of all distinguishable **association sites** in the system. The site density $\rho_\alpha$ is the density of the segment or molecule on which the association site is located times the multiplicity of the site. The fraction of non-bonded association sites $X_\alpha$ is calculated implicitly from

$$
\frac{1}{X_\alpha}=1+\sum_\beta\rho_\beta X_\beta\Delta^{\alpha\beta}
$$ (eqn:x_assoc)

where $\Delta^{\alpha\beta}$ is the association strength between sites $\alpha$ and $\beta$. The exact expression for the association strength varies between different SAFT versions. We implement the expressions used in PC-SAFT but a generalization to other models is straightforward.

The number of degrees of freedom in the association strength matrix $\Delta$ is vast and for practical purposes it is useful to reduce the number of non-zero elements in $\Delta$. In $\text{FeO}_\text{s}$ we use 3 kinds of association sites: $A$, $B$ and $C$. Association sites of kind $A$ only associate with $B$ and vice versa, whereas association sites of kind $C$ only associate with other sites of kind $C$. By sorting the sites by their kind, the entire $\Delta$ matrix can be reduced to two smaller matrices: $\Delta^{AB}$ and $\Delta^{CC}$.

```{image} FeOs_Association.png
:alt: Association strength matrix
:class: bg-primary
:width: 300px
:align: center
```

In practice, the $AB$ association can be linked to hydrogen bonding sites. The $CC$ association is less widely used but implemented to ensure that all the association schemes defined in [Huang and Radosz 1990](https://pubs.acs.org/doi/10.1021/ie00107a014) are covered.

## Calculation of the fraction of non-bonded association sites

The algorithm to solve the fraction of non-bonded sites for general association schemes follows [Michelsen 2006](https://pubs.acs.org/doi/full/10.1021/ie060029x). The problem is rewritten as an optimization problem with gradients

$$g_\alpha=\frac{1}{X_\alpha}-1-\sum_\beta\rho_\beta X_\beta\Delta^{\alpha\beta}$$

The analytic Hessian is

$$H_{\alpha\beta}=-\frac{\delta_{\alpha\beta}}{X_\alpha^2}-\rho_\beta\Delta^{\alpha\beta}$$

with the Kronecker delta $\delta_{\alpha\beta}$. However, [Michelsen 2006](https://pubs.acs.org/doi/full/10.1021/ie060029x) shows that it is beneficial to use

$$\hat{H}_{\alpha\beta}=-\frac{\delta_{\alpha\beta}}{X_\alpha}\left(1+\sum_\beta\rho_\beta X_\beta\Delta^{\alpha\beta}\right)-\rho_\beta\Delta^{\alpha\beta}$$

instead. $X_\alpha$ can then be solved robustly using a Newton algorithm with a check that ensures that the values of $X_\alpha$ remain positive. With the split in $AB$ and $CC$ association the two kinds different versions could be solved separately from each other. This is currently not implemented, because use cases are rare to nonexistent and the benefit is small.

A drastic improvement in performance, however, can be achieved by solving eq. {eq}`eqn:x_assoc` analytically for simple cases. If there is only one $A$ and one $B$ site the corresponding fractions of non-bonded association sites can be calculated from

$$X_A=\frac{2}{\sqrt{\left(1+\left(\rho_A-\rho_B\right)\Delta^{AB}\right)^2+4\rho_B\Delta^{AB}}+1+\left(\rho_B-\rho_A\right)\Delta^{AB}}$$
$$X_B=\frac{2}{\sqrt{\left(1+\left(\rho_A-\rho_B\right)\Delta^{AB}\right)^2+4\rho_B\Delta^{AB}}+1+\left(\rho_A-\rho_B\right)\Delta^{AB}}$$

The specific form of the expressions (with the square root in the denominator) is particularly robust for cases where $\Delta$ and/or $\rho$ are small or even 0.

Analogously, for a single $C$ site, the expression becomes

$$X_C=\frac{2}{\sqrt{1+4\rho_C\Delta^{CC}}+1}$$

## PC-SAFT expression for the association strength

In $\text{FeO}_\text{s}$ every association site $\alpha$ is parametrized with the dimensionless association volume $\kappa^\alpha$ and the association energy $\varepsilon^\alpha$. The association strength between sites $\alpha$ and $\beta$ is then calculated from

$$\Delta^{\alpha\beta}=\left(\frac{1}{1-\zeta_3}+\frac{\frac{3}{2}d_{ij}\zeta_2}{\left(1-\zeta_3\right)^2}+\frac{\frac{1}{2}d_{ij}^2\zeta_2^2}{\left(1-\zeta_3\right)^3}\right)\sqrt{\sigma_i^3\kappa^\alpha\sigma_j^3\kappa^\beta}\left(e^{\frac{\varepsilon^\alpha+\varepsilon^\beta}{2k_\mathrm{B}T}}-1\right)$$

with

$$d_{ij}=\frac{2d_id_j}{d_i+d_j},~~~~d_k=\sigma_k\left(1-0.12e^{\frac{-3\varepsilon_k}{k_\mathrm{B}T}}\right)$$

and

$$\zeta_2=\frac{\pi}{6}\sum_k\rho_km_kd_k^2,~~~~\zeta_3=\frac{\pi}{6}\sum_k\rho_km_kd_k^3$$

The indices $i$, $j$ and $k$ are used to index molecules or segments rather than sites. $i$ and $j$ in particular refer to the molecule or segment that contain site $\alpha$ or $\beta$ respectively.

On their own the parameters $\kappa^\alpha$ and $\varepsilon^\beta$ have no physical meaning. For a pure system of self-associating molecules it is therefore natural to define $\kappa^A=\kappa^B\equiv\kappa^{A_iB_i}$ and $\varepsilon^A=\varepsilon^B\equiv\varepsilon^{A_iB_i}$ where $\kappa^{A_iB_i}$ and $\varepsilon^{A_iB_i}$ are now parameters describing the molecule rather than individual association sites. However, for systems that are not self-associating, the more generic parametrization is required.

## SAFT-VR Mie expression for the association strength

We provide an implementation of the association strength as published by [Lafitte et al. (2013)](https://doi.org/10.1063/1.4819786).
Every association site $\alpha$ is parametrized with two distances $r_c^\alpha$ and $r_d^\alpha$, and the association energy $\varepsilon^\alpha$. Whereas $r_c^\alpha$ is a parameter that is adjusted for each substance, $r_d^\alpha$ is kept constant as $r_d^\alpha = 0.4 \sigma$. Note that the parameter $r_c^\alpha$ has to be provided as dimensionless quantity in the input, i.e. divided by the segment's $\sigma$ value. 

The association strength between sites $\alpha$ and $\beta$ is then calculated from

$$\Delta^{\alpha\beta}=\left(\frac{1}{1-\zeta_3}+\frac{\frac{3}{2}\tilde{d}_{ij}\zeta_2}{\left(1-\zeta_3\right)^2}+\frac{\frac{1}{2}\tilde{d}_{ij}^2\zeta_2^2}{\left(1-\zeta_3\right)^3}\right)K_{ij}^{\alpha\beta}\sigma_{ij}^3\left(e^{\frac{\sqrt{\varepsilon^\alpha\varepsilon^\beta}}{k_\mathrm{B}T}}-1\right)$$
 
with

$$\tilde{d}_{ij}=\frac{2d_id_j}{d_i+d_j},~~~~\sigma_{ij} = \frac{\sigma_i + \sigma_j}{2}$$

and 

$$\zeta_2=\frac{\pi}{6}\sum_k\rho_km_kd_k^2,~~~~\zeta_3=\frac{\pi}{6}\sum_k\rho_km_kd_k^3$$

where 

$$d_i = \int_{l}^{\sigma_i} \left[1 - e^{-\beta u_i^\text{Mie}(r)}\right]\mathrm{d}r.$$

The integral is solved using a Gauss-Legendre quadrature of fifth order where the lower limit, $l$, is determined using the method of [Aasen et al.](https://doi.org/10.1063/1.5111364) The dimensionless bonding volume $K^{\alpha\beta}_{ij}$ (see [A43](https://doi.org/10.1063/1.4819786)) utilizes arithmetic combining rules for $r_c^{\alpha\beta}$, $r_d^{\alpha\beta}$ and $d_{ij}$ of unlike sites and segments, respectively.

## Helmholtz energy functional

In inhomogeneous systems, the fractions of non-bonded association sites $X_\alpha(\mathbf{r})$ are treated as additional variables of the density profile (see the section on the [Euler-Lagrange equation](../dft/euler_lagrange_equation.md)). The association contribution is the stationary point with respect to $X_\alpha(\mathbf{r})$ of

$$Q[X,\rho]=\int\sum_\alpha\tilde\rho_\alpha(\mathbf{r})\left(\ln X_\alpha(\mathbf{r})-X_\alpha(\mathbf{r})+1\right)\mathrm{d}\mathbf{r}-\frac{1}{2}\iint\sum_{\alpha\beta}\tilde\rho_\alpha(\mathbf{r})X_\alpha(\mathbf{r})\tilde\rho_\beta(\mathbf{r}')X_\beta(\mathbf{r}')\Delta^{\alpha\beta}(\mathbf{r},\mathbf{r}')\mathrm{d}\mathbf{r}\mathrm{d}\mathbf{r}'$$

which reduces to the expressions at the beginning of this page for a homogeneous system ([Michelsen and Hendriks 2001](https://doi.org/10.1016/S0378-3812(01)00344-2)). The different association functionals that are available in $\text{FeO}_\text{s}$ differ in the site densities $\tilde\rho_\alpha$ and in the association strength $\Delta^{\alpha\beta}(\mathbf{r},\mathbf{r}')$ between site $\alpha$ at $\mathbf{r}$ and site $\beta$ at $\mathbf{r}'$. For all of them, the association strength is written as

$$\Delta^{\alpha\beta}(\mathbf{r},\mathbf{r}')=c^{\alpha\beta}\sqrt{y_{ij}(\mathbf{r})y_{ij}(\mathbf{r}')}K_{ij}(\mathbf{r}-\mathbf{r}')$$

Here, $c^{\alpha\beta}$ is the association strength without the contact value of the cavity correlation function (e.g., $\sqrt{\sigma_i^3\kappa^\alpha\sigma_j^3\kappa^\beta}\left(e^{\frac{\varepsilon^\alpha+\varepsilon^\beta}{2k_\mathrm{B}T}}-1\right)$ for PC-SAFT), $K_{ij}$ is a kernel that is normalized to 1, and the contact value

$$y_{ij}=\frac{1}{1-\zeta_3}+\xi\frac{\frac{3}{2}d_{ij}\zeta_2}{\left(1-\zeta_3\right)^2}+\xi\frac{\frac{1}{2}d_{ij}^2\zeta_2^2}{\left(1-\zeta_3\right)^3}$$

is evaluated at local values of $\zeta_2$, $\zeta_3$ and the inhomogeneity factor $\xi$. The geometric mean of the contact values at both positions keeps the association strength symmetric. Because the kernel is normalized, all versions reduce to the bulk expressions in a homogeneous system.

|functional|site density $\tilde\rho_\alpha$|$\zeta_2$, $\zeta_3$, $\xi$ in $y_{ij}$|kernel $K_{ij}(\mathbf{r})$|
|:-:|:-:|:-:|:-:|
|iSAFT ([Segura et al. 1997](https://www.tandfonline.com/doi/abs/10.1080/002689797172110), [Bymaster and Chapman 2010](https://doi.org/10.1021/jp102677m))|$n_\alpha\rho_i$|$\zeta_2$ and $\zeta_3$ from $\bar\rho_k$, $\xi=1$|$\frac{\delta\left(\bar d_{ij}-\vert\mathbf{r}\vert\right)}{4\pi\bar d_{ij}^2}$, $\bar d_{ij}=\frac{d_i+d_j}{2}$|
|Yu-Wu ([Yu and Wu 2002](https://aip.scitation.org/doi/abs/10.1063/1.1463435))|$\frac{n_\alpha n_0^i\xi_i}{m_i}$|$\zeta_2=\frac{n_2}{6}$, $\zeta_3=n_3$, $\xi=1-\frac{\vec n_2\cdot\vec n_2}{n_2^2}$ (FMT weighted densities)|$\delta(\mathbf{r})$|
|WDA ([Camacho Vergara et al. 2020](https://doi.org/10.1080/00268976.2020.1725668))|$n_\alpha\bar\rho_i$|$\zeta_2$ and $\zeta_3$ from $\bar\rho_k$, $\xi=1$|$\delta(\mathbf{r})$|

Here, $n_\alpha$ is the multiplicity of site $\alpha$ on segment $i$. The Yu-Wu functional uses the weighted densities $n_0^i$, $n_2$, $n_3$, and $\vec n_2$ from the [hard-sphere reference model](hard_spheres.md) (White Bear version) with the effective segment diameter $d_i$ and the segment-wise inhomogeneity factor $\xi_i=1-\frac{\vec n_2^i\cdot\vec n_2^i}{\left(n_2^i\right)^2}$. WDA and iSAFT use the average density inside a sphere with radius $d_i$

$$\bar\rho_i(\mathbf{r})=\frac{3}{4\pi d_i^3}\int\rho_i(\mathbf{r}')\Theta\left(d_i-\vert\mathbf{r}-\mathbf{r}'\vert\right)\mathrm{d}\mathbf{r}'$$

Only iSAFT uses a non-local kernel, which describes that bonding is only possible for segments at contact. In this case, the mass action equations are integral equations. The inhomogeneity factors are restricted to the interval $[0,1]$ and the site densities to non-negative values, which only matters for numerical artifacts of the convolutions inside walls.
