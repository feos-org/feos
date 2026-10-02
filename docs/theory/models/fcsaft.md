# FC-SAFT

The fused-chain SAFT (FC-SAFT) equation of state describes molecules as chains of hard spheres that overlap ("fuse") with an explicit bond length. Compared to PC-SAFT, where the segments are tangent and the segment number $m$ is a (non-integer) model parameter, FC-SAFT uses an integer number of segments and treats the bond length $l$ as the parameter that determines the shape of the molecule. The reference fluid is the [fused-sphere chain model](https://doi.org/10.1103/PhysRevE.105.034110), and the dispersion contribution follows the structure of PC-SAFT ([Gross and Sadowski, 2001](https://doi.org/10.1021/ie0003887)).

The residual Helmholtz energy is a sum of four contributions:

$$A^\mathrm{res}=A^\mathrm{hs}+A^\mathrm{chain}+A^\mathrm{disp}+A^\mathrm{assoc}$$

$\text{FeO}_\text{s}$ provides two versions of the model:
- a **heterosegmented** (segment-based) model, in which every segment $\alpha$ has its own parameters $\sigma_\alpha$ and $\varepsilon_\alpha$, and the molecular structure is given by a bond graph with bond lengths $l_{\alpha\beta}$. Segments are distinguished by their position in the molecule, i.e., two segments of the same type can have different geometries depending on their neighbors.
- a **homosegmented** model, in which every molecule $i$ consists of $s_i$ identical segments in a linear chain with a single bond length $l_i$. This version additionally includes a dipolar contribution and is available as specialized implementations for pure components (`FcSaftPure`) and binary mixtures (`FcSaftBinary`) that support automatic differentiation with respect to model parameters.

Crucially, the homosegmented version is mathematically identical to a heterosegmented molecule with uniform segment parameters and bond lengths. This sets FC-SAFT apart from PC-SAFT and SAFT-VR-Mie, because their heterosegmented counterparts (gc-PC-SAFT and SAFT-$\gamma$-Mie) do not simplify to the molecular equation of state for homosegmented chains.

## Geometry of fused spheres

Throughout the model, the temperature dependent segment diameter of PC-SAFT is used

$$d_\alpha=\sigma_\alpha\left(1-0.12e^{\frac{-3\varepsilon_\alpha}{k_\mathrm{B}T}}\right)$$

Two bonded segments $\alpha$ and $\beta$ with a bond length $l_{\alpha\beta}<\frac{1}{2}(d_\alpha+d_\beta)$ overlap. The distance of the center of segment $\alpha$ to the plane of intersection is

$$h_{\alpha\beta}=\frac{d_\alpha^2-d_\beta^2+4l_{\alpha\beta}^2}{8l_{\alpha\beta}}$$

Subtracting the spherical caps cut off by all bonded neighbors yields the reduced surface area $a_\alpha$ and the reduced volume $v_\alpha$ of every segment (relative to a free sphere)

$$a_\alpha=1-\frac{1}{2}\sum_{\beta\in\mathcal{B}_\alpha}\left(1-\frac{2h_{\alpha\beta}}{d_\alpha}\right),~~~~v_\alpha=1-\frac{1}{2}\sum_{\beta\in\mathcal{B}_\alpha}\left(1-\frac{3h_{\alpha\beta}}{d_\alpha}+4\frac{h_{\alpha\beta}^3}{d_\alpha^3}\right)$$

where $\mathcal{B}_\alpha$ is the set of segments bonded to $\alpha$. For a homosegmented chain with $s_i$ segments and $\lambda_i=l_i/d_i$, the sums over all segments of a molecule simplify to

$$m_i=\sum_{\alpha\in i}a_\alpha=1+\left(s_i-1\right)\lambda_i,~~~~m_i^*=\sum_{\alpha\in i}v_\alpha=1+\left(s_i-1\right)\frac{3\lambda_i-\lambda_i^3}{2}$$

For tangent spheres ($\lambda_i=1$), $m_i=m_i^*=s_i$ and the PC-SAFT geometry is recovered.

## Reference fluid

### Hard spheres

The hard-sphere contribution is calculated from the generalized [BMCSL equation of state](hard_spheres.md) with the geometry coefficients $C_{0,\alpha}=1$, $C_{1,\alpha}=C_{2,\alpha}=a_\alpha$, and $C_{3,\alpha}=v_\alpha$, i.e., with the packing fractions

$$\zeta_k=\frac{\pi}{6}\sum_\alpha\rho_\alpha C_{k,\alpha}d_\alpha^k$$

where $\rho_\alpha$ is the density of the molecule that contains segment $\alpha$. In the homosegmented model, the coefficients of the molecules are $s_i$, $m_i$, $m_i$, and $m_i^*$.

### Fused-sphere chain

The chain contribution is the TPT1 expression evaluated for every bond

$$\frac{\beta A^\mathrm{chain}}{V}=-\sum_i\rho_i\sum_{\alpha\beta\in\mathcal{B}_i}\ln y_{\alpha\beta}$$

where the inner sum runs over all bonds of molecule $i$. The cavity correlation function at the bond length is

$$y_{\alpha\beta}=\frac{1}{1-\zeta_3}+\frac{3}{2}\frac{b_{\alpha\beta}\zeta_2}{\left(1-\zeta_3\right)^2}+\frac{1}{2}\frac{b_{\alpha\beta}^2\zeta_2^2}{\left(1-\zeta_3\right)^3},~~~~b_{\alpha\beta}=\frac{4l_{\alpha\beta}^2-\left(d_\alpha-d_\beta\right)^2}{4l_{\alpha\beta}}$$

For tangent spheres, $b_{\alpha\beta}=\frac{2d_\alpha d_\beta}{d_\alpha+d_\beta}$ and the PC-SAFT chain term is recovered. In the homosegmented model, the sum reduces to $(s_i-1)\ln y_i$ with $b_i=l_i$.

## Dispersion

The dispersion contribution has the same structure as in PC-SAFT, but is evaluated for interactions between all segments

$$\frac{\beta A^\mathrm{disp}}{V}=-\pi\sum_\alpha\sum_\beta\rho_\alpha\rho_\beta a_\alpha a_\beta\frac{\varepsilon_{\alpha\beta}}{k_\mathrm{B}T}\sigma_{\alpha\beta}^3\left(2I_1(\bar{m},\eta)+\bar{m}C_1I_2(\bar{m},\eta)\frac{\varepsilon_{\alpha\beta}}{k_\mathrm{B}T}\right)$$

with the combining rules

$$\sigma_{\alpha\beta}=\frac{1}{2}\left(\sigma_\alpha+\sigma_\beta\right),~~~~\varepsilon_{\alpha\beta}=\sqrt{\varepsilon_\alpha\varepsilon_\beta}\left(1-k_{\alpha\beta}\right)$$

The reduced surface areas $a_\alpha$ weigh the interactions of the fused segments. The mean segment number is $\bar m=\sum_ix_im_i$ with $m_i=\sum_{\alpha\in i}a_\alpha$. The integrals $I_1$ and $I_2$ are power series in the packing fraction $\eta=\zeta_3$

$$I_1=\sum_{k=0}^6\left(a_{0k}+\frac{\bar m-1}{\bar m}a_{1k}+\frac{\bar m-1}{\bar m}\frac{\bar m-2}{\bar m}a_{2k}\right)\eta^k,~~~~I_2=\sum_{k=0}^6\left(b_{0k}+\frac{\bar m-1}{\bar m}b_{1k}+\frac{\bar m-1}{\bar m}\frac{\bar m-2}{\bar m}b_{2k}\right)\eta^k$$

The constants $a_{0k}$ and $b_{0k}$ are the PC-SAFT values, whereas $a_{1k}$, $a_{2k}$, $b_{1k}$, and $b_{2k}$ are adjusted to fused-sphere chains.

In the homosegmented model, all segments of a molecule share the same parameters, so the sums over the segments of each molecule can be carried out explicitly with $\sum_{\alpha\in i}a_\alpha=m_i$. The double sum then runs over components

$$\frac{\beta A^\mathrm{disp}}{V}=-\pi\sum_i\sum_j\rho_i\rho_jm_im_j\frac{\varepsilon_{ij}}{k_\mathrm{B}T}\sigma_{ij}^3\left(2I_1(\bar{m},\eta)+\bar{m}C_1I_2(\bar{m},\eta)\frac{\varepsilon_{ij}}{k_\mathrm{B}T}\right)$$

with $\bar m=\sum_ix_im_i$ and $\eta=\frac{\pi}{6}\sum_i\rho_im_i^*d_i^3$.

### Compressibility term

In PC-SAFT, the compressibility term $C_1$ is evaluated from a closed-form expression that is specific to tangent hard-sphere chains. For fused chains, the expression is not applicable. Instead, $C_1$ is calculated directly from its definition using the Helmholtz energy density of the reference fluid $\phi^\mathrm{ref}=\beta(A^\mathrm{hs}+A^\mathrm{chain})/V$

$$C_1=\left(\frac{\partial\left(\rho\left(1+Z^\mathrm{ref}\right)\right)}{\partial\rho}\right)_{T,x_i}^{-1}=\left(1+\rho\left(\frac{\partial^2\phi^\mathrm{ref}}{\partial\rho^2}\right)_{T,x_i}\right)^{-1}$$

The second derivative is calculated exactly using second-order dual numbers during the evaluation of the reference fluid, so that the Helmholtz energy of the reference fluid and $C_1$ are obtained in a single pass.

## Association

The association contribution uses the general expression described in the section on [association](association.md) with the PC-SAFT association strength

$$\Delta^{\alpha\beta}=\left(\frac{1}{1-\zeta_3}+\frac{\frac{3}{2}d_{ij}\zeta_2}{\left(1-\zeta_3\right)^2}+\frac{\frac{1}{2}d_{ij}^2\zeta_2^2}{\left(1-\zeta_3\right)^3}\right)\sqrt{\sigma_i^3\kappa^\alpha\sigma_j^3\kappa^\beta}\left(e^{\frac{\varepsilon^\alpha+\varepsilon^\beta}{2k_\mathrm{B}T}}-1\right)$$

where the packing fractions $\zeta_2$ and $\zeta_3$ are those of the fused-sphere reference fluid.

## Dipolar interactions

The homosegmented model includes the dipolar contribution of [Gross and Vrabec (2006)](https://doi.org/10.1002/aic.10683)

$$\frac{\beta A^\mathrm{dd}}{V}=\frac{\phi_2^\mathrm{dd}}{1-\phi_3^\mathrm{dd}/\phi_2^\mathrm{dd}}$$

The second- and third-order terms $\phi_2^\mathrm{dd}$ and $\phi_3^\mathrm{dd}$ are evaluated as in PC-SAFT with two adjustments to the fused-sphere geometry: the reduced dipole moment is defined using the reduced volume of the molecule

$$\mu_i^{*2}=\frac{\mu_i^2}{m_i^*\varepsilon_i\sigma_i^3}$$

and the integrals $J_2$ and $J_3$ are evaluated with the packing fraction $\eta=\zeta_3$ and the reduced surface $m_i$ (limited to $m_i\leq2$) in place of the PC-SAFT segment number.

## Helmholtz energy functional

The heterosegmented model is also available as Helmholtz energy functional for inhomogeneous systems. Its contributions reduce to the equation of state for homogeneous systems.

- **Hard spheres:** [fundamental measure theory](hard_spheres.md) with the geometry coefficients of the fused-sphere chain.
- **Chain:** the bulk expression with the segment densities $\rho_\alpha(\mathbf{r})$ and weighted densities $\bar\zeta_2$ and $\bar\zeta_3$ obtained from averaging over spheres with radius $d_\alpha$. Each bond is counted from both of its segments with a factor $\frac{1}{2}$.
- **Dispersion:** the bulk expression evaluated with segment densities that are averaged over spheres with radius $\psi_\alpha d_\alpha$. The default value is $\psi_\alpha=1.5$ and can be adjusted for every segment (`psi_dft`). The local mean segment number $\bar m$ and the compressibility term $C_1$ are evaluated from the local composition of the weighted densities, where $C_1$ is calculated exactly from the bulk reference fluid as described above. In the limit of vanishing density, $\bar m=1$ and $C_1=1$.
- **Association:** the functional of [Yu and Wu (2002)](https://doi.org/10.1063/1.1463435) in the formulation for heterosegmented molecules.
