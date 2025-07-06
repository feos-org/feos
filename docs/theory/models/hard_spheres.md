# Hard spheres

$\text{FeO}_\text{s}$ provides an implementation of the Boublík-Mansoori-Carnahan-Starling-Leland (BMCSL) equation of state ([Boublík, 1970](https://doi.org/10.1063/1.1673824), [Mansoori et al., 1971](https://doi.org/10.1063/1.1675048)) for hard-sphere mixtures which is often used as reference contribution in SAFT equations of state. The implementation is generalized to allow the description of non-sperical or fused-sphere reference fluids.

The reduced Helmholtz energy density is calculated according to

$$\frac{\beta A}{V}=\frac{6}{\pi}\left(\frac{3\zeta_1\zeta_2}{1-\zeta_3}+\frac{\zeta_2^3}{\zeta_3\left(1-\zeta_3\right)^2}+\left(\frac{\zeta_2^3}{\zeta_3^2}-\zeta_0\right)\ln\left(1-\zeta_3\right)\right)$$

with the packing fractions

$$\zeta_k=\frac{\pi}{6}\sum_\alpha C_{k,\alpha}\rho_\alpha d_\alpha^k,~~~~~~~~k=0\ldots 3$$

The geometry coefficients $C_{k,\alpha}$ and the segment diameters $d_\alpha$ depend on the context in which the model is used. The following table shows how the expression can be reused in various models. For details on the fused-sphere chain model, check the [repository](https://github.com/feos-org/feos-fused-chains) or the [publication](https://doi.org/10.1103/PhysRevE.105.034110).

||Hard spheres|PC-SAFT|Fused-sphere chains|
|-|:-:|:-:|:-:|
|$d_\alpha$|$\sigma_\alpha$|$\sigma_\alpha\left(1-0.12e^{\frac{-3\varepsilon_\alpha}{k_\mathrm{B}T}}\right)$|$\sigma_\alpha$|
|$C_{0,\alpha}$|$1$|$m_\alpha$|$1$|
|$C_{1,\alpha}$|$1$|$m_\alpha$|$A_\alpha^*$|
|$C_{1,\alpha}$|$1$|$m_\alpha$|$A_\alpha^*$|
|$C_{1,\alpha}$|$1$|$m_\alpha$|$V_\alpha^*$|

## Fundamental measure theory

An model for inhomogeneous mixtues of hard spheres is provided by fundamental measure theory (FMT, [Rosenfeld, 1989](https://doi.org/10.1103/PhysRevLett.63.980)). Different variants have been proposed of which only those that are consistent with the BMCSL equation of state in the homogeneous limit are currently considered in $\text{FeO}_\text{s}$ (exluding, e.g., the original Rosenfeld and White-Bear II variants).

The Helmholtz energy density is calculated according to

$$\beta f=-n_0\ln\left(1-n_3\right)+\frac{n_{12}}{1-n_3}+\frac{1}{36\pi}n_2n_{22}f_3(n_3)$$

The expressions for $n_{12}$ and $n_{22}$ depend on the FMT version.

|version|$n_{12}$|$n_{22}$|references|
|-|:-:|:-:|:-:|
|WhiteBear|$n_1n_2-\vec n_1\cdot\vec n_2$|$n_2^2-3\vec n_2\cdot\vec n_2$|[Roth et al., 2002](https://doi.org/10.1088/0953-8984/14/46/313), [Yu and Wu, 2002](https://doi.org/10.1063/1.1520530)|
|KierlikRosinberg|$n_1n_2$|$n_2^2$|[Kierlik and Rosinberg, 1990](https://doi.org/10.1103/PhysRevA.42.3382)|
|AntiSymWhiteBear|$n_1n_2-\vec n_1\cdot\vec n_2$|$n_2^2\left(1-\frac{\vec n_2\cdot\vec n_2}{n_2^2}\right)^3$|[Rosenfeld et al., 1997](https://doi.org/10.1103/PhysRevE.55.4245), [Kessler et al., 2021](https://doi.org/10.1016/j.micromeso.2021.111263)|

For small $n_3$, the value of $f(n_3)$ numerically diverges. Therefore, it is approximated with a Taylor expansion.

$$f_3=\begin{cases}\frac{n_3+\left(1-n_3\right)^2\ln\left(1-n_3\right)}{n_3^2\left(1-n_3\right)^2}&\text{if }n_3>10^{-5}\\
\frac{3}{2}+\frac{8}{3}n_3+\frac{15}{4}n_3^2+\frac{24}{5}n_3^3+\frac{35}{6}n_3^4&\text{else}\end{cases}$$

The weighted densities $n_k(\mathbf{r})$ are calculated by convolving the density profiles $\rho_\alpha(\mathbf{r})$ with weight functions $\omega_k^\alpha(\mathbf{r})$

$$n_k(\mathbf{r})=\sum_\alpha n_k^\alpha(\mathbf{r})=\sum_\alpha\int\rho_\alpha(\mathbf{r}')\omega_k^\alpha(\mathbf{r}-\mathbf{r}')\mathrm{d}\mathbf{r}'$$

which differ between the different FMT versions.

||WhiteBear/AntiSymWhiteBear|KierlikRosinberg|
|-|:-:|:-:|
|$\omega_0^\alpha(\mathbf{r})$|$\frac{C_{0,\alpha}}{\pi\sigma_\alpha^2}\,\delta\!\left(\frac{d_\alpha}{2}-\|\mathbf{r}\|\right)$|$C_{0,\alpha}\left(-\frac{1}{8\pi}\,\delta''\!\left(\frac{d_\alpha}{2}-\|\mathbf{r}\|\right)+\frac{1}{2\pi\|\mathbf{r}\|}\,\delta'\!\left(\frac{d_\alpha}{2}-\|\mathbf{r}\|\right)\right)$|
|$\omega_1^\alpha(\mathbf{r})$|$\frac{C_{1,\alpha}}{2\pi\sigma_\alpha}\,\delta\!\left(\frac{d_\alpha}{2}-\|\mathbf{r}\|\right)$|$\frac{C_{1,\alpha}}{8\pi}\,\delta'\!\left(\frac{d_\alpha}{2}-\|\mathbf{r}\|\right)$|
|$\omega_2^\alpha(\mathbf{r})$|$C_{2,\alpha}\,\delta\!\left(\frac{d_\alpha}{2}-\|\mathbf{r}\|\right)$|$C_{2,\alpha}\,\delta\!\left(\frac{d_\alpha}{2}-\|\mathbf{r}\|\right)$|
|$\omega_3^\alpha(\mathbf{r})$|$C_{3,\alpha}\,\Theta\!\left(\frac{d_\alpha}{2}-\|\mathbf{r}\|\right)$|$C_{3,\alpha}\,\Theta\!\left(\frac{d_\alpha}{2}-\|\mathbf{r}\|\right)$|
|$\vec\omega_1^\alpha(\mathbf{r})$|$C_{3,\alpha}\frac{\mathbf{r}}{2\pi\sigma_\alpha\|\mathbf{r}\|}\,\delta\!\left(\frac{d_\alpha}{2}-\|\mathbf{r}\|\right)$|-|
|$\vec\omega_2^\alpha(\mathbf{r})$|$C_{3,\alpha}\frac{\mathbf{r}}{\|\mathbf{r}\|}\,\delta\!\left(\frac{d_\alpha}{2}-\|\mathbf{r}\|\right)$|-|