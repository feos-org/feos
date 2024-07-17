# Stability and critical points

The implementation of critical points in $\text{FeO}_\text{s}$ follows the algorithm by [Michelsen and Mollerup](https://tie-tech.com/new-book-release/). A necessary condition for stability is the positive-definiteness of the quadratic form ([Heidemann and Khalil 1980](https://doi.org/10.1002/aic.690260510))

$$\sum_{ij}\left(\frac{\partial^2 A}{\partial N_i\partial N_j}\right)_{T,V}\Delta N_i\Delta N_j$$

The **spinodal** or limit of stability consists of the points for which the quadratic form is positive semi-definite. Following Michelsen and Mollerup, the matrix $M$ can be defined as

$$M_{ij}=\sqrt{z_iz_j}\left(\frac{\partial^2\beta A}{\partial N_i\partial N_j}\right)$$

with the molar compositon $z_i$. Further, the variable $s$ is introduced that acts on the mole numbers $N_i$ via

$$N_i=z_i+su_i\sqrt{z_i}$$

with $u_i$ the elements of the eigenvector of $M$ corresponding to the smallest eigenvector $\lambda_1$. Then, the limit of stability can be expressed as

$$c_1=\left.\frac{\partial^2\beta A}{\partial s^2}\right|_{s=0}=\sum_{ij}u_iu_jM_{ij}=\lambda_1=0$$

A **critical point** is defined as a stable point on the limit of stability. This leads to the second criticality condition

$$c_2=\left.\frac{\partial^3\beta A}{\partial s^3}\right|_{s=0}=0$$

The derivatives of the Helmholtz energy can be calculated efficiently in a single evaluation using [generalized hyper-dual numbers](https://doi.org/10.3389/fceng.2021.758090). The following methods of `State` are available to determine spinodal or critical points for different specifications:

||specified|unkonwns|equations|
|-|-|-|-|
|`spinodal`|$T,N_i$|$\rho$|$c_1(T,\rho,N_i)=0$|
|`critical_point`|$N_i$|$T,\rho$|$c_1(T,\rho,N_i)=0$<br/>$c_2(T,\rho,N_i)=0$|
|`critical_point_binary_t`|$T$|$\rho_1,\rho_2$|$c_1(T,\rho_1,\rho_2)=0$<br/>$c_2(T,\rho_1,\rho_2)=0$|
|`critical_point_binary_p`|$p$|$T,\rho_1,\rho_2$|$c_1(T,\rho_1,\rho_2)=0$<br/>$c_2(T,\rho_1,\rho_2)=0$<br/>$p(T,\rho_1,\rho_2)=p$|
