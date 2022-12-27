# Spinodal and critical points

The stability of an isothermal system is determined by the inequality[1]

$$A-A_0-\sum_i\mu_{i0}\Delta N_i>0$$

where $A_0$ and $\mu_{i0}$ are the Helmholtz energy and chemical potentials of the state which stability is being assessed and $A$ is the Helmholtz energy of any state that is obtained from state 0 by adding the amounts $\Delta N_i$. The Helmholtz energy can be expanded around the test point, giving

$$\sum_{ij}\left(\frac{\partial^2 A}{\partial N_i\partial N_j}\right)\Delta N_i\Delta N_j+\sum_{ijk}\left(\frac{\partial^3 A}{\partial N_i\partial N_j\partial N_k}\right)\Delta N_i\Delta N_j\Delta N_k+\mathcal{O}\left(\Delta N^4\right)>0$$(eqn:stability)

The inequality must be fulfilled for any $\Delta N_i$. Therefore, a necessary criterion for stability is that the quadratic form (the first term) in eq. {eq}`eqn:stability` is [positive definite](https://en.wikipedia.org/wiki/Definite_quadratic_form). The **spinodal** or limit of stability consists of the points for which the quadratic form is positive semi-definite. The stability is then determined by the higher-order terms in eq. {eq}`eqn:stability`. Mathematically there exists a $\Delta N_i$ that fulfills

$$\sum_i\left(\frac{\partial^2 A}{\partial N_i\partial N_j}\right)\Delta N_i=0$$

A **critical point** is defined as a stable point on the limit of stability.[2] For that to be the case, the quadratic and cubic terms in eq. {eq}`eqn:stability` need to vanish[1] which leads to the second criticality condition

$$\sum_{ijk}\left(\frac{\partial^3 A}{\partial N_i\partial N_j\partial N_k}\right)\Delta N_i\Delta N_j\Delta N_k=0$$

The criticality conditions can be reformulated by defining the matrix $M$ with[3]

$$M_{ij}=\sqrt{x_ix_j}\left(\frac{\partial^2\beta A}{\partial N_i\partial N_j}\right)$$

and $u$ as the eigenvector corresponding to the smallest eigenvalue $\lambda_1$ of $M$. $M$ and in conclusion the quadratic form in eq. {eq}`eqn:stability` is positive semi-definite if and only if its smallest eigenvalue is 0. Therefore the first criticality condition simplifies to

$$\lambda_1=0$$

For the second criticality condition $\Delta N_i$ a step $s$ is defined that acts on the mole numbers $N_i$ as

$$N_i=z_i+su_i\sqrt{z_i}$$

Then the derivative $\left.\frac{\partial^3 A}{\partial s^3}\right|_{s=0}$ can be rewritten as

$$\left.\frac{\partial^3 A}{\partial s^3}\right|_{s=0}=\sum_{ijk}\left(\frac{\partial^3 A}{\partial N_i\partial N_j\partial N_k}\right)u_iu_ju_k\sqrt{z_iz_jz_k}$$

$$C=\sum_{ij}\left(\sum_k\left(\frac{\partial^3 A}{\partial N_i\partial N_j\partial N_k}\right)\Delta N_k\right)\Delta N_i\Delta N_j$$