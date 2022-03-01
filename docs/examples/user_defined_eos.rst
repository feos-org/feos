Implementing an equation of state in python
===========================================

   In ``FeOs``, you can implement your equation of state in python,
   register it to the Rust backend, and compute properties and phase
   equilbria as if you implemented it in Rust. In this tutorial, we will
   implement the Peng-Robinson equation of state.

Table of contents 
------------------

-  `Setup <#Setup>`__
-  `Implementation <#Implementation>`__
-  `Computing properties <#Computing-properties>`__
-  `Critical point <#Critical-point>`__
-  `Phase equilibria and phase
   diagrams <#Phase-equilibria-and-phase-diagrams>`__
-  `Mixtures <#Mixtures>`__
-  `Comparison to Rust
   implementation <#Comparison-to-Rust-implementation>`__

Setup 
------

`↑ Back to top <#toc>`__

Let’s start by importing the necessary python packages. Mandatory
packages are

-  ``feos.user_defined`` for classes such as ``State``,
   ``PhaseEquilibrium``, and ``UserDefinedEos``,
-  ``feos.si`` for SI numbers,
-  ``numpy`` for multidimensional arrays.

We also recommend using

-  ``matplotlib`` and ``seaborn`` for visualization, and
-  ``pandas`` for easier handling data.

If you don’t want to use the optional packages, simply remove them from
the import statement below or set ``optional = False``.

.. code:: ipython3

    from feos.user_defined import *
    from feos.si import *
    import numpy as np
    
    optional = True
    
    if optional:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    
        sns.set_style("ticks")
        sns.set_palette("Dark2")
        sns.set_context("talk")

Implementation 
---------------

`↑ Back to top <#toc>`__

To implement an equation of state in python, we have to define a
``class`` which has to have the following methods:

.. code:: python

   class EquationOfState:
       def helmholtz_energy(self, state: StateHD) -> D
       
       def components(self) -> int
       
       def subset(self, indices: List[int]) -> Self
       
       def molar_weight(self) -> SIArray1
       
       def max_density(self, moles: SIArray1) -> f64

-  ``components(self) -> int``: Returns the number of components
   (usually inferred from the shape of the input parameters).
-  ``molar_weight(self) -> SIArray1``: Returns an ``SIArray1`` with size
   equal to the number of components containing the molar mass of each
   component.
-  ``max_density(self, moles: np.ndarray[float]) -> float``: Returns the
   maximum allowed number density in units of ``molecules/Angstrom³``.
-  ``subset(self, indices: List[int]) -> self``: Returns a new equation
   of state with parameters defined in ``indices``.
-  ``helmholtz_energy(self, state: StateHD) -> Dual``: Returns the
   helmholtz energy as (hyper)-dual number given a ``StateHD``.

.. code:: ipython3

    SQRT2 = np.sqrt(2)
    
    class PyPengRobinson: 
        def __init__(self, critical_temperature, critical_pressure, acentric_factor, molar_weight, delta_ij=None):
            """Peng-Robinson Equation of State
            
            Parameters
            ----------
            critical_temperature : SIArray1
                critical temperature of each component.
            critical_pressure : SIArray1
                critical pressure of each component.
            acentric_factor : np.array[float] 
                acentric factor of each component (dimensionless).
            molar_weight: SIArray1
                molar weight of each component.
            delta_ij : np.array[[float]], optional
                binary parameters. Shape=[n, n], n = number of components.
                defaults to zero for all binary interactions.
                
            Raises
            ------
            ValueError: if the input values have incompatible sizes.
            """
            self.n = len(critical_temperature)
            if len(set((len(critical_temperature), len(critical_pressure), len(acentric_factor)))) != 1:
                raise ValueError("Input parameters must all have the same lenght.")
            
            if self.n == 1:
                self.tc = (critical_temperature / KELVIN)[0]
                self.pc = (critical_pressure / PASCAL)[0]
                self.omega = acentric_factor[0]
                self.mw = (molar_weight / GRAM * MOL)[0]
            else:
                self.tc = critical_temperature / KELVIN
                self.pc = critical_pressure / PASCAL
                self.omega = acentric_factor
                self.mw = molar_weight / GRAM * MOL
            
            self.a_r = 0.45724 * critical_temperature**2 * RGAS / critical_pressure / ANGSTROM**3 / NAV / KELVIN
            self.b = 0.07780 * critical_temperature * RGAS / critical_pressure / ANGSTROM**3 / NAV
            self.kappa = 0.37464 + (1.54226 - 0.26992 * acentric_factor) * acentric_factor
            self.delta_ij = np.zeros((self.n, self.n)) if delta_ij is None else delta_ij
            
        def helmholtz_energy(self, state):
            """Return helmholtz energy.
            
            Parameters
            ----------
            state : StateHD
                The thermodynamic state.
                
            Returns
            -------
            helmholtz_energy: float | any dual number
                The return type depends on the input types.
            """      
            n = np.sum(state.moles)
            x = state.molefracs
            tr = 1.0 / self.tc * state.temperature
            ak = ((1.0 - np.sqrt(tr)) * self.kappa + 1.0)**2 * self.a_r
            ak_mix = 0.0
            if self.n > 1:
                for i in range(self.n):
                    for j in range(self.n):
                        ak_mix += np.sqrt(ak[i] * ak[j]) * (x[i] * x[j] * (1.0 - self.delta_ij[i, j]))
            else:
                ak_mix = ak
            b = np.sum(x * self.b)
            v = state.volume
            a = n * (np.log(v / (v - b * n)) - ak_mix / (b * SQRT2 * 2.0 * state.temperature)
                * np.log((v * (SQRT2 - 1.0) + b * n) / (v * (SQRT2 + 1.0) - b * n)))
            return a
        
        def components(self) -> int: 
            """Number of components."""
            return self.n
        
        def subset(self, i: [int]):
            """Return new equation of state containing a subset of all components."""
            if self.n > 1:
                tc = self.tc[i] 
                pc = self.pc[i]
                mw = self.mw[i]
                omega = self.omega[i]
                return PyPengRobinson(tc*KELVIN, pc*PASCAL, omega, mw*GRAM/MOL)
            else:
                return self
            
        def molar_weight(self) -> SIArray1:
            if isinstance(self.mw, float):
                return np.array([self.mw]) * GRAM / MOL
            else:
                return self.mw * GRAM / MOL
        
        def max_density(self, moles:[float]) -> float:
            b = np.sum(moles * self.b) / np.sum(moles);
            return 0.9 / b 

Computing properties 
---------------------

`↑ Back to top <#toc>`__

Let’s compute some properties. First, we have to instanciate the class
and register it to Rust. This is done using the ``UserDefinedEos``
class.

.. code:: ipython3

    # parameters for propane
    tc = np.array([369.96]) * KELVIN
    pc = np.array([4250000.0]) * PASCAL
    omega = np.array([0.153])
    molar_weight = np.array([44.0962]) * GRAM / MOL
    
    # create an instance of our python class and hand it over to rust
    eos = UserDefinedEos(PyPengRobinson(tc, pc, omega, molar_weight))

Thermodynamic state
~~~~~~~~~~~~~~~~~~~

Before we can compute a property, we create a ``State`` object. This can
be done in several ways depending on what control variables we need. If
no total amount of substance is defined, it is set to
:math:`n = \frac{1}{N_{AV}}`. For possible input combinations, you can
inspect the signature of the constructor using ``State?``.

.. code:: ipython3

    State?



.. parsed-literal::

    [0;31mInit signature:[0m [0mState[0m[0;34m([0m[0mself[0m[0;34m,[0m [0;34m/[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0;31mDocstring:[0m     
    A thermodynamic state at given conditions.
    
    Parameters
    ----------
    eos : Eos
        The equation of state to use.
    temperature : SINumber, optional
        Temperature.
    volume : SINumber, optional
        Volume.
    density : SINumber, optional
        Molar density.
    partial_density : SIArray1, optional
        Partial molar densities.
    total_moles : SINumber, optional
        Total amount of substance (of a mixture).
    moles : SIArray1, optional
        Amount of substance for each component.
    molefracs : numpy.ndarray[float]
        Molar fraction of each component.
    pressure : SINumber, optional
        System pressure.
    enthalpy : SINumber, optional
        System enthalpy.
    entropy : SInumber, optional
        System entropy.
    density_initialization : {'vapor', 'liquid', SINumber, None}, optional
        Method used to initialize density for density iteration.
        'vapor' and 'liquid' are inferred from the maximum density of the equation of state.
        If no density or keyword is provided, the vapor and liquid phase is tested and, if
        different, the Pyresult with the lower free energy is returned.
    initial_temperature : SINumber, optional
        Initial temperature for temperature iteration. Can improve convergence
        when the state is specified with pressure and entropy or enthalpy.
    
    Returns
    -------
    state at given conditions
    
    Raises
    ------
    Error
        When the state cannot be created using the combination of input.
    [0;31mType:[0m           type
    [0;31mSubclasses:[0m     



If we use input variables other than :math:`\mathbf{N}, V, T` (the
natural variables of the Helmholtz energy), creating a state is an
iterative procedure. For example, we can create a state for a give
:math:`T, p`, which will result in a iteration of the volume (density).

.. code:: ipython3

    # If no amount of substance is given, it is set to 1/NAV.
    s = State(eos, temperature=300*KELVIN, pressure=1*BAR)
    s.total_moles




.. math::

    1.6605\times10^{-24}\,\mathrm{mol}



.. code:: ipython3

    s_pt = State(eos, temperature=300*KELVIN, pressure=1*BAR, total_moles=1*MOL)
    s_pt.total_moles




.. math::

    1\,\mathrm{ mol}



We can use other variables as well. For example, we can create a state
at given :math:`h, p` (using the enthalpy from the prior computation as
input):

.. code:: ipython3

    h = s.molar_enthalpy()
    s_ph = State(eos, pressure=1*BAR, enthalpy=s_pt.molar_enthalpy())

.. code:: ipython3

    # check if states are equal
    print("rel. dev.")
    print("entropy    : ", (s_ph.molar_entropy() - s_pt.molar_entropy()) / s_pt.molar_entropy())
    print("density    : ", (s_ph.mass_density() - s_pt.mass_density()) / s_pt.mass_density())
    print("temperature: ", (s_ph.temperature - s_pt.temperature) / s_pt.temperature)


.. parsed-literal::

    rel. dev.
    entropy    :  2.405775647837077e-16
    density    :  3.335935327920463e-15
    temperature:  -3.4106051316484808e-15


Critical point 
---------------

`↑ Back to top <#toc>`__

To generate a state at critical conditions, we can use the
``critical_point`` constructor.

.. code:: ipython3

    s_cp = State.critical_point(eos)
    print("Critical point")
    print("temperature: ", s_cp.temperature)
    print("density    : ", s_cp.mass_density())
    print("pressure   : ", s_cp.pressure())


.. parsed-literal::

    Critical point
    temperature:  369.9506174234607 K
    density    :  198.18624580571773 kg/m³
    pressure   :  4.249677749116937 MPa


Phase equilibria and phase diagrams
-----------------------------------

`↑ Back to top <#toc>`__

We can also create an object, ``PhaseEquilibrium``, that contains states
that are in equilibrium.

.. code:: ipython3

    vle = PhaseEquilibrium.pure_t(eos, temperature=350*KELVIN)
    vle




+---------+-------------+----------------------------+
|         | temperature | density                    |
+=========+=============+============================+
| phase 1 | 350 K       | 1.7885829531450665 kmol/m³ |
+---------+-------------+----------------------------+
| phase 2 | 350 K       | 8.190339472897485 kmol/m³  |
+---------+-------------+----------------------------+



Each phase is a ``State`` object. We can simply access these states and
compute properties, just like before.

.. code:: ipython3

    vle.liquid # the high density phase `State`




+-------------+---------------------------+
| temperature | density                   |
+=============+===========================+
| 350 K       | 8.190339472897485 kmol/m³ |
+-------------+---------------------------+



.. code:: ipython3

    vle.vapor # the low density phase `State`




+-------------+----------------------------+
| temperature | density                    |
+=============+============================+
| 350 K       | 1.7885829531450665 kmol/m³ |
+-------------+----------------------------+



.. code:: ipython3

    # we can now easily compute any property:
    print("Heat of vaporization: ", vle.vapor.molar_enthalpy() - vle.liquid.molar_enthalpy())
    print("for T = {}".format(vle.liquid.temperature))
    print("and p = {:.2f} bar".format(vle.liquid.pressure() / BAR))


.. parsed-literal::

    Heat of vaporization:  8.591742172312552 kJ/mol
    for T = 350 K
    and p = 29.63 bar


We can also easily compute **vapor pressures** and **boiling
temperatures**:

.. code:: ipython3

    # This also works for mixtures, in which case the pure component properties are computed.
    # Hence, the result is a list - that is why we use an index [0] here.
    print("vapor pressure      (T = 300 K):", PhaseEquilibrium.vapor_pressure(eos, 300*KELVIN)[0])
    print("boiling temperature (p = 3 bar):", PhaseEquilibrium.boiling_temperature(eos, 2*BAR)[0])


.. parsed-literal::

    vapor pressure      (T = 300 K): 994.7761635610083 kPa
    boiling temperature (p = 3 bar): 247.84035574956746 K


Phase Diagram
~~~~~~~~~~~~~

We could repeatedly compute ``PhaseEquilibrium`` states for different
temperatures / pressures to generate a phase diagram. Because this a
common task, there is a object for that as well.

The ``PhaseDiagramPure`` object creates multiple ``PhaseEquilibrium``
objects (``npoints``) between a given lower temperature and the critical
point.

.. code:: ipython3

    dia = PhaseDiagramPure(eos, 230.0 * KELVIN, 500)

We can have access to each ``PhaseEquilbrium`` and can conveniently
comput any property we like:

.. code:: ipython3

    enthalpy_of_vaporization = [(vle.vapor.molar_enthalpy() - vle.liquid.molar_enthalpy()) / (KILO * JOULE) * MOL for vle in dia.states]

.. code:: ipython3

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.lineplot(x=dia.temperature / KELVIN, y=enthalpy_of_vaporization, ax=ax);
    ax.set_ylabel(r"$\Delta^{LV}h$ / kJ / mol")
    ax.set_xlabel(r"$T$ / K");



.. image:: user_defined_eos_files/user_defined_eos_30_0.png


A more convenient way is to create a dictionary. The dictionary can
conveniently be used with pandas dataframe objects. This is a bit less
flexible, because the units of the properties are rigid. You can inspect
the method signature to check what units are used.

.. code:: ipython3

    dia.to_dict?



.. parsed-literal::

    [0;31mDocstring:[0m
    Returns the phase diagram as dictionary.
    
    Returns
    -------
        dict[str, list[float]]
            Keys: property names.
            Values: property for each state.
    
    Units
    -----
    temperature : K
    pressure : Pa
    densities : mol / m³
    molar enthalpies : kJ / mol
    molar entropies : kJ / mol / K
    [0;31mType:[0m      builtin_function_or_method



.. code:: ipython3

    data_dia = pd.DataFrame(dia.to_dict())
    data_dia.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>molar enthalpy vapor</th>
          <th>molar entropy vapor</th>
          <th>density vapor</th>
          <th>molar entropy liquid</th>
          <th>temperature</th>
          <th>pressure</th>
          <th>density liquid</th>
          <th>molar enthalpy liquid</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>22.140400</td>
          <td>0.120689</td>
          <td>52.208491</td>
          <td>0.039106</td>
          <td>230.000000</td>
          <td>96625.278174</td>
          <td>14125.988947</td>
          <td>3.376293</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.135738</td>
          <td>0.120569</td>
          <td>52.811929</td>
          <td>0.039135</td>
          <td>230.280462</td>
          <td>97830.133956</td>
          <td>14118.006852</td>
          <td>3.383021</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.131064</td>
          <td>0.120449</td>
          <td>53.420767</td>
          <td>0.039164</td>
          <td>230.560924</td>
          <td>99046.729400</td>
          <td>14110.010220</td>
          <td>3.389761</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.126380</td>
          <td>0.120330</td>
          <td>54.035036</td>
          <td>0.039193</td>
          <td>230.841386</td>
          <td>100275.143120</td>
          <td>14101.999011</td>
          <td>3.396514</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.121683</td>
          <td>0.120211</td>
          <td>54.654773</td>
          <td>0.039221</td>
          <td>231.121849</td>
          <td>101515.453964</td>
          <td>14093.973182</td>
          <td>3.403278</td>
        </tr>
      </tbody>
    </table>
    </div>



Once we have a dataframe, we can store our results or create a nicely
looking plot:

.. code:: ipython3

    def phase_plot(data, x, y):
        fig, ax = plt.subplots(figsize=(12, 6))
        if x != "pressure" and x != "temperature":
            xl = f"{x} liquid"
            xv = f"{x} vapor"
        else:
            xl = x
            xv = x
        if y != "pressure" and y != "temperature":
            yl = f"{y} liquid"
            yv = f"{y} vapor"
        else:
            yv = y
            yl = y
        sns.lineplot(data=data, x=xv, y=yv, ax=ax, label="vapor")
        sns.lineplot(data=data, x=xl, y=yl, ax=ax, label="liquid")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.legend(frameon=False)
        sns.despine();

.. code:: ipython3

    phase_plot(data_dia, "density", "temperature")



.. image:: user_defined_eos_files/user_defined_eos_36_0.png


.. code:: ipython3

    phase_plot(data_dia, "molar entropy", "temperature")



.. image:: user_defined_eos_files/user_defined_eos_37_0.png


Mixtures 
---------

`↑ Back to top <#toc>`__

Fox mixtures, we have to add information about the composition, either
as molar fraction, amount of substance per component, or as partial
densities.

.. code:: ipython3

    # propane, butane mixture
    tc = np.array([369.96, 425.2]) * KELVIN
    pc = np.array([4250000.0, 3800000.0]) * PASCAL
    omega = np.array([0.153, 0.199])
    molar_weight = np.array([44.0962, 58.123]) * GRAM / MOL
    
    eos = UserDefinedEos(PyPengRobinson(tc, pc, omega, molar_weight))

.. code:: ipython3

    s = State(eos, temperature=300*KELVIN, pressure=1*BAR, molefracs=np.array([0.5, 0.5]), total_moles=MOL)
    s




+-------------+--------------------------+--------------------+
| temperature | density                  | molefracs          |
+=============+==========================+====================+
| 300 K       | 40.96869036334592 mol/m³ | [0.50000, 0.50000] |
+-------------+--------------------------+--------------------+



As before, we can compute properties by calling methods on the ``State``
object. Some return vectors or matrices - for example the chemical
potential and its derivative w.r.t amount of substance:

.. code:: ipython3

    s.chemical_potential()




.. parsed-literal::

    [-15625.347451682397, -12435.866602695123] J/mol



.. code:: ipython3

    s.dmu_dni() / (KILO * JOULE / MOL**2)




.. parsed-literal::

    array([[ 4.90827975, -0.10593968],
           [-0.10593968,  4.85467746]])



Phase equilibria are compute from different constructors:

.. code:: ipython3

    s_cp = State.critical_point(eos, moles=np.array([0.5, 0.5])*MOL)
    s_cp




+----------------------+---------------------------+--------------------+
| temperature          | density                   | molefracs          |
+======================+===========================+====================+
| 401.65486400484747 K | 3.999524081819513 kmol/m³ | [0.50000, 0.50000] |
+----------------------+---------------------------+--------------------+



.. code:: ipython3

    vle = PhaseEquilibrium.bubble_point_tx(eos, 350*KELVIN, liquid_molefracs=np.array([0.5, 0.5]))
    vle




+---------+-------------+---------------------------+--------------------+
|         | temperature | density                   | molefracs          |
+=========+=============+===========================+====================+
| phase 1 | 350 K       | 879.4750481224572 mol/m³  | [0.67625, 0.32375] |
+---------+-------------+---------------------------+--------------------+
| phase 2 | 350 K       | 8.963820901871767 kmol/m³ | [0.50000, 0.50000] |
+---------+-------------+---------------------------+--------------------+



.. code:: ipython3

    vle = PhaseDiagramBinary.new_pxy(eos, temperature=350*KELVIN, npoints=50)

.. code:: ipython3

    fig, ax = plt.subplots(1, 2, figsize=(18, 6))
    # fig.title("T = 350 K, Propane (1), Butane (2)")
    sns.lineplot(x=vle.liquid_molefracs, y=vle.pressure / BAR, ax=ax[0])
    sns.lineplot(x=vle.vapor_molefracs, y=vle.pressure / BAR, ax=ax[0])
    ax[0].set_xlabel(r"$x_1$, $y_1$")
    ax[0].set_ylabel(r"$p$ / bar")
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(5, 35)
    # ax[0].legend(frameon=False);
    
    sns.lineplot(x=vle.liquid_molefracs, y=vle.vapor_molefracs, ax=ax[1])
    sns.lineplot(x=np.linspace(0, 1, 10), y=np.linspace(0, 1, 10), color="black", alpha=0.3, ax=ax[1])
    ax[1].set_xlabel(r"$x_1$")
    ax[1].set_ylabel(r"$y_1$")
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 1);



.. image:: user_defined_eos_files/user_defined_eos_48_0.png


Comparison to Rust implementation 
----------------------------------

`↑ Back to top <#toc>`__

Implementing an equation of state in Python is nice for quick
prototyping and development but when it comes to performance,
implementing the equation of state in Rust is the way to go. For each
non-cached call to the Helmholtz energy, we have to transition between
Rust and Python with our Python implementation which generates quite
some overhead.

Here are some comparisons between the Rust and our Pyhton implemenation:

.. code:: ipython3

    # rust
    from feos.cubic import PengRobinson, State as StateR, PengRobinsonParameters, PhaseDiagramPure as PhaseDiagramPureR
    eos_rust = PengRobinson(PengRobinsonParameters.from_json(["propane"], "peng-robinson.json"))
    
    # python
    tc = np.array([369.96]) * KELVIN
    pc = np.array([4250000.0]) * PASCAL
    omega = np.array([0.153])
    molar_weight = np.array([44.0962]) * GRAM / MOL
    eos_python = UserDefinedEos(PyPengRobinson(tc, pc, omega, molar_weight))

.. code:: ipython3

    # let's first test if both actually yield the same results ;)
    assert abs(State.critical_point(eos_python).pressure() / BAR - StateR.critical_point(eos_rust).pressure() / BAR) < 1e-13
    assert abs(State.critical_point(eos_python).temperature / KELVIN - StateR.critical_point(eos_rust).temperature / KELVIN) < 1e-13

.. code:: ipython3

    import timeit
    
    time_python = timeit.timeit(lambda: State.critical_point(eos_python), number=2_500) * MILLI * SECOND
    time_rust = timeit.timeit(lambda: StateR.critical_point(eos_rust), number=2_500) * MILLI * SECOND

.. code:: ipython3

    rel_dev = (time_rust - time_python) / time_rust
    print(f"Critical point for pure substance")
    print(f"Python implementation is {'slower' if rel_dev < 0 else 'faster'} by a factor of {abs(time_python / time_rust):.0f}.")


.. parsed-literal::

    Critical point for pure substance
    Python implementation is slower by a factor of 37.


.. code:: ipython3

    time_python = timeit.timeit(lambda: PhaseDiagramPure(eos_python, 300*KELVIN, 100), number=100) * MILLI * SECOND
    time_rust = timeit.timeit(lambda: PhaseDiagramPureR(eos_rust, 300*KELVIN, 100), number=100) * MILLI * SECOND


.. parsed-literal::

     domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log domain error
    
    log dom

.. code:: ipython3

    rel_dev = (time_rust - time_python) / time_rust
    print(f"Phase diagram for pure substance")
    print(f"Python implementation is {'slower' if rel_dev < 0 else 'faster'} by a factor of {abs(time_python / time_rust):.0f}.")


.. parsed-literal::

    Phase diagram for pure substance
    Python implementation is slower by a factor of 27.


