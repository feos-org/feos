Thermodynamic states
====================

One of the most important data type of `FeOs` is the `State` struct.
A `State`

- stores (a pointer to) the equation of state,
- is used to store temperature, volume, and amount of substance per species as dimensioned quantities,
- defines constructors for different combination of inputs, e.g. :math:`(T, p)`, :math:`(h, p)`, :math:`(T, s)` or :math:`(T, p, \mathbf{x})`,
- defines constructors for critical point calculations,
- and provides methods to compute thermodynamic properties such as pressure, chemical potential or activity coefficients.

`State` is the object users interact with the most.
In what follows we will take a closer look at the specific procedures to compute a property for a given `State`.

:math:`\mathbf{N}, V, T`: the natural variables of `State`
----------------------------------------------------------

`FeOs` is build around equations of state that are formulated in terms of residual Helmholtz energies.
We harness the fact that all thermodynamic properties of a system can be computed from (partial) derivatives of the Helmholtz energy.
As a consequence, we store the natural variables of the Helmholtz energy, :math:`(\mathbf{N}, V, T)`, in our `State` object together with some other variables for convenience, such as the density or the mole fractions.

The most straight forward way to construct a `State` is by providing :math:`(\mathbf{N}, V, T)`.
Using other constructors, e.g. for given pressure, entropy or enthalpy, is more expensive since we have to perform an iteration of volume (density) and/or temperature.

Computing Thermodynamic Properties
----------------------------------

Using Dual Numbers
~~~~~~~~~~~~~~~~~~

As mentioned above, thermodynamic properties can readily be computed via (partial) derivatives once the Helmholtz energy function is known.
In our code, these derivatives are **not implemented analytically**, instead we use **generalized dual numbers** (in what follows only referred to as dual numbers).

Dual numbers, by construction, enable the simultaneous evaluation of a function and its partial derivatives.
Which derivatives of a function can be determined depends on the concrete dual number.
For example, for the first derivative a dual number consisting of a real and one non-real part is sufficient while for second partial derivatives we need a dual number consisting of a real and three non-real parts (this number is referred to as hyper-dual number).

Our implementation of dual numbers provides multiple data types that can be used depending on the derivative that is needed.
Consequently, computation of partial derivatives comes down to **writing the Helmholtz energy function in terms of generalized dual numbers**.
Technically, this is achieved by parameterising the Helmholtz energy function over the `DualNum` trait, a trait all dual number structs implement, and which provides operator overloading and a wide range of commonly used mathematical functions.

At the point of this writing, rust is still missing some features that enable us to write our implementation as if we were using floating point numbers.
For example, since *specialization* is not yet implemented, in some situations we have to change the order in which arithmetic operations are performed when working with arrays.
This is a minor inconvenience though and something you'll get quickly used to.

If you are interested in the inner workings of generalized dual numbers, check out these sources:
- [The `num-dual` crate](https://github.com/itt-ustutt/num-dual)
- [Paper of Fike and Alonso on hyper-dual numbers](http://adl.stanford.edu/hyperdual/Fike_AIAA-2011-886.pdf)
- [Jeffrey Fike's website about hyper-dual numbers](http://adl.stanford.edu/hyperdual/)

The `StateHD` Struct
~~~~~~~~~~~~~~~~~~~~

So, how can we compute a property for a given `State` using dual numbers?
Let's run through a small example.
Note thought, that the way properties are actually calculated is a bit more involved - we will talk about that in more detail below.

Let us assume we already created a `State` e.g. for a given temperature and partial density and now we want to compute the residual entropy at this condition.
The residual entropy is defined as

.. math:: S^\text{res} = -\left(\frac{\partial A^\text{res}}{\partial T}\right)_{\mathbf{N}, V} \.

We need the first derivative of the residual Helmholtz energy with respect to temperature.
The first derivative can be computed using a dual number consisting of a real part and *one non-real part* which is a `Dual64` struct.
Looking at the signature of the Helmholtz energy function, we see that it takes a `StateHD<D>` as input (not a `State`), where `D` is a generalized dual number (`D: DualNum<f64>`):

.. code::rust

/// The Helmholtz energy contribution $\beta A$ of a given state in reduced units.
fn helmholtz_energy(&self, state: &StateHD<D>) -> D;


**`StateHD`** contains the same state variables as the `State` object, but instead of dimensioned quantities, properties are stored as generalized dual numbers.
To compute the first derivative, we have to create a `StateHD<Dual64>`.

This can be done "by hand", placing the state variables of `State` as real parts of our dual numbers in `StateHD`.
This can conveniently be done using *reduced* variables.
Next, we have to modify the non-real part of the temperature dual number since we want to compute the partial derivative with respect to temperature which is done by setting the dual part of the temperature to unity.

```rust
// "state" is our `State` struct we already constructed.

// create an array of dual numbers from array of f64
let n = state.reduced_moles.mapv(Dual64::from);
let v = Dual64::from(state.reduced_volume);
// .derive() sets the dual part to unity
let t = Dual64::from(state.reduced_temperature).derive();
let state_hd = StateHD<Dual64>::new(t, v, n);
```

The result of the `helmholtz_energy` function using this `StateHD` is also a `Dual64` where the dual part contains the temperature derivative of the residual Helmholtz energy.

In our code, you won't construct `StateHD` objects "by hand" as we just did in our example.
There are methods that produce the correct dual number types and set the correct non-real parts according to the partial derivative you want to compute.

Helmholtz Energy Contributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The above example is a bit simplified and in our real code we have to do some additional legwork, mainly concerning the *contributions* to the Helmholtz energy that are computed.

We differentiate between the following contributions to a property:

- **Total**: ideal gas contribution plus contribution to the property from the residual Helmholtz energy,
- **Residual**: residual property, with respect to the ideal gas at given temperature and **volume**,
- **ResidualP**: residual property, with respect to the ideal gas at given temperature and **pressure**,
- **IdealGas**: ideal gas part of the property.

These variants are encoded in the `Contributions` enum.
Furthermore, computation of a property (for given contributions) is different for additive and non-additive properties, so we have to take this into account as well.

Entropy, again!
~~~~~~~~~~~~~~~

Let's look at the actual implementation of the entropy, the `entropy` method of `State`.
There are multiple layers we have to dig through, but hopefully everything makes sense to you in the end.
You can think of each layer as an answer to a question:

1. Which contributions should be computed and is the property additive?
    - Define how the property is computed.
2. What partial derivatives are needed?
    - Construct `StateHD` object with correct dual number types.
3. Was this property computed before?
    - Yes: retrieve value from the cache.
    - No: perform computation and update cache.
4. What's the proper unit?
    - Multiply the respective reference values.

The `entropy` method is defined in the `impl` block of `State` and looks like so:

```rust
pub fn entropy(&self, contributions: Contributions) -> QuantityScalar<U> {
    self.evaluate_property(Self::entropy_, contributions, true)
}
```

Most of the properties defined for a `State` will look similar to the above method:

- `Self::entropy_` defines the actual computation (we will look at this function below),
- `contributions` will inform if and how the ideal gas and residual contributions will be included, and
- `true` signals that entropy is an additive property.

`evaluate_property` then simply calls the function (here `entropy_`) and handles the ideal gas contribution:

```rust
fn evaluate_property<R, F>(&self, f: F, contributions: Contributions, additive: bool) -> R
where
    R: Sub<Output = R>,
    F: Fn(&Self, Evaluate) -> R,
{
    match contributions {
        Contributions::IdealGas => f(self, Evaluate::IdealGas),
        Contributions::Total => f(self, Evaluate::Total),
        Contributions::Residual => {
            if additive {
                f(self, Evaluate::Residual)
            } else {
                f(self, Evaluate::Total) - f(self, Evaluate::IdealGas)
            }
        }
        Contributions::ResidualP => {
            let p = self.pressure_(Evaluate::Total);
            let state_p = Self::new_nvt(
                &self.eos,
                self.temperature,
                self.total_moles * U::gas_constant() * self.temperature / p,
                &self.moles,
            )
            .unwrap();
            f(self, Evaluate::Total) - f(&state_p, Evaluate::IdealGas)
        }
    }
}
```

For example, the residual value of the additive property entropy can be computed by evaluating the partial derivative of the residual Helmholtz energy while a non-additive property (e.g. the heat capacity) is computed by taking the difference between the total property and the ideal gas property.

Let's look at the `entropy_` method:

```rust
fn entropy_(&self, evaluate: Evaluate) -> QuantityScalar<U> {
    -self.get_or_compute_derivative(PartialDerivative::First(DT), evaluate)
}
```

That might be a bit confusing: we used a `Contribution` in `entropy` and now we use an `Evaluate` enum in `entropy_`.
Why is that?
`Evaluate` is a subset of `Contribution` (containing `Total`, `IdealGas`, and `Residual`) and simply introduced to make `evaluate_property` and `get_or_compute_derivative` more readable.

The `get_or_compute_derivative` method does multiple things in the following order:

1. Given `Evaluate`, decide what Helmholtz energy contributions to compute: ideal gas, residual or the sum of both.
2. Given the `PartialDerivative`, the the correct `StateHD` object is built by choosing what dual number type to use and which dual parts to modify. In our example, it is a `StateHD<Dual64>` where the dual part of the temperature is unity.
3. Check if the derivative is already cached.
4. Either return the cached value or compute the derivative and add it to the cache.
5. Given the `PartialDerivative`, multiply the result with dimensioned reference quantities, e.g. for entropy, we multiply with the reference energy and divide by the reference temperature.

## How to Add a New Property

Now that we discussed how properties are computed, let's talk about the two possible ways to add new properties to a state.
The first and easiest way is to use a combination of existing properties.
As example, consider the Joule Thomson coefficient, defined as

.. math:: \mu_{JT}=\left(\frac{\partial T}{\partial p}\right)_{H,N_i} = -\frac{1}{C_p} \left(V + T \left(\frac{\partial p}{\partial T}\right) \bigg/ \left(\frac{\partial p}{\partial V}\right)\right) \,

which is implemented as

```rust
pub fn joule_thomson(&self) -> QuantityScalar<U> {
    let c = Contributions::Total;
    -(self.volume + self.temperature * self.dp_dt(c) / self.dp_dv(c))
        / (self.total_moles * self.c_p(c))
}
```

Here, we don't use `Contributions` as argument because it's nonsensical for this property, and since we already have functions for the terms needed within the calculation, we don't construct the partial derivatives by hand.
Note that all properties here are dimensioned quantities and so is the result.

In contrast to the Joule Thomson coefficient, properties such as partial derivatives of pressure are implemented similar to what we saw before for the entropy:

```rust
fn dp_dv_(&self, evaluate: Evaluate) -> QuantityScalar<U> {
    -self.get_or_compute_derivative(PartialDerivative::Second(DV, DV), evaluate)
}

fn dp_dt_(&self, evaluate: Evaluate) -> QuantityScalar<U> {
    -self.get_or_compute_derivative(PartialDerivative::Second(DV, DT), evaluate)
}
```

Array Valued Properties
~~~~~~~~~~~~~~~~~~~~~~~

Note that the return types for all given examples were scalar dimensioned values (`QuantityScalar`).
For partial derivatives w.r.t. the amount of substance of a species, you need an array as return type.
For example, take a look at the chemical potential:

```rust
pub fn chemical_potential(&self, contributions: Contributions) -> QuantityArray1<U> {
    self.evaluate_property(Self::chemical_potential_, contributions, true)
}

fn chemical_potential_(&self, evaluate: Evaluate) -> QuantityArray1<U> {
    QuantityArray::from_shape_fn(self.eos.components(), |i| {
        self.get_or_compute_derivative(PartialDerivative::First(DN(i)), evaluate)
    })
}
```

Here, `QuantityArray::from_shape:fn` will create an array given a shape (here one-dimension: the number of components of the equation of state, `self.eos.components()`) and a function that returns a scalar value.
This is not limited to one-dimensional arrays, of course, as can be seen in the implementation of the partial derivative of the chemical potential:

```rust
fn dmu_dni_(&self, evaluate: Evaluate) -> QuantityArray2<U> {
    let n = self.eos.components();
    QuantityArray::from_shape_fn((n, n), |(i, j)| {
        self.get_or_compute_derivative(
            PartialDerivative::Second(DN(i), DN(j)),
            evaluate
        )
    })
}
```

which returns a matrix (two-dimensional array of dimensioned values, `QuantityArray2`).

Summary
-------

- `State` stores state variables as dimensioned quantities.
- `StateHD` stores state variables as generalized dual numbers.
- Properties are computed by evaluating the Helmholtz energy using a `StateHD` as input.
- Properties can be computed including or excluding the ideal gas contributions, which is controlled using the `Contributions` enum.
- New properties are implemented either by calling `get_or_compute_derivative` or by combining existing properties.
