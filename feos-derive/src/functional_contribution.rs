use quote::quote;
use syn::DeriveInput;

pub(crate) fn expand_functional_contribution(input: DeriveInput) -> proc_macro2::TokenStream {
    let max_size = input
        .attrs
        .iter()
        .find(|a| a.path.segments.len() == 1 && a.path.segments[0].ident == "max_size")
        .expect("max_size attribute required for deriving MyTrait!");
    let max_size: proc_macro2::Literal = max_size
        .parse_args()
        .expect("max_size has to be an integer literal!");
    let max_size = max_size
        .to_string()
        .parse()
        .expect("max_size has to be an integer literal!");
    let functional_derivative = impl_functional_derivative(&input, max_size);
    let functional_derivative_helpers = impl_functional_derivative_helpers(&input);
    quote! {
        #functional_derivative
        #functional_derivative_helpers
    }
}

fn impl_functional_derivative(input: &DeriveInput, max_size: usize) -> proc_macro2::TokenStream {
    let name = &input.ident;
    let generics = &input.generics;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    let components: Vec<_> = (1..=max_size).collect();
    quote! {
        impl #impl_generics FunctionalContribution for #name #ty_generics #where_clause {
            fn first_partial_derivatives(
                &self,
                temperature: f64,
                weighted_densities: Array2<f64>,
                helmholtz_energy_density: ArrayViewMut1<f64>,
                first_partial_derivative: ArrayViewMut2<f64>,
            ) -> EosResult<()> {
                match weighted_densities.shape()[0] {
                    #(#components => self.first_partial_derivatives_n::<#components>(
                        temperature,
                        weighted_densities,
                        helmholtz_energy_density,
                        first_partial_derivative,
                    ),)*
                    n => self.first_partial_derivatives_dyn(
                        temperature,
                        weighted_densities,
                        helmholtz_energy_density,
                        first_partial_derivative,
                    ),
                }
            }

            fn second_partial_derivatives(
                &self,
                temperature: f64,
                weighted_densities: Array2<f64>,
                helmholtz_energy_density: ArrayViewMut1<f64>,
                first_partial_derivative: ArrayViewMut2<f64>,
                second_partial_derivative: ArrayViewMut3<f64>,
            ) -> EosResult<()> {
                match weighted_densities.shape()[0] {
                    #(#components => self.second_partial_derivatives_n::<#components>(
                        temperature,
                        weighted_densities,
                        helmholtz_energy_density,
                        first_partial_derivative,
                        second_partial_derivative,
                    ),)*
                    n => self.second_partial_derivatives_dyn(
                        temperature,
                        weighted_densities,
                        helmholtz_energy_density,
                        first_partial_derivative,
                        second_partial_derivative,
                    ),
                }
            }
        }
    }
}

fn impl_functional_derivative_helpers(input: &DeriveInput) -> proc_macro2::TokenStream {
    let name = &input.ident;
    let generics = &input.generics;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    quote! {
        impl #impl_generics #name #ty_generics #where_clause {
            fn first_partial_derivatives_n<const N: usize>(
                &self,
                temperature: f64,
                weighted_densities: Array2<f64>,
                mut helmholtz_energy_density: ArrayViewMut1<f64>,
                mut first_partial_derivative: ArrayViewMut2<f64>,
            ) -> EosResult<()> {
                let t = DualVec64::<N>::from(temperature);
                let mut wd = weighted_densities.mapv(DualVec64::<N>::from);
                for i in 0..N {
                    wd.index_axis_mut(Axis(0), i)
                        .map_inplace(|x| x.eps[i] = 1.0);
                }
                let phi = self.calculate_helmholtz_energy_density(t, wd.view())?;
                helmholtz_energy_density.assign(&phi.mapv(|p| p.re));
                for i in 0..N {
                    first_partial_derivative
                        .index_axis_mut(Axis(0), i)
                        .assign(&phi.mapv(|p| p.eps[i]));
                }

                Ok(())
            }

            fn second_partial_derivatives_n<const N: usize>(
                &self,
                temperature: f64,
                weighted_densities: Array2<f64>,
                mut helmholtz_energy_density: ArrayViewMut1<f64>,
                mut first_partial_derivative: ArrayViewMut2<f64>,
                mut second_partial_derivative: ArrayViewMut3<f64>,
            ) -> EosResult<()> {
                let t = Dual2Vec64::<N>::from(temperature);
                let mut wd = weighted_densities.mapv(Dual2Vec64::<N>::from);
                for i in 0..N {
                    wd.index_axis_mut(Axis(0), i).map_inplace(|x| x.v1[i] = 1.0);
                }
                let phi = self.calculate_helmholtz_energy_density(t, wd.view())?;
                helmholtz_energy_density.assign(&phi.mapv(|p| p.re));
                for i in 0..N {
                    first_partial_derivative
                        .index_axis_mut(Axis(0), i)
                        .assign(&phi.mapv(|p| p.v1[i]));
                    for j in 0..N {
                        second_partial_derivative
                            .index_axis_mut(Axis(0), i)
                            .index_axis_mut(Axis(0), j)
                            .assign(&phi.mapv(|p| p.v2[(i, j)]));
                    }
                }

                Ok(())
            }

            fn first_partial_derivatives_dyn(
                &self,
                temperature: f64,
                weighted_densities: Array2<f64>,
                mut helmholtz_energy_density: ArrayViewMut1<f64>,
                mut first_partial_derivative: ArrayViewMut2<f64>,
            ) -> EosResult<()> {
                let mut wd = weighted_densities.mapv(Dual64::from);
                let t = Dual64::from(temperature);
                let mut phi = Array::zeros(weighted_densities.raw_dim().remove_axis(Axis(0)));

                for i in 0..wd.shape()[0] {
                    wd.index_axis_mut(Axis(0), i)
                        .map_inplace(|x| x.eps[0] = 1.0);
                    phi = self.calculate_helmholtz_energy_density(t, wd.view())?;
                    first_partial_derivative
                        .index_axis_mut(Axis(0), i)
                        .assign(&phi.mapv(|p| p.eps[0]));
                    wd.index_axis_mut(Axis(0), i)
                        .map_inplace(|x| x.eps[0] = 0.0);
                }
                helmholtz_energy_density.assign(&phi.mapv(|p| p.re));
                Ok(())
            }

            fn second_partial_derivatives_dyn(
                &self,
                temperature: f64,
                weighted_densities: Array2<f64>,
                mut helmholtz_energy_density: ArrayViewMut1<f64>,
                mut first_partial_derivative: ArrayViewMut2<f64>,
                mut second_partial_derivative: ArrayViewMut3<f64>,
            ) -> EosResult<()> {
                let mut wd = weighted_densities.mapv(HyperDual64::from);
                let t = HyperDual64::from(temperature);
                let mut phi = Array::zeros(weighted_densities.raw_dim().remove_axis(Axis(0)));

                for i in 0..wd.shape()[0] {
                    wd.index_axis_mut(Axis(0), i)
                        .map_inplace(|x| x.eps1[0] = 1.0);
                    for j in 0..=i {
                        wd.index_axis_mut(Axis(0), j)
                            .map_inplace(|x| x.eps2[0] = 1.0);
                        phi = self.calculate_helmholtz_energy_density(t, wd.view())?;
                        let p = phi.mapv(|p| p.eps1eps2[(0, 0)]);
                        second_partial_derivative
                            .index_axis_mut(Axis(0), i)
                            .index_axis_mut(Axis(0), j)
                            .assign(&p);
                        if i != j {
                            second_partial_derivative
                                .index_axis_mut(Axis(0), j)
                                .index_axis_mut(Axis(0), i)
                                .assign(&p);
                        }
                        wd.index_axis_mut(Axis(0), j)
                            .map_inplace(|x| x.eps2[0] = 0.0);
                    }
                    first_partial_derivative
                        .index_axis_mut(Axis(0), i)
                        .assign(&phi.mapv(|p| p.eps1[0]));
                    wd.index_axis_mut(Axis(0), i)
                        .map_inplace(|x| x.eps1[0] = 0.0);
                }
                helmholtz_energy_density.assign(&phi.mapv(|p| p.re));
                Ok(())
            }
        }
    }
}
