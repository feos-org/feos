use quote::quote;
use syn::DeriveInput;

pub(crate) fn expand_functional_contribution(input: DeriveInput) -> proc_macro2::TokenStream {
    let max_size = input
        .attrs
        .iter()
        .find(|a| a.path.segments.len() == 1 && a.path.segments[0].ident == "max_size")
        .expect("max_size attribute required for deriving FunctionalContribution!");
    let max_size: proc_macro2::Literal = max_size
        .parse_args()
        .expect("max_size has to be an integer literal!");
    let max_size = max_size
        .to_string()
        .parse()
        .expect("max_size has to be an integer literal!");
    impl_functional_derivative(&input, max_size)
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
                    #(#components => <Self as PartialDerivativesDual<#components>>::first_partial_derivatives_n(
                        &self,
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
                    #(#components => <Self as PartialDerivativesDual<#components>>::second_partial_derivatives_n(
                        &self,
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
