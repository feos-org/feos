//! This crate provides derive macros used for the EosVariant and
//! FunctionalVariant enums in FeOs. The macros implement
//! the boilerplate for the EquationOfState and HelmholtzEnergyFunctional traits.
use components::expand_components;
use dft::expand_helmholtz_energy_functional;
use ideal_gas::expand_ideal_gas;
use proc_macro::TokenStream;
use residual::expand_residual;
use syn::{parse_macro_input, DeriveInput};

mod components;
mod dft;
mod ideal_gas;
mod residual;

fn implement(name: &str, variant: &syn::Variant, opts: &[&'static str]) -> syn::Result<bool> {
    let syn::Variant { attrs, .. } = variant;
    let mut implement = Ok(false);
    for attr in attrs.iter() {
        if attr.path.is_ident("implement") {
            if let Ok(syn::Meta::List(list)) = attr.parse_meta() {
                for meta in list.nested {
                    if let syn::NestedMeta::Meta(syn::Meta::Path(path)) = meta {
                        // check if all keywords are valid, return error if not
                        if !opts.iter().any(|s| path.is_ident(s)) {
                            let opts = opts.join(", ");
                            return Err(syn::Error::new_spanned(
                                path,
                                format!("expected one of: {}", opts),
                            ));
                        }

                        // "name" is present
                        if path.is_ident(name) {
                            implement = Ok(true)
                        }
                    }
                }
            } else {
                return Err(syn::Error::new_spanned(
                    &attr.tokens,
                    "expected 'implement(optional_trait, ...)'",
                ));
            }
        }
    }
    implement
}

#[proc_macro_derive(Components)]
pub fn derive_components(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand_components(input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

#[proc_macro_derive(IdealGas)]
pub fn derive_ideal_gas(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand_ideal_gas(input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

#[proc_macro_derive(Residual, attributes(implement))]
pub fn derive_residual(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand_residual(input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

#[proc_macro_derive(HelmholtzEnergyFunctional, attributes(implement))]
pub fn derive_helmholtz_energy_functional(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand_helmholtz_energy_functional(input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
