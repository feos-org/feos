use quote::quote;
use syn::{DeriveInput, Ident};

pub(crate) fn expand_functional_contribution(
    input: DeriveInput,
) -> syn::Result<proc_macro2::TokenStream> {
    let ident = input.ident;
    let variants = match input.data {
        syn::Data::Enum(syn::DataEnum { ref variants, .. }) => variants,
        _ => panic!("this derive macro only works on enums"),
    };

    let functional_contribution = impl_functional_contribution(&ident, variants);
    let display = impl_display(&ident, variants);
    let from = impl_from(&ident, variants);
    Ok(quote! {
        #functional_contribution
        #display
        #from
    })
}

fn impl_functional_contribution(
    ident: &Ident,
    variants: &syn::punctuated::Punctuated<syn::Variant, syn::token::Comma>,
) -> proc_macro2::TokenStream {
    let weight_functions = variants.iter().map(|v| {
        let name = &v.ident;
        quote! {
            Self::#name(functional_contribution) => functional_contribution.weight_functions(temperature)
        }
    });
    let weight_functions_pdgt = variants.iter().map(|v| {
        let name = &v.ident;
        quote! {
            Self::#name(functional_contribution) => functional_contribution.weight_functions_pdgt(temperature)
        }
    });
    let helmholtz_energy_density = variants.iter().map(|v| {
        let name = &v.ident;
        quote! {
            Self::#name(functional_contribution) => functional_contribution.helmholtz_energy_density(temperature, weighted_densities)
        }
    });

    quote! {
        impl FunctionalContribution for #ident {
            fn weight_functions<N: DualNum<f64> + Copy+ScalarOperand>(&self, temperature: N) -> feos_dft::WeightFunctionInfo<N> {
                match self {
                    #(#weight_functions,)*
                }
            }
            fn weight_functions_pdgt<N: DualNum<f64> + Copy+ScalarOperand>(&self, temperature: N) -> feos_dft::WeightFunctionInfo<N> {
                match self {
                    #(#weight_functions_pdgt,)*
                }
            }
            fn helmholtz_energy_density<N: DualNum<f64> + Copy+ScalarOperand>(
                &self,
                temperature: N,
                weighted_densities: ndarray::ArrayView2<N>,
            ) -> EosResult<Array1<N>> {
                match self {
                    #(#helmholtz_energy_density,)*
                }
            }
        }
    }
}

fn impl_display(
    ident: &Ident,
    variants: &syn::punctuated::Punctuated<syn::Variant, syn::token::Comma>,
) -> proc_macro2::TokenStream {
    let fmt = variants.iter().map(|v| {
        let name = &v.ident;
        quote! {
            Self::#name(functional_contribution) => functional_contribution.fmt(f)
        }
    });

    quote! {
        impl std::fmt::Display for #ident {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    #(#fmt,)*
                }
            }
        }
    }
}

fn impl_from(
    ident: &Ident,
    variants: &syn::punctuated::Punctuated<syn::Variant, syn::token::Comma>,
) -> proc_macro2::TokenStream {
    let from = variants.iter().map(|v| {
        let name = &v.ident;
        let syn::Fields::Unnamed(syn::FieldsUnnamed { unnamed, .. }) = &v.fields else {
            panic!("All variants must be tuple structs!")
        };
        let inner = &unnamed.first().unwrap().ty;
        quote! {
            impl From<#inner> for #ident {
                fn from(variant: #inner) -> Self {
                    Self::#name(variant)
                }
            }
        }
    });

    quote! {
        #(#from)*
    }
}
