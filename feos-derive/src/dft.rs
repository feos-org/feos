use quote::quote;
use syn::DeriveInput;

use crate::implement;

const OPT_IMPLS: [&str; 4] = [
    "bond_lengths",
    "molar_weight",
    "fluid_parameters",
    "pair_potential",
];

pub(crate) fn expand_helmholtz_energy_functional(
    input: DeriveInput,
) -> syn::Result<proc_macro2::TokenStream> {
    let variants = match input.data {
        syn::Data::Enum(syn::DataEnum { ref variants, .. }) => variants,
        _ => panic!("this derive macro only works on enums"),
    };

    let from = impl_from(variants)?;
    let functional = impl_helmholtz_energy_functional(variants)?;
    let fluid_parameters = impl_fluid_parameters(variants)?;
    let pair_potential = impl_pair_potential(variants)?;
    Ok(quote! {
        #from
        #functional
        #fluid_parameters
        #pair_potential
    })
}

// extract the variant name and the name of the functional,
// i.e. PcSaft(PcSaftFunctional) will return (PcSaft, PcSaftFunctional)
fn extract_names(variant: &syn::Variant) -> syn::Result<(&syn::Ident, &syn::Ident)> {
    let name = &variant.ident;
    let field = if let syn::Fields::Unnamed(syn::FieldsUnnamed { ref unnamed, .. }) = variant.fields
    {
        if unnamed.len() != 1 {
            return Err(syn::Error::new_spanned(
                unnamed,
                "expected tuple struct with single HelmholtzFunctional as variant",
            ));
        }
        &unnamed[0]
    } else {
        return Err(syn::Error::new_spanned(
            name,
            "expected variant with a HelmholtzFunctional as data",
        ));
    };

    let inner = if let syn::Type::Path(syn::TypePath { ref path, .. }) = &field.ty {
        path.get_ident()
    } else {
        None
    }
    .ok_or_else(|| syn::Error::new_spanned(field, "expected HelmholtzFunctional"))?;
    Ok((name, inner))
}

fn impl_from(
    variants: &syn::punctuated::Punctuated<syn::Variant, syn::token::Comma>,
) -> syn::Result<proc_macro2::TokenStream> {
    variants
        .iter()
        .map(|v| {
            let (variant_name, functional_name) = extract_names(v)?;
            Ok(quote! {
                impl From<#functional_name> for FunctionalVariant {
                    fn from(f: #functional_name) -> Self {
                        Self::#variant_name(f)
                    }
                }
            })
        })
        .collect()
}

fn impl_helmholtz_energy_functional(
    variants: &syn::punctuated::Punctuated<syn::Variant, syn::token::Comma>,
) -> syn::Result<proc_macro2::TokenStream> {
    let molecule_shape = variants.iter().map(|v| {
        let name = &v.ident;
        quote! {
            Self::#name(functional) => functional.molecule_shape()
        }
    });
    let compute_max_density = variants.iter().map(|v| {
        let name = &v.ident;
        quote! {
            Self::#name(functional) => functional.compute_max_density(moles)
        }
    });
    let contributions = variants.iter().map(|v| {
        let name = &v.ident;
        quote! {
            Self::#name(functional) => functional.contributions()
        }
    });

    let mut molar_weight = Vec::new();
    for v in variants.iter() {
        if implement("molar_weight", v, &OPT_IMPLS)? {
            let name = &v.ident;
            molar_weight.push(quote! {
                Self::#name(functional) => functional.molar_weight()
            });
        }
    }

    let mut bond_lengths = Vec::new();
    for v in variants.iter() {
        if implement("bond_lengths", v, &OPT_IMPLS)? {
            let name = &v.ident;
            bond_lengths.push(quote! {
                Self::#name(functional) => functional.bond_lengths(temperature)
            });
        }
    }

    Ok(quote! {
        impl HelmholtzEnergyFunctional for FunctionalVariant {
            fn molecule_shape(&self) -> MoleculeShape {
                match self {
                    #(#molecule_shape,)*
                }
            }
            fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
                match self {
                    #(#compute_max_density,)*
                }
            }
            fn contributions(&self) -> &[Box<dyn FunctionalContribution>] {
                match self {
                    #(#contributions,)*
                }
            }
            fn molar_weight(&self) -> MolarWeight<Array1<f64>> {
                match self {
                    #(#molar_weight,)*
                    _ => unimplemented!()
                }
            }
            fn bond_lengths(&self, temperature: f64) -> UnGraph<(), f64> {
                match self {
                    #(#bond_lengths,)*
                    _ => Graph::with_capacity(0, 0),
                }
            }
        }
    })
}

fn impl_fluid_parameters(
    variants: &syn::punctuated::Punctuated<syn::Variant, syn::token::Comma>,
) -> syn::Result<proc_macro2::TokenStream> {
    let mut epsilon_k_ff = Vec::new();
    let mut sigma_ff = Vec::new();

    for v in variants.iter() {
        if implement("fluid_parameters", v, &OPT_IMPLS)? {
            let name = &v.ident;
            epsilon_k_ff.push(quote! {
                Self::#name(functional) => functional.epsilon_k_ff()
            });
            sigma_ff.push(quote! {
                Self::#name(functional) => functional.sigma_ff()
            });
        }
    }
    Ok(quote! {
        impl FluidParameters for FunctionalVariant {
            fn epsilon_k_ff(&self) -> Array1<f64> {
                match self {
                    #(#epsilon_k_ff,)*
                    _ => unimplemented!()
                }
            }

            fn sigma_ff(&self) -> &Array1<f64> {
                match self {
                    #(#sigma_ff,)*
                    _ => unimplemented!()
                }
            }
        }
    })
}

fn impl_pair_potential(
    variants: &syn::punctuated::Punctuated<syn::Variant, syn::token::Comma>,
) -> syn::Result<proc_macro2::TokenStream> {
    let mut pair_potential = Vec::new();

    for v in variants.iter() {
        if implement("pair_potential", v, &OPT_IMPLS)? {
            let name = &v.ident;
            pair_potential.push(quote! {
                Self::#name(functional) => functional.pair_potential(i, r, temperature)
            });
        }
    }
    Ok(quote! {
        impl PairPotential for FunctionalVariant {
            fn pair_potential(&self, i: usize, r: &Array1<f64>, temperature: f64) -> Array2<f64> {
                match self {
                    #(#pair_potential,)*
                    _ => unimplemented!()
                }
            }
        }
    })
}
