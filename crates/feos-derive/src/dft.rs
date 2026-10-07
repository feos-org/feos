use crate::{OPT_IMPLS, implement};
use quote::quote;
use syn::DeriveInput;

pub(crate) fn expand_helmholtz_energy_functional(
    input: DeriveInput,
) -> syn::Result<proc_macro2::TokenStream> {
    let syn::Data::Enum(syn::DataEnum { ref variants, .. }) = input.data else {
        panic!("this derive macro only works on enums")
    };

    let functional = impl_helmholtz_energy_functional(&input.ident, variants)?;
    let fluid_parameters = impl_fluid_parameters(&input.ident, variants)?;
    let pair_potential = impl_pair_potential(&input.ident, variants)?;
    Ok(quote! {
        #functional
        #fluid_parameters
        #pair_potential
    })
}

pub(crate) fn impl_helmholtz_energy_functional(
    ident: &syn::Ident,
    variants: &syn::punctuated::Punctuated<syn::Variant, syn::token::Comma>,
) -> syn::Result<proc_macro2::TokenStream> {
    let mut molecule_shape = Vec::new();
    let mut contributions = Vec::new();
    let mut bond_lengths_homo = Vec::new();
    let mut bond_lengths_hetero = Vec::new();
    for v in variants.iter() {
        let name = &v.ident;
        if implement("functional", v, &OPT_IMPLS)? {
            molecule_shape.push(quote! {
                Self::#name(functional) => functional.molecule_shape()
            });
            contributions.push(quote! {
                Self::#name(functional) => functional.contributions().map(FunctionalContributionVariant::from).collect::<Vec<_>>().into_iter()
            });
            bond_lengths_homo.push(quote! {
                Self::#name(functional) => functional.bond_lengths_homo(temperature)
            });
            bond_lengths_hetero.push(quote! {
                Self::#name(functional) => functional.bond_lengths_hetero(temperature)
            });
        } else {
            let panic = quote! {
                Self::#name(functional) => panic!("{} is not a Helmholtz energy functional!", stringify!(#name))
            };
            molecule_shape.push(panic.clone());
            contributions.push(panic.clone());
            bond_lengths_homo.push(panic.clone());
            bond_lengths_hetero.push(panic);
        }
    }

    Ok(quote! {
        impl HelmholtzEnergyFunctionalDyn for #ident {
            type Contribution<'a> = FunctionalContributionVariant<'a>;
            fn molecule_shape(&self) -> feos_dft::MoleculeShape<'_> {
                match self {
                    #(#molecule_shape,)*
                }
            }
            fn contributions<'a>(&'a self) -> impl Iterator<Item = Self::Contribution<'a>> {
                match self {
                    #(#contributions,)*
                }
            }
            fn bond_lengths_homo<N: DualNum<Primitive = f64> + Copy>(&self, temperature: N) -> nalgebra::DVector<N> {
                match self {
                    #(#bond_lengths_homo,)*
                }
            }
            fn bond_lengths_hetero<N: DualNum<Primitive = f64> + Copy>(&self, temperature: N) -> petgraph::graph::UnGraph<(), N> {
                match self {
                    #(#bond_lengths_hetero,)*
                }
            }
        }
    })
}

fn impl_fluid_parameters(
    ident: &syn::Ident,
    variants: &syn::punctuated::Punctuated<syn::Variant, syn::token::Comma>,
) -> syn::Result<proc_macro2::TokenStream> {
    let mut epsilon_k_ff = Vec::new();
    let mut sigma_ff = Vec::new();

    for v in variants.iter() {
        let name = &v.ident;
        if implement("fluid_parameters", v, &OPT_IMPLS)? {
            epsilon_k_ff.push(quote! {
                Self::#name(functional) => functional.epsilon_k_ff()
            });
            sigma_ff.push(quote! {
                Self::#name(functional) => functional.sigma_ff()
            });
        } else {
            epsilon_k_ff.push(quote! {
                Self::#name(functional) => panic!("{} does not support the automatic calculation of external potentials!", stringify!(#name))
            });
            sigma_ff.push(quote! {
                Self::#name(functional) => panic!("{} does not support the automatic calculation of external potentials!", stringify!(#name))
            });
        }
    }
    Ok(quote! {
        impl feos_dft::adsorption::FluidParameters for #ident {
            fn epsilon_k_ff(&self) -> DVector<f64> {
                match self {
                    #(#epsilon_k_ff,)*
                }
            }

            fn sigma_ff(&self) -> DVector<f64> {
                match self {
                    #(#sigma_ff,)*
                }
            }
        }
    })
}

fn impl_pair_potential(
    ident: &syn::Ident,
    variants: &syn::punctuated::Punctuated<syn::Variant, syn::token::Comma>,
) -> syn::Result<proc_macro2::TokenStream> {
    let mut pair_potential = Vec::new();

    for v in variants.iter() {
        let name = &v.ident;
        if implement("pair_potential", v, &OPT_IMPLS)? {
            pair_potential.push(quote! {
                Self::#name(functional) => functional.pair_potential(i, r, temperature)
            });
        } else {
            pair_potential.push(quote! {
                Self::#name(functional) => panic!("{} does not provide pair potentials!", stringify!(#name))
            });
        }
    }
    Ok(quote! {
        impl feos_dft::solvation::PairPotential for #ident {
            fn pair_potential(&self, i: usize, r: &Array1<f64>, temperature: f64) -> quantity::Energy<ndarray::Array2<f64>> {
                match self {
                    #(#pair_potential,)*
                }
            }
        }
    })
}
