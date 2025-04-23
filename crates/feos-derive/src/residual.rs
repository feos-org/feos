use super::{implement, OPT_IMPLS};
use quote::quote;
use syn::DeriveInput;

pub(crate) fn expand_residual(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let variants = match input.data {
        syn::Data::Enum(syn::DataEnum { ref variants, .. }) => variants,
        _ => panic!("this derive macro only works on enums"),
    };

    let residual = impl_residual(&input.ident, variants);
    let molar_weight = impl_molar_weight(&input.ident, variants)?;
    let entropy_scaling = impl_entropy_scaling(&input.ident, variants)?;
    Ok(quote! {
        #residual
        #molar_weight
        #entropy_scaling
    })
}

fn impl_residual(
    ident: &syn::Ident,
    variants: &syn::punctuated::Punctuated<syn::Variant, syn::token::Comma>,
) -> proc_macro2::TokenStream {
    let compute_max_density = variants.iter().map(|v| {
        let name = &v.ident;
        quote! {
            Self::#name(residual) => residual.compute_max_density(moles)
        }
    });
    let residual_helmholtz_energy_contributions = variants.iter().map(|v| {
        let name = &v.ident;
        quote! {
            Self::#name(residual) => residual.residual_helmholtz_energy_contributions(state)
        }
    });

    quote! {
        impl Residual for #ident {
            fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
                match self {
                    #(#compute_max_density,)*
                }
            }
            fn residual_helmholtz_energy_contributions<D: DualNum<f64> + Copy + ScalarOperand>(&self, state: &StateHD<D>) -> Vec<(String, D)> {
                match self {
                    #(#residual_helmholtz_energy_contributions,)*
                }
            }
        }
    }
}

fn impl_molar_weight(
    ident: &syn::Ident,
    variants: &syn::punctuated::Punctuated<syn::Variant, syn::token::Comma>,
) -> syn::Result<proc_macro2::TokenStream> {
    let mut molar_weight = Vec::new();
    let mut has_molar_weight = Vec::new();

    for v in variants.iter() {
        let name = &v.ident;
        if implement("molar_weight", v, &OPT_IMPLS)? {
            molar_weight.push(quote! {
                Self::#name(eos) => eos.molar_weight()
            });
            has_molar_weight.push(quote! {
                Self::#name(_) => true
            });
        } else {
            molar_weight.push(quote! {
                Self::#name(eos) => panic!("{} does not provide molar weights and can not be used to calculate mass-specific properties", stringify!(#name))
            });
            has_molar_weight.push(quote! {
                Self::#name(_) => false
            });
        }
    }

    Ok(quote! {
        impl Molarweight for #ident {
            fn molar_weight(&self) -> MolarWeight<Array1<f64>> {
                match self {
                    #(#molar_weight,)*
                }
            }
        }

        impl #ident {
            pub fn has_molar_weight(&self) -> bool {
                match self {
                    #(#has_molar_weight,)*
                }
            }
        }
    })
}

fn impl_entropy_scaling(
    ident: &syn::Ident,
    variants: &syn::punctuated::Punctuated<syn::Variant, syn::token::Comma>,
) -> syn::Result<proc_macro2::TokenStream> {
    let mut etar = Vec::new();
    let mut etac = Vec::new();
    let mut dr = Vec::new();
    let mut dc = Vec::new();
    let mut thcr = Vec::new();
    let mut thcc = Vec::new();

    for v in variants.iter() {
        let name = &v.ident;
        if implement("entropy_scaling", v, &OPT_IMPLS)? {
            etar.push(quote! {
                Self::#name(eos) => eos.viscosity_reference(temperature, volume, moles)
            });
            etac.push(quote! {
                Self::#name(eos) => eos.viscosity_correlation(s_res, x)
            });
            dr.push(quote! {
                Self::#name(eos) => eos.diffusion_reference(temperature, volume, moles)
            });
            dc.push(quote! {
                Self::#name(eos) => eos.diffusion_correlation(s_res, x)
            });
            thcr.push(quote! {
                Self::#name(eos) => eos.thermal_conductivity_reference(temperature, volume, moles)
            });
            thcc.push(quote! {
                Self::#name(eos) => eos.thermal_conductivity_correlation(s_res, x)
            });
        } else {
            etar.push(quote! {
                Self::#name(eos) => panic!("{} does not implement entropy scaling for transport properties!", stringify!(#name))
            });
            etac.push(quote! {
                Self::#name(eos) => panic!("{} does not implement entropy scaling for transport properties!", stringify!(#name))
            });
            dr.push(quote! {
                Self::#name(eos) => panic!("{} does not implement entropy scaling for transport properties!", stringify!(#name))
            });
            dc.push(quote! {
                Self::#name(eos) => panic!("{} does not implement entropy scaling for transport properties!", stringify!(#name))
            });
            thcr.push(quote! {
                Self::#name(eos) => panic!("{} does not implement entropy scaling for transport properties!", stringify!(#name))
            });
            thcc.push(quote! {
                Self::#name(eos) => panic!("{} does not implement entropy scaling for transport properties!", stringify!(#name))
            });
        }
    }

    Ok(quote! {
        impl EntropyScaling for #ident {
            fn viscosity_reference(
                &self,
                temperature: Temperature,
                volume: Volume,
                moles: &Moles<Array1<f64>>,
            ) -> FeosResult<Viscosity> {
                match self {
                    #(#etar,)*
                }
            }

            fn viscosity_correlation(&self, s_res: f64, x: &Array1<f64>) -> FeosResult<f64> {
                match self {
                    #(#etac,)*
                }
            }

            fn diffusion_reference(
                &self,
                temperature: Temperature,
                volume: Volume,
                moles: &Moles<Array1<f64>>,
            ) -> FeosResult<Diffusivity> {
                match self {
                    #(#dr,)*
                }
            }

            fn diffusion_correlation(&self, s_res: f64, x: &Array1<f64>) -> FeosResult<f64> {
                match self {
                    #(#dc,)*
                }
            }

            fn thermal_conductivity_reference(
                &self,
                temperature: Temperature,
                volume: Volume,
                moles: &Moles<Array1<f64>>,
            ) -> FeosResult<ThermalConductivity> {
                match self {
                    #(#thcr,)*
                }
            }

            fn thermal_conductivity_correlation(&self, s_res: f64, x: &Array1<f64>) -> FeosResult<f64> {
                match self {
                    #(#thcc,)*
                }
            }
        }
    })
}
