use super::implement;
use quote::quote;
use syn::DeriveInput;

// possible additional traits to implement
const OPT_IMPLS: [&'static str; 2] = ["molar_weight", "entropy_scaling"];

pub(crate) fn expand_equation_of_state(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let variants = match input.data {
        syn::Data::Enum(syn::DataEnum { ref variants, .. }) => variants,
        _ => panic!("this derive macro only works on enums"),
    };

    let eos = impl_equation_of_state(variants);
    let molar_weight = impl_molar_weight(variants)?;
    let entropy_scaling = impl_entropy_scaling(variants)?;
    Ok(quote! {
        #eos
        #molar_weight
        #entropy_scaling
    })
}

fn impl_equation_of_state(
    variants: &syn::punctuated::Punctuated<syn::Variant, syn::token::Comma>,
) -> proc_macro2::TokenStream {
    let components = variants.iter().map(|v| {
        let name = &v.ident;
        quote! {
            Self::#name(eos) => eos.components()
        }
    });
    let compute_max_density = variants.iter().map(|v| {
        let name = &v.ident;
        quote! {
            Self::#name(eos) => eos.compute_max_density(moles)
        }
    });
    let subset = variants.iter().map(|v| {
        let name = &v.ident;
        quote! {
            Self::#name(eos) => Self::#name(eos.subset(component_list))
        }
    });
    let residual = variants.iter().map(|v| {
        let name = &v.ident;
        quote! {
            Self::#name(eos) => eos.residual()
        }
    });
    let ideal_gas = variants.iter().map(|v| {
        let name = &v.ident;
        quote! {
            Self::#name(eos) => eos.ideal_gas()
        }
    });

    quote! {
        impl EquationOfState for EosVariant {
            fn components(&self) -> usize {
                match self {
                    #(#components,)*
                }
            }
            fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
                match self {
                    #(#compute_max_density,)*
                }
            }
            fn subset(&self, component_list: &[usize]) -> Self {
                match self {
                    #(#subset,)*
                }
            }
            fn residual(&self) -> &[Box<dyn HelmholtzEnergy>] {
                match self {
                    #(#residual,)*
                }
            }
            fn ideal_gas(&self) -> &dyn IdealGasContribution {
                match self {
                    #(#ideal_gas,)*
                }
            }
        }
    }
}

fn impl_molar_weight(
    variants: &syn::punctuated::Punctuated<syn::Variant, syn::token::Comma>,
) -> syn::Result<proc_macro2::TokenStream> {
    let mut molar_weight = Vec::new();

    for v in variants.iter() {
        if implement("molar_weight", v, &OPT_IMPLS)? {
            let name = &v.ident;
            molar_weight.push(quote! {
                Self::#name(eos) => eos.molar_weight()
            });
        }
    }
    Ok(quote! {
        impl MolarWeight<SIUnit> for EosVariant {
            fn molar_weight(&self) -> SIArray1 {
                match self {
                    #(#molar_weight,)*
                    _ => unimplemented!()
                }
            }
        }
    })
}

fn impl_entropy_scaling(
    variants: &syn::punctuated::Punctuated<syn::Variant, syn::token::Comma>,
) -> syn::Result<proc_macro2::TokenStream> {
    let mut etar = Vec::new();
    let mut etac = Vec::new();
    let mut dr = Vec::new();
    let mut dc = Vec::new();
    let mut thcr = Vec::new();
    let mut thcc = Vec::new();

    for v in variants.iter() {
        if implement("entropy_scaling", v, &OPT_IMPLS)? {
            let name = &v.ident;
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
        }
    }

    Ok(quote! {
        impl EntropyScaling<SIUnit> for EosVariant {
            fn viscosity_reference(
                &self,
                temperature: SINumber,
                volume: SINumber,
                moles: &SIArray1,
            ) -> EosResult<SINumber> {
                match self {
                    #(#etar,)*
                    _ => unimplemented!(),
                }
            }

            fn viscosity_correlation(&self, s_res: f64, x: &Array1<f64>) -> EosResult<f64> {
                match self {
                    #(#etac,)*
                    _ => unimplemented!(),
                }
            }

            fn diffusion_reference(
                &self,
                temperature: SINumber,
                volume: SINumber,
                moles: &SIArray1,
            ) -> EosResult<SINumber> {
                match self {
                    #(#dr,)*
                    _ => unimplemented!(),
                }
            }

            fn diffusion_correlation(&self, s_res: f64, x: &Array1<f64>) -> EosResult<f64> {
                match self {
                    #(#dc,)*
                    _ => unimplemented!(),
                }
            }

            fn thermal_conductivity_reference(
                &self,
                temperature: SINumber,
                volume: SINumber,
                moles: &SIArray1,
            ) -> EosResult<SINumber> {
                match self {
                    #(#thcr,)*
                    _ => unimplemented!(),
                }
            }

            fn thermal_conductivity_correlation(&self, s_res: f64, x: &Array1<f64>) -> EosResult<f64> {
                match self {
                    #(#thcc,)*
                    _ => unimplemented!(),
                }
            }
        }
    })
}
