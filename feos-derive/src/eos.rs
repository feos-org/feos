use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

// const POSSIBLE_SKIP_STATEMENTS: [&'static str; 2] = ["molar_weight", "entropy_scaling"];
// Todo: Validate possible arguments


pub fn derive_equation_of_state(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);

    let variants = if let syn::Data::Enum(syn::DataEnum { ref variants, .. }) = ast.data {
        variants
    } else {
        // todo: proper error message
        unimplemented!()
    };

    let eos = impl_equation_of_state(variants);
    let molar_weight = impl_molar_weight(variants);
    quote! {
        #eos
        #molar_weight
    }
    .into()
}

fn skip_impl(name: &str, variant: &syn::Variant) -> bool {
    let syn::Variant { attrs, .. } = variant;
    for attr in attrs.iter() {
        if attr.path.is_ident("skip_impl") {
            if let Ok(syn::Meta::List(list)) = attr.parse_meta() {
                for meta in list.nested {
                    if let syn::NestedMeta::Meta(syn::Meta::Path(path)) = meta {
                        if path.is_ident(name) {
                            return true;
                        }
                        // else {
                        //     if !POSSIBLE_SKIP_STATEMENTS.iter().any(|s| path.is_ident(s)) {
                        //         return (
                        //             false,
                        //             Some(
                        //                 syn::Error::new_spanned(
                        //                     path,
                        //                     "expected `skip_impl(molar_weight, entropy_scaling)`",
                        //                 )
                        //                 .to_compile_error(),
                        //             ),
                        //         );
                        //     }
                        // }
                    }
                }
            } else {
                panic!("expected 'molar_weight' or 'entropy_scaling'")
            }
        }
    }
    false
    // if let Ok(syn::Meta::List(list)) = attr.parse_meta() {
    //     // last path segment is "skip_impl" - then read nested
    //     if let Some(segment) = &list.path.segments.last() {
    //         if segment.ident == "skip_impl" {
    //             // check list of segments to contain "name"
    //             for meta in list.nested {
    //                 if let NestedMeta::Meta(syn::Meta::Path(syn::Path { segments, .. })) = meta
    //                 {
    //                     for segment in segments.iter() {
    //                         if segment.ident == name {
    //                             return true;
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     };
    // }
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
) -> proc_macro2::TokenStream {
    let molar_weight = variants
        .iter()
        .filter(|v| !skip_impl("molar_weight", v))
        .map(|v| {
            let name = &v.ident;
            quote! {
                Self::#name(eos) => eos.molar_weight()
            }
        });
    quote! {
        impl MolarWeight<SIUnit> for EosVariant {
            fn molar_weight(&self) -> SIArray1 {
                match self {
                    #(#molar_weight,)*
                    _ => unimplemented!()
                }
            }
        }
    }
}
