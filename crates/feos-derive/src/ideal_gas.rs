use quote::quote;
use syn::DeriveInput;

pub(crate) fn expand_ideal_gas(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let variants = match input.data {
        syn::Data::Enum(syn::DataEnum { ref variants, .. }) => variants,
        _ => panic!("this derive macro only works on enums"),
    };

    let ideal_gas = impl_ideal_gas(variants);
    Ok(quote! {
        #ideal_gas
    })
}

fn impl_ideal_gas(
    variants: &syn::punctuated::Punctuated<syn::Variant, syn::token::Comma>,
) -> proc_macro2::TokenStream {
    let ln_lambda3 = variants.iter().map(|v| {
        let name = &v.ident;
        if name == "NoModel" {
            quote! {
                Self::#name => panic!("No ideal gas model initialized!")
            }
        } else {
            quote! {
                Self::#name(ideal_gas) => ideal_gas.ln_lambda3(temperature)
            }
        }
    });
    let string = variants.iter().map(|v| {
        let name = &v.ident;
        if name == "NoModel" {
            quote! {
                Self::#name => panic!("No ideal gas model initialized!")
            }
        } else {
            quote! {
                Self::#name(ideal_gas) => ideal_gas.ideal_gas_model()
            }
        }
    });
    quote! {
        impl IdealGas for IdealGasModel {
            fn ln_lambda3<D: DualNum<f64> + Copy>(&self, temperature: D) -> D {
                match self {
                    #(#ln_lambda3,)*
                }
            }

            fn ideal_gas_model(&self) -> &'static str {
                match self {
                    #(#string,)*
                }
            }
        }
    }
}
