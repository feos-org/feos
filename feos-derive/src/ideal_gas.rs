use quote::quote;
use syn::DeriveInput;

pub(crate) fn expand_ideal_gas(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let variants = match input.data {
        syn::Data::Enum(syn::DataEnum { ref variants, .. }) => variants,
        _ => panic!("this derive macro only works on enums"),
    };

    let ideal_gas = impl_ideal_gas(variants);
    let display = impl_display(variants);
    Ok(quote! {
        #ideal_gas
        #display
    })
}

fn impl_ideal_gas(
    variants: &syn::punctuated::Punctuated<syn::Variant, syn::token::Comma>,
) -> proc_macro2::TokenStream {
    let ln_lambda3 = variants.iter().map(|v| {
        let name = &v.ident;
        if name == "NoModel" {
            quote! {
                Self::#name(_) => panic!("No ideal gas model initialized!")
            }
        } else {
            quote! {
                Self::#name(ideal_gas) => ideal_gas.ln_lambda3(temperature)
            }
        }
    });
    quote! {
        impl IdealGas for IdealGasModel {
            fn ln_lambda3<D: DualNum<f64> + Copy>(&self, temperature: D) -> Array1<D> {
                match self {
                    #(#ln_lambda3,)*
                }
            }
        }
    }
}

fn impl_display(
    variants: &syn::punctuated::Punctuated<syn::Variant, syn::token::Comma>,
) -> proc_macro2::TokenStream {
    let string = variants.iter().map(|v| {
        let name = &v.ident;
        quote! {
            Self::#name(ideal_gas) => ideal_gas.to_string()
        }
    });
    quote! {
        impl fmt::Display for IdealGasModel {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let s = match self {
                    #(#string,)*
                };
                write!(f, "{}", s)
            }
        }
    }
}
