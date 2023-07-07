use quote::quote;
use syn::DeriveInput;

pub(crate) fn expand_components(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let variants = match input.data {
        syn::Data::Enum(syn::DataEnum { ref variants, .. }) => variants,
        _ => panic!("this derive macro only works on enums"),
    };

    let components = impl_components(input.ident, variants);
    Ok(quote! {
        #components
    })
}

fn impl_components(
    ident: syn::Ident,
    variants: &syn::punctuated::Punctuated<syn::Variant, syn::token::Comma>,
) -> proc_macro2::TokenStream {
    let components = variants.iter().map(|v| {
        let name = &v.ident;
        if name == "NoModel" {
            quote! {
                Self::#name(n) => *n
            }
        } else {
            quote! {
                Self::#name(residual) => residual.components()
            }
        }
    });
    let subset = variants.iter().map(|v| {
        let name = &v.ident;
        if name == "NoModel" {
            quote! {
                Self::#name(n) => Self::#name(component_list.len())
            }
        } else {
            quote! {
                Self::#name(residual) => Self::#name(residual.subset(component_list))
            }
        }
    });

    quote! {
        impl Components for #ident {
            fn components(&self) -> usize {
                match self {
                    #(#components,)*
                }
            }
            fn subset(&self, component_list: &[usize]) -> Self {
                match self {
                    #(#subset,)*
                }
            }
        }
    }
}
