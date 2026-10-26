use quote::quote;
use syn::{Ident, Type, Visibility};

/// Enum `AssociationFunctionalVariant` of the association functionals of the given variants
/// (name and type of the Helmholtz energy functional) that dispatches the methods of
/// `AssociationFunctional`, or `NoAssociation` if there are no such variants.
pub(crate) fn impl_association_variant(
    vis: &Visibility,
    variants: &[(&Ident, &Type)],
) -> proc_macro2::TokenStream {
    if variants.is_empty() {
        return quote! {
            #vis type AssociationFunctionalVariant<'a> = feos_dft::NoAssociation;
        };
    }

    let enum_variants = variants.iter().map(|(name, ty)| {
        quote! {
            #name(<#ty as HelmholtzEnergyFunctionalDyn>::Association<'a>)
        }
    });
    let arms = |method: proc_macro2::TokenStream| {
        variants
            .iter()
            .map(|(name, _)| {
                quote! {
                    Self::#name(ref association) => association.#method
                }
            })
            .collect::<Vec<_>>()
    };
    let sites = arms(quote!(sites()));
    let weight_functions = arms(quote!(weight_functions(temperature)));
    let contact_distances = arms(quote!(contact_distances(temperature)));
    let contact_values = arms(quote!(contact_values(temperature, weighted_densities)));
    let association_strength = arms(quote!(association_strength(temperature)));

    quote! {
        #vis enum AssociationFunctionalVariant<'a> {
            #(#enum_variants,)*
        }

        impl feos_dft::AssociationFunctional for AssociationFunctionalVariant<'_> {
            fn sites(&self) -> feos_dft::AssociationSites<'_> {
                match *self {
                    #(#sites,)*
                }
            }
            fn weight_functions<N: DualNum<Primitive = f64> + Copy>(&self, temperature: N) -> feos_dft::WeightFunctionInfo<N> {
                match *self {
                    #(#weight_functions,)*
                }
            }
            fn contact_distances<N: DualNum<Primitive = f64> + Copy>(&self, temperature: N) -> Option<ndarray::Array1<N>> {
                match *self {
                    #(#contact_distances,)*
                }
            }
            fn contact_values<N: DualNum<Primitive = f64> + Copy>(
                &self,
                temperature: N,
                weighted_densities: ndarray::ArrayView2<N>,
            ) -> (ndarray::Array2<N>, ndarray::Array2<N>) {
                match *self {
                    #(#contact_values,)*
                }
            }
            fn association_strength<N: DualNum<Primitive = f64> + Copy>(&self, temperature: N) -> ndarray::Array1<N> {
                match *self {
                    #(#association_strength,)*
                }
            }
        }
    }
}
