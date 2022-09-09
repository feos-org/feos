use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

// const POSSIBLE_SKIP_STATEMENTS: [&'static str; 2] = ["molar_weight", "entropy_scaling"];
// Todo: Validate possible arguments

pub fn derive_helmholtz_energy_functional(input: TokenStream) -> TokenStream {
    TokenStream::new()
}
