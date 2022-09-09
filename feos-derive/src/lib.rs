use proc_macro::TokenStream;

mod eos;
mod dft;

#[proc_macro_derive(EquationOfState, attributes(skip_impl))]
pub fn derive_eos(input: TokenStream) -> TokenStream {
    eos::derive_equation_of_state(input)
}

#[proc_macro_derive(HelmholtzEnergyFunctional, attributes(skip_impl))]
pub fn derive_dft(input: TokenStream) -> TokenStream {
    dft::derive_helmholtz_energy_functional(input)
}