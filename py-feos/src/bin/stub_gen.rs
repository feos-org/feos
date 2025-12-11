use std::path::{Path, PathBuf};

fn main() {
    println!("Generating stub files");
    let path = Path::new(".venv/lib/python3.12/site-packages/feos/feos.abi3.so");

    assert!(path.is_file(), "Failed to locate cdylib");
    let main_module_name = "feos";
    let python_module = pyo3_introspection::introspect_cdylib(path, main_module_name)
        .expect(format!("Failed introspection of {}", main_module_name).as_str());
    let type_stubs = pyo3_introspection::module_stub_files(&python_module);

    let stubst_string = type_stubs
        .get(&PathBuf::from("__init__.pyi"))
        .expect("Failed to get __init__.pyi");
    std::fs::write("feos.pyi", stubst_string).expect("Failed to write stubs file");
    println!("Generated stubs: {}", "feos.pyi")
}
