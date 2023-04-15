fn main() {
    let target = std::env::var("TARGET").unwrap();

    if target.contains("windows") {
        // println!("cargo:rustc-link-lib=dylib=mfplat");
        // println!("cargo:rustc-link-lib=dylib=mfuuid");
        // println!("cargo:rustc-link-lib=dylib=ole32");
        // println!("cargo:rustc-link-lib=dylib=strmiids");
        println!("cargo:rustc-link-lib=dylib=user32");
        println!("cargo:rustc-link-lib=static=libx264");
    } else if target.contains("apple") {
        println!("cargo:rustc-link-lib=dylib=z");
        println!("cargo:rustc-link-lib=dylib=bz2");
    }
}

