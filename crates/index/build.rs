//! The build identity execution plans are keyed by; see `BUILD_FINGERPRINT`.
use std::path::PathBuf;
use std::process::Command;

#[path = "src/build_profile.rs"]
mod build_profile;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    build_identity();
}

fn build_identity() {
    use sha2::{Digest, Sha256};
    fn sources(directory: &std::path::Path, files: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(directory).expect("read inference sources") {
            let path = entry.expect("source entry").path();
            if path.is_dir() {
                sources(&path, files);
            } else {
                files.push(path);
            }
        }
    }
    let mut files = vec![PathBuf::from("build.rs"), PathBuf::from("Cargo.toml")];
    sources(std::path::Path::new("src"), &mut files);
    // Registry installs need not carry the workspace files.
    for name in [
        "../../Cargo.toml",
        "../../Cargo.lock",
        ".cargo/config",
        ".cargo/config.toml",
        "../../.cargo/config",
        "../../.cargo/config.toml",
    ] {
        println!("cargo:rerun-if-changed={name}");
        if std::path::Path::new(name).is_file() {
            files.push(PathBuf::from(name));
        }
    }
    files.sort();
    let mut hash = Sha256::new();
    for path in files {
        println!("cargo:rerun-if-changed={}", path.display());
        hash.update(path.to_string_lossy().as_bytes());
        hash.update(std::fs::read(path).expect("read build identity input"));
    }
    for key in [
        "TARGET",
        "PROFILE",
        "OPT_LEVEL",
        "DEBUG",
        "CARGO_ENCODED_RUSTFLAGS",
        "CARGO_CFG_TARGET_FEATURE",
    ] {
        println!("cargo:rerun-if-env-changed={key}");
        hash.update(key.as_bytes());
        hash.update(
            std::env::var_os(key)
                .unwrap_or_default()
                .to_string_lossy()
                .as_bytes(),
        );
    }
    // The same build, readable, for a record to show beside the
    // fingerprint: each Cargo profile override set and each `cfg` the target
    // resolved to, `NAME=value`, separated by 0x1f.
    let mut readable = Vec::new();
    // `vars_os`: `vars` panics on any variable in the environment that is
    // not UTF-8, which need not be one of these. Lossy, so a non-UTF-8 value
    // is still read the same way every time.
    for (key, value) in std::env::vars_os() {
        let key = key.to_string_lossy();
        if key.starts_with("CARGO_CFG_") {
            println!("cargo:rerun-if-env-changed={key}");
            readable.push(format!("{key}={}", value.to_string_lossy()));
        }
    }
    for (key, value) in build_profile::settings(
        &std::env::var("PROFILE").unwrap_or_default(),
        std::env::vars_os().map(|(key, value)| {
            (
                key.to_string_lossy().into_owned(),
                value.to_string_lossy().into_owned(),
            )
        }),
    ) {
        println!("cargo:rerun-if-env-changed={key}");
        if !value.is_empty() {
            readable.push(format!("{key}={value}"));
        }
        hash.update((key.len() as u64).to_le_bytes());
        hash.update(key.as_bytes());
        hash.update((value.len() as u64).to_le_bytes());
        hash.update(value.as_bytes());
    }
    if let Ok(output) = Command::new(std::env::var_os("RUSTC").expect("rustc path"))
        .arg("--version")
        .output()
    {
        hash.update(output.stdout);
    }
    println!(
        "cargo:rustc-env=PAMIN_INFERENCE_BUILD={:x}",
        hash.finalize()
    );
    readable.sort();
    println!(
        "cargo:rustc-env=PAMIN_INFERENCE_CARGO={}",
        readable.join("\x1f")
    );
}
