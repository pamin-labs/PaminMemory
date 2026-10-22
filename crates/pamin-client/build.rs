//! Stamps the client with a fingerprint of the protocol it speaks.
//!
//! `pamin` and `pamin-server` are two executables, so a client can meet a
//! server from another build: `pamin` upgraded alone, or copied out of one
//! tree and run against another. The server binary's own identity -- its size
//! and modification time -- catches a server that was rebuilt; it cannot
//! catch a client that was. So each request also carries this, which both
//! binaries compile in from this crate, and a server that does not share it
//! is treated as one from another build.
//!
//! A hash over what decides the protocol: this crate's sources, which hold
//! every request and result type, and `pamin-core`'s, whose types several
//! results carry. Hashing more than the types is deliberate. A rebuild that
//! changes a comment changes the fingerprint too, and costs at most a server
//! restart; a fingerprint that missed a change would cost a misread answer.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

fn main() {
    let mut files = vec![PathBuf::from("build.rs"), PathBuf::from("Cargo.toml")];
    for directory in ["src", "../pamin-core/src"] {
        let directory = Path::new(directory);
        // A registry build would carry this crate and not its neighbour.
        if directory.is_dir() {
            println!("cargo:rerun-if-changed={}", directory.display());
            sources(directory, &mut files);
        }
    }
    files.sort();

    let mut hash = Sha256::new();
    for path in &files {
        println!("cargo:rerun-if-changed={}", path.display());
        hash.update(path.to_string_lossy().as_bytes());
        hash.update(std::fs::read(path).expect("read a protocol source"));
    }
    let digest = hash.finalize();
    let fingerprint: String = digest[..8]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    println!("cargo:rustc-env=PAMIN_WIRE={fingerprint}");
}

fn sources(directory: &Path, into: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(directory).expect("read a source directory") {
        let path = entry.expect("a directory entry").path();
        if path.is_dir() {
            sources(&path, into);
        } else {
            into.push(path);
        }
    }
}
