//! Carry the Windows ML catalog with an installed executable, including
//! `cargo install`, which does not install DLLs beside the binary.
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let (arch, dll_hash) = match std::env::var("CARGO_CFG_TARGET_ARCH").unwrap().as_str() {
        "x86_64" => (
            "x64",
            "b401a47552a745755d65100828faefcf3df3895c759f96ac625ddf990711c8dd",
        ),
        "aarch64" => (
            "arm64",
            "457295ca2aec34db4aec7064cb401520bc85f29fd18f7e7bb9e25ddb849bedb1",
        ),
        other => panic!("Windows ML has no catalog binary for {other}"),
    };
    let out = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    let host = std::env::var("HOST").unwrap();
    let status = extractor(host.contains("windows"))
        .env("PAMIN_WINML_OUT", out)
        .env("PAMIN_WINML_ARCH", arch)
        .env("PAMIN_WINML_DLL_HASH", dll_hash)
        .status()
        .expect("run the Windows ML catalog extractor");
    assert!(
        status.success(),
        "fetching the pinned Windows ML catalog failed"
    );
}

fn extractor(windows: bool) -> Command {
    let mut command = if windows {
        let powershell = PathBuf::from(std::env::var_os("SystemRoot").unwrap())
            .join("System32/WindowsPowerShell/v1.0/powershell.exe");
        let mut command = Command::new(powershell);
        command.args(["-NoProfile", "-NonInteractive", "-Command", POWERSHELL]);
        command
    } else {
        // Only cross-builds need Python; native Windows uses its system tools.
        let mut command = Command::new("python3");
        command.args(["-c", PYTHON]);
        command
    };
    command
        .env("PAMIN_WINML_URL", URL)
        .env("PAMIN_WINML_PACKAGE_HASH", PACKAGE_HASH);
    command
}

const URL: &str = "https://api.nuget.org/v3-flatcontainer/microsoft.windows.ai.machinelearning/2.4.89/microsoft.windows.ai.machinelearning.2.4.89.nupkg";
const PACKAGE_HASH: &str = "5c68ecfb947223267abf159a023f5192ad42725e4e9cc995e7c1470dd54dff63";

// Microsoft.Windows.AI.MachineLearning 2.4.89. Extract only the catalog and its
// license; keep the project's existing ONNX Runtime and DirectML versions.
const PYTHON: &str = r#"
import hashlib, os, subprocess, zipfile
from pathlib import Path
out=Path(os.environ['PAMIN_WINML_OUT'])
arch=os.environ['PAMIN_WINML_ARCH']
dll=out/'winml-catalog.dll'
license=out/'winml-license.txt'
expected=os.environ['PAMIN_WINML_DLL_HASH']
if dll.exists() and license.exists() and hashlib.sha256(dll.read_bytes()).hexdigest()==expected:
    raise SystemExit(0)
package=out/'winml.nupkg'
try:
    if not package.exists():
        subprocess.run(['curl','--fail','--location','--silent','--show-error','--proto','=https','--proto-redir','=https','--max-time','900','--output',str(package),os.environ['PAMIN_WINML_URL']],check=True)
    if hashlib.sha256(package.read_bytes()).hexdigest()!=os.environ['PAMIN_WINML_PACKAGE_HASH']:
        raise RuntimeError('Windows ML package checksum mismatch')
    with zipfile.ZipFile(package) as archive:
        image=archive.read(f'runtimes/win-{arch}/native/Microsoft.Windows.AI.MachineLearning.dll')
        if hashlib.sha256(image).hexdigest()!=expected:
            raise RuntimeError('Windows ML DLL checksum mismatch')
        license.write_bytes(archive.read('license.txt'))
        dll.write_bytes(image)
finally:
    package.unlink(missing_ok=True)
"#;

const POWERSHELL: &str = r#"
$ErrorActionPreference='Stop'
function Digest([string]$path) {
    $sha=[System.Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($sha.ComputeHash([System.IO.File]::ReadAllBytes($path))).Replace('-','').ToLowerInvariant() }
    finally { $sha.Dispose() }
}
$dll=Join-Path $env:PAMIN_WINML_OUT 'winml-catalog.dll'
$license=Join-Path $env:PAMIN_WINML_OUT 'winml-license.txt'
if ((Test-Path $dll) -and (Test-Path $license) -and ((Digest $dll) -eq $env:PAMIN_WINML_DLL_HASH)) { exit 0 }
$package=Join-Path $env:PAMIN_WINML_OUT 'winml.nupkg'
try {
    if (!(Test-Path $package)) {
        Invoke-WebRequest -UseBasicParsing -TimeoutSec 900 -Uri $env:PAMIN_WINML_URL -OutFile $package
    }
    if ((Digest $package) -ne $env:PAMIN_WINML_PACKAGE_HASH) { throw 'Windows ML package checksum mismatch' }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive=[System.IO.Compression.ZipFile]::OpenRead($package)
    try {
        $entry=$archive.GetEntry("runtimes/win-$($env:PAMIN_WINML_ARCH)/native/Microsoft.Windows.AI.MachineLearning.dll")
        [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry,$dll,$true)
        [System.IO.Compression.ZipFileExtensions]::ExtractToFile($archive.GetEntry('license.txt'),$license,$true)
    } finally { $archive.Dispose() }
    if ((Digest $dll) -ne $env:PAMIN_WINML_DLL_HASH) { throw 'Windows ML DLL checksum mismatch' }
} finally { if (Test-Path $package) { Remove-Item $package } }
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corrupted_catalog_package_is_rejected_before_extraction() {
        let out = tempfile::tempdir().unwrap();
        std::fs::write(out.path().join("winml.nupkg"), b"corrupt").unwrap();
        let output = extractor(cfg!(windows))
            .env("PAMIN_WINML_OUT", out.path())
            .env("PAMIN_WINML_ARCH", "x64")
            .env("PAMIN_WINML_DLL_HASH", "unused")
            .env("PYTHONOPTIMIZE", "1")
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("package checksum mismatch"));
        assert!(!out.path().join("winml-catalog.dll").exists());
        assert!(!out.path().join("winml.nupkg").exists());
    }
}
