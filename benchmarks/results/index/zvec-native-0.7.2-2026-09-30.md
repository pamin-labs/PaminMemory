# Native library identity check

The JSON retains the current product-search library and official archive SHA256,
byte sizes, vendor gitlink and limits. The loaded sidecar was compared byte for
byte with libzvec_c_api.dylib extracted from the official macOS arm64 archive.
Reproduce with `curl -fL <release_asset> -o release.tar.gz`,
`tar -xzf release.tar.gz -C <scratch-dir>`, then `shasum -a 256` and `cmp` on
the extracted library and actual process-loaded sidecar. Resolve the loaded
sidecar from the process mapping/loader evidence, not a different Cargo copy.

This verifies the current macOS release artifact, not the historical Linux
DiskANN run, source-build fallback, or an independent native build attestation.
Source-build fallback follows upstream's default branch; the audited ceiling
cannot be assumed for a different runtime. Current requested defaults stay
64/100 until a new effective-parameter sweep supplies its own runtime evidence.
