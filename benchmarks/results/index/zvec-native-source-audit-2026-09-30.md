# Native DiskANN requested-PQ source audit, 2026-09-30

This read-only audit corrects the interpretation of the requested PQ setting.
It makes no new accuracy, latency, memory or disk claim. No index was opened,
rebuilt or upgraded, and no model or measurement harness was run.

## Source identity

`Cargo.lock` pins `zvec-rust` and `zvec-rust-sys` 0.7.2. Their published
`.cargo_vcs_info.json` identifies wrapper commit
[`89a0ab30dbb003d4ab456aa495a5181d97389297`](https://github.com/zvec-ai/zvec-rust/tree/89a0ab30dbb003d4ab456aa495a5181d97389297).
Its [prebuilt workflow](https://github.com/zvec-ai/zvec-rust/blob/89a0ab30dbb003d4ab456aa495a5181d97389297/.github/workflows/build-prebuilt.yml#L63)
checks out recursive submodules. The `vendor/zvec` gitlink references native
commit `1ab7975dfc2d2160054bafff614831b7099cd930`.

At that immutable native commit,
[DiskAnnIndex forwards the requested PQ count](https://github.com/alibaba/zvec/blob/1ab7975dfc2d2160054bafff614831b7099cd930/src/core/interface/indexes/diskann_index.cc#L96)
and [DiskAnnBuilder maps zero to an automatic count](https://github.com/alibaba/zvec/blob/1ab7975dfc2d2160054bafff614831b7099cd930/src/core/algorithm/diskann/diskann_builder.cc#L259):
`max(1, dimension / 2)`, subsequently constrained by an explicitly supplied
builder-memory budget. Its training path calls PQ training. At 1,024 dimensions,
that source rule selects 512 chunks before a budget cap; this is a source fact,
not a measurement of an installed index's effective chunk count.

The latest upstream main observed by this audit, immutable commit
[`13fd088b6255363c0d1362724340243bc8f6d0d2`](https://github.com/alibaba/zvec/tree/13fd088b6255363c0d1362724340243bc8f6d0d2),
committed 2026-09-29, has the [same automatic-count rule](https://github.com/alibaba/zvec/blob/13fd088b6255363c0d1362724340243bc8f6d0d2/src/core/algorithm/diskann/diskann_builder.cc#L260).
This conflicts with the pinned wrapper's description of zero as disabling PQ.

## Installed identity and limits

The Linux runtime library used by the retained restart-floor disk evidence has
SHA256 `58381ac7b12afd5eeae3dc10325914a28fc3157061291bb693a9ed757d815b8a`.
The preserved Cargo build output identifies a cached prebuilt library, and
that cached library has the same SHA256 as the installed runtime copy.
This establishes binary equality and cached-prebuilt resolution; it does not
provide an independent attestation of which native source produced it.
The separate [macOS archive audit](zvec-native-0.7.2-2026-09-30.md) concerns a
different platform and library hash and cannot attest this Linux runtime.

The [official wrapper Linux release asset](https://github.com/zvec-ai/zvec-rust/releases/download/v0.7.2/zvec-prebuilt-x86_64-unknown-linux-gnu.tar.gz)
has GitHub-reported archive digest
`sha256:123bde64ed8baae5ea813907af5ab3575113281eb81fca5310f28a0fe4ffb92b`.
This audit did not download that archive or establish its contained library's
identity against the Linux runtime.

The retained disk setup records `pq_chunks: 0` using the schema getter
`zvec_index_params_get_diskann_pq_chunk_num`. It does not report the builder's
effective chunk count or native PQ sections. Stored-document bit equality
also does not establish graph-navigation equivalence or vector recall.
The vendor source makes automatic PQ plausible for this runtime; effective
native introspection is still needed to establish it.

Historical requested-zero and requested-64 arms remain observations of those
requested settings. They do not establish a comparison of PQ disabled versus
enabled. Before a future PQ comparison, retain the native library identity and
assert each arm's effective chunk count or PQ sections. Accuracy, latency,
memory and disk effects of this documentation correction are all **N/A**:
no executable behavior or measured result changed.
