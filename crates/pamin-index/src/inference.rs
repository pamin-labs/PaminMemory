//! How much of the machine one forward pass is allowed to use.
//!
//! Both models in this crate run on ONNX Runtime through `fastembed`, and
//! neither set this, so both inherited the library's default: intra-op threads
//! equal to `available_parallelism()`. One pass therefore takes every core.
//!
//! That is the right default for a single query and the wrong one for a server.
//! Measured on four cores with nothing shared -- one model per worker, no lock
//! of ours anywhere -- throughput *falls* as callers are added, 182 embeddings
//! a second at one worker to 54 at eight, because the second caller finds the
//! first caller's threads rather than an idle core. Splitting the cores between
//! callers instead of between the layers of one pass is the other way to divide
//! them, and which one wins is a property of the machine rather than of this
//! code.
//!
//! So it is a setting, and its default is what the library already did.

/// Intra-op threads per inference session, or `None` for one per core.
///
/// Read from `PAMIN_INFERENCE_THREADS`. Unset -- the default -- leaves
/// `fastembed` to use `available_parallelism()`, which is what this crate did
/// before the setting existed, so an unconfigured workspace behaves exactly as
/// it used to.
///
/// A value that is not a positive number is ignored rather than refused: this
/// is a performance knob, and a typo in it should not stop a search from
/// working.
pub(crate) fn threads() -> Option<usize> {
    std::env::var("PAMIN_INFERENCE_THREADS")
        .ok()?
        .parse::<usize>()
        .ok()
        .filter(|threads| *threads > 0)
}
