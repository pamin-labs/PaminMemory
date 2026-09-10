//! The projection has to be able to cross threads, and that is not free.
//!
//! A resident server holds one open index per project and serves requests for
//! it from whichever runtime thread picks them up. That requires `Send` and
//! `Sync`, and the projection had neither: `icu_segmenter` reference-counts
//! its compiled data with `Rc` unless `icu_provider/sync` is enabled, and
//! `icu_segmenter` does not forward that feature, so nothing in this workspace
//! turned it on by accident.
//!
//! It is asserted here rather than left to the server to discover because the
//! failure is remote from its cause. Losing the feature -- an icu upgrade, a
//! dependency edit that looks like tidying -- produces a wall of
//! `Rc<Box<[u8]>>: Send is not satisfied` errors pointing at server code that
//! did not change, and the fix is one line in a Cargo.toml nobody was reading.

#[test]
fn the_projection_crosses_threads() {
    fn require_send<T: Send>() {}
    fn require_sync<T: Sync>() {}

    require_send::<pamin_index::ProjectionIndex>();
    require_sync::<pamin_index::ProjectionIndex>();

    // The segmenter is the half that needed the feature; the collection is
    // `Send` and `Sync` by the engine's own declaration.
    require_send::<pamin_index::segmentation::Segmenter>();
    require_sync::<pamin_index::segmentation::Segmenter>();
}
