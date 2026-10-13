//! A correction for what EmbeddingGemma 2 encodes about language.
//!
//! EmbeddingGemma 2's vectors carry the language they were written in as a
//! few strong directions, so a query ranks text in its own language above a
//! translation that answers it. Removing those directions from the query is
//! the "language-agnostic representation" step of Xie et al. (2022, LSAR):
//! centre on a mean `mu`, then project out the top `k` principal directions
//! `U` of the per-language mean vectors and renormalise,
//!
//! ```text
//! x -> y = unit(x - mu) -> unit(y - (y U) U^T)
//! ```
//!
//! `PAMIN_EG2_QUERY_TRANSFORM` picks one of [`Transform`]'s variants; it is a
//! measurement switch, undocumented on purpose, like `PAMIN_EG2_DIMS`. The
//! `center*` variants and `interp-lsar2` transform queries only: stored
//! vectors are the graph's, so they share one index with `off`, and
//! [`super::identity`] does not change with them. The `both-*` variants
//! transform stored text the same way, with the same constants, after the
//! width cut and before storage; their vectors are another space, so the
//! variant is part of [`super::identity`] and an index built under one
//! refuses to open under any other.
//!
//! Provenance and attribution. The constants, per width (`language-d{768,512,256}.f32`, little-endian
//! `f32`, `mu` then `U`'s two columns, `3 d` values), were fitted on Tatoeba
//! (tatoeba.org, sentences under CC BY 2.0 FR; credit to the Tatoeba
//! contributors), not on any evaluation corpus: per language (ar de el en es
//! hi ru th tr vi zh) 200 seeded-sample statements embedded as stored text
//! (`title: none | text: `) and 200 questions under the query prompt, through
//! this export (`embeddinggemma-2-int8@13db5bcca6ab6934`) under ONNX Runtime
//! 1.28's Python wheel, one text per pass; `mu` is the mean of the 4,400 unit
//! vectors cut to the width and renormalised, `U` the top right singular
//! vectors of the 11 language means less their own mean. Scripts
//! `xl_embed.py --jobs eg2-tatoeba` then `xl_constants.py --source tato` of
//! the EG2 cross-lingual screen; the sampled sentence ids are kept with it for
//! attribution. `U`'s first column is the `k = 1` basis, both columns `k = 2`.
//! There are none for the 128-dimension width, which the screen did not
//! measure; there every variant is `off`.
//!
//! Cost: at 768 dimensions `center-lsar2` is about 3.8 thousand
//! multiply-adds per query (a subtraction, two dot products, two updates and
//! a norm over 768 values), against the hundreds of millions of the forward
//! pass in front of it, once per query (remembered with the vector) or per
//! stored text. It goes through [`crate::projection::dot`], the vector kernel
//! this crate has (a plain loop the compiler vectorises, BLAS on macOS);
//! nothing at this size would show a runtime-dispatched SIMD kernel's
//! difference. `interp-lsar2` adds about three thousand operations more (a
//! copy of the query, a blend and a second norm over the same 768 values).

use std::sync::OnceLock;

use crate::error::{IndexError, Result};

/// The variable naming the variant.
const SETTING: &str = "PAMIN_EG2_QUERY_TRANSFORM";

/// What is done to a vector after the graph and the width cut.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Transform {
    /// Nothing: the vector the graph returned, bit for bit.
    Off,
    /// `unit(x - mu)`.
    Center,
    /// Centred, then the first language direction removed.
    CenterLsar1,
    /// Centred, then the first two language directions removed.
    CenterLsar2,
    /// `CenterLsar1` on queries and on stored text.
    BothCenterLsar1,
    /// `CenterLsar2` on queries and on stored text.
    BothCenterLsar2,
    /// `unit((1 - a) q + a T(q))` on queries, `T` being `CenterLsar2` and
    /// `a` [`INTERP`]; stored text is the graph's.
    InterpLsar2,
}

/// How far `interp-lsar2` moves a query towards its `center-lsar2` image.
///
/// Chosen on a held-out half of the XQuAD-R queries in the embedder-only
/// screen, by a rule written before the table was read: the highest
/// cross-lingual nDCG@10 whose same-language nDCG@10 is at least 0.005
/// above `off`'s. Embedder-only at 768 dimensions it moved cross-lingual
/// nDCG@10 by +0.0496, same-language by +0.0056 and MuSiQue by +0.0098; the
/// search entry point has not measured it. With unit `q` and `T(q)` and stored
/// vectors left as the graph's, the blend scores a stored vector by the sum
/// of its cosines to the two views, so it fuses them in one search over the
/// unchanged index; the final renormalisation scales every score alike.
const INTERP: f32 = 0.5;

impl Transform {
    /// What runs when the setting is absent: nothing, today's vectors.
    ///
    /// Not yet a choice. Applied to queries alone, `center-lsar2` moved the
    /// screen's embedder-only XQuAD-R same-language nDCG@10 by -0.0072 (p
    /// 0.12), past the 0.005 it is allowed, and `center-lsar1` by -0.0013;
    /// the default waits on a measurement through the search entry point.
    const DEFAULT: Self = Self::Off;

    fn parse(value: &str) -> Option<Self> {
        match value.trim() {
            "off" => Some(Self::Off),
            "center" => Some(Self::Center),
            "center-lsar1" => Some(Self::CenterLsar1),
            "center-lsar2" => Some(Self::CenterLsar2),
            "both-center-lsar1" => Some(Self::BothCenterLsar1),
            "both-center-lsar2" => Some(Self::BothCenterLsar2),
            "interp-lsar2" => Some(Self::InterpLsar2),
            _ => None,
        }
    }

    fn directions(self) -> usize {
        match self {
            Self::Off | Self::Center => 0,
            Self::CenterLsar1 | Self::BothCenterLsar1 => 1,
            Self::CenterLsar2 | Self::BothCenterLsar2 | Self::InterpLsar2 => 2,
        }
    }

    /// What this variant adds to the profile identity: the name of a
    /// transform stored vectors are made with, or nothing when they are the
    /// graph's.
    pub(crate) fn stored_tag(self) -> Option<&'static str> {
        match self {
            Self::Off
            | Self::Center
            | Self::CenterLsar1
            | Self::CenterLsar2
            | Self::InterpLsar2 => None,
            Self::BothCenterLsar1 => Some("both-center-lsar1"),
            Self::BothCenterLsar2 => Some("both-center-lsar2"),
        }
    }
}

/// The variant this process applies, read once.
///
/// An unrecognised value falls back to [`Transform::DEFAULT`] with a warning.
pub(crate) fn chosen() -> Transform {
    static CHOSEN: OnceLock<Transform> = OnceLock::new();
    *CHOSEN.get_or_init(|| {
        let transform = match std::env::var(SETTING) {
            Err(_) => Transform::DEFAULT,
            Ok(value) => Transform::parse(&value).unwrap_or_else(|| {
                tracing::warn!(
                    %value,
                    "{SETTING} is not off, center, center-lsar1, center-lsar2, \
                     both-center-lsar1, both-center-lsar2 or interp-lsar2; using {:?}",
                    Transform::DEFAULT
                );
                Transform::DEFAULT
            }),
        };
        tracing::info!(?transform, "gemma language transform");
        transform
    })
}

const D768: &[u8] = include_bytes!("language-d768.f32");
const D512: &[u8] = include_bytes!("language-d512.f32");
const D256: &[u8] = include_bytes!("language-d256.f32");

/// Little-endian `f32` values.
fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

/// `mu` followed by `U`'s columns, for a width, or `None` for one with no fit.
fn constants(width: usize) -> Option<&'static [f32]> {
    static DECODED: OnceLock<[Vec<f32>; 3]> = OnceLock::new();
    let decoded = DECODED.get_or_init(|| [D768, D512, D256].map(floats));
    let index = match width {
        768 => 0,
        512 => 1,
        256 => 2,
        _ => return None,
    };
    Some(&decoded[index])
}

/// A query vector under `transform`.
pub(crate) fn query(transform: Transform, vector: Vec<f32>) -> Result<Vec<f32>> {
    apply(transform, vector)
}

/// A vector for storage under `transform`: transformed as a query is under
/// the `both-*` variants, and the graph's own otherwise.
pub(crate) fn stored(transform: Transform, vector: Vec<f32>) -> Result<Vec<f32>> {
    if transform.stored_tag().is_none() {
        return Ok(vector);
    }
    apply(transform, vector)
}

/// `vector` under `transform`, at the vector's own width.
///
/// `Off`, and any variant at a width with no constants, returns the vector
/// as it came. `InterpLsar2` blends the `CenterLsar2` result with the vector
/// it came from, then renormalises again.
fn apply(transform: Transform, mut vector: Vec<f32>) -> Result<Vec<f32>> {
    if transform == Transform::Off {
        return Ok(vector);
    }
    let width = vector.len();
    let Some(constants) = constants(width) else {
        return Ok(vector);
    };
    let (mu, basis) = constants.split_at(width);
    let original = (transform == Transform::InterpLsar2).then(|| vector.clone());
    for (value, mean) in vector.iter_mut().zip(mu) {
        *value -= mean;
    }
    for direction in basis.chunks_exact(width).take(transform.directions()) {
        let along = crate::projection::dot(&vector, direction);
        for (value, component) in vector.iter_mut().zip(direction) {
            *value -= along * component;
        }
    }
    renormalise(&mut vector)?;
    if let Some(original) = original {
        for (value, kept) in vector.iter_mut().zip(&original) {
            *value = (1.0 - INTERP) * kept + INTERP * *value;
        }
        renormalise(&mut vector)?;
    }
    Ok(vector)
}

/// `vector` scaled to unit length, or an error when nothing is left of it.
fn renormalise(vector: &mut [f32]) -> Result<()> {
    let norm = crate::projection::dot(vector, vector).sqrt();
    if norm.is_nan() || norm <= 0.0 {
        return Err(IndexError::Numerical(
            "a gemma vector with nothing left after the language transform".into(),
        ));
    }
    for value in vector {
        *value /= norm;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [Transform; 7] = [
        Transform::Off,
        Transform::Center,
        Transform::CenterLsar1,
        Transform::CenterLsar2,
        Transform::BothCenterLsar1,
        Transform::BothCenterLsar2,
        Transform::InterpLsar2,
    ];

    fn cosine(left: &[f32], right: &[f32]) -> f64 {
        let dot = |a: &[f32], b: &[f32]| -> f64 {
            a.iter()
                .zip(b)
                .map(|(x, y)| f64::from(*x) * f64::from(*y))
                .sum()
        };
        dot(left, right) / (dot(left, left) * dot(right, right)).sqrt()
    }

    fn bits(vector: &[f32]) -> Vec<u32> {
        vector.iter().map(|v| v.to_bits()).collect()
    }

    fn unit(vector: &[f32]) -> Vec<f32> {
        let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        vector.iter().map(|v| v / norm).collect()
    }

    /// The shipped constants are what the fit wrote: three vectors per width,
    /// `mu` short of unit length (a mean of unit vectors), and `U` orthonormal.
    #[test]
    fn the_constants_are_a_mean_and_an_orthonormal_basis() {
        for width in [768, 512, 256] {
            let constants = constants(width).unwrap();
            assert_eq!(constants.len(), 3 * width);
            let (mu, basis) = constants.split_at(width);
            let mu_norm = crate::projection::dot(mu, mu).sqrt();
            assert!(mu_norm > 0.1 && mu_norm < 1.0, "{width}: |mu| = {mu_norm}");
            let columns: Vec<&[f32]> = basis.chunks_exact(width).collect();
            for (i, a) in columns.iter().enumerate() {
                for (j, b) in columns.iter().enumerate() {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!((crate::projection::dot(a, b) - expected).abs() < 1e-5);
                }
            }
        }
        assert!(constants(128).is_none());
    }

    /// `off` hands back the graph's vector bit for bit, as a query and for
    /// storage, at every width; so does every variant at the width with no
    /// constants.
    #[test]
    fn off_is_the_vector_unchanged() {
        for width in [768, 512, 256, 128] {
            let vector = unit(
                &(0..width)
                    .map(|i| ((i * 37 % 101) as f32 - 50.0) / 7.0)
                    .collect::<Vec<_>>(),
            );
            let original = bits(&vector);
            let transforms: &[Transform] = if width == 128 { &ALL } else { &ALL[..1] };
            for transform in transforms {
                assert_eq!(bits(&query(*transform, vector.clone()).unwrap()), original);
                assert_eq!(bits(&stored(*transform, vector.clone()).unwrap()), original);
            }
        }
    }

    #[test]
    fn the_setting_names_each_variant() {
        assert_eq!(Transform::parse("off"), Some(Transform::Off));
        assert_eq!(Transform::parse("center"), Some(Transform::Center));
        assert_eq!(
            Transform::parse(" center-lsar1"),
            Some(Transform::CenterLsar1)
        );
        assert_eq!(
            Transform::parse("center-lsar2\n"),
            Some(Transform::CenterLsar2)
        );
        assert_eq!(
            Transform::parse("both-center-lsar1"),
            Some(Transform::BothCenterLsar1)
        );
        assert_eq!(
            Transform::parse("both-center-lsar2"),
            Some(Transform::BothCenterLsar2)
        );
        assert_eq!(
            Transform::parse("interp-lsar2"),
            Some(Transform::InterpLsar2)
        );
        assert_eq!(Transform::parse("lsar2"), None);
        assert_eq!(Transform::parse("interp"), None);
        assert_eq!(Transform::parse("both-center"), None);
    }

    /// The transform agrees with the screen's numpy arithmetic,
    /// `unit(unit(x - mu) - (y U) U^T)` in float64 on these constants.
    ///
    /// The fixture holds two EmbeddingGemma 2 query vectors from the screen's
    /// cache (an English and a Chinese XQuAD-R question, 768 dimensions), and
    /// for each width the numpy output of every non-`off` variant on the input
    /// cut to that width and renormalised. Written by the script that wrote
    /// the constants; layout in the fixture's header comment below.
    #[test]
    fn the_transform_matches_the_numpy_reference() {
        // [inputs: 2 x 768] then, per width 768, 512, 256, per input, per
        // variant center, center-lsar1, center-lsar2: `width` values.
        let bytes = include_bytes!("../../tests/fixtures/eg2-language-reference.f32");
        let values = floats(bytes);
        let (inputs, mut expected) = values.split_at(2 * 768);
        let mut checked = 0;
        let mut checked_both = 0;
        for width in [768, 512, 256] {
            for input in inputs.as_chunks::<768>().0 {
                let cut = unit(&input[..width]);
                for transform in &ALL[1..4] {
                    let (reference, rest) = expected.split_at(width);
                    expected = rest;
                    let actual = query(*transform, cut.clone()).unwrap();
                    // The query-only variants store the graph's vector.
                    let kept = stored(*transform, cut.clone()).unwrap();
                    assert_eq!(bits(&kept), bits(&cut));
                    // The both-sides variant of the same depth transforms a
                    // query and a stored text alike, to the same reference.
                    let both = match transform {
                        Transform::CenterLsar1 => Some(Transform::BothCenterLsar1),
                        Transform::CenterLsar2 => Some(Transform::BothCenterLsar2),
                        _ => None,
                    };
                    if let Some(both) = both {
                        let as_query = query(both, cut.clone()).unwrap();
                        let as_stored = stored(both, cut.clone()).unwrap();
                        assert_eq!(bits(&as_query), bits(&actual));
                        assert_eq!(bits(&as_stored), bits(&actual));
                        checked_both += 1;
                    }
                    let agreement = cosine(&actual, reference);
                    assert!(
                        1.0 - agreement < 1e-6,
                        "{transform:?} at {width}: cosine {agreement}"
                    );
                    assert!((cosine(&actual, &actual) - 1.0).abs() < 1e-6);
                    // And it moved the vector: a no-op would pass the line above
                    // only if the reference were also the input.
                    assert!(1.0 - cosine(&actual, &cut) > 1e-3);
                    checked += 1;
                }
            }
        }
        assert!(expected.is_empty());
        assert_eq!(checked, 18);
        assert_eq!(checked_both, 12);
    }

    /// `interp-lsar2` is the even blend of a query and its `center-lsar2`
    /// image, made unit length, and leaves stored text alone.
    #[test]
    fn interp_is_the_even_blend_of_a_query_and_its_center_lsar2_image() {
        let bytes = include_bytes!("../../tests/fixtures/eg2-language-reference.f32");
        let values = floats(bytes);
        let mut checked = 0;
        for width in [768, 512, 256] {
            for input in values[..2 * 768].as_chunks::<768>().0 {
                let cut = unit(&input[..width]);
                let moved = query(Transform::CenterLsar2, cut.clone()).unwrap();
                let mut blend: Vec<f32> = cut
                    .iter()
                    .zip(&moved)
                    .map(|(q, t)| 0.5 * q + 0.5 * t)
                    .collect();
                renormalise(&mut blend).unwrap();
                let actual = query(Transform::InterpLsar2, cut.clone()).unwrap();
                assert_eq!(bits(&actual), bits(&blend), "at {width}");
                assert!((cosine(&actual, &actual) - 1.0).abs() < 1e-6);
                // Distinct from each view: neither weight is zero.
                assert!(1.0 - cosine(&actual, &cut) > 1e-4);
                assert!(1.0 - cosine(&actual, &moved) > 1e-4);
                let kept = stored(Transform::InterpLsar2, cut.clone()).unwrap();
                assert_eq!(bits(&kept), bits(&cut));
                checked += 1;
            }
        }
        assert_eq!(checked, 6);
    }
}
