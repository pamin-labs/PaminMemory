//! A correction for what EmbeddingGemma 2 encodes about language.
//!
//! EmbeddingGemma 2's vectors carry the language they were written in as a
//! few strong directions, so a query ranks text in its own language above a
//! translation that answers it. Removing those directions is the
//! "language-agnostic representation" step of Xie et al. (2022, LSAR):
//! centre on a mean `mu`, then project out the top two principal directions
//! `U` of the per-language mean vectors and renormalise,
//!
//! ```text
//! T(x) = unit(y - (y U) U^T),  y = unit(x - mu)
//! ```
//!
//! and a query is searched with the even blend of itself and that image,
//! `unit((1 - a) q + a T(q))` with `a` = [`INTERP`]. Only queries: stored
//! vectors are the graph's, so the transform needs no re-index and is not
//! part of [`super::identity`].
//!
//! Why this one. Through the search entry point (the engine's XQuAD-R and
//! MuSiQue-600 tests, 768 dimensions, CPU), against no transform, it moved
//! XQuAD-R cross-lingual nDCG@10 0.6664 -> 0.6883 (+0.0218, paired sign-flip
//! p 0.0001), same-language 0.8785 -> 0.8767 (-0.0018, p 0.38) and MuSiQue
//! 0.7887 -> 0.7845 (-0.0042, p 0.14); the two losses are not significant.
//! The other variants measured there were not kept: `T` alone on queries
//! (cross-lingual 0.6912, same-language 0.8724) and with one direction
//! (0.6851, 0.8740) gave up more same-language ranking, and `T` on stored
//! text as well (0.6936, 0.8756) needs every index rebuilt for a difference
//! inside the noise.
//!
//! Provenance and attribution. The constants (`language-d768.f32`,
//! little-endian `f32`, `mu` then `U`'s two columns, `3 x 768` values) were
//! fitted on Tatoeba (tatoeba.org, sentences under CC BY 2.0 FR; credit to
//! the Tatoeba contributors), not on any evaluation corpus: per language (ar
//! de el en es hi ru th tr vi zh) 200 seeded-sample statements embedded as
//! stored text (`title: none | text: `) and 200 questions under the query
//! prompt, through this export (`embeddinggemma-2-int8@13db5bcca6ab6934`)
//! under ONNX Runtime 1.28's Python wheel, one text per pass; `mu` is the
//! mean of the 4,400 unit vectors, `U` the top right singular vectors of the
//! 11 language means less their own mean. Scripts `xl_embed.py --jobs
//! eg2-tatoeba` then `xl_constants.py --source tato` of the EG2 cross-lingual
//! screen; the sampled sentence ids are kept with it for attribution.
//!
//! Cost: about 6.8 thousand multiply-adds per query (a subtraction, two dot
//! products and two updates, a copy, a blend and two norms over 768 values),
//! against the hundreds of millions of the forward pass in front of it, once
//! per query (remembered with the vector). It goes through
//! [`crate::projection::dot`], the vector kernel this crate has (a plain loop
//! the compiler vectorises, BLAS on macOS); nothing at this size would show a
//! runtime-dispatched SIMD kernel's difference.

use std::sync::OnceLock;

use super::WIDTH;
use crate::error::{IndexError, Result};

/// How far a query moves towards its image under `T`.
///
/// Chosen on a held-out half of the XQuAD-R queries in the embedder-only
/// screen, by a rule written before the table was read: the highest
/// cross-lingual nDCG@10 whose same-language nDCG@10 is at least 0.005
/// above no transform's. Embedder-only it moved cross-lingual nDCG@10 by
/// +0.0496, same-language by +0.0056 and MuSiQue by +0.0098; the search entry
/// point's figures are in the module comment. With unit `q` and `T(q)` and
/// stored vectors left as the graph's, the blend scores a stored vector by
/// the sum of its cosines to the two views, so it fuses them in one search
/// over the unchanged index; the final renormalisation scales every score
/// alike.
const INTERP: f32 = 0.5;

const D768: &[u8] = include_bytes!("language-d768.f32");

/// `mu` followed by `U`'s two columns.
fn constants() -> &'static [f32] {
    static DECODED: OnceLock<Vec<f32>> = OnceLock::new();
    DECODED.get_or_init(|| floats(D768))
}

/// Little-endian `f32` values.
fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

/// A query vector under the transform.
pub(crate) fn query(vector: Vec<f32>) -> Result<Vec<f32>> {
    let width = WIDTH as usize;
    if vector.len() != width {
        return Err(IndexError::Engine(format!(
            "the gemma language transform takes {width} dimensions, not {}",
            vector.len()
        )));
    }
    let moved = image(vector.clone())?;
    let mut blend: Vec<f32> = vector
        .iter()
        .zip(&moved)
        .map(|(kept, moved)| (1.0 - INTERP) * kept + INTERP * moved)
        .collect();
    renormalise(&mut blend)?;
    Ok(blend)
}

/// `T(vector)`: centred, the two language directions removed, unit length.
fn image(mut vector: Vec<f32>) -> Result<Vec<f32>> {
    let (mu, basis) = constants().split_at(vector.len());
    for (value, mean) in vector.iter_mut().zip(mu) {
        *value -= mean;
    }
    for direction in basis.chunks_exact(vector.len()) {
        let along = crate::projection::dot(&vector, direction);
        for (value, component) in vector.iter_mut().zip(direction) {
            *value -= along * component;
        }
    }
    renormalise(&mut vector)?;
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

    fn cosine(left: &[f32], right: &[f32]) -> f64 {
        let dot = |a: &[f32], b: &[f32]| -> f64 {
            a.iter()
                .zip(b)
                .map(|(x, y)| f64::from(*x) * f64::from(*y))
                .sum()
        };
        dot(left, right) / (dot(left, left) * dot(right, right)).sqrt()
    }

    /// The shipped constants are what the fit wrote: three vectors, `mu`
    /// short of unit length (a mean of unit vectors), and `U` orthonormal.
    #[test]
    fn the_constants_are_a_mean_and_an_orthonormal_basis() {
        let width = WIDTH as usize;
        let constants = constants();
        assert_eq!(constants.len(), 3 * width);
        let (mu, basis) = constants.split_at(width);
        let mu_norm = crate::projection::dot(mu, mu).sqrt();
        assert!(mu_norm > 0.1 && mu_norm < 1.0, "|mu| = {mu_norm}");
        let columns: Vec<&[f32]> = basis.chunks_exact(width).collect();
        assert_eq!(columns.len(), 2);
        for (i, a) in columns.iter().enumerate() {
            for (j, b) in columns.iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((crate::projection::dot(a, b) - expected).abs() < 1e-5);
            }
        }
    }

    /// A query becomes the even blend of itself and the screen's numpy image
    /// of it, `unit(unit(x - mu) - (y U) U^T)` in float64 on these
    /// constants.
    ///
    /// The fixture holds two EmbeddingGemma 2 query vectors from the screen's
    /// cache (an English and a Chinese XQuAD-R question, 768 dimensions),
    /// then the numpy image of each, written by the script that wrote the
    /// constants: `[inputs: 2 x 768][images: 2 x 768]`.
    #[test]
    fn a_query_is_the_even_blend_of_itself_and_its_numpy_image() {
        let bytes = include_bytes!("../../tests/fixtures/eg2-language-reference.f32");
        let values = floats(bytes);
        assert_eq!(values.len(), 4 * 768);
        let (inputs, images) = values.split_at(2 * 768);
        for (input, reference) in inputs
            .as_chunks::<768>()
            .0
            .iter()
            .zip(images.as_chunks::<768>().0)
        {
            let moved = image(input.to_vec()).unwrap();
            let agreement = cosine(&moved, reference);
            assert!(1.0 - agreement < 1e-6, "image cosine {agreement}");
            let mut expected: Vec<f32> = input
                .iter()
                .zip(reference)
                .map(|(q, t)| 0.5 * q + 0.5 * t)
                .collect();
            renormalise(&mut expected).unwrap();
            let actual = query(input.to_vec()).unwrap();
            assert!(1.0 - cosine(&actual, &expected) < 1e-6);
            assert!((cosine(&actual, &actual) - 1.0).abs() < 1e-6);
            // Distinct from each view: neither weight is zero, so a no-op
            // or a bare `T` fails here.
            assert!(1.0 - cosine(&actual, input) > 1e-4);
            assert!(1.0 - cosine(&actual, &moved) > 1e-4);
        }
    }

    #[test]
    fn another_width_is_refused() {
        assert!(query(vec![0.1; 512]).is_err());
    }
}
