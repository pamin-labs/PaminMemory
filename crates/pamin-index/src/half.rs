//! Half-precision vectors: what the binding does not wrap.
//!
//! The engine stores a `VectorFp16` field as two bytes a dimension, but
//! `zvec-rust` 0.7.2 has no adder, getter or query constructor for one:
//! `Doc::add_vector_f32` tags its bytes as fp32, the engine refuses an fp32
//! value for an fp16 field, and the f32 getter returns nothing for one -- which
//! is what made an earlier measurement conclude the vectors could not be read
//! back at all. So the three crossings
//! are made here, and nowhere else: a document's vector goes in as fp16 codes
//! through the C API the binding re-exports, comes back the same way, and a
//! query vector is handed over as its fp16 codes packed two to an `f32`, which
//! the binding passes through as bytes.
//!
//! The conversion is IEEE 754 binary16, round to nearest even. An embedding is
//! a unit vector of a thousand dimensions, so its components sit around 0.03
//! and the nearest representable value is within 0.05% of every one of them;
//! values below 2^-14 become subnormals and below 2^-25 become zero, which for
//! a cosine index is a rounding of a component that contributes nothing.

use std::ffi::{CString, c_void};

use zvec_rust::{DataType, Doc};

use crate::error::{IndexError, Result};

/// The binary16 code nearest to `value`, ties to even.
pub(crate) fn encode(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;

    if exponent == 0xff {
        // Infinity stays infinity; a NaN stays a NaN.
        return sign | 0x7c00 | if mantissa == 0 { 0 } else { 0x0200 };
    }
    let rebased = exponent - 127 + 15;
    if rebased >= 0x1f {
        return sign | 0x7c00;
    }
    if rebased <= 0 {
        // Subnormal in binary16, or too small for it.
        if rebased < -10 {
            return sign;
        }
        let full = mantissa | 0x0080_0000;
        let shift = (14 - rebased) as u32;
        let halfway = 1u32 << (shift - 1);
        let rounded = full + halfway - 1 + ((full >> shift) & 1);
        return sign | (rounded >> shift) as u16;
    }
    // A carry out of the mantissa lands in the exponent, which is exactly
    // rounding up to the next binade.
    let rounded = mantissa + 0x0fff + ((mantissa >> 13) & 1);
    let code = ((rebased as u32) << 10) + (rounded >> 13);
    sign | code.min(0x7c00) as u16
}

/// The value a binary16 code stands for.
pub(crate) fn decode(code: u16) -> f32 {
    let sign = u32::from(code & 0x8000) << 16;
    let exponent = u32::from((code >> 10) & 0x1f);
    let mantissa = u32::from(code & 0x03ff);
    if exponent == 0 {
        let magnitude = mantissa as f32 / 16_777_216.0;
        return if sign == 0 { magnitude } else { -magnitude };
    }
    let bits = if exponent == 0x1f {
        sign | 0x7f80_0000 | (mantissa << 13)
    } else {
        sign | ((exponent + 127 - 15) << 23) | (mantissa << 13)
    };
    f32::from_bits(bits)
}

/// Adds `vector` to `doc` as the fp16 field `name`.
pub(crate) fn add(doc: &mut Doc, name: &str, vector: &[f32]) -> Result<()> {
    let codes: Vec<u16> = vector.iter().map(|value| encode(*value)).collect();
    let name = CString::new(name).map_err(|error| IndexError::Engine(error.to_string()))?;
    // SAFETY: the handle is the live `doc`'s, borrowed for this call only; the
    // name and the codes outlive the call, and the engine copies both.
    let code = unsafe {
        zvec_rust::sys::zvec_doc_add_field_by_value(
            doc.as_raw(),
            name.as_ptr(),
            DataType::VectorFp16 as u32,
            codes.as_ptr().cast::<c_void>(),
            std::mem::size_of_val(codes.as_slice()),
        )
    };
    if code == zvec_rust::sys::ZVEC_OK {
        Ok(())
    } else {
        Err(IndexError::Engine(format!(
            "the engine refused an fp16 vector (code {code})"
        )))
    }
}

/// The fp16 field `name` of `doc`, widened to `f32`; `None` when it has none.
pub(crate) fn get(doc: &Doc, name: &str) -> Result<Option<Vec<f32>>> {
    if !doc.has_field(name) || doc.is_field_null(name) {
        return Ok(None);
    }
    let name = CString::new(name).map_err(|error| IndexError::Engine(error.to_string()))?;
    let mut pointer: *const c_void = std::ptr::null();
    let mut size: usize = 0;
    // SAFETY: the handle is the live `doc`'s; the pointer the engine returns
    // points into it and is read, and copied out, before `doc` can be dropped.
    let code = unsafe {
        zvec_rust::sys::zvec_doc_get_field_value_pointer(
            doc.as_raw(),
            name.as_ptr(),
            DataType::VectorFp16 as u32,
            &mut pointer,
            &mut size,
        )
    };
    if code != zvec_rust::sys::ZVEC_OK || pointer.is_null() {
        return Err(IndexError::Engine(format!(
            "the engine would not read back an fp16 vector (code {code})"
        )));
    }
    // SAFETY: `size` bytes at `pointer` are the field's fp16 codes.
    let codes = unsafe { std::slice::from_raw_parts(pointer.cast::<u16>(), size / 2) };
    Ok(Some(codes.iter().map(|code| decode(*code)).collect()))
}

/// What an index stores for `vector`: each component rounded to the nearest
/// binary16 value, and widened back.
///
/// Public because it is what a document reads back as, so anything comparing
/// a stored vector against the one it wrote has to compare against this.
pub fn as_stored(vector: &[f32]) -> Vec<f32> {
    vector.iter().map(|value| decode(encode(*value))).collect()
}

/// A query vector for an fp16 field: its codes, two to an `f32`.
///
/// `SearchQuery::new` takes `&[f32]` and hands the engine the bytes behind it,
/// which the engine reads as the field's own type -- so an fp32 query against
/// an fp16 field is refused as twice the dimensions. Packing the codes' bytes,
/// in memory order, into `f32`s is what gets the engine the fp16 array it
/// reads. Every profile's width is even, so no code is left over.
pub(crate) fn query(vector: &[f32]) -> Vec<f32> {
    debug_assert!(
        vector.len().is_multiple_of(2),
        "an odd width cannot be packed"
    );
    let bytes: Vec<u8> = vector
        .iter()
        .flat_map(|value| encode(*value).to_ne_bytes())
        .collect();
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_ne_bytes(*word))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{decode, encode, query};

    /// Every code that is a number comes back as itself.
    #[test]
    fn every_finite_code_round_trips() {
        for code in 0..=u16::MAX {
            if (code >> 10) & 0x1f == 0x1f {
                continue;
            }
            assert_eq!(encode(decode(code)), code, "{code:#06x}");
        }
    }

    /// Rounding is to the nearest code, ties to even, and saturates.
    #[test]
    fn rounding_is_to_the_nearest_code() {
        assert_eq!(encode(1.0), 0x3c00);
        assert_eq!(encode(-2.0), 0xc000);
        assert_eq!(encode(65_504.0), 0x7bff);
        assert_eq!(encode(1e9), 0x7c00, "too large is infinity");
        assert_eq!(encode(1e-9), 0x0000, "too small is zero");
        // Halfway between 1.0 and the next code rounds to the even one.
        assert_eq!(encode(1.0 + 2f32.powi(-11)), 0x3c00);
        assert_eq!(encode(1.0 + 3.0 * 2f32.powi(-11)), 0x3c02);
        // What an embedding's components look like: within half a unit in
        // the last place, which is 2^-11 of the value.
        for value in [0.031_25_f32, -0.0172, 0.117, 3.1e-5] {
            let back = decode(encode(value));
            assert!((back - value).abs() <= value.abs() * 2f32.powi(-11) + 2f32.powi(-25));
        }
    }

    /// A packed query holds the codes, in order, in the bytes it hands over.
    #[test]
    fn a_packed_query_is_its_codes_in_order() {
        let vector = [0.5_f32, -0.25, 0.125, 1.0];
        let packed = query(&vector);
        assert_eq!(packed.len(), 2);
        let bytes: Vec<u8> = packed.iter().flat_map(|word| word.to_ne_bytes()).collect();
        let codes: Vec<u16> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u16::from_ne_bytes(*pair))
            .collect();
        assert_eq!(codes, vector.iter().map(|v| encode(*v)).collect::<Vec<_>>());
    }
}
