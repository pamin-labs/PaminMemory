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
//! The conversion is IEEE 754 binary16, round to nearest even:
//! `pamin_runtime::kernels::f16`, which says what that costs an embedding.

use std::ffi::{CString, c_void};

pub(crate) use pamin_runtime::kernels::f16::{decode, encode};
use zvec_rust::{DataType, Doc};

use crate::error::{IndexError, Result};

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
    // The same values `decode` gives, converted by the CPU's own instruction
    // where it has one.
    let mut vector = vec![0.0; codes.len()];
    pamin_runtime::kernels::f16::widen(codes, &mut vector);
    Ok(Some(vector))
}

/// What an index stores for `vector`: each component rounded to the nearest
/// binary16 value, and widened back.
///
/// Public because it is what a document reads back as, so anything comparing
/// a stored vector against the one it wrote has to compare against this.
pub fn as_stored(vector: &[f32]) -> Vec<f32> {
    vector.iter().map(|value| decode(encode(*value))).collect()
}

/// `vector`'s fp16 codes as little-endian bytes: how a vector is kept outside
/// an index, at what an index stores of it.
pub fn to_bytes(vector: &[f32]) -> Vec<u8> {
    vector
        .iter()
        .flat_map(|value| encode(*value).to_le_bytes())
        .collect()
}

/// How many bytes [`to_bytes`] makes of one of this model's vectors.
pub const VECTOR_BYTES: usize = 2 * crate::DIMENSIONS as usize;

/// [`to_bytes`] widened back: the vector an index would have read back, or
/// `None` for bytes that are not one of this model's vectors.
pub fn from_bytes(bytes: &[u8]) -> Option<Vec<f32>> {
    if bytes.len() != VECTOR_BYTES {
        return None;
    }
    let codes: Vec<u16> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect();
    let mut vector = vec![0.0; codes.len()];
    pamin_runtime::kernels::f16::widen(&codes, &mut vector);
    Some(vector)
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
    use super::{as_stored, encode, from_bytes, query, to_bytes};

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

    /// A vector kept as bytes reads back as an index stores it.
    #[test]
    fn a_vector_kept_as_bytes_reads_back_as_stored() {
        let vector: Vec<f32> = (0..crate::DIMENSIONS)
            .map(|at| at as f32 / 7.0 - 3.0)
            .collect();
        let bytes = to_bytes(&vector);
        assert_eq!(from_bytes(&bytes), Some(as_stored(&vector)));
        assert_eq!(from_bytes(&bytes[2..]), None, "another width is not one");
    }
}
