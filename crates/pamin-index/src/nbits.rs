//! An 8-bit `MatMulNBits` export, told to compute in int8.
//!
//! Perplexity publishes pplx-embed with every matrix product as a
//! `MatMulNBits` node over 8-bit weights, and leaves out the node's
//! `accuracy_level`. ONNX Runtime reads that as level 0 -- dequantize each
//! block of weights to fp32 and multiply in fp32 -- which on this project's
//! four-core CPU ran a query 6.5 times and a passage 4.4 times slower than a
//! dynamic int8 quantization of the same model. At level 4 the runtime
//! quantizes each block of activations to int8 as well and runs its integer
//! kernels: 1.3 and 2.0 times that quantization, and closer to the
//! full-precision export than it -- cosine 0.99925 on average and 0.9986 at the
//! lowest over 400 texts in twelve sets, against 0.9957 and 0.9893 (ONNX
//! Runtime 1.30 in Python, interleaved text by text on a shared machine; see
//! `docs/adr/0001-tech-selection.md`).
//!
//! The runtime has no session setting for it: the attribute is read from the
//! node and nowhere else. So the graph is rewritten, once, into a directory of
//! its own under the model cache, beside a link to the weights it names --
//! the graph finds its weights by a path relative to itself -- and loaded from
//! there as if it were the download. The rewrite adds one attribute to each
//! `MatMulNBits` node and changes nothing else, byte for byte.

use std::path::{Path, PathBuf};

use crate::error::{IndexError, Result};

/// What the attribute is set to: the runtime's "int8 activations, int32
/// accumulation".
const LEVEL: u64 = 4;

/// `graph` rewritten to compute in int8, with `weights` -- the files its
/// initializers name -- beside it, under `cache_dir`.
///
/// Written once and found after. The directory is named by the graph file's
/// own name once symbolic links are followed, which for a hub download is the
/// blob, named by a hash of the file's content, so a different graph never
/// finds this one's rewrite. A directory that exists is complete: it is
/// written under another name and renamed into place, and a process that
/// loses the race to rename takes the winner's.
pub(crate) fn int8_compute(graph: &Path, weights: &[PathBuf], cache_dir: &Path) -> Result<PathBuf> {
    let failed = |what: &str, error: &dyn std::fmt::Display| {
        IndexError::Engine(format!("{what} {}: {error}", graph.display()))
    };
    let blob = std::fs::canonicalize(graph)?;
    let key = blob
        .file_name()
        .ok_or_else(|| failed("naming", &"no file name"))?
        .to_string_lossy()
        .to_string();
    let name = graph
        .file_name()
        .ok_or_else(|| failed("naming", &"no file name"))?;
    let root = cache_dir.join("int8-compute");
    let done = root.join(&key);
    let rewritten = done.join(name);
    if rewritten.exists() {
        return Ok(rewritten);
    }

    let partial = root.join(format!("{key}.partial-{}", std::process::id()));
    if partial.exists() {
        std::fs::remove_dir_all(&partial)?;
    }
    std::fs::create_dir_all(&partial)?;
    let written = (|| -> Result<()> {
        for weights in weights {
            let file = weights
                .file_name()
                .ok_or_else(|| failed("naming", &"weights with no file name"))?;
            let target = std::fs::canonicalize(weights)?;
            // A link costs nothing; a copy is what a cache on another
            // filesystem gets.
            if std::fs::hard_link(&target, partial.join(file)).is_err() {
                std::fs::copy(&target, partial.join(file))?;
            }
        }
        let (model, nodes) = rewrite(&std::fs::read(&blob)?)?;
        if nodes == 0 {
            return Err(failed("rewriting", &"it has no MatMulNBits node"));
        }
        std::fs::write(partial.join(name), model)?;
        Ok(())
    })();
    if let Err(error) = written {
        let _ = std::fs::remove_dir_all(&partial);
        return Err(error);
    }
    if let Err(error) = std::fs::rename(&partial, &done) {
        let _ = std::fs::remove_dir_all(&partial);
        if !rewritten.exists() {
            return Err(error.into());
        }
    }
    tracing::info!(graph = %graph.display(), copy = %rewritten.display(), "set the export to compute in int8");
    Ok(rewritten)
}

/// A `ModelProto` with `accuracy_level` added to every `MatMulNBits` node of
/// its main graph, and how many it added it to.
///
/// Protocol buffers read a message field by field, so a field appended to a
/// node is read as part of it; what has to change is the length every
/// enclosing message records. Everything else is copied as it was read.
fn rewrite(model: &[u8]) -> Result<(Vec<u8>, usize)> {
    // `ModelProto.graph` and `GraphProto.node`.
    const GRAPH: u64 = 7;
    const NODE: u64 = 1;
    let mut nodes = 0;
    let model = rebuild(model, GRAPH, |graph| {
        rebuild(graph, NODE, |node| {
            if !is_nbits(node)? {
                return Ok(node.to_vec());
            }
            nodes += 1;
            let mut node = node.to_vec();
            node.extend(attribute());
            Ok(node)
        })
    })?;
    Ok((model, nodes))
}

/// `message` with every length-delimited field numbered `field` replaced by
/// what `change` makes of it.
fn rebuild(
    message: &[u8],
    field: u64,
    mut change: impl FnMut(&[u8]) -> Result<Vec<u8>>,
) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(message.len() + 64);
    let mut at = 0;
    while at < message.len() {
        let start = at;
        let tag = varint(message, &mut at)?;
        let body = skip(message, &mut at, tag)?;
        match body {
            Some(body) if tag >> 3 == field => {
                let changed = change(body)?;
                push_varint(&mut out, tag);
                push_varint(&mut out, changed.len() as u64);
                out.extend(changed);
            }
            _ => out.extend_from_slice(&message[start..at]),
        }
    }
    Ok(out)
}

/// Whether a `NodeProto` is a `MatMulNBits` that does not state its level.
fn is_nbits(node: &[u8]) -> Result<bool> {
    // `NodeProto.op_type` and `NodeProto.attribute`, and `AttributeProto.name`.
    const OP_TYPE: u64 = 4;
    const ATTRIBUTE: u64 = 5;
    const NAME: u64 = 1;
    let mut nbits = false;
    let mut stated = false;
    let mut at = 0;
    while at < node.len() {
        let tag = varint(node, &mut at)?;
        match (tag >> 3, skip(node, &mut at, tag)?) {
            (OP_TYPE, Some(op)) => nbits = op == b"MatMulNBits",
            (ATTRIBUTE, Some(attribute)) => {
                let mut inner = 0;
                while inner < attribute.len() {
                    let tag = varint(attribute, &mut inner)?;
                    if let (NAME, Some(name)) = (tag >> 3, skip(attribute, &mut inner, tag)?) {
                        stated |= name == b"accuracy_level";
                    }
                }
            }
            _ => {}
        }
    }
    Ok(nbits && !stated)
}

/// `NodeProto.attribute` holding `accuracy_level = LEVEL`, as an `INT`.
fn attribute() -> Vec<u8> {
    let mut attribute = Vec::new();
    // name = 1, a string.
    push_varint(&mut attribute, 1 << 3 | 2);
    push_varint(&mut attribute, "accuracy_level".len() as u64);
    attribute.extend(b"accuracy_level");
    // i = 3, a varint.
    push_varint(&mut attribute, 3 << 3);
    push_varint(&mut attribute, LEVEL);
    // type = 20, `AttributeType.INT` = 2.
    push_varint(&mut attribute, 20 << 3);
    push_varint(&mut attribute, 2);

    let mut field = Vec::new();
    push_varint(&mut field, 5 << 3 | 2);
    push_varint(&mut field, attribute.len() as u64);
    field.extend(attribute);
    field
}

/// Moves past one field's value, returning its bytes if it is
/// length-delimited.
fn skip<'a>(message: &'a [u8], at: &mut usize, tag: u64) -> Result<Option<&'a [u8]>> {
    let malformed = || IndexError::Engine("the model file is not a valid ONNX graph".into());
    match tag & 7 {
        0 => {
            varint(message, at)?;
            Ok(None)
        }
        1 | 5 => {
            *at += if tag & 7 == 1 { 8 } else { 4 };
            (*at <= message.len()).then_some(None).ok_or_else(malformed)
        }
        2 => {
            let length = usize::try_from(varint(message, at)?).map_err(|_| malformed())?;
            let end = at.checked_add(length).filter(|end| *end <= message.len());
            let end = end.ok_or_else(malformed)?;
            let body = &message[*at..end];
            *at = end;
            Ok(Some(body))
        }
        _ => Err(malformed()),
    }
}

fn varint(message: &[u8], at: &mut usize) -> Result<u64> {
    let mut value = 0u64;
    for shift in (0..64).step_by(7) {
        let byte = *message
            .get(*at)
            .ok_or_else(|| IndexError::Engine("the model file ends inside a field".into()))?;
        *at += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err(IndexError::Engine(
        "the model file holds an overlong varint".into(),
    ))
}

fn push_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn string(field: u64, value: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        push_varint(&mut out, field << 3 | 2);
        push_varint(&mut out, value.len() as u64);
        out.extend(value);
        out
    }

    fn node(op: &str, extra: &[u8]) -> Vec<u8> {
        let mut node = string(1, b"input");
        node.extend(string(4, op.as_bytes()));
        node.extend(string(
            3,
            b"a name long enough to need a two-byte length "
                .repeat(4)
                .as_slice(),
        ));
        node.extend(extra);
        node
    }

    /// A model of two nodes, one of each kind, with fields around the graph
    /// and around the nodes that the rewrite must copy untouched.
    fn model(nbits: &[u8]) -> Vec<u8> {
        let mut graph = string(1, &node("MatMul", &[]));
        graph.extend(string(1, &node("MatMulNBits", nbits)));
        graph.extend(string(2, b"main"));
        let mut model = vec![0x08, 0x0a]; // ir_version = 10
        model.extend(string(7, &graph));
        model.extend(string(2, b"producer"));
        model
    }

    /// Only the `MatMulNBits` node gains the attribute, and every other byte
    /// is where it was.
    #[test]
    fn only_the_nbits_node_gains_the_level() {
        let (rewritten, nodes) = rewrite(&model(&[])).expect("rewrite");
        assert_eq!(nodes, 1);
        assert_eq!(rewritten, model(&attribute()));
    }

    /// A node that states its level already keeps it: two attributes of one
    /// name is a graph the runtime refuses.
    #[test]
    fn a_stated_level_is_left_alone() {
        let stated = string(5, &string(1, b"accuracy_level"));
        let (rewritten, nodes) = rewrite(&model(&stated)).expect("rewrite");
        assert_eq!(nodes, 0);
        assert_eq!(rewritten, model(&stated));
    }

    #[test]
    fn a_truncated_model_is_refused() {
        let mut broken = model(&[]);
        broken.truncate(broken.len() - 3);
        assert!(rewrite(&broken).is_err());
    }
}
