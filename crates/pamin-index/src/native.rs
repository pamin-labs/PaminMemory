//! Typed numerical preparation for an accelerator's FP16 reranker export.
//! Weights remain the original bytes. Large raw initializers reference the
//! source file; only graph operators and the classifier's compute types change.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::error::{IndexError, Result};
use crate::onnx::*;

const FLOAT: i64 = 1;
const HALF: i64 = 10;
const RULE: &str = "typed-reranker-v3";

pub(crate) struct Rewritten {
    pub(crate) model: Vec<u8>,
    pub(crate) norms: usize,
    pub(crate) gelus: usize,
    pub(crate) head: bool,
}

/// Build once under a process-safe lock. The graph references a single-link
/// copy of the original GPU export, as required by ONNX external-data validation.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn prepare(source: &Path, cache: &Path) -> Result<PathBuf> {
    let digest = crate::prepared::source_digest(source)?;
    let root = cache.join(RULE);
    std::fs::create_dir_all(&root)?;
    let destination = root.join(&digest);
    let model = destination.join("model.onnx");
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(root.join(format!("{digest}.lock")))?;
    lock.lock()?;
    if model.is_file() && destination.join("source.onnx").is_file() {
        return Ok(model);
    }
    let partial = root.join(format!("{digest}.partial"));
    if partial.exists() {
        std::fs::remove_dir_all(&partial)?;
    }
    std::fs::create_dir(&partial)?;
    // Read and rewrite the private copy, so its bytes and the graph's
    // external offsets agree even if the input changes during preparation.
    let written = (|| -> Result<Rewritten> {
        let bytes = snapshot(source, &partial.join("source.onnx"), &digest)?;
        let rewritten = rewrite(&bytes)
            .map_err(|error| IndexError::Engine(format!("preparing typed reranker: {error}")))?;
        std::fs::write(partial.join("model.onnx"), &rewritten.model)?;
        Ok(rewritten)
    })();
    let rewritten = match written {
        Ok(rewritten) => rewritten,
        Err(error) => {
            std::fs::remove_dir_all(&partial)?;
            return Err(error);
        }
    };
    if destination.exists() {
        std::fs::remove_dir_all(&destination)?;
    }
    std::fs::rename(&partial, &destination)?;
    tracing::info!(
        norms = rewritten.norms,
        gelus = rewritten.gelus,
        fp32_head = rewritten.head,
        "prepared typed accelerator reranker"
    );
    Ok(model)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn snapshot(source: &Path, into: &Path, expected: &str) -> Result<Vec<u8>> {
    // ONNX rejects symlinks and multiply-linked external data files.
    std::fs::copy(std::fs::canonicalize(source)?, into)?;
    let bytes = std::fs::read(into)?;
    if format!("{:x}", Sha256::digest(&bytes)) != expected {
        return Err(IndexError::Engine(
            "source changed while preparing typed reranker".into(),
        ));
    }
    Ok(bytes)
}

pub(crate) fn rewrite(model: &[u8]) -> std::result::Result<Rewritten, String> {
    let top = fields(model)?;
    let graph_field = only(&top, MODEL_GRAPH, "graph")?;
    let graph_fields = fields(graph_field.bytes()?)?;
    let node_fields: Vec<_> = graph_fields
        .iter()
        .filter(|f| f.number == GRAPH_NODE)
        .collect();
    let nodes = node_fields
        .iter()
        .map(|f| Node::parse(f))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if nodes.iter().any(|n| n.subgraph) {
        return Err("control-flow graphs are not numerical preparation candidates".into());
    }
    let initializers: HashMap<_, _> = graph_fields
        .iter()
        .filter(|f| f.number == GRAPH_INITIALIZER)
        .map(|f| {
            Ok((
                string(&fields(f.bytes()?)?, TENSOR_NAME)?.unwrap_or(""),
                f.bytes()?,
            ))
        })
        .collect::<std::result::Result<_, String>>()?;
    let mut types = HashMap::new();
    for field in graph_fields
        .iter()
        .filter(|f| matches!(f.number, GRAPH_INPUT | GRAPH_OUTPUT | GRAPH_VALUE_INFO))
    {
        let (name, ty) = value_info(field.bytes()?)?;
        types.insert(name, ty);
    }
    let outputs: HashSet<_> = graph_fields
        .iter()
        .filter(|f| f.number == GRAPH_OUTPUT)
        .map(|f| Ok(value_info(f.bytes()?)?.0))
        .collect::<std::result::Result<_, String>>()?;
    let mut users: HashMap<&str, Vec<usize>> = HashMap::new();
    for (at, node) in nodes.iter().enumerate() {
        for input in &node.inputs {
            users.entry(input).or_default().push(at);
        }
    }
    let closed = |indices: &[usize]| {
        indices[..indices.len() - 1]
            .iter()
            .flat_map(|&at| &nodes[at].outputs)
            .all(|name| {
                !outputs.contains(name)
                    && users
                        .get(name)
                        .is_some_and(|uses| uses.iter().all(|at| indices.contains(at)))
            })
    };
    let scalar = |name: &str| {
        initializers
            .get(name)
            .and_then(|raw| scalar(raw).ok().flatten())
    };
    let mut replacement: HashMap<usize, Vec<Vec<u8>>> = HashMap::new();
    let mut removed = HashSet::new();
    let mut changed_types = HashSet::new();
    let mut norms = 0;
    let mut gelus = 0;

    for start in 0..nodes.len() {
        if let Some(indices) = chain(
            start,
            &[
                "ReduceMean",
                "Sub",
                "Pow",
                "ReduceMean",
                "Add",
                "Sqrt",
                "Div",
                "Mul",
                "Add",
            ],
            &nodes,
            &users,
        ) {
            let block: Vec<_> = indices.iter().map(|&at| &nodes[at]).collect();
            let ops = [
                "ReduceMean",
                "Sub",
                "Pow",
                "ReduceMean",
                "Add",
                "Sqrt",
                "Div",
                "Mul",
                "Add",
            ];
            if block
                .iter()
                .zip(ops)
                .all(|(n, op)| n.is(op, false) && n.outputs.len() == 1)
                && block
                    .iter()
                    .enumerate()
                    .all(|(i, n)| n.inputs.len() == if matches!(i, 0 | 3 | 5) { 1 } else { 2 })
                && block[0].int("keepdims", 1) == Some(1)
                && block[3].int("keepdims", 1) == Some(1)
                && block[0].attribute("axes").is_some_and(|a| a.ints == [-1])
                && block[3].attribute("axes").is_some_and(|a| a.ints == [-1])
                && block[1].inputs == [block[0].inputs[0], block[0].outputs[0]]
                && block[2].inputs[0] == block[1].outputs[0]
                && scalar(block[2].inputs[1]) == Some(2.)
                && block[3].inputs[0] == block[2].outputs[0]
                && block[4].inputs[0] == block[3].outputs[0]
                && block[5].inputs[0] == block[4].outputs[0]
                && block[6].inputs == [block[1].outputs[0], block[5].outputs[0]]
                && block[7].inputs[0] == block[6].outputs[0]
                && block[8].inputs[0] == block[7].outputs[0]
                && initializers.contains_key(block[7].inputs[1])
                && initializers.contains_key(block[8].inputs[1])
                && matches!(types.get(block[0].inputs[0]), Some(&HALF | &FLOAT))
                && closed(&indices)
                && let Some(epsilon) =
                    scalar(block[4].inputs[1]).filter(|e| e.is_finite() && *e > 0.)
            {
                let attributes = [
                    int_attribute("axis", -1),
                    float_attribute("epsilon", epsilon),
                    int_attribute("stash_type", FLOAT),
                ];
                replacement.insert(
                    start,
                    vec![node(
                        "LayerNormalization",
                        &[block[0].inputs[0], block[7].inputs[1], block[8].inputs[1]],
                        block[8].outputs[0],
                        &attributes,
                    )],
                );
                removed.extend(indices.iter().copied());
                norms += 1;
            }
        }
        if let Some(indices) = chain(start, &["Div", "Erf", "Add", "Mul", "Mul"], &nodes, &users) {
            let block: Vec<_> = indices.iter().map(|&at| &nodes[at]).collect();
            let ops = ["Div", "Erf", "Add", "Mul", "Mul"];
            if block
                .iter()
                .zip(ops)
                .all(|(n, op)| n.is(op, false) && n.outputs.len() == 1)
                && block
                    .iter()
                    .enumerate()
                    .all(|(i, n)| n.inputs.len() == if i == 1 { 1 } else { 2 })
                && scalar(block[0].inputs[1])
                    .is_some_and(|v| (v - std::f32::consts::SQRT_2).abs() < 0.0005)
                && block[1].inputs[0] == block[0].outputs[0]
                && block[2].inputs[0] == block[1].outputs[0]
                && scalar(block[2].inputs[1]) == Some(1.)
                && block[3].inputs == [block[0].inputs[0], block[2].outputs[0]]
                && block[4].inputs[0] == block[3].outputs[0]
                && scalar(block[4].inputs[1]) == Some(0.5)
                && types.get(block[0].inputs[0]) == Some(&HALF)
                && closed(&indices)
            {
                replacement.insert(
                    start,
                    vec![node(
                        "Gelu",
                        &[block[0].inputs[0]],
                        block[4].outputs[0],
                        &[],
                    )],
                );
                removed.extend(indices.iter().copied());
                gelus += 1;
            }
        }
    }

    // Promote only a closed terminal Gemm/Tanh/Gemm classifier, never an
    // arbitrary encoder layer. Its consumers and initializer types are checked.
    let producers: HashMap<_, _> = nodes
        .iter()
        .enumerate()
        .flat_map(|(i, n)| n.outputs.iter().map(move |&o| (o, i)))
        .collect();
    let mut head = false;
    if outputs.len() == 1 {
        let output = *outputs.iter().next().expect("one output");
        let chain = (|| {
            let cast = *producers.get(output)?;
            let c = &nodes[cast];
            if !c.is("Cast", false) || c.inputs.len() != 1 || c.int("to", 0) != Some(FLOAT) {
                return None;
            }
            let last = *producers.get(c.inputs[0])?;
            let b = &nodes[last];
            if !b.is("Gemm", false) || b.inputs.len() != 3 {
                return None;
            }
            let act = *producers.get(b.inputs[0])?;
            let a = &nodes[act];
            if !a.is("Tanh", false) || a.inputs.len() != 1 {
                return None;
            }
            let first = *producers.get(a.inputs[0])?;
            let f = &nodes[first];
            if !f.is("Gemm", false) || f.inputs.len() != 3 || types.get(f.inputs[0]) != Some(&HALF)
            {
                return None;
            }
            for (at, next) in [(first, act), (act, last), (last, cast)] {
                let n = &nodes[at];
                if n.outputs.len() != 1
                    || outputs.contains(n.outputs[0])
                    || users.get(n.outputs[0]) != Some(&vec![next])
                {
                    return None;
                }
            }
            for n in [f, b] {
                for input in &n.inputs[1..] {
                    if dtype(initializers.get(input)?).ok()? != HALF {
                        return None;
                    }
                }
            }
            Some([first, act, last])
        })();
        if let Some([first, act, last]) = chain {
            let names: HashSet<_> = nodes
                .iter()
                .flat_map(|n| n.inputs.iter().chain(&n.outputs))
                .copied()
                .collect();
            for at in [first, last] {
                let n = &nodes[at];
                let mut inputs: Vec<String> = n.inputs.iter().map(|s| s.to_string()).collect();
                let mut new = Vec::new();
                for position in if at == first { 0..3 } else { 1..3 } {
                    let name = format!("pamin_typed_head/{at}/{position}");
                    if names.contains(name.as_str()) {
                        return Err("typed classifier name collision".into());
                    }
                    new.push(node(
                        "Cast",
                        &[n.inputs[position]],
                        &name,
                        &[int_attribute("to", FLOAT)],
                    ));
                    inputs[position] = name;
                }
                let parsed = fields(node_fields[at].bytes()?)?;
                let mut body = Vec::new();
                let mut input_at = 0;
                for f in parsed {
                    if f.number == NODE_INPUT {
                        put_bytes(&mut body, NODE_INPUT, inputs[input_at].as_bytes());
                        input_at += 1;
                    } else {
                        body.extend_from_slice(f.raw);
                    }
                }
                new.push(body);
                replacement.insert(at, new);
            }
            changed_types.extend([
                nodes[first].outputs[0],
                nodes[act].outputs[0],
                nodes[last].outputs[0],
            ]);
            head = true;
        }
    }
    if norms == 0 || gelus == 0 || !head {
        return Err(
            "expected FP16 normalization, GELU and terminal classifier patterns did not all match"
                .into(),
        );
    }
    if nodes.iter().enumerate().any(|(at, node)| {
        !removed.contains(&at)
            && node.op_type.starts_with("Reduce")
            && node.attribute("axes").is_some()
    }) {
        return Err("a remaining reduction needs an explicit opset migration".into());
    }
    let removed_outputs: HashSet<_> = removed
        .iter()
        .flat_map(|&at| nodes[at].outputs.iter().copied())
        .filter(|name| {
            !replacement.values().flatten().any(|raw| {
                fields(raw).is_ok_and(|fs| {
                    fs.iter()
                        .any(|f| f.number == NODE_OUTPUT && f.str() == Ok(*name))
                })
            })
        })
        .collect();
    let mut rewritten_graph = Vec::new();
    let mut at = 0;
    for field in &graph_fields {
        match field.number {
            GRAPH_NODE => {
                if let Some(new) = replacement.get(&at) {
                    for bytes in new {
                        put_bytes(&mut rewritten_graph, GRAPH_NODE, bytes);
                    }
                } else if !removed.contains(&at) {
                    rewritten_graph.extend_from_slice(field.raw);
                }
                at += 1;
            }
            GRAPH_VALUE_INFO if removed_outputs.contains(value_info(field.bytes()?)?.0) => {}
            GRAPH_VALUE_INFO if changed_types.contains(value_info(field.bytes()?)?.0) => put_bytes(
                &mut rewritten_graph,
                GRAPH_VALUE_INFO,
                &float_info(field.bytes()?)?,
            ),
            GRAPH_INITIALIZER => put_bytes(
                &mut rewritten_graph,
                GRAPH_INITIALIZER,
                &external_initializer(field.bytes()?, model)?,
            ),
            _ => rewritten_graph.extend_from_slice(field.raw),
        }
    }
    let mut out = Vec::new();
    for field in top {
        if field.number == MODEL_GRAPH {
            put_bytes(&mut out, MODEL_GRAPH, &rewritten_graph);
        } else if field.number == MODEL_OPSET_IMPORT {
            let fs = fields(field.bytes()?)?;
            if string(&fs, OPSET_DOMAIN)?.unwrap_or("").is_empty() {
                let version = fs
                    .iter()
                    .find(|f| f.number == 2)
                    .ok_or("no default opset version")?
                    .varint()?;
                if version > 20 {
                    return Err("newer opset is not a validated preparation candidate".into());
                }
                let mut import = Vec::new();
                for f in fs {
                    if f.number != 2 {
                        import.extend_from_slice(f.raw);
                    }
                }
                put_varint_field(&mut import, 2, 20);
                put_bytes(&mut out, MODEL_OPSET_IMPORT, &import);
            } else {
                out.extend_from_slice(field.raw);
            }
        } else {
            out.extend_from_slice(field.raw);
        }
    }
    Ok(Rewritten {
        model: out,
        norms,
        gelus,
        head,
    })
}

fn chain(
    start: usize,
    ops: &[&str],
    nodes: &[Node<'_>],
    users: &HashMap<&str, Vec<usize>>,
) -> Option<Vec<usize>> {
    if !nodes[start].is(ops[0], false) {
        return None;
    }
    let mut indices = vec![start];
    for op in &ops[1..] {
        let previous = *indices.last()?;
        let output = *nodes[previous].outputs.first()?;
        let candidates: Vec<_> = users
            .get(output)?
            .iter()
            .copied()
            .filter(|&at| at > previous && nodes[at].is(op, false))
            .collect();
        if candidates.len() != 1 {
            return None;
        }
        indices.push(candidates[0]);
    }
    Some(indices)
}

fn dtype(raw: &[u8]) -> std::result::Result<i64, String> {
    Ok(only(&fields(raw)?, TENSOR_DATA_TYPE, "tensor type")?.varint()? as i64)
}
fn scalar(raw: &[u8]) -> std::result::Result<Option<f32>, String> {
    let fs = fields(raw)?;
    if int64s(&fs, TENSOR_DIMS)?.iter().any(|&dim| dim != 1) {
        return Ok(None);
    }
    let Some(data) = fs.iter().find(|f| f.number == TENSOR_RAW_DATA) else {
        return Ok(None);
    };
    let bytes = data.bytes()?;
    Ok(match (dtype(raw)?, bytes.len()) {
        (HALF, 2) => Some(crate::half::decode(u16::from_le_bytes(
            bytes.try_into().expect("two"),
        ))),
        (FLOAT, 4) => Some(f32::from_le_bytes(bytes.try_into().expect("four"))),
        _ => None,
    })
}
fn node(op: &str, inputs: &[&str], output: &str, attributes: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    for input in inputs {
        put_bytes(&mut out, NODE_INPUT, input.as_bytes());
    }
    put_bytes(&mut out, NODE_OUTPUT, output.as_bytes());
    put_bytes(&mut out, NODE_OP_TYPE, op.as_bytes());
    for attribute in attributes {
        put_bytes(&mut out, NODE_ATTRIBUTE, attribute);
    }
    out
}
fn float_info(raw: &[u8]) -> std::result::Result<Vec<u8>, String> {
    fn set(raw: &[u8], path: &[u32]) -> std::result::Result<Vec<u8>, String> {
        let mut out = Vec::new();
        for f in fields(raw)? {
            if f.number == path[0] {
                if path.len() == 1 {
                    put_varint_field(&mut out, f.number, FLOAT as u64);
                } else {
                    put_bytes(&mut out, f.number, &set(f.bytes()?, &path[1..])?);
                }
            } else {
                out.extend_from_slice(f.raw);
            }
        }
        Ok(out)
    }
    set(raw, &[VALUE_INFO_TYPE, TYPE_TENSOR, TENSOR_TYPE_ELEM_TYPE])
}
fn external_initializer(raw: &[u8], model: &[u8]) -> std::result::Result<Vec<u8>, String> {
    let fs = fields(raw)?;
    if fs.iter().any(|f| f.number == 13)
        || fs
            .iter()
            .any(|f| f.number == TENSOR_DATA_LOCATION && f.varint() == Ok(1))
    {
        return Err("already external weights cannot be relocated".into());
    }
    let Some(data) = fs
        .iter()
        .find(|f| f.number == TENSOR_RAW_DATA && f.bytes().is_ok_and(|v| v.len() > 4096))
    else {
        return Ok(raw.to_vec());
    };
    let bytes = data.bytes()?;
    let offset = (bytes.as_ptr() as usize)
        .checked_sub(model.as_ptr() as usize)
        .ok_or("initializer is outside its source")?;
    if offset
        .checked_add(bytes.len())
        .is_none_or(|end| end > model.len())
    {
        return Err("initializer extends past source".into());
    }
    let mut out = Vec::new();
    for f in fs {
        if !matches!(f.number, TENSOR_RAW_DATA | TENSOR_DATA_LOCATION | 13) {
            out.extend_from_slice(f.raw);
        }
    }
    for (key, value) in [
        ("location", "source.onnx".to_string()),
        ("offset", offset.to_string()),
        ("length", bytes.len().to_string()),
    ] {
        let mut entry = Vec::new();
        put_bytes(&mut entry, 1, key.as_bytes());
        put_bytes(&mut entry, 2, value.as_bytes());
        put_bytes(&mut out, 13, &entry);
    }
    put_varint_field(&mut out, TENSOR_DATA_LOCATION, 1);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn a_changed_source_is_not_published_under_the_previous_digest() {
        let temporary = tempfile::tempdir().unwrap();
        let source = temporary.path().join("source");
        let copy = temporary.path().join("copy");
        std::fs::write(&source, b"before").unwrap();
        let expected = crate::prepared::source_digest(&source).unwrap();
        std::fs::write(&source, b"after").unwrap();
        assert!(snapshot(&source, &copy, &expected).is_err());
        let current = crate::prepared::source_digest(&source).unwrap();
        assert_eq!(snapshot(&source, &copy, &current).unwrap(), b"after");
        assert!(
            !std::fs::symlink_metadata(copy)
                .unwrap()
                .file_type()
                .is_symlink()
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn a_rejected_graph_leaves_no_partial_weight_copy() {
        let temporary = tempfile::tempdir().unwrap();
        let source = temporary.path().join("unsupported.onnx");
        std::fs::write(&source, b"not an ONNX graph").unwrap();
        let cache = temporary.path().join("cache");
        assert!(prepare(&source, &cache).is_err());
        let entries = std::fs::read_dir(cache.join(RULE)).unwrap();
        assert!(entries.into_iter().all(|entry| {
            entry
                .unwrap()
                .path()
                .extension()
                .is_none_or(|ext| ext != "partial")
        }));
    }

    #[test]
    #[ignore = "PAMIN_NATIVE_MODEL_SOURCE supplies a cached FP16 BGE export"]
    fn cached_bge_rewrites_the_complete_graph()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let source =
            PathBuf::from(std::env::var_os("PAMIN_NATIVE_MODEL_SOURCE").expect("cached source"));
        let bytes = std::fs::read(&source)?;
        let graph = rewrite(&bytes)?;
        assert_eq!((graph.norms, graph.gelus, graph.head), (49, 24, true));
        assert!(
            graph.model.len() < 1024 * 1024,
            "raw weights were copied into the prepared graph"
        );
        let temporary = tempfile::tempdir()?;
        let cache = std::env::var_os("PAMIN_NATIVE_TEST_CACHE")
            .map(PathBuf::from)
            .unwrap_or_else(|| temporary.path().to_path_buf());
        let prepared = prepare(&source, &cache)?;
        assert_eq!(std::fs::read(&prepared)?, graph.model);
        assert_eq!(
            std::fs::metadata(prepared.parent().unwrap().join("source.onnx"))?.len(),
            bytes.len() as u64
        );
        assert!(
            !std::fs::symlink_metadata(prepared.parent().unwrap().join("source.onnx"))?
                .file_type()
                .is_symlink()
        );
        assert_eq!(
            prepare(&source, &cache)?,
            prepared,
            "cache hit returned a different graph"
        );
        println!(
            "prepared {:?}, bytes {}, norms {}, GELUs {}",
            prepared,
            graph.model.len(),
            graph.norms,
            graph.gelus
        );
        Ok(())
    }
    #[test]
    fn external_weight_offsets_preserve_bytes_and_unknown_fields() {
        let payload = vec![0x5au8; 8192];
        let mut tensor = Vec::new();
        put_varint_field(&mut tensor, TENSOR_DATA_TYPE, HALF as u64);
        put_bytes(&mut tensor, TENSOR_NAME, b"weight");
        put_bytes(&mut tensor, 100, b"future metadata");
        put_bytes(&mut tensor, TENSOR_RAW_DATA, &payload);
        let mut source = Vec::new();
        put_bytes(&mut source, MODEL_GRAPH, &tensor);
        let graph = fields(&source).unwrap();
        let raw = graph[0].bytes().unwrap();
        let external = external_initializer(raw, &source).unwrap();
        let fs = fields(&external).unwrap();
        assert!(!fs.iter().any(|f| f.number == TENSOR_RAW_DATA));
        assert_eq!(
            fs.iter()
                .find(|f| f.number == 100)
                .unwrap()
                .bytes()
                .unwrap(),
            b"future metadata"
        );
        let metadata: HashMap<_, _> = fs
            .iter()
            .filter(|f| f.number == 13)
            .map(|f| {
                let entry = fields(f.bytes().unwrap()).unwrap();
                (
                    entry[0].str().unwrap().to_string(),
                    entry[1].str().unwrap().to_string(),
                )
            })
            .collect();
        let offset: usize = metadata["offset"].parse().unwrap();
        let length: usize = metadata["length"].parse().unwrap();
        assert_eq!(&source[offset..offset + length], payload);
        assert_eq!(metadata["location"], "source.onnx");
    }

    #[test]
    fn existing_external_weights_are_not_relocated() {
        let mut tensor = Vec::new();
        put_varint_field(&mut tensor, TENSOR_DATA_LOCATION, 1);
        assert!(external_initializer(&tensor, &tensor).is_err());
    }

    #[test]
    fn small_constants_stay_inline() {
        let mut scalar = Vec::new();
        put_varint_field(&mut scalar, TENSOR_DATA_TYPE, HALF as u64);
        put_bytes(
            &mut scalar,
            TENSOR_RAW_DATA,
            &crate::half::encode(2.0).to_le_bytes(),
        );
        assert_eq!(super::scalar(&scalar).unwrap(), Some(2.0));
        assert_eq!(external_initializer(&scalar, &scalar).unwrap(), scalar);
        scalar.pop();
        assert!(super::scalar(&scalar).is_err());
    }
}
