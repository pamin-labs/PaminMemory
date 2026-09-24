//! Attention in a prepared copy, as one operator per layer instead of ten.
//!
//! ONNX Runtime fuses a transformer's attention into a single kernel when it
//! recognizes the pattern, and its C++ fusion only recognizes the pattern
//! over `f32` `MatMul`s. In the `accurate` reranker's int8 export the query,
//! key and value projections are quantized, so the fusion never fires and
//! each of the 24 layers runs attention as it was exported: a reshape and a
//! transpose per head input, a scaled `FusedMatMul`, the mask added, a
//! softmax, a `MatMul`, and a transpose and reshape back. The two matrix
//! products inside it are `f32` in that export -- only the projections around
//! it are quantized -- so the `com.microsoft` `MultiHeadAttention` kernel
//! computes the same thing in one node.
//!
//! This rewrites the graph a prepared copy holds, and nothing else. The
//! weights stay in the data file ONNX Runtime wrote, untouched and never
//! read: a node here names its inputs and outputs, and replacing ten nodes
//! with one changes which names are joined, not what any initializer holds.
//! So it is done on the protobuf's wire format directly, decoding only the
//! messages it has to read and copying every other byte through as it was.
//! A protobuf library would decode and re-encode the whole model to change a
//! few hundred nodes, and cost a dependency for it.
//!
//! It is only ever a candidate. `crate::prepared` loads it beside the graph it
//! came from, runs both on the same padded probe and keeps it only if every
//! output matches bit for bit -- so a pattern that looks like attention and is
//! not, or a kernel that rounds differently, costs the speed-up and nothing
//! else.
//!
//! Which exports it changes, measured rather than assumed: the `accurate`
//! reranker (24 layers). BGE-M3's int8 export quantizes the two products
//! inside attention as well, so a float kernel would not compute what it
//! computes, and the pattern below does not match it.

use std::collections::{HashMap, HashSet};

/// What a rewrite did, for the log.
pub(crate) struct Fused {
    /// The whole model, rewritten.
    pub(crate) model: Vec<u8>,
    /// Attention blocks replaced.
    pub(crate) layers: usize,
    /// Nodes in the graph before and after.
    pub(crate) nodes: (usize, usize),
}

/// The model with every attention block it recognizes fused, or why there is
/// nothing to fuse.
///
/// `Err` is not a failure of the copy: it says this graph keeps the nodes it
/// has, and why, so the caller can record it and not ask again.
pub(crate) fn fuse(model: &[u8]) -> Result<Fused, String> {
    let top = fields(model)?;
    let graph = only(&top, MODEL_GRAPH, "graph")?;
    let microsoft = top
        .iter()
        .filter(|field| field.number == MODEL_OPSET_IMPORT)
        .map(|field| Ok(string(&fields(field.bytes()?)?, OPSET_DOMAIN)?.unwrap_or("")))
        .collect::<Result<Vec<_>, String>>()?
        .contains(&MICROSOFT);
    if !microsoft {
        return Err("the graph imports no com.microsoft operators".into());
    }

    let graph_fields = fields(graph.bytes()?)?;
    let nodes = graph_fields
        .iter()
        .filter(|field| field.number == GRAPH_NODE)
        .map(Node::parse)
        .collect::<Result<Vec<_>, String>>()?;
    if nodes.iter().any(|node| node.subgraph) {
        // Dead-node removal below cannot see what a subgraph reads.
        return Err("the graph has control flow".into());
    }
    let initializers = graph_fields
        .iter()
        .filter(|field| field.number == GRAPH_INITIALIZER)
        .map(|field| initializer(field.bytes()?))
        .collect::<Result<HashMap<_, _>, String>>()?;
    let inputs = graph_fields
        .iter()
        .filter(|field| field.number == GRAPH_INPUT)
        .map(|field| value_info(field.bytes()?))
        .collect::<Result<HashMap<_, _>, String>>()?;
    let outputs = graph_fields
        .iter()
        .filter(|field| field.number == GRAPH_OUTPUT)
        .map(|field| Ok(value_info(field.bytes()?)?.0))
        .collect::<Result<Vec<_>, String>>()?;
    if inputs.get(MASK) != Some(&INT64) {
        return Err(format!("the graph has no int64 input named {MASK}"));
    }

    let graph = Graph::new(&nodes, &initializers, &inputs);
    let mut replaced: HashMap<usize, Attention> = HashMap::new();
    let mut masks = HashSet::new();
    for softmax in 0..nodes.len() {
        if let Some((at, attention, mask)) = graph.attention(softmax) {
            replaced.insert(at, attention);
            masks.insert(mask);
        }
    }
    if replaced.is_empty() {
        return Err("no attention block matched".into());
    }
    if masks.len() != 1 {
        return Err("the attention blocks do not share one mask".into());
    }
    let mask = masks.into_iter().next().expect("one mask");
    if !graph.depends_on_mask_alone(mask) {
        return Err(format!(
            "the attention mask is computed from inputs other than {MASK}"
        ));
    }
    if nodes
        .iter()
        .flat_map(|node| node.inputs.iter().chain(&node.outputs))
        .any(|name| *name == MASK_INT32)
    {
        return Err(format!("the graph already names a tensor {MASK_INT32}"));
    }

    // The new nodes, in the order they can run: the mask's cast first, since
    // it reads only a graph input, and each fused block where the reshape it
    // replaces was -- everything that reshape's chain read was produced
    // before it, and the block reads a subset of that.
    let cast = cast_mask();
    let mut candidates: Vec<Candidate> = vec![Candidate::New {
        bytes: cast,
        inputs: vec![MASK.to_string()],
        outputs: vec![MASK_INT32.to_string()],
    }];
    for (at, node) in nodes.iter().enumerate() {
        match replaced.get(&at) {
            Some(attention) => candidates.push(Candidate::New {
                bytes: attention.encode(at),
                inputs: vec![
                    attention.query.clone(),
                    attention.key.clone(),
                    attention.value.clone(),
                    MASK_INT32.to_string(),
                ],
                outputs: vec![attention.output.clone()],
            }),
            None => candidates.push(Candidate::Kept(node)),
        }
    }

    // What the fused blocks no longer read is dead: walk back from the
    // graph's outputs and keep only what reaches one.
    let mut needed: HashSet<String> = outputs.iter().map(|name| name.to_string()).collect();
    let mut keep = vec![false; candidates.len()];
    for (at, candidate) in candidates.iter().enumerate().rev() {
        if candidate.outputs().any(|output| needed.contains(output)) {
            keep[at] = true;
            needed.extend(candidate.inputs().map(str::to_string));
        }
    }
    let produced: HashSet<&str> = candidates
        .iter()
        .zip(&keep)
        .filter(|(_, kept)| **kept)
        .flat_map(|(candidate, _)| candidate.outputs())
        .collect();
    let removed: HashSet<&str> = nodes
        .iter()
        .flat_map(|node| node.outputs.iter().copied())
        .filter(|output| !produced.contains(output))
        .collect();

    let mut graph_out = Vec::with_capacity(model.len());
    let mut nodes_written = false;
    let mut kept_nodes = 0;
    for field in &graph_fields {
        match field.number {
            GRAPH_NODE => {
                if nodes_written {
                    continue;
                }
                nodes_written = true;
                for (candidate, _) in candidates.iter().zip(&keep).filter(|(_, kept)| **kept) {
                    kept_nodes += 1;
                    match candidate {
                        Candidate::Kept(node) => graph_out.extend_from_slice(node.raw),
                        Candidate::New { bytes, .. } => {
                            put_bytes(&mut graph_out, GRAPH_NODE, bytes)
                        }
                    }
                }
            }
            // Shapes recorded for tensors nothing produces any more.
            GRAPH_VALUE_INFO if removed.contains(value_info(field.bytes()?)?.0) => {}
            _ => graph_out.extend_from_slice(field.raw),
        }
    }

    let mut out = Vec::with_capacity(model.len());
    for field in &top {
        if field.number == MODEL_GRAPH {
            put_bytes(&mut out, MODEL_GRAPH, &graph_out);
        } else {
            out.extend_from_slice(field.raw);
        }
    }
    Ok(Fused {
        model: out,
        layers: replaced.len(),
        nodes: (nodes.len(), kept_nodes),
    })
}

/// The graph input the mask is read from, and the `int32` copy of it the
/// fused kernel takes as its key padding mask.
const MASK: &str = "attention_mask";
const MASK_INT32: &str = "pamin/attention_mask_int32";

const MICROSOFT: &str = "com.microsoft";

// Field numbers, from `onnx.proto`.
const MODEL_OPSET_IMPORT: u32 = 8;
const MODEL_GRAPH: u32 = 7;
const OPSET_DOMAIN: u32 = 1;
const GRAPH_NODE: u32 = 1;
const GRAPH_INITIALIZER: u32 = 5;
const GRAPH_INPUT: u32 = 11;
const GRAPH_OUTPUT: u32 = 12;
const GRAPH_VALUE_INFO: u32 = 13;
const NODE_INPUT: u32 = 1;
const NODE_OUTPUT: u32 = 2;
const NODE_NAME: u32 = 3;
const NODE_OP_TYPE: u32 = 4;
const NODE_ATTRIBUTE: u32 = 5;
const NODE_DOMAIN: u32 = 7;
const ATTRIBUTE_NAME: u32 = 1;
const ATTRIBUTE_F: u32 = 2;
const ATTRIBUTE_I: u32 = 3;
const ATTRIBUTE_G: u32 = 6;
const ATTRIBUTE_INTS: u32 = 8;
const ATTRIBUTE_GRAPHS: u32 = 11;
const ATTRIBUTE_TYPE: u32 = 20;
const TENSOR_DIMS: u32 = 1;
const TENSOR_DATA_TYPE: u32 = 2;
const TENSOR_INT64_DATA: u32 = 7;
const TENSOR_NAME: u32 = 8;
const TENSOR_RAW_DATA: u32 = 9;
const TENSOR_DATA_LOCATION: u32 = 14;
const VALUE_INFO_NAME: u32 = 1;
const VALUE_INFO_TYPE: u32 = 2;
const TYPE_TENSOR: u32 = 1;
const TENSOR_TYPE_ELEM_TYPE: u32 = 1;

// `AttributeProto.AttributeType` and `TensorProto.DataType`.
const ATTRIBUTE_FLOAT: u64 = 1;
const ATTRIBUTE_INT: u64 = 2;
const INT32: i64 = 6;
const INT64: i64 = 7;

/// One field of a message: its number, its payload, and the bytes it occupied
/// -- tag included -- so an untouched field is copied back verbatim.
struct Field<'a> {
    number: u32,
    value: Value<'a>,
    raw: &'a [u8],
}

enum Value<'a> {
    Varint(u64),
    /// Read past, never used: nothing matched here is a `double`.
    Fixed64,
    Bytes(&'a [u8]),
    Fixed32(u32),
}

impl<'a> Field<'a> {
    fn bytes(&self) -> Result<&'a [u8], String> {
        match self.value {
            Value::Bytes(bytes) => Ok(bytes),
            _ => Err(format!("field {} is not length-delimited", self.number)),
        }
    }

    fn str(&self) -> Result<&'a str, String> {
        std::str::from_utf8(self.bytes()?).map_err(|error| error.to_string())
    }

    fn varint(&self) -> Result<u64, String> {
        match self.value {
            Value::Varint(value) => Ok(value),
            _ => Err(format!("field {} is not a varint", self.number)),
        }
    }
}

/// Every field of one message, in the order they were written.
fn fields(mut bytes: &[u8]) -> Result<Vec<Field<'_>>, String> {
    let mut found = Vec::new();
    while !bytes.is_empty() {
        let start = bytes;
        let tag = varint(&mut bytes)?;
        let number = u32::try_from(tag >> 3).map_err(|_| "a field number out of range")?;
        let value = match tag & 7 {
            0 => Value::Varint(varint(&mut bytes)?),
            1 => {
                take(&mut bytes, 8)?;
                Value::Fixed64
            }
            2 => {
                let length = usize::try_from(varint(&mut bytes)?)
                    .map_err(|_| "a length out of range".to_string())?;
                Value::Bytes(take(&mut bytes, length)?)
            }
            5 => Value::Fixed32(u32::from_le_bytes(
                take(&mut bytes, 4)?.try_into().expect("4"),
            )),
            wire => return Err(format!("wire type {wire} is not one ONNX writes")),
        };
        let raw = &start[..start.len() - bytes.len()];
        found.push(Field { number, value, raw });
    }
    Ok(found)
}

fn varint(bytes: &mut &[u8]) -> Result<u64, String> {
    let mut value = 0u64;
    for shift in (0..64).step_by(7) {
        let (&byte, rest) = bytes.split_first().ok_or("a truncated varint")?;
        *bytes = rest;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err("a varint longer than ten bytes".into())
}

fn take<'a>(bytes: &mut &'a [u8], length: usize) -> Result<&'a [u8], String> {
    if bytes.len() < length {
        return Err("a truncated field".into());
    }
    let (taken, rest) = bytes.split_at(length);
    *bytes = rest;
    Ok(taken)
}

/// The single field `number` of a message, which must be there exactly once.
fn only<'f, 'a>(fields: &'f [Field<'a>], number: u32, what: &str) -> Result<&'f Field<'a>, String> {
    let mut matching = fields.iter().filter(|field| field.number == number);
    match (matching.next(), matching.next()) {
        (Some(field), None) => Ok(field),
        _ => Err(format!("expected exactly one {what}")),
    }
}

/// The last value of string field `number`, as protobuf reads a repeated
/// scalar written twice.
fn string<'a>(fields: &[Field<'a>], number: u32) -> Result<Option<&'a str>, String> {
    fields
        .iter()
        .rfind(|field| field.number == number)
        .map(Field::str)
        .transpose()
}

/// Repeated `int64`, packed or not.
fn int64s(fields: &[Field<'_>], number: u32) -> Result<Vec<i64>, String> {
    let mut values = Vec::new();
    for field in fields.iter().filter(|field| field.number == number) {
        match field.value {
            // Two's complement in ten bytes for a negative value.
            Value::Varint(value) => values.push(value as i64),
            Value::Bytes(mut packed) => {
                while !packed.is_empty() {
                    values.push(varint(&mut packed)? as i64);
                }
            }
            _ => return Err(format!("field {number} is not int64")),
        }
    }
    Ok(values)
}

/// An initializer's name, and its values if it is a small `int64` tensor held
/// in the graph -- a reshape's target shape. Anything else is only named.
fn initializer(bytes: &[u8]) -> Result<(&str, Option<Vec<i64>>), String> {
    let fields = fields(bytes)?;
    let name = string(&fields, TENSOR_NAME)?.unwrap_or("");
    let data_type = fields
        .iter()
        .rfind(|field| field.number == TENSOR_DATA_TYPE)
        .map(Field::varint)
        .transpose()?;
    let external = fields
        .iter()
        .rfind(|field| field.number == TENSOR_DATA_LOCATION)
        .map(Field::varint)
        .transpose()?
        == Some(1);
    if data_type != Some(INT64 as u64) || external {
        return Ok((name, None));
    }
    let dims = int64s(&fields, TENSOR_DIMS)?;
    let count: i64 = dims.iter().product();
    let values = match fields.iter().rfind(|field| field.number == TENSOR_RAW_DATA) {
        Some(raw) => raw
            .bytes()?
            .as_chunks::<8>()
            .0
            .iter()
            .map(|chunk| i64::from_le_bytes(*chunk))
            .collect(),
        None => int64s(&fields, TENSOR_INT64_DATA)?,
    };
    Ok((name, (values.len() as i64 == count).then_some(values)))
}

/// A graph input's or output's name, and its element type if it is a tensor.
fn value_info(bytes: &[u8]) -> Result<(&str, i64), String> {
    let fields = fields(bytes)?;
    let name = string(&fields, VALUE_INFO_NAME)?.unwrap_or("");
    let element = match last(&fields, VALUE_INFO_TYPE)? {
        Some(ty) => match last(&ty, TYPE_TENSOR)? {
            Some(tensor) => tensor
                .iter()
                .rfind(|field| field.number == TENSOR_TYPE_ELEM_TYPE)
                .map(Field::varint)
                .transpose()?
                .map_or(0, |element| element as i64),
            None => 0,
        },
        None => 0,
    };
    Ok((name, element))
}

/// The last field `number` of a message, decoded as a message itself.
fn last<'a>(fields: &[Field<'a>], number: u32) -> Result<Option<Vec<Field<'a>>>, String> {
    fields
        .iter()
        .rfind(|field| field.number == number)
        .map(fields_of)
        .transpose()
}

fn fields_of<'a>(field: &Field<'a>) -> Result<Vec<Field<'a>>, String> {
    fields(field.bytes()?)
}

/// A node, decoded as far as matching needs, with the bytes it came from.
struct Node<'a> {
    raw: &'a [u8],
    inputs: Vec<&'a str>,
    outputs: Vec<&'a str>,
    op_type: &'a str,
    domain: &'a str,
    attributes: Vec<Attribute<'a>>,
    /// Whether any attribute holds a graph, which makes this control flow.
    subgraph: bool,
}

struct Attribute<'a> {
    name: &'a str,
    float: Option<f32>,
    int: Option<i64>,
    ints: Vec<i64>,
}

impl<'a> Node<'a> {
    fn parse(field: &Field<'a>) -> Result<Self, String> {
        let fields = fields(field.bytes()?)?;
        let mut node = Node {
            raw: field.raw,
            inputs: Vec::new(),
            outputs: Vec::new(),
            op_type: "",
            domain: "",
            attributes: Vec::new(),
            subgraph: false,
        };
        for field in &fields {
            match field.number {
                NODE_INPUT => node.inputs.push(field.str()?),
                NODE_OUTPUT => node.outputs.push(field.str()?),
                NODE_OP_TYPE => node.op_type = field.str()?,
                NODE_DOMAIN => node.domain = field.str()?,
                NODE_ATTRIBUTE => {
                    let attribute = fields_of(field)?;
                    node.subgraph |= attribute
                        .iter()
                        .any(|field| matches!(field.number, ATTRIBUTE_G | ATTRIBUTE_GRAPHS));
                    node.attributes.push(Attribute {
                        name: string(&attribute, ATTRIBUTE_NAME)?.unwrap_or(""),
                        float: attribute
                            .iter()
                            .rfind(|field| field.number == ATTRIBUTE_F)
                            .and_then(|field| match field.value {
                                Value::Fixed32(bits) => Some(f32::from_bits(bits)),
                                _ => None,
                            }),
                        int: attribute
                            .iter()
                            .rfind(|field| field.number == ATTRIBUTE_I)
                            .map(|field| field.varint().map(|value| value as i64))
                            .transpose()?,
                        ints: int64s(&attribute, ATTRIBUTE_INTS)?,
                    });
                }
                _ => {}
            }
        }
        Ok(node)
    }

    /// `op_type` in the default domain, or in `com.microsoft` when `microsoft`.
    fn is(&self, op_type: &str, microsoft: bool) -> bool {
        let domain = if microsoft { MICROSOFT } else { "" };
        self.op_type == op_type
            && (self.domain == domain || (!microsoft && self.domain == "ai.onnx"))
    }

    fn attribute(&self, name: &str) -> Option<&Attribute<'a>> {
        self.attributes
            .iter()
            .find(|attribute| attribute.name == name)
    }

    /// An integer attribute, or `default` if the node does not set it.
    fn int(&self, name: &str, default: i64) -> Option<i64> {
        match self.attribute(name) {
            None => Some(default),
            Some(attribute) => attribute.int,
        }
    }
}

/// One attention block, as the fused node that replaces it.
struct Attention {
    query: String,
    key: String,
    value: String,
    output: String,
    heads: i64,
    scale: f32,
}

impl Attention {
    /// The `MultiHeadAttention` node, in `onnx.proto`'s encoding. The bias and
    /// the attention bias are absent -- the projections already added their
    /// bias -- and the mask is the key padding mask, one `int32` per token.
    fn encode(&self, at: usize) -> Vec<u8> {
        let mut node = Vec::new();
        for input in [&self.query, &self.key, &self.value, "", MASK_INT32] {
            put_bytes(&mut node, NODE_INPUT, input.as_bytes());
        }
        put_bytes(&mut node, NODE_OUTPUT, self.output.as_bytes());
        put_bytes(
            &mut node,
            NODE_NAME,
            format!("pamin/attention/{at}").as_bytes(),
        );
        put_bytes(&mut node, NODE_OP_TYPE, b"MultiHeadAttention");
        put_bytes(
            &mut node,
            NODE_ATTRIBUTE,
            &int_attribute("num_heads", self.heads),
        );
        put_bytes(
            &mut node,
            NODE_ATTRIBUTE,
            &float_attribute("scale", self.scale),
        );
        put_bytes(&mut node, NODE_DOMAIN, MICROSOFT.as_bytes());
        node
    }
}

/// The `Cast` that turns the graph's `int64` mask into the `int32` one the
/// fused kernel reads.
fn cast_mask() -> Vec<u8> {
    let mut node = Vec::new();
    put_bytes(&mut node, NODE_INPUT, MASK.as_bytes());
    put_bytes(&mut node, NODE_OUTPUT, MASK_INT32.as_bytes());
    put_bytes(&mut node, NODE_NAME, MASK_INT32.as_bytes());
    put_bytes(&mut node, NODE_OP_TYPE, b"Cast");
    put_bytes(&mut node, NODE_ATTRIBUTE, &int_attribute("to", INT32));
    node
}

fn int_attribute(name: &str, value: i64) -> Vec<u8> {
    let mut attribute = Vec::new();
    put_bytes(&mut attribute, ATTRIBUTE_NAME, name.as_bytes());
    put_varint_field(&mut attribute, ATTRIBUTE_I, value as u64);
    put_varint_field(&mut attribute, ATTRIBUTE_TYPE, ATTRIBUTE_INT);
    attribute
}

fn float_attribute(name: &str, value: f32) -> Vec<u8> {
    let mut attribute = Vec::new();
    put_bytes(&mut attribute, ATTRIBUTE_NAME, name.as_bytes());
    put_varint(&mut attribute, u64::from(ATTRIBUTE_F) << 3 | 5);
    attribute.extend_from_slice(&value.to_bits().to_le_bytes());
    put_varint_field(&mut attribute, ATTRIBUTE_TYPE, ATTRIBUTE_FLOAT);
    attribute
}

fn put_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn put_varint_field(out: &mut Vec<u8>, number: u32, value: u64) {
    put_varint(out, u64::from(number) << 3);
    put_varint(out, value);
}

fn put_bytes(out: &mut Vec<u8>, number: u32, bytes: &[u8]) {
    put_varint(out, u64::from(number) << 3 | 2);
    put_varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

/// A node of the rewritten graph: one kept as it was, or one written here.
enum Candidate<'n, 'a> {
    Kept(&'n Node<'a>),
    New {
        bytes: Vec<u8>,
        inputs: Vec<String>,
        outputs: Vec<String>,
    },
}

impl Candidate<'_, '_> {
    fn inputs(&self) -> Box<dyn Iterator<Item = &str> + '_> {
        match self {
            Self::Kept(node) => Box::new(node.inputs.iter().copied()),
            Self::New { inputs, .. } => Box::new(inputs.iter().map(String::as_str)),
        }
    }

    fn outputs(&self) -> Box<dyn Iterator<Item = &str> + '_> {
        match self {
            Self::Kept(node) => Box::new(node.outputs.iter().copied()),
            Self::New { outputs, .. } => Box::new(outputs.iter().map(String::as_str)),
        }
    }
}

/// The graph's nodes, indexed by what they produce and what reads it.
struct Graph<'n, 'a> {
    nodes: &'n [Node<'a>],
    producer: HashMap<&'a str, usize>,
    consumers: HashMap<&'a str, Vec<usize>>,
    initializers: &'n HashMap<&'a str, Option<Vec<i64>>>,
    inputs: &'n HashMap<&'a str, i64>,
}

impl<'n, 'a> Graph<'n, 'a> {
    fn new(
        nodes: &'n [Node<'a>],
        initializers: &'n HashMap<&'a str, Option<Vec<i64>>>,
        inputs: &'n HashMap<&'a str, i64>,
    ) -> Self {
        let mut producer = HashMap::new();
        let mut consumers: HashMap<&str, Vec<usize>> = HashMap::new();
        for (at, node) in nodes.iter().enumerate() {
            for output in &node.outputs {
                producer.insert(*output, at);
            }
            for input in &node.inputs {
                consumers.entry(*input).or_default().push(at);
            }
        }
        Self {
            nodes,
            producer,
            consumers,
            initializers,
            inputs,
        }
    }

    fn producer(&self, tensor: &str) -> Option<&'n Node<'a>> {
        self.producer.get(tensor).map(|at| &self.nodes[*at])
    }

    /// The one node reading `tensor`, if exactly one does and the graph does
    /// not also return it.
    fn sole_consumer(&self, tensor: &str) -> Option<(usize, &'n Node<'a>)> {
        match self.consumers.get(tensor).map(Vec::as_slice) {
            Some([at]) => Some((*at, &self.nodes[*at])),
            _ => None,
        }
    }

    fn constant(&self, tensor: &str) -> Option<&'n [i64]> {
        self.initializers.get(tensor)?.as_deref()
    }

    /// `Transpose` with exactly this permutation, and what it transposes.
    fn transpose(&self, tensor: &str, perm: [i64; 4]) -> Option<&'a str> {
        let node = self.producer(tensor)?;
        (node.is("Transpose", false) && node.attribute("perm")?.ints == perm)
            .then(|| node.inputs[0])
    }

    /// A `Reshape` splitting the last axis into heads -- target `[0, 0, H,
    /// D]` -- and what it reshapes, with `H` and `D`.
    fn split_heads(&self, tensor: &str) -> Option<(&'a str, i64, i64)> {
        let node = self.producer(tensor)?;
        if !node.is("Reshape", false) || node.int("allowzero", 0)? != 0 {
            return None;
        }
        match self.constant(node.inputs.get(1)?)? {
            [0, 0, heads, size] if *heads > 0 && *size > 0 => Some((node.inputs[0], *heads, *size)),
            _ => None,
        }
    }

    /// The attention block whose softmax is node `softmax`: where the fused
    /// node goes (the index of the reshape that merges the heads back), the
    /// node, and the mask it added.
    ///
    /// ```text
    /// q ─ Reshape[0,0,H,D] ─ Transpose(0,2,1,3) ─┐
    /// k ─ Reshape[0,0,H,D] ─ Transpose(0,2,3,1) ─┴ FusedMatMul(alpha) ─ Add(mask) ─ Softmax(-1) ─┐
    /// v ─ Reshape[0,0,H,D] ─ Transpose(0,2,1,3) ───────────────────────────────────────────────┴ MatMul
    ///   ─ Transpose(0,2,1,3) ─ Reshape[.., H*D]
    /// ```
    ///
    /// The block's own intermediates are not removed here but left for the
    /// dead-node pass, so one that something else in the graph also reads
    /// survives, still computed. The merging chain after the softmax has to
    /// have a single reader at each step, because that is how its end is
    /// found.
    fn attention(&self, softmax: usize) -> Option<(usize, Attention, &'a str)> {
        let node = &self.nodes[softmax];
        if !node.is("Softmax", false) || node.attribute("axis")?.int? != -1 {
            return None;
        }
        let masked = self.producer(node.inputs[0])?;
        if !masked.is("Add", false) || masked.inputs.len() != 2 {
            return None;
        }
        let (product, mask) = (masked.inputs[0], masked.inputs[1]);
        let fused = self.producer(product)?;
        if !fused.is("FusedMatMul", true)
            || ["transA", "transB", "transBatchA", "transBatchB"]
                .iter()
                .any(|flag| fused.int(flag, 0) != Some(0))
        {
            return None;
        }
        let scale = fused.attribute("alpha")?.float?;

        let (query, heads, size) =
            self.split_heads(self.transpose(fused.inputs[0], [0, 2, 1, 3])?)?;
        let (key, key_heads, key_size) =
            self.split_heads(self.transpose(fused.inputs[1], [0, 2, 3, 1])?)?;

        let (_, weighted) = self.sole_consumer(node.outputs[0])?;
        if !weighted.is("MatMul", false) || weighted.inputs[0] != node.outputs[0] {
            return None;
        }
        let (value, value_heads, value_size) =
            self.split_heads(self.transpose(weighted.inputs[1], [0, 2, 1, 3])?)?;
        if (key_heads, key_size) != (heads, size) || (value_heads, value_size) != (heads, size) {
            return None;
        }

        let (_, merged) = self.sole_consumer(weighted.outputs[0])?;
        if !merged.is("Transpose", false) || merged.attribute("perm")?.ints != [0, 2, 1, 3] {
            return None;
        }
        let (at, reshaped) = self.sole_consumer(merged.outputs[0])?;
        if !reshaped.is("Reshape", false) || !self.merges_heads(reshaped, heads * size) {
            return None;
        }

        Some((
            at,
            Attention {
                query: query.to_string(),
                key: key.to_string(),
                value: value.to_string(),
                output: reshaped.outputs[0].to_string(),
                heads,
                scale,
            },
            mask,
        ))
    }

    /// Whether `reshape` puts the heads back into one axis of `hidden`: a
    /// constant target ending in it, or one concatenated at run time from the
    /// batch and sequence lengths and a constant `[hidden]`.
    fn merges_heads(&self, reshape: &Node<'_>, hidden: i64) -> bool {
        let Some(target) = reshape.inputs.get(1) else {
            return false;
        };
        if let Some(values) = self.constant(target) {
            return values.len() == 3 && values[2] == hidden;
        }
        let Some(concat) = self.producer(target) else {
            return false;
        };
        concat.is("Concat", false)
            && concat.inputs.len() == 3
            && self.constant(concat.inputs[2]) == Some(&[hidden][..])
    }

    /// Whether `mask` is computed from the values of the `attention_mask`
    /// input and no other, reading other inputs at most for their shapes.
    ///
    /// The fused kernel masks by that input alone. A bias computed from
    /// anything else -- positions, a causal triangle -- would not be what it
    /// applies, and the probe in `crate::prepared` would say so; this says so
    /// before loading anything.
    fn depends_on_mask_alone(&self, mask: &str) -> bool {
        let mut seen = HashSet::new();
        let mut pending = vec![mask];
        let mut read = HashSet::new();
        while let Some(tensor) = pending.pop() {
            if !seen.insert(tensor) {
                continue;
            }
            if self.inputs.contains_key(tensor) {
                read.insert(tensor);
                continue;
            }
            if let Some(node) = self.producer(tensor) {
                if node.is("Shape", false) {
                    continue;
                }
                pending.extend(node.inputs.iter().filter(|input| !input.is_empty()));
            }
        }
        read.len() == 1 && read.contains(MASK)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A node as `onnx.proto` encodes it, for building graphs by hand.
    fn node(
        op_type: &str,
        domain: &str,
        inputs: &[&str],
        outputs: &[&str],
        attributes: &[Vec<u8>],
    ) -> Vec<u8> {
        let mut node = Vec::new();
        for input in inputs {
            put_bytes(&mut node, NODE_INPUT, input.as_bytes());
        }
        for output in outputs {
            put_bytes(&mut node, NODE_OUTPUT, output.as_bytes());
        }
        put_bytes(&mut node, NODE_NAME, outputs[0].as_bytes());
        put_bytes(&mut node, NODE_OP_TYPE, op_type.as_bytes());
        for attribute in attributes {
            put_bytes(&mut node, NODE_ATTRIBUTE, attribute);
        }
        if !domain.is_empty() {
            put_bytes(&mut node, NODE_DOMAIN, domain.as_bytes());
        }
        node
    }

    fn ints_attribute(name: &str, values: &[i64]) -> Vec<u8> {
        let mut attribute = Vec::new();
        put_bytes(&mut attribute, ATTRIBUTE_NAME, name.as_bytes());
        for value in values {
            put_varint_field(&mut attribute, ATTRIBUTE_INTS, *value as u64);
        }
        put_varint_field(&mut attribute, ATTRIBUTE_TYPE, 7);
        attribute
    }

    fn int64_initializer(name: &str, values: &[i64]) -> Vec<u8> {
        let mut tensor = Vec::new();
        put_varint_field(&mut tensor, TENSOR_DIMS, values.len() as u64);
        put_varint_field(&mut tensor, TENSOR_DATA_TYPE, INT64 as u64);
        put_bytes(&mut tensor, TENSOR_NAME, name.as_bytes());
        let raw: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        put_bytes(&mut tensor, TENSOR_RAW_DATA, &raw);
        tensor
    }

    fn tensor_value(name: &str, element: i64) -> Vec<u8> {
        let mut tensor_type = Vec::new();
        put_varint_field(&mut tensor_type, TENSOR_TYPE_ELEM_TYPE, element as u64);
        let mut ty = Vec::new();
        put_bytes(&mut ty, TYPE_TENSOR, &tensor_type);
        let mut value = Vec::new();
        put_bytes(&mut value, VALUE_INFO_NAME, name.as_bytes());
        put_bytes(&mut value, VALUE_INFO_TYPE, &ty);
        value
    }

    /// One layer of attention the way the `accurate` reranker's prepared copy
    /// holds it, two heads of four, with `softmax_axis` on its softmax.
    fn layer(softmax_axis: i64) -> Vec<u8> {
        let perm = |values: &[i64]| ints_attribute("perm", values);
        let nodes = [
            node("Cast", "", &["attention_mask"], &["mask_float"], &[]),
            node("Add", "", &["q_in", "zero"], &["q"], &[]),
            node("Add", "", &["k_in", "zero"], &["k"], &[]),
            node("Add", "", &["v_in", "zero"], &["v"], &[]),
            node("Reshape", "", &["q", "heads"], &["q_split"], &[]),
            node(
                "Transpose",
                "",
                &["q_split"],
                &["q_t"],
                &[perm(&[0, 2, 1, 3])],
            ),
            node("Reshape", "", &["k", "heads"], &["k_split"], &[]),
            node(
                "Transpose",
                "",
                &["k_split"],
                &["k_t"],
                &[perm(&[0, 2, 3, 1])],
            ),
            node(
                "FusedMatMul",
                MICROSOFT,
                &["q_t", "k_t"],
                &["scores"],
                &[float_attribute("alpha", 0.5)],
            ),
            node("Add", "", &["scores", "mask_float"], &["masked"], &[]),
            node(
                "Softmax",
                "",
                &["masked"],
                &["probs"],
                &[int_attribute("axis", softmax_axis)],
            ),
            node("Reshape", "", &["v", "heads"], &["v_split"], &[]),
            node(
                "Transpose",
                "",
                &["v_split"],
                &["v_t"],
                &[perm(&[0, 2, 1, 3])],
            ),
            node("MatMul", "", &["probs", "v_t"], &["context"], &[]),
            node(
                "Transpose",
                "",
                &["context"],
                &["context_t"],
                &[perm(&[0, 2, 1, 3])],
            ),
            node("Reshape", "", &["context_t", "hidden"], &["attended"], &[]),
            node("Add", "", &["attended", "q_in"], &["out"], &[]),
        ];
        let mut graph = Vec::new();
        for node in &nodes {
            put_bytes(&mut graph, GRAPH_NODE, node);
        }
        put_bytes(
            &mut graph,
            GRAPH_INITIALIZER,
            &int64_initializer("heads", &[0, 0, 2, 4]),
        );
        put_bytes(
            &mut graph,
            GRAPH_INITIALIZER,
            &int64_initializer("hidden", &[0, 0, 8]),
        );
        for (name, element) in [
            ("attention_mask", INT64),
            ("q_in", 1),
            ("k_in", 1),
            ("v_in", 1),
        ] {
            put_bytes(&mut graph, GRAPH_INPUT, &tensor_value(name, element));
        }
        put_bytes(&mut graph, GRAPH_OUTPUT, &tensor_value("out", 1));
        put_bytes(&mut graph, GRAPH_VALUE_INFO, &tensor_value("probs", 1));
        put_bytes(&mut graph, GRAPH_VALUE_INFO, &tensor_value("q", 1));

        let mut opset = Vec::new();
        put_bytes(&mut opset, OPSET_DOMAIN, MICROSOFT.as_bytes());
        put_varint_field(&mut opset, 2, 1);
        let mut model = Vec::new();
        put_varint_field(&mut model, 1, 7);
        put_bytes(&mut model, MODEL_OPSET_IMPORT, &opset);
        put_bytes(&mut model, MODEL_GRAPH, &graph);
        model
    }

    fn graph_nodes(model: &[u8]) -> Vec<(String, Vec<String>, Vec<String>)> {
        let top = fields(model).expect("the model");
        let graph = fields(
            only(&top, MODEL_GRAPH, "graph")
                .expect("a graph")
                .bytes()
                .unwrap(),
        )
        .expect("the graph");
        graph
            .iter()
            .filter(|field| field.number == GRAPH_NODE)
            .map(|field| {
                let node = Node::parse(field).expect("a node");
                let names = |names: &[&str]| names.iter().map(|name| name.to_string()).collect();
                (
                    node.op_type.to_string(),
                    names(&node.inputs),
                    names(&node.outputs),
                )
            })
            .collect()
    }

    /// The block becomes one node reading the projections and the mask, and
    /// everything only it read is gone -- the mask's float cast included --
    /// while the rest of the model is copied through byte for byte.
    #[test]
    fn a_layer_of_attention_becomes_one_node() {
        let model = layer(-1);
        let fused = fuse(&model).expect("the layer fuses");
        assert_eq!(fused.layers, 1);

        let nodes = graph_nodes(&fused.model);
        let ops: Vec<&str> = nodes.iter().map(|(op, _, _)| op.as_str()).collect();
        assert_eq!(
            ops,
            ["Cast", "Add", "Add", "Add", "MultiHeadAttention", "Add"]
        );
        let (_, inputs, outputs) = &nodes[4];
        assert_eq!(inputs, &["q", "k", "v", "", MASK_INT32]);
        assert_eq!(outputs, &["attended"]);
        assert_eq!(fused.nodes, (17, 6));

        let top = fields(&fused.model).expect("the rewritten model");
        let original = fields(&model).expect("the model");
        for (one, other) in top.iter().zip(&original) {
            if one.number != MODEL_GRAPH {
                assert_eq!(one.raw, other.raw, "field {} changed", one.number);
            }
        }
        let graph = fields(only(&top, MODEL_GRAPH, "graph").unwrap().bytes().unwrap()).unwrap();
        let described: Vec<&str> = graph
            .iter()
            .filter(|field| field.number == GRAPH_VALUE_INFO)
            .map(|field| value_info(field.bytes().unwrap()).unwrap().0)
            .collect();
        assert_eq!(described, ["q"], "the removed tensor's shape was kept");

        let attention = Node::parse(
            fields(only(&top, MODEL_GRAPH, "graph").unwrap().bytes().unwrap())
                .unwrap()
                .iter()
                .filter(|field| field.number == GRAPH_NODE)
                .nth(4)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(attention.domain, MICROSOFT);
        assert_eq!(attention.attribute("num_heads").unwrap().int, Some(2));
        assert_eq!(attention.attribute("scale").unwrap().float, Some(0.5));
    }

    /// Softmax over another axis is not attention, and nothing is rewritten.
    #[test]
    fn a_softmax_over_another_axis_is_left_alone() {
        assert_eq!(
            fuse(&layer(1)).err().as_deref(),
            Some("no attention block matched")
        );
    }

    /// Bytes that are not a model are refused, never rewritten.
    #[test]
    fn a_file_that_is_not_a_model_is_refused() {
        assert!(fuse(b"this is not a protobuf").is_err());
    }
}
