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
//! reranker and BGE-M3's (24 layers each). Both keep the two products inside
//! attention `f32`; BGE-M3's export was constant-folded and given constant
//! head-split targets so that its blocks have the reranker's shape.

use std::collections::{HashMap, HashSet};

use crate::onnx::*;

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

/// What [`trim`] did, for the log.
pub(crate) struct Trimmed {
    /// The whole model, rewritten.
    pub(crate) model: Vec<u8>,
    /// Nodes of the last layer that now run on the first row only, the
    /// classifier's `Gather` of it included.
    pub(crate) rows: usize,
    /// Fused attention nodes that no longer read the key padding mask.
    pub(crate) unmasked: usize,
}

/// A fused graph (see [`fuse`]) that scores from its first token, rewritten
/// for passes of one unpadded row: its last layer computed for the first
/// token alone, and its attention run without the key padding mask.
///
/// The classifier reads the last layer's first row and nothing else, so of
/// that layer only its keys and values need every token. The query
/// projection, the attention's queries, its output projection, both layer
/// norms and the feed-forward block run on a `Slice` of row 0 instead: one
/// before the layer's input reaches its residual, one after the quantized
/// copy of that input the three projections share, so the keys and values
/// still read it whole. Found by walking back from the `Gather` of row 0
/// through the operators that work row by row, stopping at the attention's
/// keys and values, which need every row; the walk has to cross exactly one
/// attention and end at those two tensors, or nothing is rewritten.
///
/// Without padding a mask of ones masks nothing, and the kernel takes its
/// faster path without one. So every fused attention node drops the mask, and
/// the `Cast` that made it goes. That is exact only where no row is padded:
/// the caller loads this graph only for passes of one pair (see
/// `crate::prepared::Passes`).
///
/// Neither half scores bit-identically to the graph it came from. The output
/// projection and the feed-forward block quantize their activations per
/// tensor, so over one row they quantize with that row's range rather than
/// the sequence's; and the kernel's path without a mask sums in another
/// order. `crate::prepared` checks only that the result loads and scores.
pub(crate) fn trim(model: &[u8]) -> Result<Trimmed, String> {
    let top = fields(model)?;
    let graph_field = only(&top, MODEL_GRAPH, "graph")?;
    let graph_fields = fields(graph_field.bytes()?)?;
    let nodes = graph_fields
        .iter()
        .filter(|field| field.number == GRAPH_NODE)
        .map(Node::parse)
        .collect::<Result<Vec<_>, String>>()?;
    if nodes.iter().any(|node| node.subgraph) {
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
        .collect::<Result<HashSet<_>, String>>()?;
    let graph = Graph::new(&nodes, &initializers, &inputs);
    let read_only_by = |tensor: &str, region: &HashMap<usize, &[usize]>| {
        !outputs.contains(tensor)
            && graph
                .consumers
                .get(tensor)
                .is_none_or(|readers| readers.iter().all(|at| region.contains_key(at)))
    };

    // The classifier's `Gather` of the last hidden state's first row.
    let heads: Vec<usize> = (0..nodes.len())
        .filter(|at| {
            let node = &nodes[*at];
            node.is("Gather", false)
                && node.int("axis", 0) == Some(1)
                && node.inputs.get(1).and_then(|index| graph.constant(index)) == Some(&[0][..])
                && graph
                    .producer(node.inputs[0])
                    .is_some_and(|last| last.is("SkipLayerNormalization", true))
        })
        .collect();
    let [head] = heads[..] else {
        return Err(format!(
            "expected one Gather of the first token, found {}",
            heads.len()
        ));
    };
    if graph.sole_consumer(nodes[head].inputs[0]).map(|(at, _)| at) != Some(head)
        || outputs.contains(nodes[head].inputs[0])
    {
        return Err(
            "the last hidden state is read by more than the Gather of its first token".into(),
        );
    }

    // Walk back from it, in reverse topological order. A node joins the
    // region when it works row by row and only the region reads it; a tensor
    // the region reads by row and anything else also reads is a frontier,
    // which the region reads a slice of instead.
    let row_reads = |at: usize, rows: &[usize]| -> Vec<&str> {
        rows.iter()
            .filter_map(|row| nodes[at].inputs.get(*row).copied())
            .filter(|input| !input.is_empty() && !initializers.contains_key(input))
            .collect()
    };
    let mut region: HashMap<usize, &[usize]> = HashMap::from([(head, &[0][..])]);
    let mut wanted: HashSet<&str> = row_reads(head, &[0]).into_iter().collect();
    let mut inside: HashSet<&str> = HashSet::new();
    let mut frontier: Vec<&str> = Vec::new();
    for at in (0..head).rev() {
        let node = &nodes[at];
        let mine: Vec<&str> = node
            .outputs
            .iter()
            .copied()
            .filter(|output| wanted.contains(output))
            .collect();
        if mine.is_empty() {
            continue;
        }
        let joins = row_wise(node).filter(|_| {
            mine.iter().all(|output| read_only_by(output, &region))
                && node.outputs.iter().all(|output| {
                    output.is_empty()
                        || mine.contains(output)
                        || graph.consumers.get(output).is_none_or(Vec::is_empty)
                })
                && (!(node.is("Add", false) || node.is("Mul", false))
                    || node
                        .inputs
                        .iter()
                        .any(|input| initializers.contains_key(input)))
        });
        match joins {
            Some(rows) => {
                region.insert(at, rows);
                inside.extend(&mine);
                wanted.extend(row_reads(at, rows));
            }
            None => frontier.extend(mine),
        }
    }
    if wanted.iter().any(|tensor| inputs.contains_key(tensor)) {
        return Err("the first token's path reads a graph input by row".into());
    }
    let attentions = region
        .keys()
        .filter(|at| nodes[**at].is("MultiHeadAttention", true))
        .count();
    if attentions != 1 {
        return Err(format!(
            "the first token's path crosses {attentions} attention nodes, not one"
        ));
    }
    frontier.sort_by_key(|tensor| graph.producer[tensor]);
    frontier.dedup();
    let mut kinds: Vec<&str> = frontier
        .iter()
        .map(|tensor| nodes[graph.producer[tensor]].op_type)
        .collect();
    kinds.sort_unstable();
    if kinds != ["DynamicQuantizeLinear", "SkipLayerNormalization"] {
        return Err(format!(
            "the first token's path ends at {kinds:?}, not at the layer's input and its quantized copy"
        ));
    }

    // A slice of each frontier tensor, named for it, after what produces it;
    // the region's readers of it read the slice.
    let sliced: HashMap<&str, String> = frontier
        .iter()
        .enumerate()
        .map(|(k, tensor)| (*tensor, format!("{TRIM}{k}/{tensor}")))
        .collect();
    let mut rewired: HashMap<usize, Vec<&str>> = HashMap::new();
    let mut changed = Vec::new();
    for (at, rows) in &region {
        let node = &nodes[*at];
        let mut read = node.inputs.clone();
        for row in *rows {
            if let Some(slice) = read.get(*row).and_then(|input| sliced.get(input)) {
                read[*row] = slice;
                changed.push((node.op_type, *row));
            }
        }
        if read != node.inputs {
            rewired.insert(*at, read);
        }
    }
    changed.sort_unstable();
    if changed != [("MatMulIntegerToFloat", 0), ("SkipLayerNormalization", 1)] {
        return Err(format!(
            "the slices would be read as {changed:?}, not by the query projection and the residual"
        ));
    }

    // Without the key padding mask: the inputs after the key and value go.
    let mut unmasked = 0;
    for (at, node) in nodes.iter().enumerate() {
        if !node.is("MultiHeadAttention", true) {
            continue;
        }
        let read = rewired.entry(at).or_insert_with(|| node.inputs.clone());
        if read.len() != 5 || !read[3].is_empty() || read[4] != MASK_INT32 {
            return Err(format!(
                "an attention node reads {read:?}, not its mask alone after its value"
            ));
        }
        read.truncate(3);
        unmasked += 1;
    }
    let casts: Vec<usize> = (0..nodes.len())
        .filter(|at| nodes[*at].outputs.contains(&MASK_INT32))
        .collect();
    let [cast] = casts[..] else {
        return Err(format!("nothing makes {MASK_INT32}"));
    };
    if graph.consumers.get(MASK_INT32).is_some_and(|readers| {
        readers
            .iter()
            .any(|at| !nodes[*at].is("MultiHeadAttention", true))
    }) || outputs.contains(MASK_INT32)
    {
        return Err(format!("{MASK_INT32} is read by more than attention"));
    }

    let after: HashMap<usize, Vec<Vec<u8>>> =
        frontier
            .iter()
            .enumerate()
            .fold(HashMap::new(), |mut after, (k, tensor)| {
                let slice = slice(tensor, &sliced[tensor], k);
                after.entry(graph.producer[tensor]).or_default().push(slice);
                after
            });
    let mut graph_out = Vec::with_capacity(graph_field.raw.len());
    let mut nodes_written = false;
    for field in &graph_fields {
        match field.number {
            GRAPH_NODE => {
                if nodes_written {
                    continue;
                }
                nodes_written = true;
                for (name, value) in [("starts", 0), ("ends", 1), ("axes", 1)] {
                    put_bytes(
                        &mut graph_out,
                        GRAPH_NODE,
                        &int64_constant(&format!("{TRIM}{name}"), value),
                    );
                }
                for (at, node) in nodes.iter().enumerate() {
                    match rewired.get(&at) {
                        _ if at == cast => {}
                        Some(read) => {
                            put_bytes(&mut graph_out, GRAPH_NODE, &with_inputs(node, read)?)
                        }
                        None => graph_out.extend_from_slice(node.raw),
                    }
                    for slice in after.get(&at).into_iter().flatten() {
                        put_bytes(&mut graph_out, GRAPH_NODE, slice);
                    }
                }
            }
            // Shapes recorded for a whole sequence the region now computes
            // one row of, and for the mask nothing makes any more.
            GRAPH_VALUE_INFO => {
                let name = value_info(field.bytes()?)?.0;
                if !inside.contains(name) && name != MASK_INT32 {
                    graph_out.extend_from_slice(field.raw);
                }
            }
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
    Ok(Trimmed {
        model: out,
        rows: region.len(),
        unmasked,
    })
}

/// What [`trim`] names the tensors and nodes it adds with.
const TRIM: &str = "pamin/cls1/";

/// The inputs of `node` that hold one value per token, if it computes each
/// token's from that token's alone; its weights, scales and per-tensor
/// scalars are the rest. Attention reads every token's key and value, so only
/// its query is.
fn row_wise(node: &Node<'_>) -> Option<&'static [usize]> {
    Some(match node.op_type {
        "SkipLayerNormalization" | "Add" | "Mul" => &[0, 1],
        "LayerNormalization"
        | "Gelu"
        | "BiasGelu"
        | "FastGelu"
        | "DynamicQuantizeMatMul"
        | "MatMulIntegerToFloat"
        | "MultiHeadAttention" => &[0],
        _ => return None,
    })
}

/// `Constant` node `name` of a one-element `int64` tensor, also named `name`.
fn int64_constant(name: &str, value: i64) -> Vec<u8> {
    let mut tensor = Vec::new();
    put_varint_field(&mut tensor, TENSOR_DIMS, 1);
    put_varint_field(&mut tensor, TENSOR_DATA_TYPE, INT64 as u64);
    put_bytes(&mut tensor, TENSOR_NAME, name.as_bytes());
    put_bytes(&mut tensor, TENSOR_RAW_DATA, &value.to_le_bytes());
    let mut attribute = Vec::new();
    put_bytes(&mut attribute, ATTRIBUTE_NAME, b"value");
    put_bytes(&mut attribute, ATTRIBUTE_T, &tensor);
    put_varint_field(&mut attribute, ATTRIBUTE_TYPE, ATTRIBUTE_TENSOR);
    let mut node = Vec::new();
    put_bytes(&mut node, NODE_OUTPUT, name.as_bytes());
    put_bytes(&mut node, NODE_NAME, name.as_bytes());
    put_bytes(&mut node, NODE_OP_TYPE, b"Constant");
    put_bytes(&mut node, NODE_ATTRIBUTE, &attribute);
    node
}

/// The `Slice` of row 0 of `tensor`, the `k`th, into `into`.
fn slice(tensor: &str, into: &str, k: usize) -> Vec<u8> {
    let mut node = Vec::new();
    for input in [
        tensor,
        &format!("{TRIM}starts"),
        &format!("{TRIM}ends"),
        &format!("{TRIM}axes"),
    ] {
        put_bytes(&mut node, NODE_INPUT, input.as_bytes());
    }
    put_bytes(&mut node, NODE_OUTPUT, into.as_bytes());
    put_bytes(&mut node, NODE_NAME, format!("{TRIM}slice{k}").as_bytes());
    put_bytes(&mut node, NODE_OP_TYPE, b"Slice");
    node
}

/// The graph input the mask is read from, and the `int32` copy of it the
/// fused kernel takes as its key padding mask.
const MASK: &str = "attention_mask";
const MASK_INT32: &str = "pamin/attention_mask_int32";

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

    /// The last layer of a fused graph, as `fuse` leaves the `accurate`
    /// reranker's: the mask's cast, the layer's input from the layer before,
    /// its quantized copy shared by the three projections, attention, the
    /// output projection and the feed-forward block between two skip layer
    /// norms, and the classifier's `Gather` of the first token. Its graph also
    /// returns `extra`, if given. `smoothed` puts a per-channel `Mul` by a
    /// constant before the quantized copy and before the feed-forward block's
    /// first projection, as a SmoothQuant export does.
    fn last_layer(extra: Option<&str>, smoothed: bool) -> Vec<u8> {
        let ms = MICROSOFT;
        let projection = |weight: &str, output: &str| {
            node(
                "MatMulIntegerToFloat",
                ms,
                &["xq", weight, "xs", "ws", "xz", "wz", "bias"],
                &[output],
                &[],
            )
        };
        let dense = |input: &str, weight: &str, output: &str| {
            node(
                "DynamicQuantizeMatMul",
                ms,
                &[input, weight, "ws", "wz", "bias"],
                &[output],
                &[],
            )
        };
        let norm = |input: &str, skip: &str, output: &str| {
            node(
                "SkipLayerNormalization",
                ms,
                &[input, skip, "gamma", "beta"],
                &[output],
                &[],
            )
        };
        let (quantized, expanded) = if smoothed { ("xm", "h1m") } else { ("x", "h1") };
        let mut nodes = vec![
            node(
                "Cast",
                "",
                &["attention_mask"],
                &[MASK_INT32],
                &[int_attribute("to", INT32)],
            ),
            norm("h_in", "skip_in", "x"),
            node(
                "DynamicQuantizeLinear",
                "",
                &[quantized],
                &["xq", "xs", "xz"],
                &[],
            ),
            projection("wq", "q"),
            projection("wk", "k"),
            projection("wv", "v"),
            node(
                "MultiHeadAttention",
                ms,
                &["q", "k", "v", "", MASK_INT32],
                &["attended"],
                &[int_attribute("num_heads", 2)],
            ),
            dense("attended", "wo", "o"),
            norm("o", "x", "h1"),
            dense(expanded, "w1", "f1"),
            node("Gelu", ms, &["f1"], &["f1g"], &[]),
            dense("f1g", "w2", "f2"),
            norm("f2", "h1", "h2"),
            node(
                "Gather",
                "",
                &["h2", "first"],
                &["cls"],
                &[int_attribute("axis", 1)],
            ),
            node("MatMul", "", &["cls", "wc"], &["logits"], &[]),
        ];
        if smoothed {
            nodes.insert(9, node("Mul", "", &["h1", "s1"], &["h1m"], &[]));
            nodes.insert(2, node("Mul", "", &["x", "sx"], &["xm"], &[]));
        }
        let mut graph = Vec::new();
        for node in &nodes {
            put_bytes(&mut graph, GRAPH_NODE, node);
        }
        put_bytes(
            &mut graph,
            GRAPH_INITIALIZER,
            &int64_initializer("first", &[0]),
        );
        for weight in [
            "wq", "wk", "wv", "wo", "w1", "w2", "wc", "ws", "wz", "bias", "gamma", "beta", "sx",
            "s1",
        ] {
            put_bytes(
                &mut graph,
                GRAPH_INITIALIZER,
                &int64_initializer(weight, &[1]),
            );
        }
        for (name, element) in [("attention_mask", INT64), ("h_in", 1), ("skip_in", 1)] {
            put_bytes(&mut graph, GRAPH_INPUT, &tensor_value(name, element));
        }
        for output in std::iter::once("logits").chain(extra) {
            put_bytes(&mut graph, GRAPH_OUTPUT, &tensor_value(output, 1));
        }
        for described in ["x", "q", "h1", MASK_INT32] {
            put_bytes(&mut graph, GRAPH_VALUE_INFO, &tensor_value(described, 1));
        }
        let mut opset = Vec::new();
        put_bytes(&mut opset, OPSET_DOMAIN, MICROSOFT.as_bytes());
        put_varint_field(&mut opset, 2, 1);
        let mut model = Vec::new();
        put_varint_field(&mut model, 1, 7);
        put_bytes(&mut model, MODEL_OPSET_IMPORT, &opset);
        put_bytes(&mut model, MODEL_GRAPH, &graph);
        model
    }

    /// The last layer's query, attention output, output projection, layer
    /// norms and feed-forward block read a slice of row 0 -- of the layer's
    /// input where the residual reads it, and of its quantized copy where the
    /// query projection does -- while the keys and values still read every
    /// row. The attention reads no mask, and the cast that made it is gone,
    /// as are the shapes recorded for a whole sequence the region now
    /// computes one row of.
    #[test]
    fn the_last_layer_runs_on_the_first_token_without_a_mask() {
        let trimmed = trim(&last_layer(None, false)).expect("the layer trims");
        assert_eq!((trimmed.rows, trimmed.unmasked), (9, 1));

        let nodes = graph_nodes(&trimmed.model);
        let ops: Vec<&str> = nodes.iter().map(|(op, _, _)| op.as_str()).collect();
        assert_eq!(
            ops,
            [
                "Constant",
                "Constant",
                "Constant",
                "SkipLayerNormalization",
                "Slice",
                "DynamicQuantizeLinear",
                "Slice",
                "MatMulIntegerToFloat",
                "MatMulIntegerToFloat",
                "MatMulIntegerToFloat",
                "MultiHeadAttention",
                "DynamicQuantizeMatMul",
                "SkipLayerNormalization",
                "DynamicQuantizeMatMul",
                "Gelu",
                "DynamicQuantizeMatMul",
                "SkipLayerNormalization",
                "Gather",
                "MatMul",
            ]
        );
        let constants: Vec<&str> = nodes[..3]
            .iter()
            .map(|(_, _, out)| out[0].as_str())
            .collect();
        assert_eq!(
            constants,
            ["pamin/cls1/starts", "pamin/cls1/ends", "pamin/cls1/axes"]
        );
        let slice_of = |tensor: &str, into: &str| {
            (
                "Slice".to_string(),
                vec![
                    tensor.to_string(),
                    "pamin/cls1/starts".to_string(),
                    "pamin/cls1/ends".to_string(),
                    "pamin/cls1/axes".to_string(),
                ],
                vec![into.to_string()],
            )
        };
        assert_eq!(nodes[4], slice_of("x", "pamin/cls1/0/x"));
        assert_eq!(nodes[6], slice_of("xq", "pamin/cls1/1/xq"));
        let first_input = |at: usize| nodes[at].1[0].as_str();
        assert_eq!(
            first_input(7),
            "pamin/cls1/1/xq",
            "the query reads every row"
        );
        assert_eq!(
            (first_input(8), first_input(9)),
            ("xq", "xq"),
            "keys or values lost rows"
        );
        assert_eq!(
            nodes[10].1,
            ["q", "k", "v"],
            "the attention still reads a mask"
        );
        assert_eq!(nodes[12].1, ["o", "pamin/cls1/0/x", "gamma", "beta"]);

        let top = fields(&trimmed.model).unwrap();
        let graph = fields(only(&top, MODEL_GRAPH, "graph").unwrap().bytes().unwrap()).unwrap();
        let described: Vec<&str> = graph
            .iter()
            .filter(|field| field.number == GRAPH_VALUE_INFO)
            .map(|field| value_info(field.bytes().unwrap()).unwrap().0)
            .collect();
        assert_eq!(described, ["x"]);
    }

    /// A last hidden state something besides the classifier reads -- here the
    /// graph returns it -- needs every row, and nothing is rewritten.
    #[test]
    fn a_last_layer_read_whole_is_left_alone() {
        assert!(trim(&last_layer(Some("h2"), false)).is_err());
    }

    /// A smoothed export's per-channel `Mul`s by a constant are computed per
    /// row like a bias `Add`: the feed-forward block's runs on row 0, and the
    /// one before the quantized copy stays whole, since the keys and values
    /// read every row of it.
    #[test]
    fn a_smoothed_last_layer_trims_through_its_muls() {
        let trimmed = trim(&last_layer(None, true)).expect("the layer trims");
        assert_eq!((trimmed.rows, trimmed.unmasked), (10, 1));
        let nodes = graph_nodes(&trimmed.model);
        let muls: Vec<(&[String], &[String])> = nodes
            .iter()
            .filter(|(op, _, _)| op == "Mul")
            .map(|(_, inputs, outputs)| (&inputs[..], &outputs[..]))
            .collect();
        assert_eq!(
            muls,
            [
                (
                    &["x".to_string(), "sx".to_string()][..],
                    &["xm".to_string()][..]
                ),
                (
                    &["h1".to_string(), "s1".to_string()][..],
                    &["h1m".to_string()][..]
                ),
            ]
        );
    }

    /// A graph `fuse` has not rewritten has no attention to unmask.
    #[test]
    fn an_unfused_graph_is_not_trimmed() {
        assert!(trim(&layer(-1)).is_err());
    }

    /// Bytes that are not a model are refused, never rewritten.
    #[test]
    fn a_file_that_is_not_a_model_is_refused() {
        assert!(fuse(b"this is not a protobuf").is_err());
    }
}
