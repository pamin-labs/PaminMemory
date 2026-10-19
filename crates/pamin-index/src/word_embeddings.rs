//! Word embeddings in a prepared copy, read by this process instead of mapped
//! by the runtime.
//!
//! Both int8 exports the CPU loads -- the `accurate` reranker and BGE-M3 --
//! start the same way: the token ids gather rows of a quantized table,
//! `word_embeddings.weight_quantized`, 250,002 rows of 1,024 `uint8`, and a
//! `DequantizeLinear` with one scale and one zero point for the whole table
//! turns them into `f32`. That table is 244 MiB of the copy's data file, and
//! ONNX Runtime maps it as one initializer. A gather reads one 1 KiB row a
//! token, but the kernel maps in the 64 KiB around every page it faults, and
//! `MADV_RANDOM` does not stop it, so the few thousand rows of a search leave
//! most of the table resident. Estimated from the mapped files of a search
//! server, not measured per mapping: about 210 of the 244 MiB in the
//! reranker, and likely as much in the embedder after an ingest.
//!
//! So a prepared copy gets a graph without those two nodes. What the
//! `DequantizeLinear` produced becomes a graph input, under the same name, so
//! nothing that read it changes; and [`WordTable`] fills it. It reads only the
//! rows a batch names, with positional reads of the data file the copy already
//! has, at the offset the initializer recorded, and dequantizes them. The
//! runtime never maps the table, and the process holds the rows of the batch
//! it is running.
//!
//! Dequantized as `DequantizeLinear` defines it for one scale and one zero
//! point: the difference from the zero point, converted to `f32`, times the
//! scale. For 8-bit values the difference is an integer of at most nine bits,
//! exact in `f32` however it is computed, so the multiply is the only
//! rounding, and Rust never fuses it into anything. That this is what the
//! runtime's kernel computes is not argued here but checked: like the
//! attention fusion (`crate::attention`), the split graph is only a candidate,
//! which `crate::prepared` keeps only if it scores a probe bit for bit as the
//! graph it came from does -- the probe's word embeddings read through
//! [`WordTable`], so the check covers this code as well as the rewrite.
//!
//! The graph is rewritten on the protobuf's wire format, as the fusion is
//! (see `crate::onnx`), and where the table is goes in a small file beside the
//! split graph ([`description`]), because the graph no longer names the
//! initializer that said so.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::{Path, PathBuf};

use crate::error::{IndexError, Result};
use crate::onnx::*;

/// The graph input the table is gathered by, which the encoder feeds and the
/// table is read for.
const IDS: &str = "input_ids";

/// What a rewrite produced.
pub(crate) struct Split {
    /// The whole model, rewritten.
    pub(crate) model: Vec<u8>,
    /// Where the table it no longer holds is, for [`description`].
    pub(crate) table: Table,
}

/// Where a split graph's word embeddings are and how to dequantize them, as
/// the file beside the graph records it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Table {
    /// The graph input the rows are fed to: the name the `DequantizeLinear`'s
    /// output had.
    pub(crate) input: String,
    /// The data file, beside the graph, as the initializer named it.
    pub(crate) data: String,
    /// Where the table starts in it, in bytes.
    pub(crate) offset: u64,
    pub(crate) rows: usize,
    /// Values a row, one byte each.
    pub(crate) width: usize,
    /// `int8` rather than `uint8`.
    pub(crate) signed: bool,
    pub(crate) scale: f32,
    pub(crate) zero_point: i32,
}

impl Table {
    /// One line each, the scale as its bits so it reads back exactly.
    pub(crate) fn encode(&self) -> String {
        format!(
            "input {}\ndata {}\noffset {}\nrows {}\nwidth {}\nsigned {}\nscale {:08x}\nzero_point {}\n",
            self.input,
            self.data,
            self.offset,
            self.rows,
            self.width,
            self.signed,
            self.scale.to_bits(),
            self.zero_point
        )
    }

    pub(crate) fn parse(text: &str) -> Option<Self> {
        let line = |prefix: &str| text.lines().find_map(|line| line.strip_prefix(prefix));
        let table = Self {
            input: line("input ")?.to_string(),
            data: line("data ")?.to_string(),
            offset: line("offset ")?.parse().ok()?,
            rows: line("rows ")?.parse().ok()?,
            width: line("width ")?.parse().ok()?,
            signed: line("signed ")?.parse().ok()?,
            scale: f32::from_bits(u32::from_str_radix(line("scale ")?, 16).ok()?),
            zero_point: line("zero_point ")?.parse().ok()?,
        };
        (plain(&table.data) && table.rows > 0 && table.width > 0).then_some(table)
    }

    /// Where the table ends in the data file.
    fn end(&self) -> Option<u64> {
        (self.rows as u64)
            .checked_mul(self.width as u64)?
            .checked_add(self.offset)
    }
}

/// The file beside a split graph that says where its table is:
/// `model.split1.onnx` has `model.split1.table`.
pub(crate) fn description(graph: &Path) -> PathBuf {
    graph.with_extension("table")
}

/// Whether `name` is a file name alone, so a data file is read from beside the
/// graph and from nowhere else.
fn plain(name: &str) -> bool {
    !name.is_empty() && Path::new(name).file_name() == Some(std::ffi::OsStr::new(name))
}

/// The model without its word-embedding gather and dequantize, the rows they
/// produced taken as an input instead, or why there is nothing to split.
///
/// The pattern, found by structure rather than by name:
///
/// ```text
/// input_ids ─┐
/// table ─────┴ Gather(axis 0) ─ DequantizeLinear(scale, zero point) ─ ...
/// ```
///
/// where the table is a two-dimensional `uint8` or `int8` initializer held in
/// a data file, read by the gather alone; the scale is one `f32` and the zero
/// point one value of the table's type, or absent; and neither the gather's
/// output nor the dequantize's is anything else's -- the dequantize the
/// gather's only reader, neither a graph output. Exactly one such gather, or
/// none is split.
///
/// `Err` is not a failure of the copy: it says this graph keeps the nodes it
/// has, and why: the EmbeddingGemma export, which scales each row on its
/// own, and any graph that gathers no quantized table.
pub(crate) fn split(model: &[u8]) -> std::result::Result<Split, String> {
    let top = fields(model)?;
    let graph = only(&top, MODEL_GRAPH, "graph")?;
    let graph_fields = fields(graph.bytes()?)?;
    let nodes = graph_fields
        .iter()
        .filter(|field| field.number == GRAPH_NODE)
        .map(Node::parse)
        .collect::<std::result::Result<Vec<_>, String>>()?;
    if nodes.iter().any(|node| node.subgraph) {
        // A subgraph reads the graph's tensors by name, unseen here.
        return Err("the graph has control flow".into());
    }
    let tensors = graph_fields
        .iter()
        .filter(|field| field.number == GRAPH_INITIALIZER)
        .map(|field| Tensor::parse(field.bytes()?))
        .collect::<std::result::Result<Vec<_>, String>>()?;
    let tensors: HashMap<&str, &Tensor> =
        tensors.iter().map(|tensor| (tensor.name, tensor)).collect();
    let inputs = graph_fields
        .iter()
        .filter(|field| field.number == GRAPH_INPUT)
        .map(|field| Ok((value_info(field.bytes()?)?, field.bytes()?)))
        .collect::<std::result::Result<Vec<_>, String>>()?;
    let outputs = graph_fields
        .iter()
        .filter(|field| field.number == GRAPH_OUTPUT)
        .map(|field| Ok(value_info(field.bytes()?)?.0))
        .collect::<std::result::Result<HashSet<_>, String>>()?;
    let Some(ids) = inputs
        .iter()
        .find(|((name, element), _)| *name == IDS && *element == INT64)
        .map(|(_, bytes)| *bytes)
    else {
        return Err(format!("the graph has no int64 input named {IDS}"));
    };
    let mut consumers: HashMap<&str, Vec<usize>> = HashMap::new();
    for (at, node) in nodes.iter().enumerate() {
        for input in &node.inputs {
            consumers.entry(*input).or_default().push(at);
        }
    }
    let sole = |tensor: &str, reader: usize| matches!(consumers.get(tensor).map(Vec::as_slice), Some([only]) if *only == reader);

    let matched = |gather: usize| -> Option<Found> {
        let node = &nodes[gather];
        if !node.is("Gather", false)
            || node.int("axis", 0)? != 0
            || node.inputs.len() != 2
            || node.inputs[1] != IDS
            || node.outputs.len() != 1
        {
            return None;
        }
        let weights = *tensors.get(node.inputs[0])?;
        let signed = match weights.data_type {
            UINT8 => false,
            INT8 => true,
            _ => return None,
        };
        if weights.dims.len() != 2 || weights.external.is_none() || !sole(weights.name, gather) {
            return None;
        }
        let gathered = node.outputs[0];
        let dequantize = *consumers.get(gathered)?.as_slice().first()?;
        let dequantizing = &nodes[dequantize];
        if !sole(gathered, dequantize)
            || outputs.contains(gathered)
            || !dequantizing.is("DequantizeLinear", false)
            || dequantizing.inputs.first() != Some(&gathered)
            || !matches!(dequantizing.int("output_dtype", 0)?, 0 | FLOAT)
            || dequantizing.outputs.len() != 1
            || outputs.contains(dequantizing.outputs[0])
        {
            return None;
        }
        let scale = *dequantizing
            .inputs
            .get(1)
            .and_then(|name| tensors.get(name))?;
        let zero_point = match dequantizing.inputs.get(2).filter(|name| !name.is_empty()) {
            Some(name) => Some(*tensors.get(name)?),
            None => None,
        };
        Some(Found {
            gather,
            dequantize,
            weights,
            signed,
            scale: (scale.name, scale.scalar_float()?),
            zero_point: match zero_point {
                Some(tensor) if tensor.data_type == weights.data_type => {
                    (Some(tensor.name), tensor.scalar_byte()?)
                }
                Some(_) => return None,
                None => (None, 0),
            },
            input: dequantizing.outputs[0],
        })
    };
    let found: Vec<Found> = (0..nodes.len()).filter_map(matched).collect();
    let found = match found.as_slice() {
        [one] => one,
        [] => return Err(format!("no gather of a quantized table by {IDS} matched")),
        _ => {
            return Err(format!(
                "more than one quantized table is gathered by {IDS}"
            ));
        }
    };
    let table = found.table()?;

    // The new input's shape is the ids' with the row width after it, so the
    // batch and sequence lengths stay the symbols the rest of the graph names.
    let ids_dims: Vec<Vec<u8>> = match shape(ids)? {
        Some(dims) if dims.len() == 2 => dims.into_iter().map(<[u8]>::to_vec).collect(),
        Some(_) => return Err(format!("{IDS} is not two-dimensional")),
        None => vec![symbolic("batch"), symbolic("sequence")],
    };
    let input = float_input(found.input, &ids_dims, table.width);

    // The table, and the scale and zero point if only the dequantize read
    // them; and the shapes recorded for the two tensors no node produces now.
    let mut removed_tensors: HashSet<&str> = HashSet::from([found.weights.name]);
    for name in [Some(found.scale.0), found.zero_point.0]
        .into_iter()
        .flatten()
    {
        if sole(name, found.dequantize) {
            removed_tensors.insert(name);
        }
    }
    let removed_values: HashSet<&str> =
        HashSet::from([nodes[found.gather].outputs[0], found.input]);
    let last_input = graph_fields
        .iter()
        .rposition(|field| field.number == GRAPH_INPUT);

    let mut graph_out = Vec::with_capacity(graph.bytes()?.len());
    let mut node = 0;
    for (at, field) in graph_fields.iter().enumerate() {
        match field.number {
            GRAPH_NODE => {
                if node != found.gather && node != found.dequantize {
                    graph_out.extend_from_slice(field.raw);
                }
                node += 1;
            }
            GRAPH_INITIALIZER
                if removed_tensors
                    .contains(string(&fields(field.bytes()?)?, TENSOR_NAME)?.unwrap_or("")) => {}
            // An older IR lists every initializer among the inputs as well.
            GRAPH_INPUT if removed_tensors.contains(value_info(field.bytes()?)?.0) => {}
            GRAPH_VALUE_INFO if removed_values.contains(value_info(field.bytes()?)?.0) => {}
            _ => graph_out.extend_from_slice(field.raw),
        }
        if Some(at) == last_input {
            put_bytes(&mut graph_out, GRAPH_INPUT, &input);
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
    Ok(Split { model: out, table })
}

/// The gather and dequantize [`split`] takes out, and what it reads of them.
struct Found<'t> {
    gather: usize,
    dequantize: usize,
    weights: &'t Tensor<'t>,
    signed: bool,
    /// Each with the initializer's name, the zero point's absent if the node
    /// leaves it out.
    scale: (&'t str, f32),
    zero_point: (Option<&'t str>, i32),
    /// The dequantize's output, which becomes the graph input.
    input: &'t str,
}

impl Found<'_> {
    /// Where the table is, from its initializer's `external_data` entries.
    fn table(&self) -> std::result::Result<Table, String> {
        let entries = self.weights.external.as_deref().unwrap_or_default();
        let entry = |key: &str| {
            entries
                .iter()
                .rfind(|(named, _)| *named == key)
                .map(|(_, value)| *value)
        };
        let data = entry("location").unwrap_or("");
        if !plain(data) {
            return Err(format!(
                "the table's data file {data:?} is not a file beside the graph"
            ));
        }
        let number = |key: &str| {
            entry(key)
                .map(|value| {
                    value
                        .parse::<u64>()
                        .map_err(|_| format!("the table's {key} {value:?} is not a number"))
                })
                .transpose()
        };
        let [rows, width] = &self.weights.dims[..] else {
            unreachable!("matched as two-dimensional")
        };
        let (Ok(rows), Ok(width)) = (usize::try_from(*rows), usize::try_from(*width)) else {
            return Err("the table has a negative dimension".into());
        };
        if rows == 0 || width == 0 || self.input.contains('\n') {
            return Err("the table is empty, or its input's name spans lines".into());
        }
        let table = Table {
            input: self.input.to_string(),
            data: data.to_string(),
            offset: number("offset")?.unwrap_or(0),
            rows,
            width,
            signed: self.signed,
            scale: self.scale.1,
            zero_point: self.zero_point.1,
        };
        let length = (rows as u64).checked_mul(width as u64);
        if let Some(recorded) = number("length")?
            && Some(recorded) != length
        {
            return Err(format!(
                "the table's recorded length {recorded} is not {rows} rows of {width}"
            ));
        }
        table.end().ok_or("the table's end overflows")?;
        Ok(table)
    }
}

/// An initializer, decoded as far as [`split`] reads one.
struct Tensor<'a> {
    name: &'a str,
    data_type: i64,
    dims: Vec<i64>,
    /// Its `external_data` entries, if its values are in a data file.
    external: Option<Vec<(&'a str, &'a str)>>,
    raw: Option<&'a [u8]>,
    floats: Vec<f32>,
    /// `int32_data`, which holds an 8-bit tensor's values when `raw_data`
    /// does not.
    ints: Vec<i64>,
}

impl<'a> Tensor<'a> {
    fn parse(bytes: &'a [u8]) -> std::result::Result<Self, String> {
        let fields = fields(bytes)?;
        let varint = |number: u32| {
            fields
                .iter()
                .rfind(|field| field.number == number)
                .map(Field::varint)
                .transpose()
        };
        let external = varint(TENSOR_DATA_LOCATION)? == Some(1);
        let mut floats = Vec::new();
        for field in fields
            .iter()
            .filter(|field| field.number == TENSOR_FLOAT_DATA)
        {
            match field.value {
                Value::Fixed32(bits) => floats.push(f32::from_bits(bits)),
                Value::Bytes(packed) => floats.extend(
                    packed
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|chunk| f32::from_le_bytes(*chunk)),
                ),
                _ => return Err("float_data is not float".into()),
            }
        }
        Ok(Self {
            name: string(&fields, TENSOR_NAME)?.unwrap_or(""),
            data_type: varint(TENSOR_DATA_TYPE)?.map_or(0, |value| value as i64),
            dims: int64s(&fields, TENSOR_DIMS)?,
            external: if external {
                Some(
                    fields
                        .iter()
                        .filter(|field| field.number == TENSOR_EXTERNAL_DATA)
                        .map(|field| {
                            let entry = fields_of(field)?;
                            Ok((
                                string(&entry, ENTRY_KEY)?.unwrap_or(""),
                                string(&entry, ENTRY_VALUE)?.unwrap_or(""),
                            ))
                        })
                        .collect::<std::result::Result<Vec<_>, String>>()?,
                )
            } else {
                None
            },
            raw: fields
                .iter()
                .rfind(|field| field.number == TENSOR_RAW_DATA)
                .map(Field::bytes)
                .transpose()?,
            floats,
            ints: int64s(&fields, TENSOR_INT32_DATA)?,
        })
    }

    /// One value, held in the graph: a scalar, or a tensor of one.
    fn single(&self) -> bool {
        self.external.is_none() && self.dims.len() <= 1 && self.dims.iter().product::<i64>() == 1
    }

    /// The value of a single `f32`.
    fn scalar_float(&self) -> Option<f32> {
        if self.data_type != FLOAT || !self.single() {
            return None;
        }
        match self.raw {
            Some(raw) => Some(f32::from_le_bytes(raw.try_into().ok()?)),
            None => match self.floats[..] {
                [value] => Some(value),
                _ => None,
            },
        }
    }

    /// The value of a single `uint8` or `int8`.
    fn scalar_byte(&self) -> Option<i32> {
        if !self.single() {
            return None;
        }
        let signed = match self.data_type {
            UINT8 => false,
            INT8 => true,
            _ => return None,
        };
        match (self.raw, &self.ints[..]) {
            (Some(&[byte]), _) if signed => Some(i32::from(byte as i8)),
            (Some(&[byte]), _) => Some(i32::from(byte)),
            (None, [value]) => i32::try_from(*value).ok(),
            _ => None,
        }
    }
}

/// The `dim` fields of a graph input's shape, each as written, if it records
/// one.
fn shape(input: &[u8]) -> std::result::Result<Option<Vec<&[u8]>>, String> {
    let fields = fields(input)?;
    let Some(ty) = last(&fields, VALUE_INFO_TYPE)? else {
        return Ok(None);
    };
    let Some(tensor) = last(&ty, TYPE_TENSOR)? else {
        return Ok(None);
    };
    let Some(shape) = last(&tensor, TENSOR_TYPE_SHAPE)? else {
        return Ok(None);
    };
    Ok(Some(
        shape
            .iter()
            .filter(|field| field.number == SHAPE_DIM)
            .map(|field| field.raw)
            .collect(),
    ))
}

/// A `dim` field naming a symbol, for an ids input that records no shape.
fn symbolic(name: &str) -> Vec<u8> {
    let mut dim = Vec::new();
    put_bytes(&mut dim, DIM_PARAM, name.as_bytes());
    let mut field = Vec::new();
    put_bytes(&mut field, SHAPE_DIM, &dim);
    field
}

/// The graph input the rows are fed to: `f32`, the ids' `dims` and then
/// `width`.
fn float_input(name: &str, dims: &[Vec<u8>], width: usize) -> Vec<u8> {
    let mut shape = Vec::new();
    for dim in dims {
        shape.extend_from_slice(dim);
    }
    let mut last = Vec::new();
    put_varint_field(&mut last, DIM_VALUE, width as u64);
    put_bytes(&mut shape, SHAPE_DIM, &last);
    let mut tensor_type = Vec::new();
    put_varint_field(&mut tensor_type, TENSOR_TYPE_ELEM_TYPE, FLOAT as u64);
    put_bytes(&mut tensor_type, TENSOR_TYPE_SHAPE, &shape);
    let mut ty = Vec::new();
    put_bytes(&mut ty, TYPE_TENSOR, &tensor_type);
    let mut value = Vec::new();
    put_bytes(&mut value, VALUE_INFO_NAME, name.as_bytes());
    put_bytes(&mut value, VALUE_INFO_TYPE, &ty);
    value
}

/// A split graph's table, open for reading rows: what the encoder feeds the
/// graph's word-embedding input from, shared by every session over that
/// graph.
pub(crate) struct WordTable {
    file: File,
    table: Table,
}

impl WordTable {
    /// The table of the split graph at `graph`, from the file [`description`]
    /// names; `None` if there is no such file, as for any graph that is not a
    /// split one.
    ///
    /// The data file is opened once and kept open, and checked to hold the
    /// whole table, so a row read later fails only if the file changes under
    /// it.
    pub(crate) fn beside(graph: &Path) -> Result<Option<Self>> {
        let described = description(graph);
        let text = match std::fs::read_to_string(&described) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let table = Table::parse(&text).ok_or_else(|| {
            IndexError::Engine(format!(
                "{} does not describe a word-embedding table",
                described.display()
            ))
        })?;
        let data = graph.with_file_name(&table.data);
        let file = File::open(&data).map_err(|error| {
            IndexError::Engine(format!(
                "opening the word-embedding table in {}: {error}",
                data.display()
            ))
        })?;
        let length = file.metadata()?.len();
        if table.end().is_none_or(|end| end > length) {
            return Err(IndexError::Engine(format!(
                "{} is {length} bytes, too short for the word-embedding table {} describes",
                data.display(),
                described.display()
            )));
        }
        Ok(Some(Self { file, table }))
    }

    /// The graph input [`embed`](Self::embed)'s rows are fed to.
    pub(crate) fn input(&self) -> &str {
        &self.table.input
    }

    /// Values a row.
    pub(crate) fn width(&self) -> usize {
        self.table.width
    }

    /// The row each of `ids` names, dequantized, one after another:
    /// `ids.len()` rows of [`width`](Self::width) values, in the layout of a
    /// `[batch, sequence, width]` tensor when `ids` is a `[batch, sequence]`
    /// one.
    ///
    /// Each distinct id is read from the file once -- padding repeats one id
    /// down every short row -- and only the rows named are read. An id outside
    /// the table is an error, as it is for the gather this replaces; so is a
    /// negative one, which ONNX's `Gather` would count from the end and no
    /// tokenizer produces.
    pub(crate) fn embed(&self, ids: &[i64]) -> Result<Vec<f32>> {
        let width = self.table.width;
        let mut rows = vec![0f32; ids.len() * width];
        let mut first: HashMap<i64, usize> = HashMap::new();
        let mut quantized = vec![0u8; width];
        for (at, &id) in ids.iter().enumerate() {
            let into = at * width;
            if let Some(&seen) = first.get(&id) {
                rows.copy_within(seen * width..(seen + 1) * width, into);
                continue;
            }
            let row = usize::try_from(id)
                .ok()
                .filter(|row| *row < self.table.rows)
                .ok_or_else(|| {
                    IndexError::Engine(format!(
                        "token id {id} is outside the word-embedding table's {} rows",
                        self.table.rows
                    ))
                })?;
            read_at(
                &self.file,
                &mut quantized,
                self.table.offset + row as u64 * width as u64,
            )?;
            dequantize(
                &quantized,
                self.table.signed,
                self.table.zero_point,
                self.table.scale,
                &mut rows[into..into + width],
            );
            first.insert(id, at);
        }
        Ok(rows)
    }
}

/// `quantized` dequantized into `into`: `(q - zero_point) * scale`, the
/// difference in `i32` and so exact, converted to `f32` exactly, and rounded
/// once by the multiply.
///
/// Element by element with nothing carried from one to the next, so the
/// compiler vectorizes it for the SIMD the target has -- a widening, an
/// integer-to-float conversion and a multiply a lane -- and it has no kernel
/// of its own, as `crate::vector_math`'s reduction has: a hand-written one
/// could compute nothing else, since every lane's bits are fixed. Not timed
/// against one; a row is 1,024 values. Rust contracts no multiply into a
/// fused one, so the bits are the ones written here.
fn dequantize(quantized: &[u8], signed: bool, zero_point: i32, scale: f32, into: &mut [f32]) {
    if signed {
        for (value, &q) in into.iter_mut().zip(quantized) {
            *value = (i32::from(q as i8) - zero_point) as f32 * scale;
        }
    } else {
        for (value, &q) in into.iter_mut().zip(quantized) {
            *value = (i32::from(q) - zero_point) as f32 * scale;
        }
    }
}

/// Fills `buffer` from `file` at `offset`, without moving a cursor another
/// thread's read would share.
#[cfg(unix)]
fn read_at(file: &File, buffer: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buffer, offset)
}

/// Fills `buffer` from `file` at `offset`. Windows has no `read_exact_at`;
/// `seek_read` moves the handle's cursor, which nothing here reads.
#[cfg(windows)]
fn read_at(file: &File, mut buffer: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buffer.is_empty() {
        match file.seek_read(buffer, offset) {
            Ok(0) => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            Ok(read) => {
                buffer = &mut buffer[read..];
                offset += read as u64;
            }
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(op_type: &str, inputs: &[&str], outputs: &[&str]) -> Vec<u8> {
        let mut node = Vec::new();
        for input in inputs {
            put_bytes(&mut node, NODE_INPUT, input.as_bytes());
        }
        for output in outputs {
            put_bytes(&mut node, NODE_OUTPUT, output.as_bytes());
        }
        put_bytes(&mut node, NODE_NAME, outputs[0].as_bytes());
        put_bytes(&mut node, NODE_OP_TYPE, op_type.as_bytes());
        node
    }

    fn tensor(name: &str, data_type: i64, dims: &[i64], raw: &[u8]) -> Vec<u8> {
        let mut tensor = Vec::new();
        for dim in dims {
            put_varint_field(&mut tensor, TENSOR_DIMS, *dim as u64);
        }
        put_varint_field(&mut tensor, TENSOR_DATA_TYPE, data_type as u64);
        put_bytes(&mut tensor, TENSOR_NAME, name.as_bytes());
        put_bytes(&mut tensor, TENSOR_RAW_DATA, raw);
        tensor
    }

    /// A tensor whose values are in `model.onnx.data` at `offset`.
    fn external(name: &str, data_type: i64, dims: &[i64], offset: u64, length: u64) -> Vec<u8> {
        let mut tensor = Vec::new();
        for dim in dims {
            put_varint_field(&mut tensor, TENSOR_DIMS, *dim as u64);
        }
        put_varint_field(&mut tensor, TENSOR_DATA_TYPE, data_type as u64);
        put_bytes(&mut tensor, TENSOR_NAME, name.as_bytes());
        for (key, value) in [
            ("location", "model.onnx.data".to_string()),
            ("offset", offset.to_string()),
            ("length", length.to_string()),
        ] {
            let mut entry = Vec::new();
            put_bytes(&mut entry, ENTRY_KEY, key.as_bytes());
            put_bytes(&mut entry, ENTRY_VALUE, value.as_bytes());
            put_bytes(&mut tensor, TENSOR_EXTERNAL_DATA, &entry);
        }
        put_varint_field(&mut tensor, TENSOR_DATA_LOCATION, 1);
        tensor
    }

    fn value(name: &str, element: i64, dims: &[&str]) -> Vec<u8> {
        let mut shape = Vec::new();
        for dim in dims {
            shape.extend_from_slice(&symbolic(dim));
        }
        let mut tensor_type = Vec::new();
        put_varint_field(&mut tensor_type, TENSOR_TYPE_ELEM_TYPE, element as u64);
        put_bytes(&mut tensor_type, TENSOR_TYPE_SHAPE, &shape);
        let mut ty = Vec::new();
        put_bytes(&mut ty, TYPE_TENSOR, &tensor_type);
        let mut value = Vec::new();
        put_bytes(&mut value, VALUE_INFO_NAME, name.as_bytes());
        put_bytes(&mut value, VALUE_INFO_TYPE, &ty);
        value
    }

    /// The start of an encoder as both int8 exports hold it: four rows of
    /// three, `uint8`, at offset 8 of the data file, gathered by the ids and
    /// dequantized, then a bias added. With `quantized` false the table is
    /// `f32` and gathered straight into the add, as an unquantized export's is.
    fn embeddings(quantized: bool) -> Vec<u8> {
        let mut graph = Vec::new();
        let table = if quantized { UINT8 } else { FLOAT };
        let gathered = if quantized { "gathered" } else { "embedded" };
        put_bytes(
            &mut graph,
            GRAPH_NODE,
            &node("Gather", &["table", IDS], &[gathered]),
        );
        if quantized {
            put_bytes(
                &mut graph,
                GRAPH_NODE,
                &node(
                    "DequantizeLinear",
                    &["gathered", "scale", "zero_point"],
                    &["embedded"],
                ),
            );
        }
        put_bytes(&mut graph, GRAPH_NODE, &node("Shape", &[IDS], &["shape"]));
        put_bytes(
            &mut graph,
            GRAPH_NODE,
            &node("Add", &["embedded", "bias"], &["out"]),
        );
        put_bytes(
            &mut graph,
            GRAPH_INITIALIZER,
            &external("table", table, &[4, 3], 8, 12),
        );
        put_bytes(
            &mut graph,
            GRAPH_INITIALIZER,
            &tensor("scale", FLOAT, &[], &0.25f32.to_le_bytes()),
        );
        put_bytes(
            &mut graph,
            GRAPH_INITIALIZER,
            &tensor("zero_point", UINT8, &[], &[127]),
        );
        let bias: Vec<u8> = [0.5f32; 3]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        put_bytes(
            &mut graph,
            GRAPH_INITIALIZER,
            &tensor("bias", FLOAT, &[3], &bias),
        );
        put_bytes(
            &mut graph,
            GRAPH_INPUT,
            &value(IDS, INT64, &["batch", "sequence"]),
        );
        put_bytes(
            &mut graph,
            GRAPH_OUTPUT,
            &value("out", FLOAT, &["batch", "sequence", "width"]),
        );
        put_bytes(&mut graph, GRAPH_VALUE_INFO, &value("gathered", UINT8, &[]));
        put_bytes(&mut graph, GRAPH_VALUE_INFO, &value("embedded", FLOAT, &[]));
        put_bytes(&mut graph, GRAPH_VALUE_INFO, &value("shape", INT64, &[]));

        let mut opset = Vec::new();
        put_bytes(&mut opset, OPSET_DOMAIN, b"");
        put_varint_field(&mut opset, 2, 13);
        let mut model = Vec::new();
        put_varint_field(&mut model, 1, 8);
        put_bytes(&mut model, MODEL_OPSET_IMPORT, &opset);
        put_bytes(&mut model, MODEL_GRAPH, &graph);
        model
    }

    /// The gather and the dequantize go, and the table, scale and zero point
    /// with them; what the dequantize produced is an `f32` input of the ids'
    /// shape and the row width; and the rest is copied through byte for byte.
    #[test]
    fn a_quantized_gather_becomes_an_input() {
        let model = embeddings(true);
        let split = split(&model).expect("the gather splits");
        assert_eq!(
            split.table,
            Table {
                input: "embedded".into(),
                data: "model.onnx.data".into(),
                offset: 8,
                rows: 4,
                width: 3,
                signed: false,
                scale: 0.25,
                zero_point: 127,
            }
        );
        assert_eq!(
            Table::parse(&split.table.encode()),
            Some(split.table.clone())
        );

        let top = fields(&split.model).unwrap();
        for (one, other) in top.iter().zip(&fields(&model).unwrap()) {
            if one.number != MODEL_GRAPH {
                assert_eq!(one.raw, other.raw, "field {} changed", one.number);
            }
        }
        let graph = fields(only(&top, MODEL_GRAPH, "graph").unwrap().bytes().unwrap()).unwrap();
        let of = |number: u32| graph.iter().filter(move |field| field.number == number);
        let ops: Vec<&str> = of(GRAPH_NODE)
            .map(|field| Node::parse(field).unwrap().op_type)
            .collect();
        assert_eq!(ops, ["Shape", "Add"]);
        let initializers: Vec<&str> = of(GRAPH_INITIALIZER)
            .map(|field| Tensor::parse(field.bytes().unwrap()).unwrap().name)
            .collect();
        assert_eq!(initializers, ["bias"]);
        let described: Vec<&str> = of(GRAPH_VALUE_INFO)
            .map(|field| value_info(field.bytes().unwrap()).unwrap().0)
            .collect();
        assert_eq!(described, ["shape"]);
        let inputs: Vec<(&str, i64)> = of(GRAPH_INPUT)
            .map(|field| value_info(field.bytes().unwrap()).unwrap())
            .collect();
        assert_eq!(inputs, [(IDS, INT64), ("embedded", FLOAT)]);
        let added = of(GRAPH_INPUT).nth(1).unwrap().bytes().unwrap();
        let dims = shape(added).unwrap().expect("a shape");
        assert_eq!(dims.len(), 3);
        assert_eq!(
            dims[..2],
            shape(&value(IDS, INT64, &["batch", "sequence"]))
                .unwrap()
                .unwrap()[..]
        );
        let mut width = Vec::new();
        put_varint_field(&mut width, DIM_VALUE, 3);
        assert_eq!(
            fields_of(&fields(dims[2]).unwrap()[0]).unwrap()[0].raw,
            width
        );
    }

    /// A graph that gathers no quantized table is left as it is.
    #[test]
    fn a_graph_without_the_pattern_is_left_alone() {
        assert_eq!(
            split(&embeddings(false)).err().as_deref(),
            Some("no gather of a quantized table by input_ids matched")
        );
        assert!(split(b"this is not a protobuf").is_err());
    }

    /// A table of `rows` rows of three after eight bytes that are not the
    /// table, and its description, written as a split graph has them.
    fn written(dir: &Path, rows: &[[u8; 3]], signed: bool) -> PathBuf {
        let mut data = vec![0xee; 8];
        data.extend(rows.iter().flatten());
        std::fs::write(dir.join("model.onnx.data"), data).unwrap();
        let graph = dir.join("model.split1.onnx");
        let table = Table {
            input: "embedded".into(),
            data: "model.onnx.data".into(),
            offset: 8,
            rows: rows.len(),
            width: 3,
            signed,
            scale: 0.0123,
            zero_point: if signed { -3 } else { 127 },
        };
        std::fs::write(description(&graph), table.encode()).unwrap();
        graph
    }

    /// Rows read back as the reference definition computes them, bit for bit:
    /// the first row, the second, the last, and repeats; and an id outside the
    /// table is refused rather than read.
    #[test]
    fn rows_dequantize_as_the_reference_does() {
        let rows = [[0, 127, 255], [1, 128, 254], [7, 200, 33], [255, 0, 128]];
        for signed in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let graph = written(dir.path(), &rows, signed);
            let table = WordTable::beside(&graph).unwrap().expect("the table");
            assert_eq!((table.input(), table.width()), ("embedded", 3));

            let ids = [0, 1, 3, 1, 0, 2];
            let (zero_point, scale) = (table.table.zero_point, table.table.scale);
            let expected: Vec<u32> = ids
                .iter()
                .flat_map(|id| rows[*id as usize])
                .map(|q| {
                    let q = if signed {
                        i32::from(q as i8)
                    } else {
                        i32::from(q)
                    };
                    ((q - zero_point) as f32 * scale).to_bits()
                })
                .collect();
            let embedded: Vec<u32> = table
                .embed(&ids)
                .unwrap()
                .iter()
                .map(|value| value.to_bits())
                .collect();
            assert_eq!(embedded, expected, "signed: {signed}");

            for outside in [rows.len() as i64, -1] {
                assert!(table.embed(&[0, outside]).is_err(), "id {outside} was read");
            }
        }
    }

    /// No description, no table; a data file too short for the one described
    /// is refused at load rather than at the first row past its end.
    #[test]
    fn a_table_is_found_only_where_it_is_described_and_fits() {
        let dir = tempfile::tempdir().unwrap();
        assert!(
            WordTable::beside(&dir.path().join("model.onnx"))
                .unwrap()
                .is_none()
        );
        let graph = written(dir.path(), &[[1, 2, 3]], false);
        std::fs::write(dir.path().join("model.onnx.data"), [0u8; 10]).unwrap();
        assert!(WordTable::beside(&graph).is_err());
    }
}
