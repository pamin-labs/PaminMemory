//! Small ONNX wire helpers shared by model graph rewrites.
//! Unknown fields remain byte-for-byte unchanged in the caller.

pub(crate) const MICROSOFT: &str = "com.microsoft";

// Field numbers, from `onnx.proto`.
pub(crate) const MODEL_OPSET_IMPORT: u32 = 8;
pub(crate) const MODEL_GRAPH: u32 = 7;
pub(crate) const OPSET_DOMAIN: u32 = 1;
pub(crate) const GRAPH_NODE: u32 = 1;
pub(crate) const GRAPH_INITIALIZER: u32 = 5;
pub(crate) const GRAPH_INPUT: u32 = 11;
pub(crate) const GRAPH_OUTPUT: u32 = 12;
pub(crate) const GRAPH_VALUE_INFO: u32 = 13;
pub(crate) const NODE_INPUT: u32 = 1;
pub(crate) const NODE_OUTPUT: u32 = 2;
pub(crate) const NODE_NAME: u32 = 3;
pub(crate) const NODE_OP_TYPE: u32 = 4;
pub(crate) const NODE_ATTRIBUTE: u32 = 5;
pub(crate) const NODE_DOMAIN: u32 = 7;
pub(crate) const ATTRIBUTE_NAME: u32 = 1;
pub(crate) const ATTRIBUTE_F: u32 = 2;
pub(crate) const ATTRIBUTE_I: u32 = 3;
pub(crate) const ATTRIBUTE_G: u32 = 6;
pub(crate) const ATTRIBUTE_INTS: u32 = 8;
pub(crate) const ATTRIBUTE_GRAPHS: u32 = 11;
pub(crate) const ATTRIBUTE_TYPE: u32 = 20;
pub(crate) const TENSOR_DIMS: u32 = 1;
pub(crate) const TENSOR_DATA_TYPE: u32 = 2;
pub(crate) const TENSOR_INT64_DATA: u32 = 7;
pub(crate) const TENSOR_NAME: u32 = 8;
pub(crate) const TENSOR_RAW_DATA: u32 = 9;
pub(crate) const TENSOR_DATA_LOCATION: u32 = 14;
pub(crate) const VALUE_INFO_NAME: u32 = 1;
pub(crate) const VALUE_INFO_TYPE: u32 = 2;
pub(crate) const TYPE_TENSOR: u32 = 1;
pub(crate) const TENSOR_TYPE_ELEM_TYPE: u32 = 1;

// `AttributeProto.AttributeType` and `TensorProto.DataType`.
pub(crate) const ATTRIBUTE_FLOAT: u64 = 1;
pub(crate) const ATTRIBUTE_INT: u64 = 2;
pub(crate) const INT32: i64 = 6;
pub(crate) const INT64: i64 = 7;

/// One field of a message: its number, its payload, and the bytes it occupied
/// -- tag included -- so an untouched field is copied back verbatim.
pub(crate) struct Field<'a> {
    pub(crate) number: u32,
    pub(crate) value: Value<'a>,
    pub(crate) raw: &'a [u8],
}

pub(crate) enum Value<'a> {
    Varint(u64),
    /// Read past, never used: nothing matched here is a `double`.
    Fixed64,
    Bytes(&'a [u8]),
    Fixed32(u32),
}

impl<'a> Field<'a> {
    pub(crate) fn bytes(&self) -> Result<&'a [u8], String> {
        match self.value {
            Value::Bytes(bytes) => Ok(bytes),
            _ => Err(format!("field {} is not length-delimited", self.number)),
        }
    }

    pub(crate) fn str(&self) -> Result<&'a str, String> {
        std::str::from_utf8(self.bytes()?).map_err(|error| error.to_string())
    }

    pub(crate) fn varint(&self) -> Result<u64, String> {
        match self.value {
            Value::Varint(value) => Ok(value),
            _ => Err(format!("field {} is not a varint", self.number)),
        }
    }
}

/// Every field of one message, in the order they were written.
pub(crate) fn fields(mut bytes: &[u8]) -> Result<Vec<Field<'_>>, String> {
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

pub(crate) fn varint(bytes: &mut &[u8]) -> Result<u64, String> {
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

pub(crate) fn take<'a>(bytes: &mut &'a [u8], length: usize) -> Result<&'a [u8], String> {
    if bytes.len() < length {
        return Err("a truncated field".into());
    }
    let (taken, rest) = bytes.split_at(length);
    *bytes = rest;
    Ok(taken)
}

/// The single field `number` of a message, which must be there exactly once.
pub(crate) fn only<'f, 'a>(
    fields: &'f [Field<'a>],
    number: u32,
    what: &str,
) -> Result<&'f Field<'a>, String> {
    let mut matching = fields.iter().filter(|field| field.number == number);
    match (matching.next(), matching.next()) {
        (Some(field), None) => Ok(field),
        _ => Err(format!("expected exactly one {what}")),
    }
}

/// The last value of string field `number`, as protobuf reads a repeated
/// scalar written twice.
pub(crate) fn string<'a>(fields: &[Field<'a>], number: u32) -> Result<Option<&'a str>, String> {
    fields
        .iter()
        .rfind(|field| field.number == number)
        .map(Field::str)
        .transpose()
}

/// Repeated `int64`, packed or not.
pub(crate) fn int64s(fields: &[Field<'_>], number: u32) -> Result<Vec<i64>, String> {
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
pub(crate) fn initializer(bytes: &[u8]) -> Result<(&str, Option<Vec<i64>>), String> {
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
pub(crate) fn value_info(bytes: &[u8]) -> Result<(&str, i64), String> {
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
pub(crate) fn last<'a>(
    fields: &[Field<'a>],
    number: u32,
) -> Result<Option<Vec<Field<'a>>>, String> {
    fields
        .iter()
        .rfind(|field| field.number == number)
        .map(fields_of)
        .transpose()
}

pub(crate) fn fields_of<'a>(field: &Field<'a>) -> Result<Vec<Field<'a>>, String> {
    fields(field.bytes()?)
}

/// A node, decoded as far as matching needs, with the bytes it came from.
pub(crate) struct Node<'a> {
    pub(crate) raw: &'a [u8],
    pub(crate) inputs: Vec<&'a str>,
    pub(crate) outputs: Vec<&'a str>,
    pub(crate) op_type: &'a str,
    pub(crate) domain: &'a str,
    pub(crate) attributes: Vec<Attribute<'a>>,
    /// Whether any attribute holds a graph, which makes this control flow.
    pub(crate) subgraph: bool,
}

pub(crate) struct Attribute<'a> {
    pub(crate) name: &'a str,
    pub(crate) float: Option<f32>,
    pub(crate) int: Option<i64>,
    pub(crate) ints: Vec<i64>,
}

impl<'a> Node<'a> {
    pub(crate) fn parse(field: &Field<'a>) -> Result<Self, String> {
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
    pub(crate) fn is(&self, op_type: &str, microsoft: bool) -> bool {
        let domain = if microsoft { MICROSOFT } else { "" };
        self.op_type == op_type
            && (self.domain == domain || (!microsoft && self.domain == "ai.onnx"))
    }

    pub(crate) fn attribute(&self, name: &str) -> Option<&Attribute<'a>> {
        self.attributes
            .iter()
            .find(|attribute| attribute.name == name)
    }

    /// An integer attribute, or `default` if the node does not set it.
    pub(crate) fn int(&self, name: &str, default: i64) -> Option<i64> {
        match self.attribute(name) {
            None => Some(default),
            Some(attribute) => attribute.int,
        }
    }
}

pub(crate) fn int_attribute(name: &str, value: i64) -> Vec<u8> {
    let mut attribute = Vec::new();
    put_bytes(&mut attribute, ATTRIBUTE_NAME, name.as_bytes());
    put_varint_field(&mut attribute, ATTRIBUTE_I, value as u64);
    put_varint_field(&mut attribute, ATTRIBUTE_TYPE, ATTRIBUTE_INT);
    attribute
}

pub(crate) fn float_attribute(name: &str, value: f32) -> Vec<u8> {
    let mut attribute = Vec::new();
    put_bytes(&mut attribute, ATTRIBUTE_NAME, name.as_bytes());
    put_varint(&mut attribute, u64::from(ATTRIBUTE_F) << 3 | 5);
    attribute.extend_from_slice(&value.to_bits().to_le_bytes());
    put_varint_field(&mut attribute, ATTRIBUTE_TYPE, ATTRIBUTE_FLOAT);
    attribute
}

pub(crate) fn put_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

pub(crate) fn put_varint_field(out: &mut Vec<u8>, number: u32, value: u64) {
    put_varint(out, u64::from(number) << 3);
    put_varint(out, value);
}

pub(crate) fn put_bytes(out: &mut Vec<u8>, number: u32, bytes: &[u8]) {
    put_varint(out, u64::from(number) << 3 | 2);
    put_varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}
