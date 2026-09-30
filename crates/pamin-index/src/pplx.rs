//! Explicit int8 MatMulNBits arithmetic for the immutable PPLX export.
//! Only node attributes change; external weights and output encoding remain.
use crate::error::{IndexError, Result};
use crate::onnx::*;
use std::path::{Path, PathBuf};

pub(crate) fn prepare(source: &Path, cache: &Path) -> Result<PathBuf> {
    let digest = crate::prepared::source_digest(source)?;
    // Derived files belong to the writable workspace. ONNX rejects external
    // data symlinks that escape its graph directory and multiply-linked files;
    // keep one independently owned copy, leaving a shared/read-only hub alone.
    let directory = cache.join("pplx-int8-v1").join(&digest);
    std::fs::create_dir_all(&directory)?;
    let output = directory.join("model.onnx");
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(output.with_extension("lock"))?;
    lock.lock()?;
    let bytes = std::fs::read(source)?;
    let (rewritten, count) = rewrite(&bytes).map_err(IndexError::Engine)?;
    if count != 196 {
        return Err(IndexError::Engine(format!(
            "pinned PPLX export has {count} MatMulNBits nodes, expected 196"
        )));
    }
    let data = source
        .parent()
        .expect("snapshot graph parent")
        .join("model_quantized.onnx_data");
    let data = std::fs::canonicalize(data)?;
    let linked = directory.join("model_quantized.onnx_data");
    if !linked.is_file() || std::fs::symlink_metadata(&linked)?.file_type().is_symlink() {
        if std::fs::symlink_metadata(&linked).is_ok() {
            std::fs::remove_file(&linked)?;
        }
        let pending = linked.with_extension("partial");
        std::fs::copy(&data, &pending)?;
        std::fs::rename(pending, &linked)?;
    }
    if std::fs::read(&output).is_ok_and(|existing| existing == rewritten) {
        return Ok(output);
    }
    let pending = output.with_extension("partial");
    std::fs::write(&pending, rewritten)?;
    std::fs::rename(&pending, &output)?;
    Ok(output)
}

fn rewrite(bytes: &[u8]) -> std::result::Result<(Vec<u8>, usize), String> {
    let mut result = Vec::new();
    let mut count = 0;
    for field in fields(bytes)? {
        if field.number != MODEL_GRAPH {
            result.extend_from_slice(field.raw);
            continue;
        }
        let mut graph = Vec::new();
        for item in fields(field.bytes()?)? {
            if item.number != GRAPH_NODE {
                graph.extend_from_slice(item.raw);
                continue;
            }
            let node = Node::parse(&item)?;
            if !node.is("MatMulNBits", true) {
                graph.extend_from_slice(item.raw);
                continue;
            }
            let mut rewritten = Vec::new();
            for property in fields(item.bytes()?)? {
                if property.number == NODE_ATTRIBUTE {
                    let attribute = fields(property.bytes()?)?;
                    if string(&attribute, ATTRIBUTE_NAME)? == Some("accuracy_level") {
                        continue;
                    }
                }
                rewritten.extend_from_slice(property.raw);
            }
            put_bytes(
                &mut rewritten,
                NODE_ATTRIBUTE,
                &int_attribute("accuracy_level", 4),
            );
            put_bytes(&mut graph, GRAPH_NODE, &rewritten);
            count += 1;
        }
        put_bytes(&mut result, MODEL_GRAPH, &graph);
    }
    Ok((result, count))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    #[test]
    fn preparation_keeps_a_read_only_hub_snapshot_unchanged() {
        use std::os::unix::fs::PermissionsExt;
        let root = tempfile::tempdir().unwrap();
        let snapshot = root.path().join("snapshot");
        std::fs::create_dir(&snapshot).unwrap();
        let mut node = Vec::new();
        put_bytes(&mut node, NODE_OP_TYPE, b"MatMulNBits");
        put_bytes(&mut node, NODE_DOMAIN, MICROSOFT.as_bytes());
        let mut graph = Vec::new();
        for _ in 0..196 {
            put_bytes(&mut graph, GRAPH_NODE, &node);
        }
        let mut model = Vec::new();
        put_bytes(&mut model, MODEL_GRAPH, &graph);
        let source = snapshot.join("model_quantized.onnx");
        std::fs::write(&source, &model).unwrap();
        std::fs::write(
            snapshot.join("model_quantized.onnx_data"),
            b"immutable weights",
        )
        .unwrap();
        std::fs::set_permissions(&snapshot, std::fs::Permissions::from_mode(0o555)).unwrap();
        let cache = root.path().join("workspace");
        let output = prepare(&source, &cache).unwrap();
        assert!(output.starts_with(&cache));
        let owned = output.parent().unwrap().join("model_quantized.onnx_data");
        assert!(
            !std::fs::symlink_metadata(&owned)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert!(
            owned
                .canonicalize()
                .unwrap()
                .starts_with(output.parent().unwrap().canonicalize().unwrap())
        );
        assert_eq!(
            std::fs::read(output.parent().unwrap().join("model_quantized.onnx_data")).unwrap(),
            b"immutable weights"
        );
        assert_eq!(std::fs::read(&source).unwrap(), model);
        assert_eq!(std::fs::read_dir(&snapshot).unwrap().count(), 2);
        std::fs::set_permissions(&snapshot, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    #[test]
    fn int8_mode_replaces_the_attribute_and_preserves_weights_and_other_nodes() {
        let mut node = Vec::new();
        put_bytes(&mut node, NODE_OP_TYPE, b"MatMulNBits");
        put_bytes(&mut node, NODE_DOMAIN, MICROSOFT.as_bytes());
        put_bytes(
            &mut node,
            NODE_ATTRIBUTE,
            &int_attribute("accuracy_level", 0),
        );
        let mut other = Vec::new();
        put_bytes(&mut other, NODE_OP_TYPE, b"Identity");
        let mut graph = Vec::new();
        put_bytes(&mut graph, GRAPH_NODE, &node);
        put_bytes(&mut graph, GRAPH_NODE, &other);
        put_bytes(&mut graph, GRAPH_INITIALIZER, b"original external weights");
        let mut model = Vec::new();
        put_bytes(&mut model, MODEL_GRAPH, &graph);
        let (output, count) = rewrite(&model).unwrap();
        assert_eq!(count, 1);
        assert_eq!(
            rewrite(&output).unwrap().0,
            output,
            "rewrite must be idempotent"
        );
        let graph = fields(&output).unwrap();
        let graph = fields(graph[0].bytes().unwrap()).unwrap();
        let node = Node::parse(&graph[0]).unwrap();
        assert_eq!(node.int("accuracy_level", 0), Some(4));
        assert_eq!(graph[1].bytes().unwrap(), other);
        assert_eq!(graph[2].bytes().unwrap(), b"original external weights");
    }
}
