//! What a caller decides about a server process, shared by the libraries that
//! honour it and the command-line client that shows and forwards it.
//!
//! The client resolves these settings, shows their defaults in `pamin config`
//! and asks a running server for them, and it links none of `pamin-index`,
//! `pamin-store` or `pamin-engine` -- which is what keeps it from paying for
//! the inference runtime, the vector engine and TLS at every start. So the
//! choices it names and the defaults it shows are defined here, once, and each
//! library reads them from here rather than holding a copy.

use std::path::{Path, PathBuf};
use std::time::Duration;

/// Which devices a model may be planned on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DeviceChoice {
    /// Every viable plan, measured: the mixed accelerator policy in ADR 0001.
    #[default]
    Auto,
    /// Optimized CPU only, for a measurement that must be comparable with a
    /// CPU one or a machine whose accelerator belongs to something else.
    Cpu,
}

impl DeviceChoice {
    /// Both, in the order the setting documents them.
    pub const ALL: [Self; 2] = [Self::Auto, Self::Cpu];

    /// The name the setting takes.
    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
        }
    }

    /// The choice a name stands for, ignoring ASCII case.
    pub fn parse(label: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|choice| choice.label().eq_ignore_ascii_case(label.trim()))
    }
}

/// Which vector index a project's index is built with, and what its marker
/// records.
///
/// A setting (`--vector-index`, `PAMIN_VECTOR_INDEX`) with two values, because
/// the choice is a trade between resident memory on one side and query and
/// build time on the other that only the person running a project can make.
/// Both store vectors in half precision and both are searched the same way:
/// the index proposes `RESCORE` times the candidates asked for, with their
/// stored vectors, and those are ranked again by an exact f32 cosine. The two
/// differ in their index and query parameters and in nothing else. ADR 0001,
/// "Two vector indexes, both half precision", has the measurements.
///
/// The marker records which one an index was built with, and an index is
/// never searched as the other: opening one under the other is refused with a
/// message naming `pamin reindex`, as a model change is. Reading an index
/// built one way as another is the silent-wrong-answer shape ADR 0001 records
/// from the first attempt at quantization, which returned recall@10 of 0.000
/// with no error. `pamin reindex` lends every vector the old index holds to
/// the new one (see `pamin_index::Previous`), so changing costs a build and no embedding.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum VectorIndex {
    /// A DiskANN graph kept on disk and read per query.
    ///
    /// Holds almost nothing resident, for a project whose memory is scarce.
    /// Not the default: every `optimize` after a working drain rebuilds a
    /// great deal of the graph (114 to 310 s after 64 new documents on 25,000,
    /// against about a second for `memory`), and upkeep issues one after a
    /// drain that did work whenever the file budget, the unmerged blocks or
    /// the unindexed remainder asks for it.
    Disk,
    /// An HNSW graph held in memory.
    ///
    /// The default: the fastest to query and to build, and the smallest on
    /// disk, at the cost of holding the graph and the vectors resident --
    /// still half what the full-precision graph held.
    #[default]
    Memory,
}

impl VectorIndex {
    /// Both, in the order the setting documents them.
    pub const ALL: [Self; 2] = [Self::Memory, Self::Disk];

    /// The name the setting takes and the marker records.
    pub fn label(self) -> &'static str {
        match self {
            Self::Disk => "disk",
            Self::Memory => "memory",
        }
    }

    /// The index a name stands for, `None` for any other name -- including
    /// what an index built before these existed records.
    pub fn parse(label: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|index| index.label() == label.trim())
    }
}

/// Whether a CPU load reads a mapped copy of the model rather than its
/// download, unless a caller says otherwise; `pamin_index::InferenceSettings`.
pub const PREPARED: bool = true;

/// Whether a mapped copy's attention-fused graph is loaded where one was
/// kept, unless a caller says otherwise; `pamin_index::InferenceSettings`.
pub const FUSED_ATTENTION: bool = true;

/// Whether a mapped copy's graph is loaded with its word embeddings read
/// rather than mapped, where one was kept, unless a caller says otherwise;
/// `pamin_index::InferenceSettings`.
pub const SPLIT_EMBEDDINGS: bool = true;

/// Whether PostgreSQL compiles query expressions with LLVM, unless a caller
/// says otherwise; `pamin_store::StoreSettings`, and its `jit` for why not.
pub const JIT: bool = false;

/// How long a model may sit unused before the process gives it back.
///
/// It is worth giving back because of what it costs while it sits there, and
/// because the giving back was measured rather than assumed -- dropping a
/// model frees it to the allocator, which is not the same as to the operating
/// system. On a resident server over a 13,014-document project, four cores:
///
/// | | resident |
/// | --- | --- |
/// | serving `--rerank off` | 1,625-1,626 MB, stable to a megabyte |
/// | after one `fast` search | 2,007 MB or 2,271 MB |
/// | after this releases it | 1,627-1,903 MB |
///
/// **One `fast` search costs either about 380 MB or about 645 MB of resident
/// memory for a 130 MB model cache**, and the bimodality is the runtime's
/// arenas rather than the weights -- six samples over one fixed query pair,
/// three at each value, so it is not the shortlist either. Releasing gives
/// back 368 to 380 MB of it, three samples, consistently: the model's own
/// footprint returns to the operating system and the arena growth above it
/// does not. So this reclaims about 380 MB reliably and sometimes all 645,
/// against a server that otherwise holds it for as long as it lives.
///
/// `accurate` is four and a half times the weights and was not measurable
/// here, its download being unreachable from the machine this ran on.
///
/// Thirty minutes, and the number moved once the other side of the trade was
/// measured rather than guessed. Five was written here first, on the strength
/// of "a caller pays about a second" -- **it is 4,528 ms**, median of three,
/// against 116 ms for a warm search and 5,123 ms for a server starting from
/// nothing. The sweep puts the next caller back to 88% of cold, so this is not
/// a second of latency for a gigabyte, it is four and a half.
///
/// Thirty is where that trade stops being close. An agent working in bursts
/// almost never waits half an hour between searches, so it almost never pays;
/// a server left running overnight pays once and gives back 1.6 GB for the
/// hours in between. Five minutes would have charged 4.5 seconds to anyone who
/// stopped to read what they found.
///
/// One window for models and for the indexes that pin them, because the
/// model dominates: reopening an index is part of the 4,528 ms and reloading
/// the weights is most of it.
pub const MODEL_IDLE: Duration = Duration::from_secs(30 * 60);

/// `paths` as one value a key or a comparison can hold: a JSON array of each
/// path [`spelled_path`], so a separator inside a path cannot make one path
/// read as two, and no two paths spell the same.
pub fn spelled_paths(paths: &[PathBuf]) -> String {
    let paths: Vec<String> = paths.iter().map(|path| spelled_path(path)).collect();
    serde_json::to_string(&paths).expect("a list of strings serializes")
}

/// `path` for a person or a program to read back: as written when it is
/// UTF-8, so the path it names is the path given, and [`spelled_path`] only
/// when it is not, where nothing as written could name it. Not an identity:
/// a UTF-8 path spelling a byte's escape reads like the path with that byte.
pub fn shown_path(path: &Path) -> String {
    match path.to_str() {
        Some(text) => text.to_string(),
        None => spelled_path(path),
    }
}

/// `path` as text that identifies it: as written when it is UTF-8, and
/// losslessly when it is not. For a key, an identity or a comparison, where
/// `display` or `to_string_lossy` would spell every byte it cannot decode as
/// the same U+FFFD and make two paths one.
///
/// On Unix a byte that is not UTF-8 is `\xNN`, and so that no UTF-8 path can
/// spell the same, a backslash is `\\` -- a backslash being the one change
/// to how a UTF-8 path is spelled. On Windows a path is UTF-16, and a unit
/// that does not decode is `<NNNN>`: `<` cannot be in a Windows path, so a
/// path that decodes cannot spell the same and nothing needs escaping.
pub fn spelled_path(path: &Path) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let mut out = String::new();
        for chunk in path.as_os_str().as_bytes().utf8_chunks() {
            out.push_str(&chunk.valid().replace('\\', "\\\\"));
            for byte in chunk.invalid() {
                out.push_str(&format!("\\x{byte:02x}"));
            }
        }
        out
    }
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        spelled_wide(path.as_os_str().encode_wide())
    }
    #[cfg(not(any(unix, windows)))]
    {
        path.display().to_string()
    }
}

/// A Windows path's UTF-16 `units` as [`spelled_path`] spells them: decoded,
/// and each unpaired surrogate as `<NNNN>`. The unit named is the one the
/// decoder refused, not the one at the same position: a surrogate pair is two
/// units and one character, so counting positions goes out of step after one.
#[cfg(any(windows, test))]
fn spelled_wide(units: impl IntoIterator<Item = u16>) -> String {
    char::decode_utf16(units)
        .map(|decoded| match decoded {
            Ok(character) => character.to_string(),
            Err(error) => format!("<{:04x}>", error.unpaired_surrogate()),
        })
        .collect()
}

/// What a PostgreSQL installation prefix lacks of what the store runs from
/// it, `bin/initdb` and `bin/pg_ctl`, or `None` if it has both. The one
/// check, for the store and for a caller validating a prefix before it is
/// recorded.
pub fn installation_missing(prefix: &std::path::Path) -> Option<String> {
    ["initdb", "pg_ctl"]
        .into_iter()
        .map(|program| format!("bin/{program}{}", std::env::consts::EXE_SUFFIX))
        .find(|relative| !prefix.join(relative).is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Paths that differ only in bytes that are not UTF-8 spell differently,
    /// and a UTF-8 path is spelled as written.
    #[cfg(unix)]
    #[test]
    fn a_path_that_is_not_utf8_is_spelled_byte_for_byte() {
        use std::os::unix::ffi::OsStrExt;
        let path = |bytes: &[u8]| PathBuf::from(std::ffi::OsStr::from_bytes(bytes));
        let x80 = path(b"/opt/ep-\x80.so");
        let x81 = path(b"/opt/ep-\x81.so");
        assert_eq!(spelled_path(&x80), "/opt/ep-\\x80.so");
        assert_ne!(spelled_path(&x80), spelled_path(&x81));
        assert_ne!(
            spelled_paths(std::slice::from_ref(&x80)),
            spelled_paths(&[x81])
        );
        // Spelled as written, but for a backslash, which is escaped so that
        // no UTF-8 path spells what a byte that is not UTF-8 does.
        assert_eq!(spelled_path(Path::new("/opt/ép.so")), "/opt/ép.so");
        assert_ne!(spelled_path(&path(br"/opt/ep-\x80.so")), spelled_path(&x80));
    }

    /// A UTF-8 path is shown as written, a backslash included, and one that
    /// is not is shown as it is spelled.
    #[cfg(unix)]
    #[test]
    fn a_path_is_shown_as_written_when_it_can_be() {
        use std::os::unix::ffi::OsStrExt;
        assert_eq!(shown_path(Path::new(r"/tmp/a\b")), r"/tmp/a\b");
        let x80 = PathBuf::from(std::ffi::OsStr::from_bytes(b"/opt/ep-\x80.so"));
        assert_eq!(shown_path(&x80), "/opt/ep-\\x80.so");
    }

    /// A separator inside a path does not make it two paths.
    #[test]
    fn a_separator_in_a_path_keeps_it_one_path() {
        let one = spelled_paths(&[PathBuf::from("/tmp/a, /tmp/b")]);
        let two = spelled_paths(&[PathBuf::from("/tmp/a"), PathBuf::from("/tmp/b")]);
        assert_eq!(one, r#"["/tmp/a, /tmp/b"]"#);
        assert_eq!(two, r#"["/tmp/a","/tmp/b"]"#);
    }

    /// Each unpaired surrogate is spelled as itself, after a surrogate pair
    /// as anywhere else.
    #[test]
    fn a_windows_path_spells_the_surrogate_it_could_not_decode() {
        let grin = [0xD83D, 0xDE00];
        let after = |unit: u16| spelled_wide(grin.into_iter().chain([unit]));
        assert_eq!(after(0xD800), "😀<d800>");
        assert_eq!(after(0xDE00), "😀<de00>");
        assert_ne!(after(0xD800), after(0xDE00));
        assert_eq!(spelled_wide([0xDC00, 0x61]), "<dc00>a");
        assert_eq!(spelled_wide([0x61, 0xD83D]), "a<d83d>");
        assert_eq!(spelled_wide("C:\\ep.dll".encode_utf16()), "C:\\ep.dll");
    }

    #[test]
    fn a_device_is_read_by_name_ignoring_case() {
        assert_eq!(DeviceChoice::parse("CPU"), Some(DeviceChoice::Cpu));
        assert_eq!(DeviceChoice::parse("auto"), Some(DeviceChoice::Auto));
        assert_eq!(DeviceChoice::parse("gpu"), None);
    }
}
