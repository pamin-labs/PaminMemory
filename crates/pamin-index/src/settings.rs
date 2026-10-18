//! What a caller decides about inference, passed in rather than read.
//!
//! Nothing in this crate reads the process environment for a setting. A
//! value read from the environment deep inside a model load is a value every
//! caller in the process shares, which a test can only vary by writing the
//! environment -- process-wide, and racing every other test thread in the same
//! binary. So the binary decides (flags, its configuration file) and hands the
//! result down, and a harness builds the value it wants.
//!
//! Each default is the constant documented beside the measurement that settled
//! it, so this file holds none of its own.

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

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

/// How the reranker batches and truncates. See the constants in
/// [`crate::reranking`] for what settled each default.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RerankSettings {
    /// The most pairs a pass takes, or `None` for the measured rule: one pair
    /// a pass where CPU passes overlap, four otherwise.
    pub batch: Option<NonZeroUsize>,
    /// How many padded tokens one pass may hold. Never zero: a pass that
    /// may hold nothing would take no pair at all.
    pub batch_tokens: NonZeroUsize,
    /// The longest candidate the model reads, in tokens. Never zero: a
    /// reranker that reads nothing of a candidate scores none of them.
    pub max_tokens: NonZeroUsize,
}

impl Default for RerankSettings {
    fn default() -> Self {
        Self {
            batch: None,
            batch_tokens: NonZeroUsize::new(crate::reranking::BATCH_TOKENS)
                .expect("a pass holds at least one token"),
            max_tokens: NonZeroUsize::new(crate::reranking::MAX_TOKENS)
                .expect("the reranker reads at least one token"),
        }
    }
}

/// Everything about running a model that is the caller's to decide.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InferenceSettings {
    /// Which devices may be planned on.
    pub device: DeviceChoice,
    /// Threads one forward pass may use, or on the CPU how many single-thread
    /// passes run at once; `None` is one per core.
    pub threads: Option<NonZeroUsize>,
    /// Absolute paths of NPU plugin execution-provider libraries to register.
    pub ep_libraries: Vec<PathBuf>,
    /// Whether a CPU load reads a mapped copy of the model (see
    /// `crate::prepared`) rather than its download.
    pub prepared: bool,
    /// Whether a mapped copy's attention-fused graph is loaded where one was
    /// kept.
    pub fused_attention: bool,
    /// The reranker's batching and truncation.
    pub rerank: RerankSettings,
}

impl Default for InferenceSettings {
    fn default() -> Self {
        Self {
            device: DeviceChoice::Auto,
            threads: None,
            ep_libraries: Vec::new(),
            prepared: true,
            fused_attention: true,
            rerank: RerankSettings::default(),
        }
    }
}

impl InferenceSettings {
    /// How many threads inference may keep busy: [`threads`](Self::threads),
    /// or one per core.
    pub fn budget(&self) -> std::io::Result<usize> {
        Ok(match self.threads {
            Some(threads) => threads.get(),
            None => std::thread::available_parallelism()?.get(),
        })
    }

    /// Every setting that differs from the default, as `(name, value)` pairs
    /// in a fixed order.
    ///
    /// What execution plans are keyed by, and what names an evaluation arm.
    /// Empty at the defaults, so a default configuration keys its plans
    /// exactly as a process with no settings in its environment did before
    /// settings were passed in, and plans measured then are reused.
    pub fn overrides(&self) -> Vec<(String, String)> {
        let default = Self::default();
        let mut overrides = Vec::new();
        let mut differs = |name: &str, differs: bool, value: String| {
            if differs {
                overrides.push((name.to_string(), value));
            }
        };
        differs(
            "device",
            self.device != default.device,
            self.device.label().to_string(),
        );
        differs(
            "inference_threads",
            self.threads != default.threads,
            self.threads.map_or(String::new(), |n| n.to_string()),
        );
        differs(
            "ep_libraries",
            self.ep_libraries != default.ep_libraries,
            spelled_paths(&self.ep_libraries),
        );
        differs(
            "prepared",
            self.prepared != default.prepared,
            self.prepared.to_string(),
        );
        differs(
            "fused_attention",
            self.fused_attention != default.fused_attention,
            self.fused_attention.to_string(),
        );
        differs(
            "rerank_batch",
            self.rerank.batch != default.rerank.batch,
            self.rerank.batch.map_or(String::new(), |n| n.to_string()),
        );
        differs(
            "rerank_batch_tokens",
            self.rerank.batch_tokens != default.rerank.batch_tokens,
            self.rerank.batch_tokens.to_string(),
        );
        differs(
            "rerank_max_tokens",
            self.rerank.max_tokens != default.rerank.max_tokens,
            self.rerank.max_tokens.to_string(),
        );
        overrides
    }
}

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
    fn the_defaults_override_nothing() {
        assert!(InferenceSettings::default().overrides().is_empty());
    }

    #[test]
    fn each_setting_that_differs_is_named_once() {
        let settings = InferenceSettings {
            device: DeviceChoice::Cpu,
            threads: NonZeroUsize::new(2),
            prepared: false,
            rerank: RerankSettings {
                max_tokens: NonZeroUsize::new(192).unwrap(),
                ..RerankSettings::default()
            },
            ..InferenceSettings::default()
        };
        assert_eq!(
            settings.overrides(),
            [
                ("device", "cpu"),
                ("inference_threads", "2"),
                ("prepared", "false"),
                ("rerank_max_tokens", "192"),
            ]
            .map(|(name, value)| (name.to_string(), value.to_string()))
        );
    }

    #[test]
    fn a_device_is_read_by_name_ignoring_case() {
        assert_eq!(DeviceChoice::parse("CPU"), Some(DeviceChoice::Cpu));
        assert_eq!(DeviceChoice::parse("auto"), Some(DeviceChoice::Auto));
        assert_eq!(DeviceChoice::parse("gpu"), None);
    }
}
