//! What the server runs with: its flags, the workspace's `config.toml`, and
//! the defaults, in that order.
//!
//! The one place the command line turns input into the settings a server
//! process runs with (`pamin-cli` hands them to the engine as
//! `pamin_engine::Settings`); the libraries read no environment. Resolved on
//! the client's side, which links none of the libraries, so every value and
//! default here is a plain one from `pamin_core::settings`. A setting that
//! governs the server process --
//! which device, how many threads, how long a model stays loaded -- is fixed
//! when the server starts, so it is decided here, from (highest first):
//!
//! 1. a flag on the command line, or for `--device` its `PAMIN_DEVICE`
//!    fallback;
//! 2. the `[serve]` table of `config.toml` in the workspace;
//! 3. the default, the constant each library documents beside its
//!    measurement.
//!
//! Every command takes these flags, because any command may be the one that
//! starts the server. A command forwards the ones it was given to the server
//! it starts, and tells a running server what it asked for, which refuses the
//! request if it is running with something else (see [`mismatch`]).

use std::collections::BTreeMap;
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::parser::ValueSource;
use pamin_core::Workspace;
use pamin_core::settings::{
    self as defaults, DeviceChoice, shown_path, spelled_path, spelled_paths,
};
use serde::{Deserialize, Serialize};

/// The configuration file, relative to the workspace root.
pub const CONFIG: &str = "config.toml";

/// How many indexes stay open at once.
///
/// An open index is not free and does not become free by being idle. It holds
/// 27 file descriptors, which is why the bound exists: against the
/// 1024-descriptor limit a process usually starts with, an unbounded server
/// ran out at the thirty-seventh project and the engine refused to create its
/// lexical indexer. Sixteen leaves that a wide margin and is large enough that
/// a server round-robining a working set keeps it; the seventeenth project
/// closes the one nobody has touched for longest rather than failing.
///
/// The descriptors are the only quantity this number bounds, and they are not
/// the one that hurts. Memory is, and it does not follow the count:
///
/// | profile    | per index | sixteen of them |
/// | ---------- | --------- | --------------- |
/// | `speed`    | ~100 MB   | ~1.6 GB         |
/// | `accuracy` | ~205 MB   | ~3.3 GB         |
///
/// The second row was measured the hard way. Two benchmark servers on the
/// shipping profile reached 5.6 GB resident apiece -- sixteen indexes on top
/// of a 2.3 GB floor of model and runtime -- and the kernel killed both. The
/// figure the paragraph above used to quote, 100 MB, was the smallest
/// profile's, and nothing said so.
///
/// Most of the first row is a floor rather than a function of size. The
/// full-text store is a RocksDB with twelve column families, and zvec gives
/// each one's memtable a hash index of a million buckets, an 8,000,000-byte
/// array allocated and zeroed when the memtable is: 96 MB before the first
/// memory is written. A one-memory `speed` index measured 93 MiB anonymous
/// (98 MB), and nothing pamin configures changes the bucket count. In the
/// hundred-project end-to-end test, one-memory `accuracy` indexes added 103 to
/// 152 MiB apiece while the first sixteen opened.
///
/// A count cannot be made to mean bytes here: what an index costs depends on
/// the model, the quantization, and how much has been written, and this
/// process learns the last of those only by opening it. So the number stays a
/// count, and `--open-indexes` lets whoever knows their machine say a
/// smaller one. Sizing it from a memory budget is [ADR 0001]'s hot-set
/// residency work, which is where it belongs.
pub const OPEN_INDEXES: usize = 16;

/// The server's settings as flags. Every one is global, so any command takes
/// it, and optional, so an absent flag leaves the decision to the file and
/// then the default.
#[derive(clap::Args, Clone, Debug, Default)]
pub struct ServerArgs {
    /// Devices models may be planned on: `auto` measures every viable
    /// accelerator and CPU plan, `cpu` uses the optimized CPU only.
    #[arg(
        long,
        env = "PAMIN_DEVICE",
        global = true,
        value_parser = parse_device,
        help_heading = "Server settings"
    )]
    pub device: Option<DeviceChoice>,

    /// Threads one forward pass may use; on the CPU, how many one-thread
    /// passes run at once. Default: one per core.
    #[arg(long, global = true, help_heading = "Server settings")]
    pub inference_threads: Option<NonZeroUsize>,

    /// A path to an NPU plugin execution-provider library to register,
    /// relative to the current directory if not absolute. Repeat for several.
    #[arg(
        long = "ep-library",
        value_name = "PATH",
        global = true,
        help_heading = "Server settings"
    )]
    pub ep_libraries: Vec<PathBuf>,

    /// Load CPU models from a mapped copy (`true`, the default) or from the
    /// download (`false`).
    #[arg(
        long,
        global = true,
        action = clap::ArgAction::Set,
        value_name = "BOOL",
        value_parser = clap::builder::BoolishValueParser::new(),
        help_heading = "Server settings"
    )]
    pub prepared: Option<bool>,

    /// Load a mapped copy's attention-fused graph where one was kept. For
    /// measuring the fusion against the graph ONNX Runtime wrote.
    #[arg(
        long,
        global = true,
        hide = true,
        action = clap::ArgAction::Set,
        value_name = "BOOL",
        value_parser = clap::builder::BoolishValueParser::new()
    )]
    pub fused_attention: Option<bool>,

    /// Load a mapped copy's graph with its word embeddings read from the data
    /// file rather than mapped, where one was kept. For measuring the split
    /// against the graph it came from.
    #[arg(
        long,
        global = true,
        hide = true,
        action = clap::ArgAction::Set,
        value_name = "BOOL",
        value_parser = clap::builder::BoolishValueParser::new()
    )]
    pub split_embeddings: Option<bool>,

    /// Seconds a server keeps a model, and the indexes pinning it, that
    /// nothing has asked for. Default: 1800.
    #[arg(
        long,
        value_name = "SECONDS",
        global = true,
        help_heading = "Server settings"
    )]
    pub model_idle: Option<NonZeroU64>,

    /// How many project indexes a server keeps open at once. Default: 16.
    #[arg(long, global = true, help_heading = "Server settings")]
    pub open_indexes: Option<NonZeroUsize>,

    /// Let PostgreSQL compile query expressions with LLVM. Takes effect when
    /// the database next starts.
    #[arg(
        long,
        global = true,
        action = clap::ArgAction::Set,
        value_name = "BOOL",
        value_parser = clap::builder::BoolishValueParser::new(),
        help_heading = "Server settings"
    )]
    pub jit: Option<bool>,

    /// A PostgreSQL installation prefix holding `bin/initdb` to use instead
    /// of installing one into the workspace, relative to the current
    /// directory if not absolute.
    #[arg(
        long,
        value_name = "PATH",
        global = true,
        help_heading = "Server settings"
    )]
    pub postgres_dir: Option<PathBuf>,
}

fn parse_device(name: &str) -> std::result::Result<DeviceChoice, String> {
    DeviceChoice::parse(name).ok_or_else(|| {
        format!(
            "unknown device {name:?}: expected one of {}",
            DeviceChoice::ALL.map(DeviceChoice::label).join(", ")
        )
    })
}

/// `config.toml`. Unknown keys are refused rather than ignored: a misspelled
/// setting that silently did nothing is the failure this file must not have.
#[derive(Deserialize, Default, Debug)]
#[serde(deny_unknown_fields)]
struct File {
    #[serde(default)]
    serve: Serve,
}

/// The `[serve]` table: the same settings as the flags, by the same names
/// with underscores.
#[derive(Deserialize, Default, Debug)]
#[serde(deny_unknown_fields)]
struct Serve {
    device: Option<String>,
    inference_threads: Option<NonZeroUsize>,
    ep_libraries: Option<Vec<PathBuf>>,
    prepared: Option<bool>,
    model_idle: Option<NonZeroU64>,
    open_indexes: Option<NonZeroUsize>,
    jit: Option<bool>,
    postgres_dir: Option<PathBuf>,
}

/// Where a setting's value came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    Default,
    Config,
    Env,
    Flag,
}

impl Source {
    pub fn name(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Config => "config",
            Self::Env => "env",
            Self::Flag => "flag",
        }
    }
}

/// One setting as resolved: its name (the config key), its value as text, and
/// where the value came from.
#[derive(Clone, Debug, Serialize)]
pub struct Entry {
    pub key: &'static str,
    /// The value for a person or a program to read back.
    pub value: String,
    /// The value a running server's is compared by: [`Self::value`], except
    /// for a path, where it is [`spelled_path`]'s spelling, so two
    /// paths cannot compare equal by reading alike.
    #[serde(skip)]
    pub compared: String,
    pub source: Source,
}

/// What a server runs with, and where each value came from.
///
/// Each value as the library that honours it takes it; `pamin-cli` builds the
/// engine's settings from these, and the libraries' own defaults fill in the
/// rest.
#[derive(Clone, Debug)]
pub struct Server {
    pub device: DeviceChoice,
    pub threads: Option<NonZeroUsize>,
    pub ep_libraries: Vec<PathBuf>,
    pub prepared: bool,
    pub fused_attention: bool,
    pub split_embeddings: bool,
    pub model_idle: NonZeroU64,
    pub open_indexes: usize,
    pub jit: bool,
    pub postgres_dir: Option<PathBuf>,
    pub entries: Vec<Entry>,
}

impl Server {
    /// Resolves the flags `args` (parsed from `matches`) over the workspace's
    /// configuration file over the defaults.
    pub fn resolve(
        workspace_root: &Path,
        args: &ServerArgs,
        matches: &clap::ArgMatches,
    ) -> Result<Self> {
        let file = read(&workspace_root.join(CONFIG))?;
        let given = |id: &str| match matches.value_source(id) {
            Some(ValueSource::CommandLine) => Some(Source::Flag),
            Some(ValueSource::EnvVariable) => Some(Source::Env),
            _ => None,
        };
        let mut entries = Vec::new();
        let mut pick = |key: &'static str,
                        flag: Option<String>,
                        flag_source: Option<Source>,
                        config: Option<String>,
                        default: String| {
            let (value, source) = match (flag, config) {
                (Some(flag), _) => (flag, flag_source.unwrap_or(Source::Flag)),
                (None, Some(config)) => (config, Source::Config),
                (None, None) => (default, Source::Default),
            };
            entries.push(Entry {
                key,
                compared: value.clone(),
                value,
                source,
            });
        };

        let serve = file.serve;

        let device = match (&args.device, &serve.device) {
            (Some(device), _) => *device,
            (None, Some(name)) => DeviceChoice::parse(name).with_context(|| {
                format!(
                    "{CONFIG}: unknown device {name:?}; expected one of {}",
                    DeviceChoice::ALL.map(DeviceChoice::label).join(", ")
                )
            })?,
            (None, None) => DeviceChoice::default(),
        };
        pick(
            "device",
            args.device.map(|d| d.label().to_string()),
            given("device"),
            serve.device.as_ref().map(|_| device.label().to_string()),
            DeviceChoice::default().label().to_string(),
        );

        let threads = args.inference_threads.or(serve.inference_threads);
        pick(
            "inference_threads",
            args.inference_threads.map(|n| n.to_string()),
            given("inference_threads"),
            serve.inference_threads.map(|n| n.to_string()),
            "one per core".to_string(),
        );

        // A relative path in a flag is the caller's, against the directory it
        // ran in; one in the file is the file's, against the workspace. Both
        // are made absolute here, so the server and the comparison with a
        // running one see the path meant, whatever directory they run in.
        let from_flag = |path: &PathBuf| anchored(path, None);
        let from_file = |path: &PathBuf| anchored(path, Some(workspace_root));
        let flag_libraries = args
            .ep_libraries
            .iter()
            .map(from_flag)
            .collect::<Result<Vec<_>>>()?;
        let file_libraries = serve
            .ep_libraries
            .as_ref()
            .map(|paths| paths.iter().map(from_file).collect::<Result<Vec<_>>>())
            .transpose()?;
        let ep_libraries = if flag_libraries.is_empty() {
            file_libraries.clone().unwrap_or_default()
        } else {
            flag_libraries.clone()
        };
        pick(
            "ep_libraries",
            (!flag_libraries.is_empty()).then(|| libraries(&flag_libraries)),
            given("ep_libraries"),
            file_libraries.as_deref().map(libraries),
            libraries(&[]),
        );

        let prepared = args
            .prepared
            .or(serve.prepared)
            .unwrap_or(defaults::PREPARED);
        pick(
            "prepared",
            args.prepared.map(|b| b.to_string()),
            given("prepared"),
            serve.prepared.map(|b| b.to_string()),
            defaults::PREPARED.to_string(),
        );

        let fused_attention = args.fused_attention.unwrap_or(defaults::FUSED_ATTENTION);
        pick(
            "fused_attention",
            args.fused_attention.map(|b| b.to_string()),
            given("fused_attention"),
            None,
            defaults::FUSED_ATTENTION.to_string(),
        );

        let split_embeddings = args.split_embeddings.unwrap_or(defaults::SPLIT_EMBEDDINGS);
        pick(
            "split_embeddings",
            args.split_embeddings.map(|b| b.to_string()),
            given("split_embeddings"),
            None,
            defaults::SPLIT_EMBEDDINGS.to_string(),
        );

        let default_idle = NonZeroU64::new(defaults::MODEL_IDLE.as_secs())
            .expect("a model may sit idle for at least a second");
        let model_idle = args.model_idle.or(serve.model_idle).unwrap_or(default_idle);
        pick(
            "model_idle",
            args.model_idle.map(|n| n.to_string()),
            given("model_idle"),
            serve.model_idle.map(|n| n.to_string()),
            default_idle.to_string(),
        );

        let open_indexes = args
            .open_indexes
            .or(serve.open_indexes)
            .map_or(OPEN_INDEXES, NonZeroUsize::get);
        pick(
            "open_indexes",
            args.open_indexes.map(|n| n.to_string()),
            given("open_indexes"),
            serve.open_indexes.map(|n| n.to_string()),
            OPEN_INDEXES.to_string(),
        );

        let jit = args.jit.or(serve.jit).unwrap_or(defaults::JIT);
        pick(
            "jit",
            args.jit.map(|b| b.to_string()),
            given("jit"),
            serve.jit.map(|b| b.to_string()),
            defaults::JIT.to_string(),
        );

        let flag_postgres = args.postgres_dir.as_ref().map(from_flag).transpose()?;
        let file_postgres = serve.postgres_dir.as_ref().map(from_file).transpose()?;
        let postgres_dir = flag_postgres.clone().or(file_postgres.clone());
        // Refused here rather than when the database starts, where a prefix
        // that is not an installation is an error in a log nobody reads.
        if let Some(prefix) = &postgres_dir
            && let Some(missing) = defaults::installation_missing(prefix)
        {
            let named = match given("postgres_dir") {
                Some(Source::Env) => "postgres_dir from the environment".to_string(),
                Some(_) => "--postgres-dir".to_string(),
                None => format!("{CONFIG}: postgres_dir"),
            };
            bail!(
                "{named} {} is not a PostgreSQL installation: it has no {missing}",
                prefix.display()
            );
        }
        pick(
            "postgres_dir",
            flag_postgres.as_deref().map(spelled_path),
            given("postgres_dir"),
            file_postgres.as_deref().map(spelled_path),
            "none (installs its own)".to_string(),
        );
        // Paths are compared by their spelling, which doubles a backslash, and
        // shown as written.
        for entry in &mut entries {
            match entry.key {
                "ep_libraries" if !ep_libraries.is_empty() => {
                    let shown: Vec<String> =
                        ep_libraries.iter().map(|path| shown_path(path)).collect();
                    entry.value =
                        serde_json::to_string(&shown).expect("a list of strings serializes");
                }
                "postgres_dir" => {
                    if let Some(prefix) = &postgres_dir {
                        entry.value = shown_path(prefix);
                    }
                }
                _ => {}
            }
        }

        Ok(Self {
            device,
            threads,
            ep_libraries,
            prepared,
            fused_attention,
            split_embeddings,
            model_idle,
            open_indexes,
            jit,
            postgres_dir,
            entries,
        })
    }

    /// What a client asks a running server to be running with: every
    /// setting, and whether it was chosen or left to the default.
    pub fn asked(&self) -> BTreeMap<String, Asked> {
        self.entries
            .iter()
            .map(|entry| {
                let asked = Asked {
                    value: entry.compared.clone(),
                    chosen: entry.source != Source::Default,
                };
                (entry.key.to_string(), asked)
            })
            .collect()
    }

    /// Why this server cannot answer a client that asked for `asked`, or
    /// `None` if it can.
    ///
    /// A value the client chose -- by flag, by `PAMIN_DEVICE`, or in the file
    /// -- has to be the one running. A value the client left to the default
    /// has to be too, unless the server was started with a flag for it: a
    /// server started as `pamin serve --model-idle 60`, or by a command given
    /// `--model-idle 60`, still answers commands that name no idle window,
    /// but one started from a file that has since lost the key does not.
    pub fn mismatch(&self, asked: &BTreeMap<String, Asked>) -> Option<String> {
        let differing: Vec<String> = self
            .entries
            .iter()
            .filter_map(|running| {
                let wanted = asked.get(running.key)?;
                let tolerated =
                    !wanted.chosen && matches!(running.source, Source::Flag | Source::Env);
                (wanted.value != running.compared && !tolerated).then(|| {
                    format!(
                        "{} = {} (running with {}, from {})",
                        running.key,
                        wanted.value,
                        running.value,
                        running.source.name()
                    )
                })
            })
            .collect();
        (!differing.is_empty()).then(|| {
            format!(
                "the running server was started with other settings than this command \
                 asks for: {}. Settings are fixed when the server starts: run `pamin \
                 stop`, then run the command again to start a server with them.",
                differing.join("; ")
            )
        })
    }
}

/// One setting as a client asks for it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Asked {
    pub value: String,
    /// Whether the client chose it, rather than leaving it to the default.
    pub chosen: bool,
}

/// The flags a command was given for the server, as arguments to the
/// `pamin serve` it starts. The file is not forwarded: the server reads it.
pub fn forwarded(args: &ServerArgs) -> Vec<std::ffi::OsString> {
    let mut forwarded: Vec<std::ffi::OsString> = Vec::new();
    let mut flag = |name: &str, value: std::ffi::OsString| {
        forwarded.push(format!("--{name}").into());
        forwarded.push(value);
    };
    if let Some(device) = args.device {
        flag("device", device.label().into());
    }
    if let Some(threads) = args.inference_threads {
        flag("inference-threads", threads.to_string().into());
    }
    for library in &args.ep_libraries {
        flag("ep-library", library.clone().into());
    }
    if let Some(prepared) = args.prepared {
        flag("prepared", prepared.to_string().into());
    }
    if let Some(fused) = args.fused_attention {
        flag("fused-attention", fused.to_string().into());
    }
    if let Some(split) = args.split_embeddings {
        flag("split-embeddings", split.to_string().into());
    }
    if let Some(idle) = args.model_idle {
        flag("model-idle", idle.to_string().into());
    }
    if let Some(open) = args.open_indexes {
        flag("open-indexes", open.to_string().into());
    }
    if let Some(jit) = args.jit {
        flag("jit", jit.to_string().into());
    }
    if let Some(dir) = &args.postgres_dir {
        flag("postgres-dir", dir.clone().into());
    }
    forwarded
}

/// What `pamin config` shows.
#[derive(Serialize)]
pub struct Shown {
    /// The file read, whether or not it exists, shown by
    /// [`shown_path`]: as written, or spelled byte for byte when
    /// a workspace path that is not UTF-8 could not otherwise serialize.
    pub file: String,
    pub exists: bool,
    pub settings: Vec<Entry>,
}

impl Shown {
    pub fn new(workspace: &Workspace, server: &Server) -> Self {
        let file = workspace.root().join(CONFIG);
        Self {
            exists: file.is_file(),
            file: shown_path(&file),
            settings: server.entries.clone(),
        }
    }

    pub fn render(&self) -> String {
        let width = self
            .settings
            .iter()
            .map(|entry| entry.key.len())
            .max()
            .unwrap_or(0);
        let mut out = format!(
            "{} ({})\n",
            self.file,
            if self.exists { "read" } else { "absent" }
        );
        for entry in &self.settings {
            out.push_str(&format!(
                "{:width$}  {}  [{}]\n",
                entry.key,
                entry.value,
                entry.source.name()
            ));
        }
        out.pop();
        out
    }
}

/// `path` made absolute: against the current directory when `base` is
/// `None`, otherwise against `base`.
fn anchored(path: &Path, base: Option<&Path>) -> Result<PathBuf> {
    let joined = match base {
        Some(base) => base.join(path),
        None => path.to_path_buf(),
    };
    std::path::absolute(&joined).with_context(|| format!("resolving {}", path.display()))
}

/// A list of libraries as a setting's value. An empty list is `none` however
/// it was reached, so `ep_libraries = []` and no setting at all compare equal.
///
/// Otherwise [`spelled_paths`], which is what a running server's
/// value is compared by, so a separator inside a path or a byte that is not
/// UTF-8 cannot make two lists of libraries one.
fn libraries(paths: &[PathBuf]) -> String {
    if paths.is_empty() {
        return "none".to_string();
    }
    spelled_paths(paths)
}

fn read(path: &Path) -> Result<File> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(File::default()),
        Err(error) => return Err(error).with_context(|| format!("reading {}", path.display())),
    };
    match toml::from_str(&text) {
        Ok(file) => Ok(file),
        Err(error) => bail!("{}: {error}", path.display()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::{CommandFactory, FromArgMatches, Parser};

    #[derive(Parser)]
    struct Test {
        #[command(flatten)]
        server: ServerArgs,
    }

    fn resolve(root: &Path, line: &[&str]) -> Result<Server> {
        let matches = Test::command()
            .try_get_matches_from(std::iter::once("pamin").chain(line.iter().copied()))?;
        let parsed = Test::from_arg_matches(&matches)?;
        Server::resolve(root, &parsed.server, &matches)
    }

    fn source(server: &Server, key: &str) -> Source {
        server
            .entries
            .iter()
            .find(|entry| entry.key == key)
            .expect("every key has an entry")
            .source
    }

    #[test]
    fn a_flag_beats_the_file_and_the_file_beats_the_default() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(
            root.path().join(CONFIG),
            "[serve]\nmodel_idle = 60\njit = true\nopen_indexes = 4\n",
        )
        .unwrap();
        let server = resolve(root.path(), &["--model-idle", "5"]).unwrap();
        assert_eq!(server.model_idle.get(), 5);
        assert_eq!(source(&server, "model_idle"), Source::Flag);
        assert!(server.jit);
        assert_eq!(source(&server, "jit"), Source::Config);
        assert_eq!(server.open_indexes, 4);
        assert_eq!(source(&server, "prepared"), Source::Default);
        let asked = server.asked();
        assert_eq!(asked["model_idle"].value, "5");
        assert!(asked["model_idle"].chosen);
        assert!(!asked["prepared"].chosen);
    }

    #[test]
    fn a_misspelled_or_invalid_setting_is_refused() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join(CONFIG), "[serve]\nmodel_idel = 60\n").unwrap();
        assert!(resolve(root.path(), &[]).is_err());
        std::fs::write(root.path().join(CONFIG), "[serve]\nmodel_idle = 0\n").unwrap();
        assert!(resolve(root.path(), &[]).is_err());
        std::fs::write(root.path().join(CONFIG), "[serve]\ndevice = \"gpu\"\n").unwrap();
        assert!(resolve(root.path(), &[]).is_err());
        std::fs::remove_file(root.path().join(CONFIG)).unwrap();
        assert!(resolve(root.path(), &["--device", "gpu"]).is_err());
        assert!(resolve(root.path(), &["--inference-threads", "0"]).is_err());
    }

    #[test]
    fn flags_are_forwarded_to_the_server_as_given() {
        let args = ServerArgs {
            device: Some(DeviceChoice::Cpu),
            model_idle: NonZeroU64::new(60),
            ..ServerArgs::default()
        };
        let line: Vec<String> = forwarded(&args)
            .into_iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        assert_eq!(line, ["--device", "cpu", "--model-idle", "60"]);
        let root = tempfile::tempdir().unwrap();
        let mut command: Vec<&str> = line.iter().map(String::as_str).collect();
        command.insert(0, "--jit=false");
        let server = resolve(root.path(), &command).unwrap();
        assert_eq!(server.model_idle.get(), 60);
    }

    /// A relative path is made absolute before it is stored or compared: a
    /// flag's against the directory the command ran in, the file's against
    /// the workspace.
    #[test]
    fn relative_paths_are_resolved_before_they_are_stored() {
        let root = tempfile::tempdir().unwrap();
        let here = std::env::current_dir().unwrap();
        let server = resolve(root.path(), &["--ep-library", "plugins/npu.so"]).unwrap();
        assert_eq!(server.ep_libraries, [here.join("plugins/npu.so")]);
        assert_eq!(
            server.asked()["ep_libraries"].value,
            spelled_paths(&[here.join("plugins/npu.so")])
        );

        installation(&root.path().join("pg"));
        std::fs::write(
            root.path().join(CONFIG),
            "[serve]\npostgres_dir = \"pg\"\nep_libraries = [\"npu.so\"]\n",
        )
        .unwrap();
        let server = resolve(root.path(), &[]).unwrap();
        assert_eq!(server.postgres_dir, Some(root.path().join("pg")));
        assert_eq!(server.ep_libraries, [root.path().join("npu.so")]);
    }

    /// `pamin --json config` in a workspace whose path is not UTF-8
    /// serializes, with the path spelled byte for byte.
    #[cfg(unix)]
    #[test]
    fn config_output_serializes_a_path_that_is_not_utf8() {
        use std::os::unix::ffi::OsStrExt;
        let root = tempfile::tempdir().unwrap();
        let home = root.path().join(std::ffi::OsStr::from_bytes(b"home-\x80"));
        let server = resolve(root.path(), &[]).unwrap();
        let shown = Shown::new(&Workspace::at(&home), &server);
        let json = serde_json::to_value(&shown).expect("serializes");
        assert!(
            json["file"]
                .as_str()
                .unwrap()
                .ends_with("home-\\x80/config.toml"),
            "{json}"
        );
    }

    /// A PostgreSQL prefix holding what the store runs from one.
    fn installation(prefix: &Path) {
        std::fs::create_dir_all(prefix.join("bin")).unwrap();
        for program in ["initdb", "pg_ctl"] {
            std::fs::write(prefix.join("bin").join(program), b"").unwrap();
        }
    }

    /// A PostgreSQL prefix that is not an installation is refused when the
    /// settings are resolved, naming where it came from and what it lacks,
    /// rather than recorded and then replaced by a downloaded installation.
    #[test]
    fn a_postgres_prefix_without_an_installation_is_refused() {
        let root = tempfile::tempdir().unwrap();
        let absent = root.path().join("absent");
        let Err(error) = resolve(root.path(), &["--postgres-dir", absent.to_str().unwrap()]) else {
            panic!("an absent prefix is refused");
        };
        let error = error.to_string();
        assert!(error.contains("--postgres-dir"), "{error}");
        assert!(error.contains("bin/initdb"), "{error}");

        let half = root.path().join("half");
        installation(&half);
        std::fs::remove_file(half.join("bin/pg_ctl")).unwrap();
        std::fs::write(
            root.path().join(CONFIG),
            format!("[serve]\npostgres_dir = {:?}\n", half.to_str().unwrap()),
        )
        .unwrap();
        let Err(error) = resolve(root.path(), &[]) else {
            panic!("a prefix without pg_ctl is refused");
        };
        let error = error.to_string();
        assert!(error.contains("config.toml: postgres_dir"), "{error}");
        assert!(error.contains("bin/pg_ctl"), "{error}");

        installation(&half);
        let server = resolve(root.path(), &[]).unwrap();
        assert_eq!(server.postgres_dir, Some(half));
    }

    /// A backslash in a PostgreSQL prefix or a library is shown as written,
    /// while what a running server is compared by stays the escaped spelling.
    #[cfg(unix)]
    #[test]
    fn a_path_setting_is_shown_as_written() {
        let root = tempfile::tempdir().unwrap();
        let prefix = root.path().join(r"pg\17");
        installation(&prefix);
        let server = resolve(root.path(), &["--postgres-dir", prefix.to_str().unwrap()]).unwrap();
        let entry = server
            .entries
            .iter()
            .find(|entry| entry.key == "postgres_dir")
            .unwrap();
        assert_eq!(entry.value, prefix.to_str().unwrap());
        assert_eq!(entry.compared, spelled_path(&prefix));
        assert_ne!(entry.value, entry.compared);
        assert_eq!(server.asked()["postgres_dir"].value, entry.compared);

        let library = root.path().join(r"ep\npu.so");
        let server = resolve(root.path(), &["--ep-library", library.to_str().unwrap()]).unwrap();
        let entry = server
            .entries
            .iter()
            .find(|entry| entry.key == "ep_libraries")
            .unwrap();
        let shown: Vec<String> = serde_json::from_str(&entry.value).unwrap();
        assert_eq!(shown, [library.to_str().unwrap()]);
        assert_eq!(entry.compared, spelled_paths(&[library]));
    }

    /// A comma in a library's path does not make it two libraries: a server
    /// with the one does not answer a client asking for the two.
    #[test]
    fn a_comma_in_a_library_path_keeps_it_one_library() {
        let root = tempfile::tempdir().unwrap();
        let one = resolve(root.path(), &["--ep-library", "/tmp/a, /tmp/b"]).unwrap();
        let two = resolve(
            root.path(),
            &["--ep-library", "/tmp/a", "--ep-library", "/tmp/b"],
        )
        .unwrap();
        assert_eq!(one.asked()["ep_libraries"].value, r#"["/tmp/a, /tmp/b"]"#);
        assert_eq!(two.asked()["ep_libraries"].value, r#"["/tmp/a","/tmp/b"]"#);
        assert!(one.mismatch(&two.asked()).is_some());
        assert!(two.mismatch(&one.asked()).is_some());
    }

    /// Paths that differ only in bytes that are not UTF-8 are different
    /// settings. How a path is spelled is `pamin_core::settings::spelled_path`'s,
    /// and tested there.
    #[cfg(unix)]
    #[test]
    fn a_path_that_is_not_utf8_is_compared_byte_for_byte() {
        use std::os::unix::ffi::OsStrExt;
        let path = |bytes: &[u8]| PathBuf::from(std::ffi::OsStr::from_bytes(bytes));
        let root = tempfile::tempdir().unwrap();
        let x80 = root.path().join(path(b"ep-\x80"));
        let x81 = root.path().join(path(b"ep-\x81"));
        installation(&x80);
        installation(&x81);
        let resolve_with = |library: &PathBuf, prefix: &PathBuf| {
            let matches = Test::command()
                .try_get_matches_from([
                    std::ffi::OsStr::new("pamin"),
                    "--ep-library".as_ref(),
                    library.as_os_str(),
                    "--postgres-dir".as_ref(),
                    prefix.as_os_str(),
                ])
                .unwrap();
            let parsed = Test::from_arg_matches(&matches).unwrap();
            Server::resolve(root.path(), &parsed.server, &matches).unwrap()
        };
        let one = resolve_with(&x80, &x80);
        let other = resolve_with(&x81, &x81);
        let message = one.mismatch(&other.asked()).expect("a mismatch");
        assert!(message.contains("ep_libraries"), "{message}");
        assert!(message.contains("postgres_dir"), "{message}");
    }

    /// `ep_libraries = []` is the same setting as none at all, so a server
    /// started from either answers a client with the other.
    #[test]
    fn an_empty_library_list_is_spelled_as_none() {
        let root = tempfile::tempdir().unwrap();
        let unset = resolve(root.path(), &[]).unwrap();
        std::fs::write(root.path().join(CONFIG), "[serve]\nep_libraries = []\n").unwrap();
        let empty = resolve(root.path(), &[]).unwrap();
        assert_eq!(empty.asked()["ep_libraries"].value, "none");
        assert_eq!(empty.mismatch(&unset.asked()), None);
        assert_eq!(unset.mismatch(&empty.asked()), None);
    }

    #[test]
    fn a_running_server_answers_only_a_client_it_matches() {
        let root = tempfile::tempdir().unwrap();
        let running = resolve(root.path(), &["--model-idle", "60"]).unwrap();
        let client = |line: &[&str]| resolve(root.path(), line).unwrap().asked();

        // Nothing chosen: the server's own flag stands.
        assert_eq!(running.mismatch(&client(&[])), None);
        assert_eq!(running.mismatch(&client(&["--model-idle", "60"])), None);
        assert_eq!(running.mismatch(&BTreeMap::new()), None);

        let message = running
            .mismatch(&client(&["--model-idle", "5"]))
            .expect("a mismatch");
        assert!(
            message.contains("model_idle = 5 (running with 60, from flag)"),
            "{message}"
        );
        assert!(message.contains("pamin stop"), "{message}");

        // A server that took a value from the file does not answer a client
        // that no longer finds it there.
        std::fs::write(root.path().join(CONFIG), "[serve]\njit = true\n").unwrap();
        let from_file = resolve(root.path(), &[]).unwrap();
        std::fs::remove_file(root.path().join(CONFIG)).unwrap();
        assert!(from_file.mismatch(&client(&[])).is_some());
    }
}
