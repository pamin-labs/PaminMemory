//! What the server runs with: its flags, the workspace's `config.toml`, and
//! the defaults, in that order.
//!
//! The one place the binary turns input into [`pamin_engine::Settings`]; the
//! libraries read no environment. A setting that governs the server process --
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
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::parser::ValueSource;
use pamin_index::DeviceChoice;

use crate::session::OPEN_INDEXES;
use serde::{Deserialize, Serialize};

/// The configuration file, relative to the workspace root.
pub const CONFIG: &str = "config.toml";

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
    pub value: String,
    pub source: Source,
}

/// What a server runs with, and where each value came from.
#[derive(Clone, Debug)]
pub struct Server {
    pub settings: pamin_engine::Settings,
    pub open_indexes: usize,
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
            entries.push(Entry { key, value, source });
        };

        let default = pamin_engine::Settings::default();
        let serve = file.serve;

        let device = match (&args.device, &serve.device) {
            (Some(device), _) => *device,
            (None, Some(name)) => DeviceChoice::parse(name).with_context(|| {
                format!(
                    "{CONFIG}: unknown device {name:?}; expected one of {}",
                    DeviceChoice::ALL.map(DeviceChoice::label).join(", ")
                )
            })?,
            (None, None) => default.inference.device,
        };
        pick(
            "device",
            args.device.map(|d| d.label().to_string()),
            given("device"),
            serve.device.as_ref().map(|_| device.label().to_string()),
            default.inference.device.label().to_string(),
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
            .unwrap_or(default.inference.prepared);
        pick(
            "prepared",
            args.prepared.map(|b| b.to_string()),
            given("prepared"),
            serve.prepared.map(|b| b.to_string()),
            default.inference.prepared.to_string(),
        );

        let fused_attention = args
            .fused_attention
            .unwrap_or(default.inference.fused_attention);
        pick(
            "fused_attention",
            args.fused_attention.map(|b| b.to_string()),
            given("fused_attention"),
            None,
            default.inference.fused_attention.to_string(),
        );

        let model_idle = args
            .model_idle
            .or(serve.model_idle)
            .map_or(default.model_idle, |seconds| {
                Duration::from_secs(seconds.get())
            });
        pick(
            "model_idle",
            args.model_idle.map(|n| n.to_string()),
            given("model_idle"),
            serve.model_idle.map(|n| n.to_string()),
            default.model_idle.as_secs().to_string(),
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

        let jit = args.jit.or(serve.jit).unwrap_or(default.store.jit);
        pick(
            "jit",
            args.jit.map(|b| b.to_string()),
            given("jit"),
            serve.jit.map(|b| b.to_string()),
            default.store.jit.to_string(),
        );

        let flag_postgres = args.postgres_dir.as_ref().map(from_flag).transpose()?;
        let file_postgres = serve.postgres_dir.as_ref().map(from_file).transpose()?;
        let postgres_dir = flag_postgres.clone().or(file_postgres.clone());
        // Refused here rather than when the database starts, where a prefix
        // that is not an installation is an error in a log nobody reads.
        if let Some(prefix) = &postgres_dir
            && let Some(missing) = pamin_store::installation_missing(prefix)
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
            flag_postgres.as_deref().map(pamin_index::spelled_path),
            given("postgres_dir"),
            file_postgres.as_deref().map(pamin_index::spelled_path),
            "none (installs its own)".to_string(),
        );

        let settings = pamin_engine::Settings {
            inference: pamin_index::InferenceSettings {
                device,
                threads,
                ep_libraries,
                prepared,
                fused_attention,
                ..default.inference
            },
            store: pamin_store::StoreSettings { jit, postgres_dir },
            model_idle,
            ..default
        };
        Ok(Self {
            settings,
            open_indexes,
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
                    value: entry.value.clone(),
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
                (wanted.value != running.value && !tolerated).then(|| {
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
    /// The file read, whether or not it exists, spelled by
    /// [`pamin_index::spelled_path`] so a workspace path that is not UTF-8
    /// still serializes.
    pub file: String,
    pub exists: bool,
    pub settings: Vec<Entry>,
}

impl Shown {
    pub fn new(workspace: &pamin_store::Workspace, server: &Server) -> Self {
        let file = workspace.root().join(CONFIG);
        Self {
            exists: file.is_file(),
            file: pamin_index::spelled_path(&file),
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
/// Otherwise [`pamin_index::spelled_paths`], which is what a running server's
/// value is compared by, so a separator inside a path or a byte that is not
/// UTF-8 cannot make two lists of libraries one.
fn libraries(paths: &[PathBuf]) -> String {
    if paths.is_empty() {
        return "none".to_string();
    }
    pamin_index::spelled_paths(paths)
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
    fn nothing_set_is_the_library_defaults() {
        let root = tempfile::tempdir().unwrap();
        let server = resolve(root.path(), &[]).unwrap();
        // `PAMIN_DEVICE` may be set by whoever runs the tests; nothing else is
        // read from the environment.
        let settings = pamin_engine::Settings {
            inference: pamin_index::InferenceSettings {
                device: server.settings.inference.device,
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(server.settings, settings);
        assert_eq!(server.open_indexes, OPEN_INDEXES);
        assert_eq!(source(&server, "model_idle"), Source::Default);
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
        assert_eq!(server.settings.model_idle, Duration::from_secs(5));
        assert_eq!(source(&server, "model_idle"), Source::Flag);
        assert!(server.settings.store.jit);
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
        assert_eq!(server.settings.model_idle, Duration::from_secs(60));
    }

    /// A relative path is made absolute before it is stored or compared: a
    /// flag's against the directory the command ran in, the file's against
    /// the workspace.
    #[test]
    fn relative_paths_are_resolved_before_they_are_stored() {
        let root = tempfile::tempdir().unwrap();
        let here = std::env::current_dir().unwrap();
        let server = resolve(root.path(), &["--ep-library", "plugins/npu.so"]).unwrap();
        assert_eq!(
            server.settings.inference.ep_libraries,
            [here.join("plugins/npu.so")]
        );
        assert_eq!(
            server.asked()["ep_libraries"].value,
            pamin_index::spelled_paths(&[here.join("plugins/npu.so")])
        );

        installation(&root.path().join("pg"));
        std::fs::write(
            root.path().join(CONFIG),
            "[serve]\npostgres_dir = \"pg\"\nep_libraries = [\"npu.so\"]\n",
        )
        .unwrap();
        let server = resolve(root.path(), &[]).unwrap();
        assert_eq!(
            server.settings.store.postgres_dir,
            Some(root.path().join("pg"))
        );
        assert_eq!(
            server.settings.inference.ep_libraries,
            [root.path().join("npu.so")]
        );
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
        let shown = Shown::new(&pamin_store::Workspace::at(&home), &server);
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
        assert_eq!(server.settings.store.postgres_dir, Some(half));
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
    /// settings. How a path is spelled is `pamin_index::spelled_path`'s, and
    /// tested there.
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
