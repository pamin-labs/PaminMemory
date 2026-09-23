//! Fetching a model's files from the hub.
//!
//! Both kinds of model in this crate are downloaded on first use into the
//! workspace's model directory and read from there afterwards. One place does
//! that, so the two cannot disagree about where the weights live.

use std::path::{Path, PathBuf};

use fastembed::TokenizerFiles;

use crate::error::{IndexError, Result};

/// One model repository on the hub, cached under a workspace's model directory.
pub(crate) struct Repository {
    repo: hf_hub::api::sync::ApiRepo,
    name: String,
}

impl Repository {
    pub(crate) fn open(cache_dir: &Path, name: &str) -> Result<Self> {
        let repo = hf_hub::api::sync::ApiBuilder::new()
            .with_cache_dir(cache_dir.to_path_buf())
            .with_progress(false)
            .build()
            .map_err(|error| IndexError::Engine(format!("reaching the model hub: {error}")))?
            .model(name.to_string());
        Ok(Self {
            repo,
            name: name.to_string(),
        })
    }

    /// The local path of one of the repository's files, downloading it first
    /// if it is not cached yet.
    pub(crate) fn get(&self, file: &str) -> Result<PathBuf> {
        self.repo.get(file).map_err(|error| {
            IndexError::Engine(format!("fetching {file} from {}: {error}", self.name))
        })
    }

    /// The four files a tokenizer is built from, read into memory.
    pub(crate) fn tokenizer(&self) -> Result<TokenizerFiles> {
        let read = |file: &str| -> Result<Vec<u8>> { Ok(std::fs::read(self.get(file)?)?) };
        Ok(TokenizerFiles {
            tokenizer_file: read("tokenizer.json")?,
            config_file: read("config.json")?,
            special_tokens_map_file: read("special_tokens_map.json")?,
            tokenizer_config_file: read("tokenizer_config.json")?,
        })
    }
}
