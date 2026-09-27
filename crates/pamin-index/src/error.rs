//! Index errors.

/// Anything that can go wrong talking to the projection index.
#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    #[error("projection index: {0}")]
    Engine(String),

    #[error(
        "index was built with embedding model {indexed} but {requested} was requested; \
         run `pamin reindex` to rebuild it"
    )]
    ProfileMismatch { indexed: String, requested: String },

    #[error(
        "this index holds one document per {indexed} and this build expects one per {expected}; \
         run `pamin reindex` to rebuild it"
    )]
    GrainMismatch { indexed: String, expected: String },

    #[error(
        "this index was built as vector index {indexed} and {requested} was requested; \
         run `pamin reindex` to rebuild it"
    )]
    VectorIndexMismatch { indexed: String, requested: String },

    #[error(
        "this workspace has an index from before projects were separated; \
         run `pamin reindex` to rebuild it per project"
    )]
    LegacyLayout,

    #[error(
        "another process is holding this project's index and did not release it \
         in time ({0}); if a pamin server was just stopped or replaced, retry"
    )]
    Busy(String),

    #[error(
        "this project's index could not be reopened after it was reshaped ({0}); \
         restart the server, or run `pamin reindex` to rebuild it"
    )]
    Unavailable(String),

    #[error("index io: {0}")]
    Io(#[from] std::io::Error),
}

impl From<zvec_rust::Error> for IndexError {
    fn from(error: zvec_rust::Error) -> Self {
        Self::Engine(error.to_string())
    }
}

/// Result alias for index operations.
pub type Result<T> = std::result::Result<T, IndexError>;
