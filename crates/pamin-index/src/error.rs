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
