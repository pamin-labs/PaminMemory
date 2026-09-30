//! Index errors.

/// Anything that can go wrong talking to the projection index.
#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    #[error("incompatible compute plan: {0}")]
    Incompatible(String),
    #[error("numerical accelerator validation: {0}")]
    Numerical(String),
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

impl IndexError {
    pub(crate) fn context(self, context: &str) -> Self {
        match self {
            Self::Incompatible(message) => Self::Incompatible(format!("{context}: {message}")),
            Self::Numerical(message) => Self::Numerical(format!("{context}: {message}")),
            other => Self::Engine(format!("{context}: {other}")),
        }
    }
}

impl From<zvec_rust::Error> for IndexError {
    fn from(error: zvec_rust::Error) -> Self {
        Self::Engine(error.to_string())
    }
}

/// Result alias for index operations.
pub type Result<T> = std::result::Result<T, IndexError>;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn loader_context_preserves_plan_failure_classification() {
        assert!(matches!(
            IndexError::Incompatible("shape".into()).context("loader"),
            IndexError::Incompatible(_)
        ));
        assert!(matches!(
            IndexError::Numerical("drift".into()).context("loader"),
            IndexError::Numerical(_)
        ));
        assert!(matches!(
            IndexError::Engine("pressure".into()).context("loader"),
            IndexError::Engine(_)
        ));
    }
}
