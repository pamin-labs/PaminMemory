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

    /// The index was built with an embedding model this build does not run.
    #[error("{}", model_mismatch(.indexed))]
    ModelMismatch { indexed: String },

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

/// What [`IndexError::ModelMismatch`] says: the retired profile by name when
/// the recorded model is one a profile shipped, since that is what a user
/// chose and will recognise, and the model id otherwise.
fn model_mismatch(indexed: &str) -> String {
    let current = crate::embedding::MODEL_ID;
    let built = match crate::embedding::retired_profile(indexed) {
        Some(profile) => format!("the retired `{profile}` profile's embedding model ({indexed})"),
        None => format!("embedding model {indexed}"),
    };
    format!(
        "this index was built with {built}, and every project now uses BGE-M3 ({current}); \
         run `pamin reindex` to rebuild it"
    )
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

    /// An index from a retired profile is named by the profile the user
    /// chose, and the remedy is the same for any model this build lacks.
    #[test]
    fn a_retired_model_is_named_with_its_profile_and_the_remedy() {
        for (model, profile) in [
            ("intfloat/multilingual-e5-small+p1", "speed"),
            ("intfloat/multilingual-e5-base+p1", "balanced"),
            (
                "bge-m3-int8@2b34e84df040034d4b9eabb62383a87c18955822+pplx-0.6b@2c4d510dd4a732063c31a0f70193e35067b51fd8:pool-int8-single-v2-level4",
                "dual_accuracy",
            ),
        ] {
            let message = IndexError::ModelMismatch {
                indexed: model.to_string(),
            }
            .to_string();
            assert!(
                message.contains(&format!("`{profile}` profile")),
                "{message}"
            );
            assert!(message.contains(model), "{message}");
            assert!(message.contains("pamin reindex"), "{message}");
        }

        let unknown = IndexError::ModelMismatch {
            indexed: "someone/else".to_string(),
        }
        .to_string();
        assert!(!unknown.contains("profile"), "{unknown}");
        assert!(unknown.contains("someone/else"), "{unknown}");
        assert!(unknown.contains("pamin reindex"), "{unknown}");
    }
}
