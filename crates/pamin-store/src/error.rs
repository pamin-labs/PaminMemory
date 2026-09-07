//! Store errors.

/// Anything that can go wrong talking to the authoritative store.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("database error: {0}")]
    Database(#[from] tokio_postgres::Error),

    #[error("migration failed: {0}")]
    Migration(#[from] refinery::Error),

    #[error("embedded postgres: {0}")]
    EmbeddedPostgres(#[from] postgresql_embedded::Error),

    #[error("workspace io: {0}")]
    Io(#[from] std::io::Error),
}

/// Result alias for store operations.
pub type Result<T> = std::result::Result<T, StoreError>;
