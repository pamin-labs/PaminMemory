//! Domain model, version ledger semantics, and retrieval fusion for PaminMemory.
//!
//! This crate holds the parts that must stay stable regardless of which store or
//! index sits behind them, and it deliberately carries no heavy dependencies:
//! it is the crate edited most often, so its rebuild cost sets the development
//! loop. That is also why the command-line client can share it: the settings,
//! search depths and workspace layout the client shows and the libraries
//! honour are defined here, once, so the client links none of the libraries.

pub mod cascade;
pub mod channel;
pub mod filter;
pub mod fusion;
pub mod graph;
pub mod id;
pub mod ledger;
pub mod search;
pub mod settings;
pub mod version;
pub mod workspace;

pub use cascade::{JobKind, LAGGING_AT, MAX_ATTEMPTS, may_defer};
pub use channel::{Channel, ChannelResults, Scored};
pub use filter::{Rejection, SensoryFilter, Verdict};
pub use fusion::{Combine, DEFAULT_K, FusedResult, Fusion, Why, sort_results};
pub use graph::{Derivation, EdgeKind, Relationship, RelationshipVersion, TombstoneReason};
pub use id::{
    IndexJobId, ProjectId, RelationshipId, RelationshipVersionId, SourceId, SourceSpanId,
    SourceVersionId, TopicId, TopicStateId,
};
pub use ledger::{
    FilterDecision, Project, Source, SourceKind, SourceSpan, SourceVersion, Topic, TopicState,
    Validity,
};
pub use search::Depths;
pub use settings::{DeviceChoice, VectorIndex};
pub use version::{ResolvedVersion, VersionOffset, resolve};
pub use workspace::Workspace;
