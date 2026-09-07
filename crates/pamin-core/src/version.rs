//! Version resolution over a topic's ledger.
//!
//! Latest is computed from the newest undeleted version rather than stored in a
//! flag. A stored `is_latest` column has to be maintained transactionally on
//! every insert and every soft delete, and any path that forgets leaves two rows
//! claiming to be current.

use serde::{Deserialize, Serialize};

/// How far back from the latest version to read. Zero means latest.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct VersionOffset(pub u32);

impl VersionOffset {
    /// The latest version.
    pub const LATEST: Self = Self(0);
}

impl From<u32> for VersionOffset {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

/// Which version an offset selected, and how it relates to the rest of the ledger.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedVersion {
    /// The version number that was selected.
    pub version: u32,
    /// The offset actually applied. Differs from the request when it ran past
    /// the oldest version, so a caller can tell "three back" from "as far back
    /// as this topic goes".
    pub actual_offset: VersionOffset,
    /// Whether the selected version is the current one.
    pub is_current: bool,
    pub oldest_version: u32,
    pub latest_version: u32,
    /// How many undeleted versions the topic has.
    pub available_versions: u32,
}

/// Resolves an offset against a topic's undeleted version numbers.
///
/// `versions` must be sorted ascending and contain no soft-deleted versions.
/// Returns `None` only when every version has been deleted, which is the one
/// case with nothing to return rather than something to clamp to.
pub fn resolve(versions: &[u32], offset: VersionOffset) -> Option<ResolvedVersion> {
    let latest_index = versions.len().checked_sub(1)?;

    // Running past the oldest version clamps rather than errors: an agent asking
    // for more history than exists wants the oldest state, not a failure. The
    // reported offset tells it how far it actually got.
    let index = latest_index.saturating_sub(offset.0 as usize);

    Some(ResolvedVersion {
        version: versions[index],
        actual_offset: VersionOffset((latest_index - index) as u32),
        is_current: index == latest_index,
        oldest_version: versions[0],
        latest_version: versions[latest_index],
        available_versions: versions.len() as u32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latest_is_the_newest_undeleted_version() {
        let resolved = resolve(&[1, 2, 3], VersionOffset::LATEST).unwrap();
        assert_eq!(resolved.version, 3);
        assert!(resolved.is_current);
        assert_eq!(resolved.actual_offset, VersionOffset(0));
        assert_eq!(resolved.available_versions, 3);
    }

    #[test]
    fn an_offset_steps_back_from_latest() {
        let resolved = resolve(&[1, 2, 3], VersionOffset(1)).unwrap();
        assert_eq!(resolved.version, 2);
        assert!(!resolved.is_current);
        assert_eq!(resolved.actual_offset, VersionOffset(1));
    }

    #[test]
    fn an_offset_past_the_oldest_clamps_and_reports_how_far_it_got() {
        let resolved = resolve(&[1, 2, 3], VersionOffset(9)).unwrap();
        assert_eq!(resolved.version, 1);
        assert_eq!(resolved.actual_offset, VersionOffset(2));
        assert!(!resolved.is_current);
    }

    #[test]
    fn deleting_the_latest_promotes_the_previous_version() {
        // Version 3 was soft deleted, so the caller passes the surviving ones.
        let resolved = resolve(&[1, 2], VersionOffset::LATEST).unwrap();
        assert_eq!(resolved.version, 2);
        assert!(resolved.is_current);
        assert_eq!(resolved.latest_version, 2);
    }

    #[test]
    fn version_numbers_need_not_be_contiguous() {
        // Soft deletes leave gaps; offsets count surviving versions, not numbers.
        let resolved = resolve(&[1, 4, 7], VersionOffset(1)).unwrap();
        assert_eq!(resolved.version, 4);
        assert_eq!(resolved.oldest_version, 1);
        assert_eq!(resolved.latest_version, 7);
    }

    #[test]
    fn a_fully_deleted_topic_resolves_to_nothing() {
        assert!(resolve(&[], VersionOffset::LATEST).is_none());
    }
}
