//! Mapping domain enums onto the TEXT columns that store them.
//!
//! Every such column carries a CHECK constraint naming the same set of labels,
//! so the SQL and the Rust have to agree. Writing the two directions by hand
//! per enum is how they stop agreeing: a variant gets added, one of the two
//! matches gets updated, and the mismatch surfaces as a value that reads back
//! as something else. Here one list generates both directions, and a test
//! holds the SQL to the same list.

/// A domain enum with a stable textual representation in the database.
pub(crate) trait SqlLabel: Sized {
    /// The label written to the column.
    fn label(self) -> &'static str;

    /// Reads a label back.
    ///
    /// `None` means the column held something its CHECK constraint should have
    /// rejected. Callers decide what to do about that; nothing here guesses.
    fn from_label(label: &str) -> Option<Self>;

    /// Every label this enum can produce. Exists for the drift test below.
    #[cfg(test)]
    fn labels() -> &'static [&'static str];
}

macro_rules! sql_enum {
    ($ty:ident { $($variant:ident => $label:literal),+ $(,)? }) => {
        impl crate::sql::SqlLabel for $ty {
            fn label(self) -> &'static str {
                match self {
                    $($ty::$variant => $label,)+
                }
            }

            fn from_label(label: &str) -> Option<Self> {
                match label {
                    $($label => Some($ty::$variant),)+
                    _ => None,
                }
            }

            #[cfg(test)]
            fn labels() -> &'static [&'static str] {
                &[$($label),+]
            }
        }
    };
}

pub(crate) use sql_enum;

#[cfg(test)]
mod tests {
    use super::SqlLabel;
    use pamin_core::{Derivation, EdgeKind, FilterDecision, TombstoneReason};

    /// Enums whose column is constrained to their label set.
    ///
    /// `SourceKind` is absent on purpose: kinds beyond `manual` arrive with an
    /// ingest path that does not exist yet, so its column stays open until one
    /// does.
    fn constrained() -> Vec<&'static [&'static str]> {
        vec![
            FilterDecision::labels(),
            EdgeKind::labels(),
            Derivation::labels(),
            TombstoneReason::labels(),
        ]
    }

    fn migrations() -> String {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/migrations");
        std::fs::read_dir(dir)
            .expect("migrations directory")
            .filter_map(|entry| {
                let path = entry.expect("directory entry").path();
                (path.extension()? == "sql").then(|| std::fs::read_to_string(&path).expect("sql"))
            })
            .collect()
    }

    /// A label the schema rejects is a value that cannot be written, and a
    /// constraint listing a label no variant produces is a rule nothing
    /// enforces. Both are silent: each half compiles and runs on its own.
    #[test]
    fn every_constrained_label_appears_in_the_schema() {
        let sql = migrations();
        let mut checked = 0;

        for labels in constrained() {
            for label in labels {
                assert!(
                    sql.contains(&format!("'{label}'")),
                    "no migration admits the label {label:?}"
                );
                checked += 1;
            }
        }

        assert!(checked > 0, "no labels were checked");
    }
}
