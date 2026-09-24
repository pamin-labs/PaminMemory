//! Numeric settings read from the environment.

use std::str::FromStr;

/// A positive number from the environment variable `name`, or `None` when it
/// is unset or is not one.
///
/// Every numeric knob this workspace reads from the environment has the same
/// contract: unset means the shipped constant, and a value that is not a
/// positive number is ignored rather than refused -- a typo in a tuning
/// setting should not stop a search, and a long-running server that refuses
/// to start over one has failed worse than one that ignores it. Zero counts as
/// not a number here: every one of these is a size, a width or a window, and
/// zero of any of them is a broken process rather than a smaller one. The
/// caller supplies its constant with `unwrap_or`.
///
/// Read on every call and never cached, and that is a requirement rather than
/// an oversight: the evaluation harnesses change several of these between
/// passes in one process -- `PAMIN_SEARCH_EFFORT` between the widths
/// `monolingual.rs` compares, on an engine that stays open -- and a value
/// cached at first read would silently measure the first width every time.
pub fn positive<T: FromStr + PartialOrd + Default>(name: &str) -> Option<T> {
    parse_positive(std::env::var(name).ok().as_deref())
}

/// Split from the lookup so it can be tested without setting a variable the
/// rest of the process shares.
fn parse_positive<T: FromStr + PartialOrd + Default>(raw: Option<&str>) -> Option<T> {
    raw?.trim()
        .parse::<T>()
        .ok()
        .filter(|value| *value > T::default())
}

#[cfg(test)]
mod tests {
    use super::parse_positive;

    #[test]
    fn a_positive_number_is_read_whatever_surrounds_it() {
        assert_eq!(parse_positive::<usize>(Some("4")), Some(4));
        assert_eq!(parse_positive::<usize>(Some("  4 ")), Some(4));
        assert_eq!(parse_positive::<i32>(Some("700")), Some(700));
    }

    #[test]
    fn anything_that_is_not_a_positive_number_is_ignored() {
        for raw in [None, Some(""), Some("0"), Some("-1"), Some("lots")] {
            assert_eq!(parse_positive::<usize>(raw), None, "{raw:?} was read");
            assert_eq!(parse_positive::<i32>(raw), None, "{raw:?} was read");
        }
    }
}
