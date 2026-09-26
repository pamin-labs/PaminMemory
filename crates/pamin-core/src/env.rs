//! Settings read from the environment.
//!
//! Every one of them is a tuning knob on a measured default -- a width, a
//! batch, a bound, an idle window -- or a switch a measurement uses to turn one
//! mechanism off. So each reads the same way: a value that does not parse is
//! ignored rather than refused, because a typo in a performance knob should
//! not stop a search or a long-running server from working.

use std::str::FromStr;

/// The positive number `name` holds, or `None` when it is unset or holds
/// anything else.
///
/// Zero counts as anything else: for every knob read this way zero is a
/// setting that would break something -- a window that releases a model the
/// tick after it loads, a bound that evicts what was just opened -- rather
/// than a smaller value of it. Surrounding whitespace is ignored.
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

/// Whether `name` is set to `value`, ignoring ASCII case.
///
/// For the switches that turn a mechanism off for a measurement:
/// `PAMIN_PREPARED=off`, `PAMIN_DEVICE=cpu`. Unset, or set to anything else,
/// is `false`.
pub fn is(name: &str, value: &str) -> bool {
    std::env::var(name).is_ok_and(|set| set.eq_ignore_ascii_case(value))
}

#[cfg(test)]
mod tests {
    use super::{is, parse_positive};

    #[test]
    fn a_positive_number_is_read_whatever_the_width() {
        assert_eq!(parse_positive::<usize>(Some("4")), Some(4));
        assert_eq!(parse_positive::<usize>(Some("  4 ")), Some(4));
        assert_eq!(parse_positive::<u64>(Some("300")), Some(300));
        assert_eq!(parse_positive::<i32>(Some("700")), Some(700));
    }

    #[test]
    fn anything_that_is_not_a_positive_number_is_ignored() {
        for raw in [
            None,
            Some(""),
            Some("0"),
            Some("-1"),
            Some("lots"),
            Some("4.5"),
        ] {
            assert_eq!(parse_positive::<usize>(raw), None, "{raw:?} for usize");
            assert_eq!(parse_positive::<i32>(raw), None, "{raw:?} for i32");
        }
    }

    #[test]
    fn a_number_too_wide_for_the_type_is_ignored() {
        assert_eq!(parse_positive::<i32>(Some("4294967296")), None);
    }

    #[test]
    fn a_switch_is_read_ignoring_case() {
        const NAME: &str = "PAMIN_CORE_ENV_SWITCH_TEST";
        assert!(!is(NAME, "off"), "unset is not a switch");

        // SAFETY: nothing else in this process reads or writes this variable.
        unsafe { std::env::set_var(NAME, "OFF") };
        let (off, on) = (is(NAME, "off"), is(NAME, "on"));
        unsafe { std::env::remove_var(NAME) };

        assert!(off, "OFF should read as off");
        assert!(!on, "OFF should not read as on");
    }
}
