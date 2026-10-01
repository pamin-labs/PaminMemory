//! Cargo profile overrides included in the inference build identity.
use std::collections::BTreeMap;

pub(crate) fn settings(
    profile: &str,
    environment: impl IntoIterator<Item = (String, String)>,
) -> BTreeMap<String, String> {
    let mut settings = BTreeMap::new();
    // Declare absent overrides too, so adding one invalidates the build script.
    for name in [
        "DEV",
        "RELEASE",
        "TEST",
        "BENCH",
        &profile.to_uppercase().replace('-', "_"),
    ] {
        for field in [
            "OPT_LEVEL",
            "DEBUG",
            "SPLIT_DEBUGINFO",
            "STRIP",
            "DEBUG_ASSERTIONS",
            "OVERFLOW_CHECKS",
            "LTO",
            "PANIC",
            "INCREMENTAL",
            "CODEGEN_UNITS",
            "RPATH",
        ] {
            settings.insert(format!("CARGO_PROFILE_{name}_{field}"), String::new());
            settings.insert(
                format!("CARGO_PROFILE_{name}_BUILD_OVERRIDE_{field}"),
                String::new(),
            );
        }
    }
    for (key, value) in environment {
        if key.starts_with("CARGO_PROFILE_") {
            settings.insert(key, value);
        }
    }
    settings
}

#[cfg(test)]
mod tests {
    use super::settings;

    #[test]
    fn profile_codegen_overrides_change_identity_inputs() {
        let baseline = settings("release", []);
        for (key, value) in [
            ("CARGO_PROFILE_RELEASE_LTO", "thin"),
            ("CARGO_PROFILE_RELEASE_CODEGEN_UNITS", "1"),
            ("CARGO_PROFILE_CUSTOM_LTO", "fat"),
        ] {
            assert_ne!(baseline, settings("release", [(key.into(), value.into())]));
        }
        assert!(baseline.contains_key("CARGO_PROFILE_RELEASE_LTO"));
        assert_eq!(
            baseline,
            settings("release", [("PAMIN_DEVICE".into(), "cpu".into())])
        );
    }

    #[test]
    fn profile_override_order_does_not_change_identity_inputs() {
        let overrides = [
            ("CARGO_PROFILE_RELEASE_LTO".into(), "thin".into()),
            ("CARGO_PROFILE_RELEASE_CODEGEN_UNITS".into(), "1".into()),
        ];
        assert_eq!(
            settings("release", overrides.clone()),
            settings("release", overrides.into_iter().rev())
        );
    }
}
