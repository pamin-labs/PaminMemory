//! Compare the same embedding export on the automatic provider and CPU.
//! Each arm runs in its own process; native inference threads never see an
//! environment mutation after initialization.
use pamin_index::{Device, Embedder, Profile};

#[test]
#[ignore = "loads embedding models; PAMIN_TEST_MODEL_CACHE reuses an existing cache"]
fn accelerated_embeddings_preserve_the_embedding_space() {
    if let Some(output) = std::env::var_os("PAMIN_EMBEDDING_TEST_OUTPUT") {
        let models = std::path::PathBuf::from(
            std::env::var_os("PAMIN_TEST_MODEL_CACHE").expect("the parent supplies a cache"),
        );
        let profile = Profile::parse(
            &std::env::var("PAMIN_TEST_PROFILE").unwrap_or_else(|_| "accuracy".into()),
        )
        .unwrap();
        let mut model = Embedder::load(profile, &models).expect("embedding model");
        if std::env::var("PAMIN_DEVICE").as_deref() == Ok("cpu") {
            assert_eq!(model.device(), Device::Cpu);
        } else {
            if let Ok(expected) = std::env::var("PAMIN_EXPECT_DEVICE") {
                if expected == "accelerated" {
                    assert_ne!(
                        model.device(),
                        Device::Cpu,
                        "explicit accelerator premise failed"
                    );
                } else {
                    assert_eq!(
                        model.device().name(),
                        expected,
                        "unexpected provider fallback"
                    );
                }
            }
            eprintln!("automatic embedding provider: {}", model.device().name());
        }
        let mut vectors = Vec::new();
        for text in texts() {
            vectors.push(model.embed_passage(text).expect("passage embedding"));
            vectors.push(model.embed_query(text).expect("query embedding"));
        }
        std::fs::write(output, serde_json::to_vec(&vectors).unwrap()).unwrap();
        return;
    }
    assert!(
        std::env::var_os("PAMIN_DEVICE").is_none(),
        "unset PAMIN_DEVICE for the accelerator arm"
    );
    let temporary = tempfile::tempdir().unwrap();
    let models = std::env::var_os("PAMIN_TEST_MODEL_CACHE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| temporary.path().join("models"));
    let profile =
        Profile::parse(&std::env::var("PAMIN_TEST_PROFILE").unwrap_or_else(|_| "accuracy".into()))
            .unwrap();
    let run = |arm: &str| {
        let output = temporary.path().join(format!("{arm}.json"));
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "accelerated_embeddings_preserve_the_embedding_space",
                "--exact",
                "--ignored",
                "--nocapture",
            ])
            .env("PAMIN_EMBEDDING_TEST_OUTPUT", &output)
            .env("PAMIN_TEST_MODEL_CACHE", &models);
        if arm == "cpu" {
            command.env("PAMIN_DEVICE", "cpu");
        } else {
            command.env_remove("PAMIN_DEVICE");
        }
        assert!(command.status().unwrap().success(), "{arm} arm failed");
        let vectors: Vec<Vec<f32>> =
            serde_json::from_slice(&std::fs::read(output).unwrap()).unwrap();
        assert_eq!(vectors.len(), texts().len() * 2);
        vectors
    };
    let chosen = run("automatic");
    let cpu = run("cpu");
    for (index, (actual, expected)) in chosen.iter().zip(&cpu).enumerate() {
        assert_eq!(actual.len(), profile.dimensions() as usize);
        assert_eq!(actual.len(), expected.len());
        assert!(actual.iter().chain(expected).all(|v| v.is_finite()));
        let dot: f64 = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        let norm = |values: &[f32]| {
            values
                .iter()
                .map(|v| f64::from(*v).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        let cosine = dot / (norm(actual) * norm(expected));
        println!("case {index} cosine {cosine:.9}");
        assert!(
            cosine >= 0.99999,
            "accelerator changed the embedding space: {cosine}"
        );
    }
}

fn texts() -> [&'static str; 5] {
    [
        "How does a failed database migration roll back?",
        "数据库迁移失败后如何回滚？",
        "Πώς λειτουργεί η ανάκτηση μνήμης;",
        "Kumbukumbu zinahifadhiwa kwa matumizi ya baadaye.",
        "a_deferred_write_is_found_with_the_next_search memory_record_42",
    ]
}
