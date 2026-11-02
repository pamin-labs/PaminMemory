//! Where a process's resident memory is, read from the kernel.
//!
//! One resident figure has been published for years and never divided, so
//! nothing could say what a reduction would have to target. This splits it the
//! two ways that decide what to do about it:
//!
//! - **Anonymous against file-backed.** File-backed pages -- an index read
//!   through a memory map, the binary, its libraries -- are counted in RSS but
//!   are page cache the kernel can drop and read back; anonymous pages are the
//!   heap, the model weights once parsed, and the inference runtime's arenas,
//!   which only the process can give back.
//! - **By mapping.** The largest mappings, grouped by the file behind them or
//!   by kind for anonymous ones, so a figure is attributed to something that
//!   can be changed.
//!
//! Linux only: it reads `/proc/self/smaps`, which is what can say this at all.

// Included by harnesses that use only some of it, like the other shared
// modules here.
#![allow(dead_code)]

use std::collections::BTreeMap;

/// Resident memory at one moment, in KiB.
pub struct Resident {
    pub rss: u64,
    pub anonymous: u64,
    /// Resident pages by what is behind them: a file's path, or `[heap]`,
    /// `[stack]`, or `[anon]` for mappings with no name.
    pub by_mapping: BTreeMap<String, u64>,
}

impl Resident {
    pub fn now() -> Self {
        let smaps = std::fs::read_to_string("/proc/self/smaps").expect("reading /proc/self/smaps");
        let mut rss = 0;
        let mut anonymous = 0;
        let mut by_mapping: BTreeMap<String, u64> = BTreeMap::new();
        let mut current = String::from("[anon]");
        for line in smaps.lines() {
            let mut fields = line.split_whitespace();
            let Some(first) = fields.next() else { continue };
            if first.contains('-') && !first.ends_with(':') {
                // A mapping's header: address range, perms, offset, device,
                // inode, and then the path if there is one.
                current = line
                    .split_whitespace()
                    .nth(5)
                    .map_or_else(|| "[anon]".to_string(), group);
                continue;
            }
            let kib = || -> u64 {
                fields
                    .clone()
                    .next()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(0)
            };
            match first {
                "Rss:" => {
                    let value = kib();
                    rss += value;
                    *by_mapping.entry(current.clone()).or_default() += value;
                }
                "Anonymous:" => anonymous += kib(),
                _ => {}
            }
        }
        Self {
            rss,
            anonymous,
            by_mapping,
        }
    }

    /// One line for this stage, and its largest mappings.
    pub fn print(&self, stage: &str) {
        println!(
            "  {stage:<40} rss {:>7.0} MB   anonymous {:>7.0} MB   file-backed {:>7.0} MB",
            self.rss as f64 / 1024.0,
            self.anonymous as f64 / 1024.0,
            (self.rss - self.anonymous.min(self.rss)) as f64 / 1024.0,
        );
        let mut largest: Vec<(&String, &u64)> = self.by_mapping.iter().collect();
        largest.sort_by(|left, right| right.1.cmp(left.1));
        for (mapping, kib) in largest.into_iter().take(6) {
            if *kib >= 10 * 1024 {
                println!("      {:>7.0} MB  {mapping}", *kib as f64 / 1024.0);
            }
        }
    }
}

/// Groups a mapped path into what it belongs to: every file of one index is
/// one index, and every library is its own name.
fn group(path: &str) -> String {
    if path.starts_with('[') {
        return path.to_string();
    }
    for marker in ["/index/", "/models/"] {
        if let Some(at) = path.find(marker) {
            let rest = &path[at + marker.len()..];
            let head = rest.split('/').next().unwrap_or(rest);
            return format!("{}{head}/...", &marker[1..]);
        }
    }
    path.rsplit('/').next().unwrap_or(path).to_string()
}
