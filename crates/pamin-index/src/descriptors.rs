//! How many files this process may hold open.
//!
//! A projection index keeps one file per segment and a search reads across all
//! of them, so the descriptors a process needs grow with the corpus: 131,924
//! documents is 2,111 segment files and 2,733 descriptors held at once,
//! against the 1,024 a Linux process is given by default. Nothing degrades at
//! that boundary. A search fails outright with `Too many open files`, and so
//! does an index build -- measured, the hard way, on exactly that corpus:
//! indexing MIRACL's Swahili dev split died 65 minutes in with RocksDB
//! reporting `While open a file for appending: ... Too many open files`, and
//! what it left behind was a segment that could not be flushed or closed.
//!
//! This used to live in `pamin-cli`'s server module and be called from the
//! serve path alone, which meant the one process that raised the limit was the
//! one that had it raised and every other way of opening an index did not:
//! `PAMIN_NO_SERVER`, and every evaluation harness in this workspace. The
//! failure above is what that cost. It lives here now because this is the
//! crate whose files these are, and it is a call rather than something that
//! happens on its own -- a library changing a process-wide limit behind its
//! caller's back is a surprise, and the caller saying so is one line.

/// Raises the open-file limit to what this process is already allowed.
///
/// Returns the soft limit before and after, so a caller can log the change
/// rather than assume it.
///
/// The soft limit is the process's to raise, up to the hard limit, with no
/// privilege: this asks for what the kernel has already agreed to. Beyond the
/// hard limit is the operator's to grant, so a caller should log a failure
/// rather than treat it as fatal -- a smaller index still works, and refusing
/// to start would take away the case that does.
#[cfg(unix)]
pub fn raise_open_file_limit() -> std::io::Result<(u64, u64)> {
    // SAFETY: both calls write only through the pointer given, which is a
    // local of exactly the type they expect.
    unsafe {
        let mut limit = std::mem::zeroed::<libc::rlimit>();
        if libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) != 0 {
            return Err(std::io::Error::last_os_error());
        }
        let before = limit.rlim_cur as u64;
        if limit.rlim_cur >= limit.rlim_max {
            return Ok((before, before));
        }
        limit.rlim_cur = limit.rlim_max;
        if libc::setrlimit(libc::RLIMIT_NOFILE, &limit) != 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok((before, limit.rlim_max as u64))
    }
}

/// Nothing to raise where there is no such limit.
#[cfg(not(unix))]
pub fn raise_open_file_limit() -> std::io::Result<(u64, u64)> {
    Ok((0, 0))
}

#[cfg(all(test, unix))]
mod tests {
    use super::raise_open_file_limit;

    /// The process's current soft and hard open-file limits.
    fn limits() -> (u64, u64) {
        // SAFETY: writes only through the pointer given, to a local of the
        // type the call expects.
        unsafe {
            let mut limit = std::mem::zeroed::<libc::rlimit>();
            assert_eq!(libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit), 0);
            (limit.rlim_cur as u64, limit.rlim_max as u64)
        }
    }

    fn set_soft(soft: u64, hard: u64) {
        // SAFETY: reads only through the pointer given, from a local of the
        // type the call expects.
        unsafe {
            let limit = libc::rlimit {
                rlim_cur: soft as libc::rlim_t,
                rlim_max: hard as libc::rlim_t,
            };
            assert_eq!(libc::setrlimit(libc::RLIMIT_NOFILE, &limit), 0);
        }
    }

    /// The reproduction, at the mechanism rather than at the corpus.
    ///
    /// Indexing 131,924 documents to find out takes half an hour and four
    /// gigabytes; what actually failed was a search running under a soft limit
    /// of 1,024 while the kernel would have allowed twenty times that. This
    /// lowers the limit, asks the server's startup to raise it, and checks the
    /// process is really running under the higher one afterwards.
    ///
    /// The limit is process-wide, so it is put back -- and the two tests here
    /// are the only ones in this crate's library that touch it, so neither is
    /// racing the other over a starting point.
    ///
    /// The corpus that produced the failure is in
    /// `pamin-engine/tests/monolingual.rs`, which found it: indexing MIRACL's
    /// Swahili dev split died 65 minutes in with RocksDB unable to append,
    /// under a soft limit of 1,024 that this call would have taken to 20,000.
    #[test]
    fn raising_takes_the_open_file_limit_the_kernel_already_allows() {
        let (original, hard) = limits();
        // A box whose hard limit is this low has nothing to raise, and the
        // no-op is covered by the test below.
        if hard <= 512 {
            return;
        }

        set_soft(512, hard);
        let (before, after) = raise_open_file_limit().expect("raising within the hard limit");
        assert_eq!(before, 512, "reports the limit it found");
        assert_eq!(after, hard, "takes everything the hard limit allows");
        assert_eq!(
            limits().0,
            hard,
            "the process is running under the raised limit, not merely told about it"
        );

        set_soft(original, hard);
    }

    /// Already at the ceiling is not a failure, and must not be reported as a
    /// raise: the startup logs one only when the number actually moved.
    #[test]
    fn a_limit_already_at_the_ceiling_is_left_alone() {
        let (original, hard) = limits();

        set_soft(hard, hard);
        let (before, after) = raise_open_file_limit().expect("a no-op still succeeds");
        assert_eq!(before, hard);
        assert_eq!(after, hard);

        set_soft(original, hard);
    }
}
