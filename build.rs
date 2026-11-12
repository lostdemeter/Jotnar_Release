//! Build identity for receipts.
//!
//! Resolves the short git HEAD into `JOTNAR_GIT_HASH`. No dependency and no
//! timestamp/dirty state: a receipt must name the binary deterministically,
//! not the working tree. Falls back to `unknown` when git is unavailable
//! (for example a source tarball).
//!
//! Cargo caches build-script output, so we watch the real ref files (HEAD,
//! the current branch ref, and packed-refs). Watching only `.git/HEAD` is
//! not enough: it names the branch, and that file does not change when the
//! branch advances.

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

fn main() {
    let hash = git(&["rev-parse", "--short=12", "HEAD"]).unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=JOTNAR_GIT_HASH={hash}");

    println!("cargo:rerun-if-changed=build.rs");
    // Detached HEAD / branch switch.
    if let Some(p) = git(&["rev-parse", "--git-path", "HEAD"]) {
        println!("cargo:rerun-if-changed={p}");
    }
    // The branch ref itself: this is what changes on commit.
    if let Some(refname) = git(&["symbolic-ref", "-q", "HEAD"]) {
        if let Some(p) = git(&["rev-parse", "--git-path", &refname]) {
            println!("cargo:rerun-if-changed={p}");
        }
    }
    // Packed refs (after gc) hold the branch tip when the loose ref is gone.
    if let Some(p) = git(&["rev-parse", "--git-path", "packed-refs"]) {
        println!("cargo:rerun-if-changed={p}");
    }
}
