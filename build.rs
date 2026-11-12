//! Build identity for receipts.
//!
//! Resolves the short git HEAD into `JOTNAR_GIT_HASH`. No dependency and no
//! timestamp/dirty state: a receipt must name the binary deterministically,
//! not the working tree. Falls back to `unknown` when git is unavailable
//! (for example a source tarball).

use std::process::Command;

fn main() {
    let hash = Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=JOTNAR_GIT_HASH={hash}");

    // Re-run when HEAD moves (branch switch) or the build script changes.
    // A normal source change already invalidates the build; this keeps the
    // identity reasonably fresh without shelling out on every compile.
    println!("cargo:rerun-if-changed=build.rs");
    if let Ok(out) = Command::new("git")
        .args(["rev-parse", "--git-dir"])
        .output()
    {
        if out.status.success() {
            let dir = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !dir.is_empty() {
                println!("cargo:rerun-if-changed={dir}/HEAD");
            }
        }
    }
}
