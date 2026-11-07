//! A quality-diversity archive of adopted harness attractors.
//!
//! The driver adopts one harness at a time. A single incumbent collapses the
//! search back to one basin: two harnesses that are equally correct on the
//! selection stream can differ on a held-out family. The archive keeps several
//! gate-correct harnesses, selected by **quality** (exact correctness) and
//! **novelty** (distance in harness-state space). It is analysis/proposal
//! state; the exact gate still decides what is adopted.

use serde::{Deserialize, Serialize};

use crate::driver::{EvalReport, Experience, Proposer};
use crate::harness::{Harness, Mutation};
use crate::view::CaseView;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArchiveEntry {
    pub label: String,
    pub source: String,
    pub harness: Harness,
    /// Capability descriptor: the same legible state vector the world model
    /// consumes.
    pub descriptor: Vec<i64>,
    pub correct: usize,
    pub wrong: usize,
    pub heldout_correct: usize,
    pub heldout_wrong: usize,
    /// L1 distance to the nearest other entry at insertion time.
    pub novelty: usize,
}

impl ArchiveEntry {
    /// Lexicographic quality: correctness first, then held-out, then fewer
    /// wrong.
    pub fn quality(&self) -> (usize, usize, usize) {
        (self.correct, self.heldout_correct, usize::MAX - self.wrong)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Archive {
    pub entries: Vec<ArchiveEntry>,
    #[serde(default = "default_capacity")]
    pub capacity: usize,
    #[serde(default = "default_novelty")]
    pub min_novelty: usize,
}

fn default_capacity() -> usize {
    16
}

fn default_novelty() -> usize {
    2
}

pub enum Insert {
    Added { novelty: usize },
    Duplicate,
    Rejected,
}

impl Archive {
    pub fn new(capacity: usize, min_novelty: usize) -> Self {
        Self {
            entries: Vec::new(),
            capacity: capacity.max(1),
            min_novelty,
        }
    }

    pub fn novelty_of(&self, harness: &Harness) -> usize {
        let descriptor = harness.state_features();
        self.entries
            .iter()
            .map(|entry| distance(&descriptor, &entry.descriptor))
            .min()
            .unwrap_or(descriptor.len())
    }

    /// Insert a candidate if it is a new descriptor and either has the best
    /// quality seen or clears the novelty threshold. A duplicate fingerprint
    /// is dropped. When the archive is full, the worst entry is evicted only
    /// for a strictly better quality.
    pub fn insert(&mut self, mut entry: ArchiveEntry) -> Insert {
        entry.descriptor = entry.harness.state_features();
        if self
            .entries
            .iter()
            .any(|e| e.harness.fingerprint() == entry.harness.fingerprint())
        {
            return Insert::Duplicate;
        }
        let novelty = self
            .entries
            .iter()
            .map(|e| distance(&entry.descriptor, &e.descriptor))
            .min()
            .unwrap_or(entry.descriptor.len());
        entry.novelty = novelty;
        let best = self
            .entries
            .iter()
            .map(|e| e.quality())
            .max()
            .unwrap_or((0, 0, 0));
        let accepted = novelty >= self.min_novelty || entry.quality() > best;
        if !accepted {
            return Insert::Rejected;
        }
        if self.entries.len() >= self.capacity {
            // Evict the worst entry only if the newcomer is better.
            let (worst, _) = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.quality())
                .map(|(i, e)| (i, e.quality()))
                .unwrap_or((0, (0, 0, 0)));
            if entry.quality() <= self.entries[worst].quality() {
                return Insert::Rejected;
            }
            self.entries.remove(worst);
        }
        self.entries.push(entry);
        self.entries.sort_by_key(|e| std::cmp::Reverse(e.quality()));
        Insert::Added { novelty }
    }

    pub fn best(&self) -> Option<&ArchiveEntry> {
        self.entries.first()
    }

    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, text + "\n").map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        serde_json::from_str(&text).map_err(|e| e.to_string())
    }
}

fn distance(a: &[i64], b: &[i64]) -> usize {
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x - *y).unsigned_abs() as usize)
        .sum()
}

/// Basin hopping from archived attractors.
///
/// The proposer seeds the loop with the best archive entries and then proposes
/// their bounded mutations (including the neutral-drift toggles). The driver
/// still evaluates every candidate with the exact gate; an archived harness is
/// adopted only if it beats the incumbent under the normal retention and
/// held-out checks.
#[derive(Clone, Debug)]
pub struct ArchiveProposer {
    pub archive: Archive,
    pub max_seeds: usize,
}

impl ArchiveProposer {
    pub fn new(archive: Archive, max_seeds: usize) -> Self {
        Self {
            archive,
            max_seeds: max_seeds.max(1),
        }
    }
}

impl Proposer for ArchiveProposer {
    fn propose(
        &self,
        base: &Harness,
        _train: &EvalReport,
        _experience: &[Experience],
    ) -> Vec<Mutation> {
        base.mutations()
    }

    fn propose_cases(
        &self,
        _views: &[CaseView],
        base: &Harness,
        _experience: &[Experience],
    ) -> Option<Vec<Mutation>> {
        let mut out: Vec<Mutation> = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        let mut push = |mutation: Mutation, out: &mut Vec<Mutation>| {
            let key = mutation.harness.fingerprint();
            if mutation.harness != *base && seen.insert(key) {
                out.push(mutation);
            }
        };
        for entry in self.archive.entries.iter().take(self.max_seeds) {
            push(
                Mutation {
                    label: format!("qd-seed: {}", entry.label),
                    harness: entry.harness.clone(),
                },
                &mut out,
            );
            for mutation in entry.harness.mutations() {
                push(
                    Mutation {
                        label: format!("qd-hop[{}]: {}", entry.label, mutation.label),
                        harness: mutation.harness,
                    },
                    &mut out,
                );
            }
        }
        if out.is_empty() {
            return Some(base.mutations());
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proposer::apply_action;

    fn harness(actions: &[usize]) -> Harness {
        let mut h = Harness::default();
        for action in actions {
            h = apply_action(&h, *action).unwrap().harness;
        }
        h
    }

    fn entry(label: &str, h: Harness, correct: usize, heldout: usize) -> ArchiveEntry {
        let descriptor = h.state_features();
        ArchiveEntry {
            label: label.to_string(),
            source: "test".to_string(),
            descriptor,
            harness: h,
            correct,
            wrong: 0,
            heldout_correct: heldout,
            heldout_wrong: 0,
            novelty: 0,
        }
    }

    #[test]
    fn keeps_diverse_correct_harnesses() {
        let mut archive = Archive::new(8, 1);
        assert!(matches!(
            archive.insert(entry("reduce_sum", harness(&[0]), 15, 5)),
            Insert::Added { .. }
        ));
        assert!(matches!(
            archive.insert(entry("reduce_sum+select", harness(&[0, 6]), 15, 6)),
            Insert::Added { .. }
        ));
        assert!(matches!(
            archive.insert(entry("reduce_sum+select again", harness(&[0, 6]), 15, 6)),
            Insert::Duplicate
        ));
        assert!(matches!(
            archive.insert(entry("reduce_sum+depth", harness(&[0, 1]), 15, 6)),
            Insert::Added { .. }
        ));
        assert_eq!(archive.entries.len(), 3);
        assert!(archive.best().is_some());
        assert!(archive.entries.iter().all(|e| e.novelty > 0));
    }

    #[test]
    fn rejects_a_low_quality_duplicate_descriptor() {
        let mut archive = Archive::new(8, 100);
        archive.insert(entry("reduce_sum", harness(&[0]), 15, 6));
        assert!(matches!(
            archive.insert(entry("tags off", harness(&[3]), 10, 2)),
            Insert::Rejected
        ));
    }

    #[test]
    fn archive_roundtrips() {
        let mut archive = Archive::new(8, 1);
        archive.insert(entry("reduce_sum", harness(&[0]), 15, 6));
        let path =
            std::env::temp_dir().join(format!("jotnar-agent-archive-{}.json", std::process::id()));
        archive.save(&path).unwrap();
        let loaded = Archive::load(&path).unwrap();
        assert_eq!(loaded.entries.len(), 1);
        assert_eq!(loaded.entries[0].label, "reduce_sum");
        let _ = std::fs::remove_file(path);
    }
}
