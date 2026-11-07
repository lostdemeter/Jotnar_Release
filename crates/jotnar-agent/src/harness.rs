//! The editable circuit harness.
//!
//! This is the analog of Harness²'s editable component tree. The model/substrate
//! (the Jotnar interpreter and the target listing) is fixed. What changes is the
//! family set, the bounded step set, the search bound, the refinement policy,
//! and the tag policy.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

/// Legible harness-state features for the state-dependent world model.
///
/// The task vector describes the case; these describe the harness that the
/// next action would be applied to. The state-dependent effect model consumes
/// both, so it can learn that `sum4` is useless once the consumer is present,
/// that `depth + 1` is the remaining fix after a composition, and so on.
pub const STATE_FEATURE_NAMES: [&str; 14] = [
    "has_sum4",
    "has_select_min",
    "has_select_max",
    "has_saxpy_dup",
    "has_affine_dup",
    "has_matmul2_dup",
    "families",
    "consumers",
    "depth",
    "refinements",
    "refine_enabled",
    "use_tags",
    "depth_cap",
    "has_add2",
];

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Harness {
    pub families: Vec<String>,
    pub compose_consumers: Vec<String>,
    pub max_depth: usize,
    pub max_refinements: usize,
    pub refine_enabled: bool,
    pub use_tags: bool,
}

impl Default for Harness {
    fn default() -> Self {
        Self {
            families: vec![
                "saxpy".into(),
                "reduce".into(),
                "matmul2".into(),
                "select".into(),
                "select_max".into(),
                "add2".into(),
            ],
            compose_consumers: vec!["add2".into()],
            max_depth: 2,
            max_refinements: 1,
            refine_enabled: true,
            use_tags: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Mutation {
    pub label: String,
    pub harness: Harness,
}

impl Harness {
    pub fn fingerprint(&self) -> String {
        format!(
            "fam={} comp={} depth={} refinements={} refine={} tags={}",
            self.families.join(","),
            self.compose_consumers.join(","),
            self.max_depth,
            self.max_refinements,
            self.refine_enabled,
            self.use_tags
        )
    }

    /// The state half of the state-dependent effect model's input vector.
    pub fn state_features(&self) -> Vec<i64> {
        let has = |name: &str| i64::from(self.families.iter().any(|f| f == name));
        vec![
            i64::from(self.compose_consumers.iter().any(|c| c == "sum4")),
            has("select_min"),
            has("select_max"),
            has("saxpy_dup"),
            has("affine_dup"),
            has("matmul2_dup"),
            self.families.len() as i64,
            self.compose_consumers.len() as i64,
            self.max_depth as i64,
            self.max_refinements as i64,
            i64::from(self.refine_enabled),
            i64::from(self.use_tags),
            i64::from(self.max_depth >= 6),
            has("add2"),
        ]
    }

    /// Deterministic bounded mutations. No model is required for this layer.
    pub fn mutations(&self) -> Vec<Mutation> {
        let mut out = Vec::new();

        if !self.compose_consumers.iter().any(|c| c == "sum4") {
            let mut h = self.clone();
            h.compose_consumers.push("sum4".into());
            out.push(Mutation {
                label: "add compose[sum4]".into(),
                harness: h,
            });
        }

        if !self.compose_consumers.iter().any(|c| c == "sum9") {
            let mut h = self.clone();
            h.compose_consumers.push("sum9".into());
            out.push(Mutation {
                label: "add compose[sum9]".into(),
                harness: h,
            });
        }

        if !self.compose_consumers.iter().any(|c| c == "sum16") {
            let mut h = self.clone();
            h.compose_consumers.push("sum16".into());
            out.push(Mutation {
                label: "add compose[sum16]".into(),
                harness: h,
            });
        }

        {
            let mut h = self.clone();
            h.max_depth = (h.max_depth + 1).min(6);
            out.push(Mutation {
                label: "depth + 1".into(),
                harness: h,
            });
        }

        if self.refine_enabled {
            let mut h = self.clone();
            h.refine_enabled = false;
            out.push(Mutation {
                label: "refine off".into(),
                harness: h,
            });
        }

        if self.use_tags {
            let mut h = self.clone();
            h.use_tags = false;
            out.push(Mutation {
                label: "tags off".into(),
                harness: h,
            });
        }

        {
            let mut h = self.clone();
            h.families.push("select_max".into());
            h.families = dedup(h.families);
            out.push(Mutation {
                label: "add family select_max".into(),
                harness: h,
            });
        }

        if self.max_refinements < 3 {
            let mut h = self.clone();
            h.max_refinements += 1;
            out.push(Mutation {
                label: "refinements + 1".into(),
                harness: h,
            });
        }

        for family in ["select_min", "saxpy_dup", "affine_dup", "matmul2_dup"] {
            let mut h = self.clone();
            if !h.families.iter().any(|f| f == family) {
                h.families.push(family.to_string());
                h.families = dedup(h.families);
                out.push(Mutation {
                    label: format!("add family {family}"),
                    harness: h,
                });
            }
        }

        out
    }

    pub fn diff(&self, other: &Harness) -> Vec<String> {
        let mut out = Vec::new();
        if self.families != other.families {
            out.push(format!(
                "families {:?} -> {:?}",
                self.families, other.families
            ));
        }
        if self.compose_consumers != other.compose_consumers {
            out.push(format!(
                "consumers {:?} -> {:?}",
                self.compose_consumers, other.compose_consumers
            ));
        }
        if self.max_depth != other.max_depth {
            out.push(format!("depth {} -> {}", self.max_depth, other.max_depth));
        }
        if self.max_refinements != other.max_refinements {
            out.push(format!(
                "refinements {} -> {}",
                self.max_refinements, other.max_refinements
            ));
        }
        if self.refine_enabled != other.refine_enabled {
            out.push(format!(
                "refine {} -> {}",
                self.refine_enabled, other.refine_enabled
            ));
        }
        if self.use_tags != other.use_tags {
            out.push(format!("tags {} -> {}", self.use_tags, other.use_tags));
        }
        out
    }
}

fn dedup(v: Vec<String>) -> Vec<String> {
    let mut seen = BTreeSet::new();
    v.into_iter().filter(|x| seen.insert(x.clone())).collect()
}
