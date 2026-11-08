//! A state-dependent action-effect world model.
//!
//! The first effect model was `(task, action) -> outcome`; it could not tell
//! that `reduce_sum` is useless once the consumer is already present, or that after
//! a composition the remaining fix is `depth + 1`. This model is
//! `(task, harness_state, action) -> outcome`:
//!
//! ```text
//! 20 task features + 14 state features -> 10 action scores
//! ```
//!
//! It is trained on the exact transition table from `gen-transitions` and
//! remains an ordinary Jotnar listing. `BeamProposer` uses it to rank
//! state/action expansions and proposes the resulting harnesses; the exact
//! gate still selects the candidate.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::cost_model::CostModel;
use crate::driver::{EvalReport, Experience, OutcomeKind, Proposer};
use crate::effect_model::EffectModel;
use crate::effects::StateEffectRow;
use crate::harness::{Harness, Mutation, STATE_FEATURE_NAMES};
use crate::proposer::{action_applies, apply_action, apply_action_set, ACTIONS};
use crate::task_encoder::{task_vector, TASK_FEATURE_NAMES};
use crate::view::CaseView;

pub const STATE_MODEL_INPUTS: usize = TASK_FEATURE_NAMES.len() + STATE_FEATURE_NAMES.len();

#[derive(Clone, Debug, Default)]
pub struct DiscriminatingEval {
    pub groups: usize,
    pub hits: usize,
    pub accuracy: f64,
    /// Per-group `max(wrong score) - max(correct score)`. Positive means the
    /// wrong action outranked the correct one.
    pub residual: Vec<i64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateEffectModel {
    /// ACTIONS.len() x feature_len (task + state + self-similar products)
    pub w: Vec<Vec<i64>>,
    pub b: Vec<i64>,
    #[serde(default)]
    pub listing: String,
    /// WBP self-similar product lags over `[task, state]`. Empty is the plain
    /// additive model. A lag `L` appends `f[i] * f[i-L]` for `i >= L`.
    #[serde(default)]
    pub interaction_lags: Vec<usize>,
}

impl StateEffectModel {
    pub fn prior() -> Self {
        let base = EffectModel::prior();
        let mut w = vec![vec![0i64; STATE_MODEL_INPUTS]; ACTIONS.len()];
        for (action, row) in base.w.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                w[action][j] = *value;
            }
        }
        let mut b = base.b.clone();

        // Applicability / no-op priors. These are priors, not the gate: the
        // transition fit can move them, and `apply_action` still refuses
        // actions that cannot be applied.
        let s = TASK_FEATURE_NAMES.len();
        let has_reduce_sum = s;
        let has_select_min = s + 1;
        let has_select_max = s + 2;
        let has_saxpy_dup = s + 3;
        let has_affine_dup = s + 4;
        let has_matmul2_dup = s + 5;
        let refinements = s + 9;
        let depth_cap = s + 12;
        w[0][has_reduce_sum] = -64;
        w[1][depth_cap] = -64;
        w[4][has_select_max] = -64;
        w[5][refinements] = -8;
        w[6][has_select_min] = -64;
        w[7][has_saxpy_dup] = -64;
        w[8][has_affine_dup] = -64;
        w[9][has_matmul2_dup] = -64;
        b.fill(0);

        let mut model = Self {
            w,
            b,
            listing: String::new(),
            interaction_lags: Vec::new(),
        };
        model.listing = build_listing(&model);
        model
    }

    /// Add WBP self-similar product features over `[task, state]`.
    pub fn with_interactions(mut self, lags: Vec<usize>) -> Self {
        let mut valid: Vec<usize> = lags
            .into_iter()
            .filter(|lag| *lag > 0 && *lag < STATE_MODEL_INPUTS)
            .collect();
        valid.sort_unstable();
        valid.dedup();
        self.interaction_lags = valid;
        let len = self.feature_len();
        self.w.truncate(ACTIONS.len());
        while self.w.len() < ACTIONS.len() {
            self.w.push(vec![0; len]);
        }
        for row in &mut self.w {
            row.resize(len, 0);
        }
        self.b.truncate(ACTIONS.len());
        while self.b.len() < ACTIONS.len() {
            self.b.push(0);
        }
        self.listing = build_listing(&self);
        self
    }

    pub fn feature_len(&self) -> usize {
        let base = STATE_MODEL_INPUTS;
        let extra: usize = self
            .interaction_lags
            .iter()
            .filter(|lag| **lag > 0 && **lag < base)
            .map(|lag| base - *lag)
            .sum();
        base + extra
    }

    pub fn features(&self, task: &[i64], state: &[i64]) -> Vec<i64> {
        let mut base = Vec::with_capacity(STATE_MODEL_INPUTS);
        base.extend_from_slice(task);
        base.extend_from_slice(state);
        let mut out = base.clone();
        for lag in &self.interaction_lags {
            if *lag == 0 || *lag >= base.len() {
                continue;
            }
            for i in *lag..base.len() {
                out.push(base[i].wrapping_mul(base[i - lag]));
            }
        }
        out
    }

    pub fn scores(&self, task: &[i64], state: &[i64]) -> Vec<i64> {
        let f = self.features(task, state);
        self.w
            .iter()
            .enumerate()
            .map(|(action, row)| {
                let mut sum = self.b.get(action).copied().unwrap_or(0);
                for (j, value) in f.iter().enumerate() {
                    sum += value * row[j];
                }
                sum
            })
            .collect()
    }

    /// Fit a per-`(case, state)` ranking perceptron on the exact transition
    /// table. Correct children are positives, wrong children are negatives.
    pub fn fit(&mut self, rows: &[StateEffectRow], epochs: usize, lr: i64) -> f64 {
        let lr = lr.max(1);
        let mut groups: BTreeMap<(String, String), Vec<&StateEffectRow>> = BTreeMap::new();
        for row in rows {
            groups
                .entry((row.case.clone(), row.state_key.clone()))
                .or_default()
                .push(row);
        }

        for _ in 0..epochs.max(1) {
            for group in groups.values() {
                let sample = &group[0];
                let positives: Vec<&&StateEffectRow> = group
                    .iter()
                    .filter(|row| row.correct && action_applies(row.action, &sample.task))
                    .collect();
                let negatives: Vec<&&StateEffectRow> = group
                    .iter()
                    .filter(|row| row.wrong && action_applies(row.action, &sample.task))
                    .collect();
                if positives.is_empty() || negatives.is_empty() {
                    continue;
                }
                let scores = self.scores(&sample.task, &sample.state);
                let best_positive = positives
                    .iter()
                    .max_by_key(|row| scores[row.action])
                    .map(|row| row.action)
                    .expect("positive");
                let best_negative = negatives
                    .iter()
                    .max_by_key(|row| scores[row.action])
                    .map(|row| row.action)
                    .expect("negative");
                if scores[best_negative] < scores[best_positive] {
                    continue;
                }
                // Breaking an already-correct state is the costly mistake.
                let weight = if sample.parent_correct { 2 } else { 1 };
                let f = self.features(&sample.task, &sample.state);
                for (j, value) in f.iter().enumerate() {
                    self.w[best_positive][j] =
                        (self.w[best_positive][j] + lr * weight * value).clamp(-128, 128);
                    self.w[best_negative][j] =
                        (self.w[best_negative][j] - lr * weight * value).clamp(-128, 128);
                }
                self.b[best_positive] = (self.b[best_positive] + lr * weight).clamp(-128, 128);
                self.b[best_negative] = (self.b[best_negative] - lr * weight).clamp(-128, 128);
            }
        }

        self.listing = build_listing(self);
        // Score only the groups where the transition table actually contains
        // a choice (both a correct and a wrong child). Groups with no wrong
        // child are safety checks, not ranking problems.
        let mut discriminating = 0usize;
        let mut hits = 0usize;
        for group in groups.values() {
            let sample = &group[0];
            let has_positive = group
                .iter()
                .any(|row| row.correct && action_applies(row.action, &sample.task));
            let has_negative = group
                .iter()
                .any(|row| row.wrong && action_applies(row.action, &sample.task));
            if !has_positive || !has_negative {
                continue;
            }
            discriminating += 1;
            let scores = self.scores(&sample.task, &sample.state);
            // Rank only the actions the transition table actually offers from
            // this state. An action that `apply_action` would reject (already
            // present, inapplicable shape) is not a valid choice.
            let top = group
                .iter()
                .filter(|row| action_applies(row.action, &sample.task))
                .max_by_key(|row| scores[row.action])
                .map(|row| row.action);
            if let Some(top) = top {
                if group
                    .iter()
                    .find(|row| row.action == top)
                    .map(|row| row.correct)
                    .unwrap_or(false)
                {
                    hits += 1;
                }
            }
        }
        if discriminating == 0 {
            1.0
        } else {
            hits as f64 / discriminating as f64
        }
    }

    /// Evaluate on held-out transition rows. Returns the discriminating top-1
    /// accuracy and the per-group score residual (wrong-best minus right-best).
    /// Only groups with both a correct and a wrong action are counted; groups
    /// where every applicable action is safe are not ranking problems.
    pub fn evaluate(&self, rows: &[StateEffectRow]) -> DiscriminatingEval {
        let mut groups: BTreeMap<(String, String), Vec<&StateEffectRow>> = BTreeMap::new();
        for row in rows {
            groups
                .entry((row.case.clone(), row.state_key.clone()))
                .or_default()
                .push(row);
        }
        let mut total = 0usize;
        let mut hits = 0usize;
        let mut residual = Vec::new();
        for group in groups.values() {
            if group.is_empty() {
                continue;
            }
            let sample = group[0];
            let scores = self.scores(&sample.task, &sample.state);
            let applicable: Vec<&&StateEffectRow> = group
                .iter()
                .filter(|row| action_applies(row.action, &sample.task))
                .collect();
            let has_pos = applicable.iter().any(|row| row.correct);
            let has_neg = applicable.iter().any(|row| row.wrong);
            if !has_pos || !has_neg {
                continue;
            }
            total += 1;
            let best_pos = applicable
                .iter()
                .filter(|row| row.correct)
                .map(|row| scores[row.action])
                .max()
                .unwrap_or(i64::MIN);
            let best_neg = applicable
                .iter()
                .filter(|row| row.wrong)
                .map(|row| scores[row.action])
                .max()
                .unwrap_or(i64::MIN);
            residual.push(best_neg.saturating_sub(best_pos));
            let top = applicable
                .iter()
                .max_by_key(|row| scores[row.action])
                .map(|row| row.action)
                .unwrap_or(0);
            if applicable
                .iter()
                .find(|row| row.action == top)
                .map(|row| row.correct)
                .unwrap_or(false)
            {
                hits += 1;
            }
        }
        DiscriminatingEval {
            groups: total,
            hits,
            accuracy: if total == 0 {
                1.0
            } else {
                hits as f64 / total as f64
            },
            residual,
        }
    }

    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, text + "\n").map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut model: Self = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        model.normalize();
        Ok(model)
    }

    /// Pad or truncate to the current action space. Old artifact files remain
    /// loadable when a new action is added; the new rows start at zero.
    pub fn normalize(&mut self) {
        let len = self.feature_len();
        self.w.truncate(ACTIONS.len());
        while self.w.len() < ACTIONS.len() {
            self.w.push(vec![0; len]);
        }
        for row in &mut self.w {
            row.resize(len, 0);
        }
        self.b.truncate(ACTIONS.len());
        while self.b.len() < ACTIONS.len() {
            self.b.push(0);
        }
        self.listing = build_listing(self);
    }
}

fn build_listing(model: &StateEffectModel) -> String {
    let mut lines = vec![
        "# jotnar-agent state-dependent action-effect world model".to_string(),
        "# inputs: task t0..tN, harness state u0..uM; outputs: action scores s0..sK".to_string(),
    ];
    for (j, name) in TASK_FEATURE_NAMES.iter().enumerate() {
        lines.push(format!("IN t{j}  # {name}"));
    }
    for (j, name) in STATE_FEATURE_NAMES.iter().enumerate() {
        lines.push(format!("IN u{j}  # {name}"));
    }
    lines.push(String::new());
    let base_names: Vec<String> = (0..TASK_FEATURE_NAMES.len())
        .map(|j| format!("t{j}"))
        .chain((0..STATE_FEATURE_NAMES.len()).map(|j| format!("u{j}")))
        .collect();
    let mut names = base_names.clone();
    // WBP self-similar products: x_i * x_{i-lag} over [task, state].
    for lag in &model.interaction_lags {
        if *lag == 0 || *lag >= base_names.len() {
            continue;
        }
        for i in *lag..base_names.len() {
            let name = format!("x{lag}_{i}");
            lines.push(format!(
                "{name} = IMUL({}, {})",
                base_names[i],
                base_names[i - lag]
            ));
            names.push(name);
        }
    }
    if !model.interaction_lags.is_empty() {
        lines.push(String::new());
    }
    for (action, row) in model.w.iter().enumerate() {
        let mut products = Vec::new();
        for (j, weight) in row.iter().enumerate() {
            if *weight != 0 {
                let name = format!("p{action}_{j}");
                lines.push(format!("{name} = IMUL({}, {weight})", names[j]));
                products.push(name);
            }
        }
        let bias = model.b.get(action).copied().unwrap_or(0);
        if products.is_empty() {
            let zero = format!("z{action}");
            lines.push(format!("{zero} = IMUL(t0, 0)"));
            lines.push(format!("s{action} = IADD({zero}, {bias})"));
        } else {
            let mut acc = products[0].clone();
            for (k, term) in products.iter().enumerate().skip(1) {
                let name = format!("a{action}_{k}");
                lines.push(format!("{name} = IADD({acc}, {term})"));
                acc = name;
            }
            lines.push(format!("s{action} = IADD({acc}, {bias})"));
        }
    }
    lines.join("\n") + "\n"
}

#[derive(Clone, Debug)]
struct BeamNode {
    path: Vec<usize>,
    harness: Harness,
    state: Vec<i64>,
    score: i64,
}

/// Rank state/action expansions for one case and return the best paths.
fn search_case<S>(
    task: &[i64],
    base: &Harness,
    beam: usize,
    max_depth: usize,
    score: &S,
) -> Vec<BeamNode>
where
    S: Fn(&[i64], &[i64]) -> Vec<i64>,
{
    let mut frontier = vec![BeamNode {
        path: Vec::new(),
        harness: base.clone(),
        state: base.state_features(),
        score: 0,
    }];
    let mut visited: Vec<BeamNode> = frontier.clone();
    for _ in 0..max_depth {
        let mut next: BTreeMap<String, BeamNode> = BTreeMap::new();
        for node in &frontier {
            let scores = score(task, &node.state);
            for action in 0..ACTIONS.len() {
                if !action_applies(action, task) {
                    continue;
                }
                let Some(mutation) = apply_action(&node.harness, action) else {
                    continue;
                };
                let key = mutation.harness.fingerprint();
                let predicted = scores.get(action).copied().unwrap_or(0);
                let candidate = BeamNode {
                    path: node.path.iter().copied().chain([action]).collect(),
                    state: mutation.harness.state_features(),
                    harness: mutation.harness,
                    score: node.score + predicted,
                };
                let replace = next
                    .get(&key)
                    .map(|old| candidate.score > old.score)
                    .unwrap_or(true);
                if replace {
                    next.insert(key, candidate);
                }
            }
        }
        if next.is_empty() {
            break;
        }
        let mut ranked: Vec<BeamNode> = next.into_values().collect();
        ranked.sort_by_key(|node| std::cmp::Reverse(node.score));
        ranked.truncate(beam);
        visited.extend(ranked.iter().cloned());
        frontier = ranked;
    }
    visited.sort_by_key(|node| std::cmp::Reverse(node.score));
    visited
}

/// Coarse capability group for the coverage proposal. Among gate-equivalent
/// harnesses, covering a new group is the neutral-drift move that preserves
/// generalization potential without changing exact correctness.
fn capability_group(action: usize) -> usize {
    match action {
        0 => 0,     // geometric compose consumer (reduce_sum)
        1 => 1,     // depth
        2 | 3 => 2, // refine/tags policy
        5 => 3,     // refinements
        _ => 4,     // family adds
    }
}

/// Shared beam-search proposal path for the state model and the joint model.
///
/// `score(task, state)` returns one predicted correctness score per action.
/// The exact gate remains the judge: this function only orders the candidates.
/// Structural safety-net options for [`propose_from_scores`]. The learned-only
/// ablation disables all three.
#[derive(Clone, Copy, Debug)]
pub struct ProposalOptions {
    pub include_pairs: bool,
    pub include_complete: bool,
    pub include_singles: bool,
}

impl Default for ProposalOptions {
    fn default() -> Self {
        Self {
            include_pairs: true,
            include_complete: true,
            include_singles: true,
        }
    }
}

impl ProposalOptions {
    pub fn learned_only() -> Self {
        Self {
            include_pairs: false,
            include_complete: false,
            include_singles: false,
        }
    }
}

pub fn propose_from_scores<S>(
    base: &Harness,
    views: &[CaseView],
    beam: usize,
    max_depth: usize,
    options: ProposalOptions,
    prefix: &str,
    score: S,
) -> Vec<Mutation>
where
    S: Fn(&[i64], &[i64]) -> Vec<i64>,
{
    let beam = beam.max(1);
    let max_depth = max_depth.max(1);
    // A geometric action is only meaningful when its shape contract holds for
    // this case. Mask the inapplicable ones so the learned ranking is among
    // applicable actions. This is a shape rule, not a per-arity table.
    let scored = |task: &[i64], state: &[i64]| {
        let mut scores = score(task, state);
        for (action, value) in scores.iter_mut().enumerate() {
            if !action_applies(action, task) {
                *value = i64::MIN / 4;
            }
        }
        scores
    };
    let mut out: Vec<Mutation> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let push = |mutation: Mutation, out: &mut Vec<Mutation>, seen: &mut BTreeSet<String>| {
        let key = mutation.harness.fingerprint();
        if mutation.harness != *base && seen.insert(key) {
            out.push(mutation);
        }
    };

    // The model's top action from the *base* state for each wrong case. Their
    // union is the first candidate, so it wins exact ties. The coverage union
    // adds the best positive action from each capability group; it is the
    // neutral-drift candidate that keeps a second capability alive even when
    // the selection stream does not require it.
    let mut union: BTreeSet<usize> = BTreeSet::new();
    let mut coverage_union: BTreeSet<usize> = BTreeSet::new();
    let mut best_group: BTreeMap<usize, (i64, usize)> = BTreeMap::new();
    let mut per_case: Vec<(String, Vec<BeamNode>)> = Vec::new();
    for view in views {
        if view.outcome != OutcomeKind::WrongRefusal {
            continue;
        }
        let task = task_vector(view, base);
        let state = base.state_features();
        let scores = scored(&task, &state);
        let mut order: Vec<usize> = (0..ACTIONS.len()).collect();
        order.sort_by_key(|action| std::cmp::Reverse(scores[*action]));
        let applicable: Vec<usize> = order
            .into_iter()
            .filter(|action| {
                apply_action(base, *action).is_some() && action_applies(*action, &task)
            })
            .collect();
        if let Some(&top) = applicable.first() {
            union.insert(top);
        }
        for action in applicable {
            let group = capability_group(action);
            // Policy disables (`refine off`, `tags off`) are not capabilities.
            if group == 2 {
                continue;
            }
            let entry = best_group.entry(group).or_insert((i64::MIN, action));
            if scores[action] > entry.0 {
                *entry = (scores[action], action);
            }
        }
        // Structural deep prior: a one-output target with more ops than the
        // start may need the depth capability even when the learned model has
        // down-weighted it in favour of refinements. The exact gate still
        // decides whether the coverage candidate is safe.
        if view.start_outputs == 1 && view.target_ops > view.start_ops {
            if let Some(action) = (0..ACTIONS.len())
                .find(|a| capability_group(*a) == 1 && apply_action(base, *a).is_some())
            {
                coverage_union.insert(action);
            }
        }
        per_case.push((
            view.case_id.clone(),
            search_case(&task, base, beam, max_depth, &scored),
        ));
    }
    coverage_union.extend(union.iter().copied());
    for (score, action) in best_group.values() {
        if *score > 0 {
            coverage_union.insert(*action);
        }
    }
    if !union.is_empty() {
        let names = union
            .iter()
            .map(|action| ACTIONS[*action])
            .collect::<Vec<_>>()
            .join(" + ");
        push(
            Mutation {
                label: format!("{prefix}-union: {names}"),
                harness: apply_action_set(base, &union),
            },
            &mut out,
            &mut seen,
        );
    }
    if coverage_union != union && !coverage_union.is_empty() {
        let names = coverage_union
            .iter()
            .map(|action| ACTIONS[*action])
            .collect::<Vec<_>>()
            .join(" + ");
        push(
            Mutation {
                label: format!("{prefix}-coverage: {names}"),
                harness: apply_action_set(base, &coverage_union),
            },
            &mut out,
            &mut seen,
        );
    }

    // Union the learned top path per wrong case, preserving repeated
    // `depth + 1` / `refinements + 1` actions. The plain union is a set and
    // collapses repeats; this candidate keeps the counts the beam found.
    let mut path_actions: Vec<usize> = Vec::new();
    for (_, paths) in &per_case {
        if let Some(best) = paths.first() {
            path_actions.extend(best.path.iter().copied());
        }
    }
    if !path_actions.is_empty() {
        let mut harness = base.clone();
        let mut applied = Vec::new();
        for action in path_actions {
            if let Some(mutation) = apply_action(&harness, action) {
                harness = mutation.harness;
                applied.push(action);
            }
        }
        if harness != *base {
            let names = applied
                .iter()
                .map(|action| ACTIONS[*action])
                .collect::<Vec<_>>()
                .join(" + ");
            push(
                Mutation {
                    label: format!("{prefix}-path-union: {names}"),
                    harness,
                },
                &mut out,
                &mut seen,
            );
        }
    }

    // Compose completeness: keep every available compose consumer alive. A
    // consumer that does not match the current output arity is a safe no-op,
    // and it is exactly what a feature-distant shape may need. This is a
    // structural prior, not a learned ranking; the ablation disables it.
    if options.include_complete {
        let mut complete_union = coverage_union.clone();
        for action in 0..ACTIONS.len() {
            if capability_group(action) == 0 && apply_action(base, action).is_some() {
                complete_union.insert(action);
            }
        }
        if complete_union != coverage_union && !complete_union.is_empty() {
            let names = complete_union
                .iter()
                .map(|action| ACTIONS[*action])
                .collect::<Vec<_>>()
                .join(" + ");
            push(
                Mutation {
                    label: format!("{prefix}-complete: {names}"),
                    harness: apply_action_set(base, &complete_union),
                },
                &mut out,
                &mut seen,
            );
        }
    }

    // Exhaustive singles: the safety net for an action the learned model has
    // never seen (e.g. a geometric consumer before any shaped training case). The
    // ablation disables it to measure the learned ranking alone.
    if options.include_singles {
        for (action, label) in ACTIONS.iter().enumerate() {
            if let Some(mutation) = apply_action(base, action) {
                push(
                    Mutation {
                        label: format!("{prefix}-single: {label}"),
                        harness: mutation.harness,
                    },
                    &mut out,
                    &mut seen,
                );
            }
        }
    }

    // Top predicted paths per case, then the global best.
    let mut global: BTreeMap<String, BeamNode> = BTreeMap::new();
    for (case, paths) in &per_case {
        for (rank, node) in paths.iter().take(2).enumerate() {
            let names = node
                .path
                .iter()
                .map(|action| ACTIONS[*action])
                .collect::<Vec<_>>()
                .join(" + ");
            push(
                Mutation {
                    label: format!("{prefix}[{case}#{rank}]: {names}"),
                    harness: node.harness.clone(),
                },
                &mut out,
                &mut seen,
            );
        }
        for node in paths.iter().take(beam) {
            global
                .entry(node.harness.fingerprint())
                .and_modify(|old| {
                    if node.score > old.score {
                        *old = node.clone();
                    }
                })
                .or_insert_with(|| node.clone());
        }
    }
    let mut ranked: Vec<BeamNode> = global.into_values().collect();
    ranked.sort_by_key(|node| std::cmp::Reverse(node.score));
    for node in ranked.into_iter().take(beam) {
        let names = node
            .path
            .iter()
            .map(|action| ACTIONS[*action])
            .collect::<Vec<_>>()
            .join(" + ");
        push(
            Mutation {
                label: format!("{prefix}-global: {names}"),
                harness: node.harness,
            },
            &mut out,
            &mut seen,
        );
    }

    // Keep the exhaustive pair set as a gate safety net. The beam covers
    // triples; the pairs are cheap and were the first accepted attractor.
    if options.include_pairs {
        for (a, a_name) in ACTIONS.iter().enumerate() {
            for (b, b_name) in ACTIONS.iter().enumerate().skip(a + 1) {
                let pair: BTreeSet<usize> = [a, b].into_iter().collect();
                push(
                    Mutation {
                        label: format!("{prefix}-pair: {a_name} + {b_name}"),
                        harness: apply_action_set(base, &pair),
                    },
                    &mut out,
                    &mut seen,
                );
            }
        }
    }
    out
}

#[derive(Clone, Debug)]
pub struct BeamProposer {
    pub model: StateEffectModel,
    /// Width of the predicted-effect frontier at each depth.
    pub beam: usize,
    /// Number of distinct actions in a path (pairs, triples, ...).
    pub max_depth: usize,
    /// Include the exhaustive pair set as a gate safety net.
    pub include_pairs: bool,
    /// Include the compose-completeness union (all consumers) as a structural
    /// safety net. Disable it for the learned-only ablation.
    pub include_complete: bool,
    /// Include the exhaustive single-action set as a structural safety net.
    /// Disable it for the learned-only ablation.
    pub include_singles: bool,
    /// Optional learned cost head. The beam score becomes
    /// `effect - cost_weight * predicted_steps`.
    pub cost: Option<CostModel>,
    pub cost_weight: i64,
}

impl BeamProposer {
    pub fn new(model: StateEffectModel, beam: usize, max_depth: usize) -> Self {
        Self {
            model,
            beam: beam.max(1),
            max_depth: max_depth.max(1),
            include_pairs: true,
            include_complete: true,
            include_singles: true,
            cost: None,
            cost_weight: 0,
        }
    }

    /// Disable the structural safety nets, leaving only the learned union,
    /// coverage, beam paths, and global proposal.
    pub fn learned_only(mut self) -> Self {
        self.include_complete = false;
        self.include_singles = false;
        self.include_pairs = false;
        self
    }

    pub fn with_cost(mut self, cost: CostModel, weight: i64) -> Self {
        self.cost = Some(cost);
        self.cost_weight = weight.max(0);
        self
    }
}

impl Proposer for BeamProposer {
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
        views: &[CaseView],
        base: &Harness,
        _experience: &[Experience],
    ) -> Option<Vec<Mutation>> {
        Some(propose_from_scores(
            base,
            views,
            self.beam,
            self.max_depth,
            ProposalOptions {
                include_pairs: self.include_pairs,
                include_complete: self.include_complete,
                include_singles: self.include_singles,
            },
            "beam",
            |task, state| {
                let mut scores = self.model.scores(task, state);
                if let Some(cost) = &self.cost {
                    let predicted = cost.scores(task, state);
                    for (score, cost) in scores.iter_mut().zip(predicted) {
                        *score -= self.cost_weight * cost;
                    }
                }
                scores
            },
        ))
    }
}
