//! Reproducible end-to-end demo.
//!
//! This is intentionally a *narrow* demo: it shows the loop we actually built
//! (certified harness composition under an exact interpreter gate), not an
//! open-ended self-improving agent. The demo fits the state model from the
//! training stream, runs the learned beam, checks retention and held-out
//! non-regression, and audits the adopted harness on the separate hard set.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::json;

use crate::{
    cases, evaluate, generated_corpus, hard_cases, reachable_states, run_round, run_round_with,
    split_cases, state_effects, BeamProposer, EvalReport, Harness, OutcomeKind, Split,
    StateEffectModel,
};

pub struct DemoSummary {
    pub out_dir: PathBuf,
    pub quick_base: usize,
    pub quick_adopted: usize,
    pub quick_heldout: usize,
    pub generated_base: usize,
    pub generated_adopted: usize,
    pub generated_heldout: usize,
    pub corpus_base: usize,
    pub corpus_adopted: usize,
    pub corpus_heldout: usize,
    pub hard_correct: usize,
    pub hard_total: usize,
    pub learned_only_adopted: usize,
    pub learned_only_heldout: usize,
    pub learned_only_hard: usize,
    pub model_cases: usize,
    pub model_states: usize,
    pub model_transitions: usize,
    pub model_accuracy: f64,
    pub checks: Vec<(String, bool)>,
}

impl DemoSummary {
    pub fn all_ok(&self) -> bool {
        self.checks.iter().all(|(_, ok)| *ok)
    }
}

fn hard_total() -> usize {
    hard_cases().len()
}

fn clear_dir(path: &Path) -> Result<(), String> {
    if path.exists() {
        fs::remove_dir_all(path).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    Ok(())
}

fn case_table(report: &crate::RoundReport) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "| case | incumbent | adopted |\n| --- | --- | --- |");
    for (base, adopted) in report.base.outcomes.iter().zip(&report.adopted.outcomes) {
        let _ = writeln!(
            out,
            "| `{}` | {} | {} |",
            base.case,
            base.kind.as_str(),
            adopted.kind.as_str()
        );
    }
    out
}

fn outcome_table(report: &EvalReport) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "| case | outcome | steps |\n| --- | --- | ---: |");
    for outcome in &report.outcomes {
        let _ = writeln!(
            out,
            "| `{}` | {} | {} |",
            outcome.case,
            outcome.kind.as_str(),
            outcome.steps
        );
    }
    out
}

fn report_json(report: &EvalReport) -> serde_json::Value {
    json!({
        "solved": report.solved,
        "refused": report.refused,
        "wrong_solve": report.wrong_solve,
        "wrong_refusal": report.wrong_refusal,
        "correct": report.correct(),
        "wrong": report.wrong(),
        "steps": report.total_steps,
    })
}

fn hard_json(report: &EvalReport) -> serde_json::Value {
    json!({
        "correct": report.correct(),
        "wrong": report.wrong(),
        "outcomes": report.outcomes.iter().map(|o| json!({
            "case": o.case,
            "family": o.family,
            "kind": o.kind.as_str(),
            "steps": o.steps,
            "note": o.note,
        })).collect::<Vec<_>>(),
    })
}

fn certificate_excerpt(path: &Path) -> Option<String> {
    let text = fs::read_to_string(path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&text).ok()?;
    let outcome = value.get("outcome")?.as_str()?;
    let verdict = value.get("verdict")?.as_str()?;
    let gate = value.get("gate")?.get("name")?.as_str()?;
    let can_say_no = value.get("gate")?.get("can_say_no")?.as_bool()?;
    Some(format!(
        "outcome={outcome}, verdict={verdict}, gate={gate}, can_say_no={can_say_no}"
    ))
}

/// Run the complete demo. Returns a summary; writes `report.md`,
/// `summary.json`, model weights, round artifacts, certificates, and traces
/// under `out`.
pub fn run_demo(out: &Path) -> Result<DemoSummary, String> {
    fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    for sub in ["quick", "generated", "corpus", "learned-only"] {
        clear_dir(&out.join(sub))?;
    }
    let report_path = out.join("report.md");
    let summary_path = out.join("summary.json");
    let model_path = out.join("state-effect-model.weights.json");

    let base = Harness::default();
    let mut checks = Vec::new();

    println!("[1/5] quick loop: 5 cases, deterministic proposer");
    let quick = run_round(&base, &cases(), &out.join("quick"));
    println!(
        "      base {}/{} -> adopted {}/{}; held-out {}/{} -> {}/{}; selected: {}",
        quick.base.correct(),
        quick.base.outcomes.len(),
        quick.adopted.correct(),
        quick.adopted.outcomes.len(),
        quick.base_heldout.correct(),
        quick.base_heldout.outcomes.len(),
        quick.adopted_heldout.correct(),
        quick.adopted_heldout.outcomes.len(),
        quick.chosen_label
    );
    checks.push(("quick base is 3/5".into(), quick.base.correct() == 3));
    checks.push(("quick adopts to 5/5".into(), quick.adopted.correct() == 5));
    checks.push((
        "quick held-out is 5/5".into(),
        quick.adopted_heldout.correct() == quick.adopted_heldout.outcomes.len(),
    ));
    let _ = crate::write_round_traces(
        &out.join("quick/traces.jsonl"),
        &cases(),
        &[],
        &quick,
        "deterministic",
        1,
    );
    checks.push((
        "quick keeps a first-class refusal".into(),
        quick
            .adopted
            .outcomes
            .iter()
            .any(|o| o.kind == OutcomeKind::Refused),
    ));

    println!("[2/5] fit the state-dependent model on the generated training split");
    let generated = generated_corpus();
    let train_cases: Vec<_> = generated
        .iter()
        .filter(|c| matches!(c.split, Split::Train | Split::Calibration))
        .map(|c| c.case.clone())
        .collect();
    let rows = state_effects(&train_cases, &base, 3);
    let states = reachable_states(&base, 3);
    let state_groups: BTreeSet<(String, String)> = rows
        .iter()
        .map(|row| (row.case.clone(), row.state_key.clone()))
        .collect();
    let mut model = StateEffectModel::prior();
    let accuracy = model.fit(&rows, 20, 1);
    model.save(&model_path)?;
    fs::write(out.join("state-effect-model.jotnar"), &model.listing)
        .map_err(|e| format!("state-effect-model.jotnar: {e}"))?;
    println!(
        "      {} cases, {} states, {} transitions, {} groups, top-1 {:.3}",
        train_cases.len(),
        states.len(),
        rows.len(),
        state_groups.len(),
        accuracy
    );
    checks.push(("state model is at least 0.99".into(), accuracy >= 0.99));

    println!("[3/5] generated stream: learned beam with exact-gate adoption");
    let generated_heldout = generated
        .iter()
        .filter(|c| c.split == Split::Heldout)
        .map(|c| c.case.clone())
        .collect::<Vec<_>>();
    let beam = BeamProposer::new(model.clone(), 8, 4);
    let generated_round = run_round_with(&base, &train_cases, &out.join("generated"), &beam);
    let generated_held = evaluate(&generated_round.adopted_harness, &generated_heldout, false);
    let hard = evaluate(&generated_round.adopted_harness, &hard_cases(), false);
    println!(
        "      base {}/{} -> adopted {}/{}; held-out {}/{}; hard {}/{}",
        generated_round.base.correct(),
        generated_round.base.outcomes.len(),
        generated_round.adopted.correct(),
        generated_round.adopted.outcomes.len(),
        generated_held.correct(),
        generated_held.outcomes.len(),
        hard.correct(),
        hard_total()
    );
    let _ = crate::write_round_traces(
        &out.join("generated/traces.jsonl"),
        &train_cases,
        &generated_heldout,
        &generated_round,
        "beam",
        1,
    );
    checks.push((
        "generated adopts to 19/19".into(),
        generated_round.adopted.correct() == 19,
    ));
    checks.push((
        "generated held-out is 6/6".into(),
        generated_held.correct() == 6,
    ));
    checks.push(("hard audit is 13/13".into(), hard.correct() == hard_total()));

    println!("[4/5] original 24-case acceptance corpus");
    let mut corpus_train = split_cases(Split::Train);
    corpus_train.extend(split_cases(Split::Calibration));
    let corpus_heldout = split_cases(Split::Heldout);
    let corpus_round = run_round_with(&base, &corpus_train, &out.join("corpus"), &beam);
    let corpus_held = evaluate(&corpus_round.adopted_harness, &corpus_heldout, false);
    println!(
        "      base {}/{} -> adopted {}/{}; held-out {}/{}; selected: {}",
        corpus_round.base.correct(),
        corpus_round.base.outcomes.len(),
        corpus_round.adopted.correct(),
        corpus_round.adopted.outcomes.len(),
        corpus_held.correct(),
        corpus_held.outcomes.len(),
        corpus_round.chosen_label
    );
    let _ = crate::write_round_traces(
        &out.join("corpus/traces.jsonl"),
        &corpus_train,
        &corpus_heldout,
        &corpus_round,
        "beam",
        1,
    );
    checks.push((
        "corpus adopts to 18/18".into(),
        corpus_round.adopted.correct() == 18,
    ));
    checks.push(("corpus held-out is 6/6".into(), corpus_held.correct() == 6));

    println!("[5/5] learned-only ablation: no complete/singles/pairs safety nets");
    let learned_beam = BeamProposer::new(model, 8, 4).learned_only();
    let learned_round = run_round_with(
        &base,
        &train_cases,
        &out.join("learned-only"),
        &learned_beam,
    );
    let learned_held = evaluate(&learned_round.adopted_harness, &generated_heldout, false);
    let learned_hard = evaluate(&learned_round.adopted_harness, &hard_cases(), false);
    println!(
        "      adopted {}/{}; held-out {}/{}; hard {}/{}",
        learned_round.adopted.correct(),
        learned_round.adopted.outcomes.len(),
        learned_held.correct(),
        learned_held.outcomes.len(),
        learned_hard.correct(),
        hard_total()
    );
    let _ = crate::write_round_traces(
        &out.join("learned-only/traces.jsonl"),
        &train_cases,
        &generated_heldout,
        &learned_round,
        "beam-learned-only",
        1,
    );
    checks.push((
        "learned-only adopts to 19/19".into(),
        learned_round.adopted.correct() == 19,
    ));
    checks.push((
        "learned-only held-out is 6/6".into(),
        learned_held.correct() == 6,
    ));
    checks.push((
        "learned-only hard audit is 13/13".into(),
        learned_hard.correct() == hard_total(),
    ));

    let solved_cert =
        certificate_excerpt(&out.join("quick/certificates/adopted-train/compose-matmul.json"));
    let refused_cert =
        certificate_excerpt(&out.join("quick/certificates/adopted-train/refuse-saxpy-matmul.json"));

    let mut report = String::new();
    let _ = writeln!(report, "# Jotnar agent demo\n");
    let _ = writeln!(
        report,
        "This is the complete fixed menu: propose -> exact gate -> compose -> blind \
         select -> adopt -> retain -> held out. It is a **certified harness tuner**, \
         not an open-ended self-improving agent.\n"
    );

    let _ = writeln!(report, "## 1. Quick loop (5 cases)\n");
    let _ = write!(report, "{}", case_table(&quick));
    let _ = writeln!(
        report,
        "\nBase: **{}/{} correct**, adopted: **{}/{} correct**, held-out: \
         **{}/{} -> {}/{}**. Selected: `{}`.\n",
        quick.base.correct(),
        quick.base.outcomes.len(),
        quick.adopted.correct(),
        quick.adopted.outcomes.len(),
        quick.base_heldout.correct(),
        quick.base_heldout.outcomes.len(),
        quick.adopted_heldout.correct(),
        quick.adopted_heldout.outcomes.len(),
        quick.chosen_label
    );
    if let Some(cert) = &solved_cert {
        let _ = writeln!(report, "Solved certificate: `{cert}`.\n");
    }
    if let Some(cert) = &refused_cert {
        let _ = writeln!(report, "Refused certificate: `{cert}`.\n");
    }

    let _ = writeln!(report, "## 2. Generated stream (25 cases)\n");
    let _ = writeln!(
        report,
        "State model: **{} cases, {} states, {} transitions**, top-1 **{:.3}** \
         after 20 epochs at depth 3.\n",
        train_cases.len(),
        states.len(),
        rows.len(),
        accuracy
    );
    let _ = writeln!(
        report,
        "Base: **{}/{} correct**, adopted: **{}/{} correct**, held-out: \
         **{}/{}**. Selected: `{}`.\n",
        generated_round.base.correct(),
        generated_round.base.outcomes.len(),
        generated_round.adopted.correct(),
        generated_round.adopted.outcomes.len(),
        generated_held.correct(),
        generated_held.outcomes.len(),
        generated_round.chosen_label
    );

    let _ = writeln!(report, "## 3. Acceptance corpus (24 cases)\n");
    let _ = writeln!(
        report,
        "Base: **{}/{} correct**, adopted: **{}/{} correct**, held-out: \
         **{}/{}**. Selected: `{}`.\n",
        corpus_round.base.correct(),
        corpus_round.base.outcomes.len(),
        corpus_round.adopted.correct(),
        corpus_round.adopted.outcomes.len(),
        corpus_held.correct(),
        corpus_held.outcomes.len(),
        corpus_round.chosen_label
    );

    let _ = writeln!(report, "## 4. Learned-only ablation\n");
    let _ = writeln!(
        report,
        "No completeness, singles, or pairs safety nets. Adopted: **{}/{}**, \
         held-out: **{}/{}**, hard audit: **{}/{}**.\n",
        learned_round.adopted.correct(),
        learned_round.adopted.outcomes.len(),
        learned_held.correct(),
        learned_held.outcomes.len(),
        learned_hard.correct(),
        hard_total()
    );

    let _ = writeln!(report, "## 5. Hard audit (13 cases, audit-only)\n");
    let _ = write!(report, "{}", outcome_table(&hard));

    let _ = writeln!(report, "\n## What this demonstrates\n");
    let _ = writeln!(
        report,
        "- The exact interpreter gate is the only judge; every candidate is certified.\n\
         - Two independent failures are solved by composing the top proposals, not by a \
         single hand-picked edit.\n\
         - The learned `(task, state, action)` model drives the deep and shaped cases; the \
         held-out stream stays clean.\n\
         - One geometric `reduce_sum` action transfers to the held-out `[3,3]`, `[4,4]`, \
         and `[16]` shapes.\n\
         - A correctly impossible task produces a first-class refusal certificate.\n\
         - The learned-only run keeps the result without the completeness/singles/pairs nets.\n"
    );
    let _ = writeln!(report, "## What this does not demonstrate\n");
    let _ = writeln!(
        report,
        "- It does not invent new opcodes or families; the action menu is fixed and \
         human-authored.\n\
         - The hard set is 13 hand-authored audit cases, not an external benchmark.\n\
         - The deep refine cases still rely on the coverage/neutral-drift tie-break; this is \
         not a proof of general planning.\n\
         - No hardware cost or performance claim: the cost head predicts bounded steps, not \
         machine time.\n\
         - No claim of open-ended self-improvement; the human still supplies mechanisms, \
         features, and training cases.\n"
    );
    let _ = writeln!(report, "## Artifacts\n");
    let _ = writeln!(
        report,
        "- `quick/round.json`, `quick/traces.jsonl`, `quick/certificates/`\n\
         - `generated/round.json`, `generated/traces.jsonl`, `generated/certificates/`\n\
         - `corpus/...`, `learned-only/...`\n\
         - `state-effect-model.weights.json`, `state-effect-model.jotnar`\n\
         - `hard.json`, `summary.json`\n"
    );

    fs::write(&report_path, &report).map_err(|e| format!("{}: {e}", report_path.display()))?;

    let summary = json!({
        "quick": {
            "base": report_json(&quick.base),
            "adopted": report_json(&quick.adopted),
            "base_heldout": report_json(&quick.base_heldout),
            "adopted_heldout": report_json(&quick.adopted_heldout),
            "selected": quick.chosen_label,
            "adopted_flag": quick.adopted_flag,
        },
        "generated": {
            "base": report_json(&generated_round.base),
            "adopted": report_json(&generated_round.adopted),
            "base_heldout": report_json(&generated_round.base_heldout),
            "adopted_heldout": report_json(&generated_held),
            "selected": generated_round.chosen_label,
            "adopted_flag": generated_round.adopted_flag,
            "hard": hard_json(&hard),
        },
        "corpus": {
            "base": report_json(&corpus_round.base),
            "adopted": report_json(&corpus_round.adopted),
            "base_heldout": report_json(&corpus_round.base_heldout),
            "adopted_heldout": report_json(&corpus_held),
            "selected": corpus_round.chosen_label,
            "adopted_flag": corpus_round.adopted_flag,
        },
        "learned_only": {
            "adopted": report_json(&learned_round.adopted),
            "adopted_heldout": report_json(&learned_held),
            "selected": learned_round.chosen_label,
            "hard": hard_json(&learned_hard),
        },
        "model": {
            "cases": train_cases.len(),
            "states": states.len(),
            "transitions": rows.len(),
            "accuracy": accuracy,
            "epochs": 20,
            "depth": 3,
        },
        "checks": checks.iter().map(|(name, ok)| json!({"name": name, "ok": ok}))
            .collect::<Vec<_>>(),
        "all_ok": checks.iter().all(|(_, ok)| *ok),
    });
    fs::write(
        &summary_path,
        serde_json::to_string_pretty(&summary).unwrap() + "\n",
    )
    .map_err(|e| format!("{}: {e}", summary_path.display()))?;

    let _ = fs::write(
        out.join("hard.json"),
        serde_json::to_string_pretty(&json!({
            "generated": hard_json(&hard),
            "learned_only": hard_json(&learned_hard),
        }))
        .unwrap()
            + "\n",
    );

    let summary = DemoSummary {
        out_dir: out.to_path_buf(),
        quick_base: quick.base.correct(),
        quick_adopted: quick.adopted.correct(),
        quick_heldout: quick.adopted_heldout.correct(),
        generated_base: generated_round.base.correct(),
        generated_adopted: generated_round.adopted.correct(),
        generated_heldout: generated_held.correct(),
        corpus_base: corpus_round.base.correct(),
        corpus_adopted: corpus_round.adopted.correct(),
        corpus_heldout: corpus_held.correct(),
        hard_correct: hard.correct(),
        hard_total: hard_total(),
        learned_only_adopted: learned_round.adopted.correct(),
        learned_only_heldout: learned_held.correct(),
        learned_only_hard: learned_hard.correct(),
        model_cases: train_cases.len(),
        model_states: states.len(),
        model_transitions: rows.len(),
        model_accuracy: accuracy,
        checks,
    };

    if !summary.all_ok() {
        let failed: Vec<String> = summary
            .checks
            .iter()
            .filter(|(_, ok)| !ok)
            .map(|(name, _)| name.clone())
            .collect();
        return Err(format!("demo checks failed: {}", failed.join(", ")));
    }
    Ok(summary)
}
