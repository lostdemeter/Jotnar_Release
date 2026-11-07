use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::process::ExitCode;

use jotnar_agent::{
    action_effects, cases, evaluate, fit_joint, run_round, run_round_with, run_round_with_memory,
    split_cases, state_effects, Archive, ArchiveEntry, ArchiveProposer, Backbone, BeamProposer,
    BudgetPredictor, CostModel, DeterministicProposer, EffectModel, EvalReport, Harness,
    JointProposer, JotnarProposer, LoopedProposer, LoopedTaskEncoder, Proposer, RoundReport, Split,
    StateEffectModel, TaskEncoder, TeacherActionProposer, WorldModelProposer,
};

fn usage() -> &'static str {
    "jotnar-agent — listing-level autonomous circuit\n\
     \n\
     usage:\n\
       jotnar-agent run [--out DIR] [--rounds N] [--proposer jotnar|looped|task|teacher|world|beam|joint|qd|deterministic] [--loops N] [--encoder-loops N] [--fit] [--deep-train] [--weights FILE] [--encoder-weights FILE] [--looped-encoder-weights FILE] [--budget-predictor FILE] [--backbone FILE] [--teacher-actions FILE] [--effect-weights FILE] [--state-effect-weights FILE] [--joint-weights FILE] [--cost-weights FILE] [--cost-weight N] [--beam N] [--depth N] [--archive FILE] [--archive-novelty N] [--archive-capacity N] [--qd-seeds N] [--no-complete] [--no-singles] [--no-pairs] [--hard] [--corpus] [--generated]\n\
       jotnar-agent archive --file FILE\n\
       jotnar-agent feature-audit [--corpus] [--generated] [--state-effect-weights FILE] [--top-k N]\n\
       jotnar-agent fit-recurrent [--out DIR] [--deep-train] [--efficient]\n\
       jotnar-agent fit-encoder [--out DIR] [--corpus] [--generated] [--epochs N] [--lr N]\n\
       jotnar-agent fit-looped-encoder [--out DIR] [--corpus] [--generated] [--max-loops N] [--encoder-weights FILE]\n\
       jotnar-agent fit-budget-predictor [--out DIR] [--corpus] [--generated] [--max-loops N] [--epochs N] [--backbone FILE]\n\
       jotnar-agent fit-backbone [--out DIR] [--corpus] [--generated] [--qwen] [--hidden N] [--mix N] [--dim N] [--inter N] [--lr N] [--max-loops N]\n\
       jotnar-agent fit-qwen [--out DIR] [--generated] [--masks N] [--mask-rate F] [--ridge F] [--rounds N] [--seed N] [--scale F] [--dim N] [--inter N] [--max-loops N]\n\
       jotnar-agent fit-effect-model [--out DIR] [--corpus] [--generated] [--epochs N] [--lr N]\n\
       jotnar-agent fit-state-effect-model [--out DIR] [--corpus] [--generated] [--depth N] [--epochs N] [--lr N]\n\
       jotnar-agent fit-cost-model [--out DIR] [--corpus] [--generated] [--depth N] [--lambda F]\n\
       jotnar-agent fit-joint-model --out DIR --state-effect-weights FILE --backbone FILE [--corpus] [--generated] [--depth N] [--beam N]\n\
       jotnar-agent gen-tasks [--out FILE]\n\
       jotnar-agent gen-effects [--out FILE] [--corpus]\n\
       jotnar-agent gen-transitions [--out FILE] [--corpus] [--depth N]\n\
       jotnar-agent gen-teacher [--out FILE] [--generated] [--source NAME]\n\
       jotnar-agent distill-qwen --teacher FILE [--out DIR] [--readout|--end-to-end] [--corpus|--generated] [--epochs N] [--lr F] [--dim N] [--inter N] [--seed N] [--scale F] [--ridge F] [--rounds N] [--max-loops N]\n\
       jotnar-agent list\n"
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<ExitCode, String> {
    let mut args = std::env::args().skip(1);
    let cmd = args.next().ok_or_else(|| usage().to_string())?;
    match cmd.as_str() {
        "-h" | "--help" | "help" => {
            println!("{}", usage());
            Ok(ExitCode::SUCCESS)
        }
        "list" => {
            for case in cases() {
                println!("{:<22} {:?}", case.intent.id, case.expect);
            }
            Ok(ExitCode::SUCCESS)
        }
        "gen-effects" => {
            let mut out: Option<PathBuf> = None;
            let mut corpus_mode = false;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--out" => {
                        out = Some(PathBuf::from(args.next().ok_or("--out needs a path")?));
                    }
                    "--corpus" => corpus_mode = true,
                    other => return Err(format!("unknown option {other}")),
                }
            }
            let cases: Vec<jotnar_agent::Case> = if corpus_mode {
                jotnar_agent::corpus().into_iter().map(|c| c.case).collect()
            } else {
                jotnar_agent::generated_corpus()
                    .into_iter()
                    .map(|c| c.case)
                    .collect()
            };
            let effects = jotnar_agent::action_effects(&cases, &Harness::default());
            let mut text = String::new();
            for effect in &effects {
                text.push_str(&serde_json::to_string(effect).map_err(|e| e.to_string())?);
                text.push('\n');
            }
            match out {
                Some(path) => {
                    std::fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))?;
                }
                None => print!("{text}"),
            }
            Ok(ExitCode::SUCCESS)
        }
        "gen-teacher" => {
            let mut out: Option<PathBuf> = None;
            let mut generated = false;
            let mut source = "named".to_string();
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--out" => {
                        out = Some(PathBuf::from(args.next().ok_or("--out needs a path")?));
                    }
                    "--generated" => generated = true,
                    "--source" => {
                        source = args.next().ok_or("--source needs a value")?;
                    }
                    other => return Err(format!("unknown option {other}")),
                }
            }
            let cases: Vec<jotnar_agent::Case> = if generated {
                jotnar_agent::generated_corpus()
                    .into_iter()
                    .map(|c| c.case)
                    .collect()
            } else {
                jotnar_agent::corpus().into_iter().map(|c| c.case).collect()
            };
            let splits: BTreeMap<String, String> = if generated {
                jotnar_agent::generated_corpus()
                    .into_iter()
                    .map(|c| (c.case.intent.id.clone(), c.split.as_str().to_string()))
                    .collect()
            } else {
                jotnar_agent::corpus()
                    .into_iter()
                    .map(|c| (c.case.intent.id.clone(), c.split.as_str().to_string()))
                    .collect()
            };
            let base = Harness::default();
            let report = evaluate(&base, &cases, false);
            let views = jotnar_agent::case_views(&cases, &report);
            let teacher = TaskEncoder::prior();
            let mut text = String::new();
            for view in &views {
                let task = jotnar_agent::task_encoder::task_vector(view, &base);
                let latent: Vec<f64> = teacher
                    .latent(&task)
                    .map_err(|e| e.to_string())?
                    .into_iter()
                    .map(|v| v as f64)
                    .collect();
                let record = jotnar_agent::TeacherRecord {
                    case: view.case_id.clone(),
                    task,
                    latent,
                    split: splits
                        .get(&view.case_id)
                        .cloned()
                        .unwrap_or_else(|| "stream".to_string()),
                    source: source.clone(),
                };
                text.push_str(&serde_json::to_string(&record).map_err(|e| e.to_string())?);
                text.push('\n');
            }
            match out {
                Some(path) => {
                    std::fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))?;
                }
                None => print!("{text}"),
            }
            Ok(ExitCode::SUCCESS)
        }
        "distill-qwen" => {
            let mut out = PathBuf::from("runs/qwen-distill");
            let mut teacher: Option<PathBuf> = None;
            let mut dim = 10usize;
            let mut inter = 16usize;
            let mut seed = 1u64;
            let mut scale = 0.2f64;
            let mut ridge = 1e-3f64;
            let mut rounds = 3usize;
            let mut max_loops = 3usize;
            let mut end_to_end = true;
            let mut epochs = 20usize;
            let mut lr = 0.001f64;
            let mut corpus_mode = false;
            let mut generated = false;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--out" => {
                        out = PathBuf::from(args.next().ok_or("--out needs a path")?);
                    }
                    "--teacher" => {
                        teacher = Some(PathBuf::from(args.next().ok_or("--teacher needs a path")?));
                    }
                    "--corpus" => corpus_mode = true,
                    "--generated" => generated = true,
                    "--readout" => end_to_end = false,
                    "--end-to-end" => end_to_end = true,
                    "--epochs" => {
                        epochs = args
                            .next()
                            .ok_or("--epochs needs a number")?
                            .parse()
                            .map_err(|_| "--epochs needs a number".to_string())?;
                    }
                    "--lr" => {
                        lr = args
                            .next()
                            .ok_or("--lr needs a number")?
                            .parse()
                            .map_err(|_| "--lr needs a number".to_string())?;
                    }
                    "--dim" => {
                        dim = args
                            .next()
                            .ok_or("--dim needs a number")?
                            .parse()
                            .map_err(|_| "--dim needs a number".to_string())?;
                    }
                    "--inter" => {
                        inter = args
                            .next()
                            .ok_or("--inter needs a number")?
                            .parse()
                            .map_err(|_| "--inter needs a number".to_string())?;
                    }
                    "--seed" => {
                        seed = args
                            .next()
                            .ok_or("--seed needs a number")?
                            .parse()
                            .map_err(|_| "--seed needs a number".to_string())?;
                    }
                    "--scale" => {
                        scale = args
                            .next()
                            .ok_or("--scale needs a number")?
                            .parse()
                            .map_err(|_| "--scale needs a number".to_string())?;
                    }
                    "--ridge" => {
                        ridge = args
                            .next()
                            .ok_or("--ridge needs a number")?
                            .parse()
                            .map_err(|_| "--ridge needs a number".to_string())?;
                    }
                    "--rounds" => {
                        rounds = args
                            .next()
                            .ok_or("--rounds needs a number")?
                            .parse()
                            .map_err(|_| "--rounds needs a number".to_string())?;
                    }
                    "--max-loops" => {
                        max_loops = args
                            .next()
                            .ok_or("--max-loops needs a number")?
                            .parse()
                            .map_err(|_| "--max-loops needs a number".to_string())?;
                    }
                    other => return Err(format!("unknown option {other}")),
                }
            }
            distill_qwen(
                &out,
                teacher.as_deref().ok_or("--teacher is required")?,
                dim,
                inter,
                seed,
                scale,
                ridge,
                rounds,
                max_loops,
                end_to_end,
                epochs,
                lr,
                corpus_mode,
                generated,
            )
        }
        "gen-tasks" => {
            let mut out: Option<PathBuf> = None;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--out" => {
                        out = Some(PathBuf::from(args.next().ok_or("--out needs a path")?));
                    }
                    other => return Err(format!("unknown option {other}")),
                }
            }
            let mut text = String::new();
            for spec in jotnar_agent::task_specs() {
                text.push_str(&serde_json::to_string(&spec).map_err(|e| e.to_string())?);
                text.push('\n');
            }
            match out {
                Some(path) => {
                    std::fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))?;
                }
                None => print!("{text}"),
            }
            Ok(ExitCode::SUCCESS)
        }
        "fit-budget-predictor" => {
            let mut out = PathBuf::from("runs/encoder");
            let mut corpus_mode = false;
            let mut generated = false;
            let mut max_loops = 3usize;
            let mut epochs = 10usize;
            let mut backbone: Option<PathBuf> = None;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--out" => {
                        out = PathBuf::from(args.next().ok_or("--out needs a path")?);
                    }
                    "--corpus" => corpus_mode = true,
                    "--generated" => generated = true,
                    "--max-loops" => {
                        max_loops = args
                            .next()
                            .ok_or("--max-loops needs a number")?
                            .parse()
                            .map_err(|_| "--max-loops needs a number".to_string())?;
                    }
                    "--epochs" => {
                        epochs = args
                            .next()
                            .ok_or("--epochs needs a number")?
                            .parse()
                            .map_err(|_| "--epochs needs a number".to_string())?;
                    }
                    "--backbone" => {
                        backbone =
                            Some(PathBuf::from(args.next().ok_or("--backbone needs a path")?));
                    }
                    other => return Err(format!("unknown option {other}")),
                }
            }
            fit_budget_predictor(
                &out,
                corpus_mode,
                generated,
                max_loops,
                epochs,
                backbone.as_deref(),
            )
        }
        "fit-backbone" => {
            let mut out = PathBuf::from("runs/backbone");
            let mut corpus_mode = false;
            let mut generated = false;
            let mut qwen = false;
            let mut hidden = 32usize;
            let mut mix = 1i64;
            let mut dim = 10usize;
            let mut inter = 16usize;
            let mut lr = 1i64;
            let mut max_loops = 3usize;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--out" => {
                        out = PathBuf::from(args.next().ok_or("--out needs a path")?);
                    }
                    "--corpus" => corpus_mode = true,
                    "--generated" => generated = true,
                    "--qwen" => qwen = true,
                    "--dim" => {
                        dim = args
                            .next()
                            .ok_or("--dim needs a number")?
                            .parse()
                            .map_err(|_| "--dim needs a number".to_string())?;
                    }
                    "--inter" => {
                        inter = args
                            .next()
                            .ok_or("--inter needs a number")?
                            .parse()
                            .map_err(|_| "--inter needs a number".to_string())?;
                    }
                    "--hidden" => {
                        hidden = args
                            .next()
                            .ok_or("--hidden needs a number")?
                            .parse()
                            .map_err(|_| "--hidden needs a number".to_string())?;
                    }
                    "--mix" => {
                        mix = args
                            .next()
                            .ok_or("--mix needs a number")?
                            .parse()
                            .map_err(|_| "--mix needs a number".to_string())?;
                    }
                    "--lr" => {
                        lr = args
                            .next()
                            .ok_or("--lr needs a number")?
                            .parse()
                            .map_err(|_| "--lr needs a number".to_string())?;
                    }
                    "--max-loops" => {
                        max_loops = args
                            .next()
                            .ok_or("--max-loops needs a number")?
                            .parse()
                            .map_err(|_| "--max-loops needs a number".to_string())?;
                    }
                    other => return Err(format!("unknown option {other}")),
                }
            }
            fit_backbone(
                &out,
                corpus_mode,
                generated,
                qwen,
                hidden,
                mix,
                dim,
                inter,
                lr,
                max_loops,
            )
        }
        "fit-qwen" => {
            let mut out = PathBuf::from("runs/qwen");
            let mut generated = false;
            let mut masks = 8usize;
            let mut mask_rate = 0.3f64;
            let mut ridge = 1e-3f64;
            let mut rounds = 3usize;
            let mut seed = 1u64;
            let mut scale = 0.2f64;
            let mut dim = 10usize;
            let mut inter = 16usize;
            let mut max_loops = 3usize;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--out" => {
                        out = PathBuf::from(args.next().ok_or("--out needs a path")?);
                    }
                    "--generated" => generated = true,
                    "--masks" => {
                        masks = args
                            .next()
                            .ok_or("--masks needs a number")?
                            .parse()
                            .map_err(|_| "--masks needs a number".to_string())?;
                    }
                    "--mask-rate" => {
                        mask_rate = args
                            .next()
                            .ok_or("--mask-rate needs a number")?
                            .parse()
                            .map_err(|_| "--mask-rate needs a number".to_string())?;
                    }
                    "--ridge" => {
                        ridge = args
                            .next()
                            .ok_or("--ridge needs a number")?
                            .parse()
                            .map_err(|_| "--ridge needs a number".to_string())?;
                    }
                    "--rounds" => {
                        rounds = args
                            .next()
                            .ok_or("--rounds needs a number")?
                            .parse()
                            .map_err(|_| "--rounds needs a number".to_string())?;
                    }
                    "--seed" => {
                        seed = args
                            .next()
                            .ok_or("--seed needs a number")?
                            .parse()
                            .map_err(|_| "--seed needs a number".to_string())?;
                    }
                    "--scale" => {
                        scale = args
                            .next()
                            .ok_or("--scale needs a number")?
                            .parse()
                            .map_err(|_| "--scale needs a number".to_string())?;
                    }
                    "--dim" => {
                        dim = args
                            .next()
                            .ok_or("--dim needs a number")?
                            .parse()
                            .map_err(|_| "--dim needs a number".to_string())?;
                    }
                    "--inter" => {
                        inter = args
                            .next()
                            .ok_or("--inter needs a number")?
                            .parse()
                            .map_err(|_| "--inter needs a number".to_string())?;
                    }
                    "--max-loops" => {
                        max_loops = args
                            .next()
                            .ok_or("--max-loops needs a number")?
                            .parse()
                            .map_err(|_| "--max-loops needs a number".to_string())?;
                    }
                    other => return Err(format!("unknown option {other}")),
                }
            }
            fit_qwen(
                &out, generated, masks, mask_rate, ridge, rounds, seed, scale, dim, inter,
                max_loops,
            )
        }
        "fit-effect-model" => {
            let mut out = PathBuf::from("runs/effect-model");
            let mut corpus_mode = false;
            let mut generated = true;
            let mut epochs = 10usize;
            let mut lr = 1i64;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--out" => {
                        out = PathBuf::from(args.next().ok_or("--out needs a path")?);
                    }
                    "--corpus" => {
                        corpus_mode = true;
                        generated = false;
                    }
                    "--generated" => {
                        generated = true;
                        corpus_mode = false;
                    }
                    "--epochs" => {
                        epochs = args
                            .next()
                            .ok_or("--epochs needs a number")?
                            .parse()
                            .map_err(|_| "--epochs needs a number".to_string())?;
                    }
                    "--lr" => {
                        lr = args
                            .next()
                            .ok_or("--lr needs a number")?
                            .parse()
                            .map_err(|_| "--lr needs a number".to_string())?;
                    }
                    other => return Err(format!("unknown option {other}")),
                }
            }
            fit_effect_model(&out, corpus_mode, generated, epochs, lr)
        }
        "gen-transitions" => {
            let mut out: Option<PathBuf> = None;
            let mut corpus_mode = false;
            let mut depth = 2usize;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--out" => {
                        out = Some(PathBuf::from(args.next().ok_or("--out needs a path")?));
                    }
                    "--corpus" => corpus_mode = true,
                    "--depth" => {
                        depth = args
                            .next()
                            .ok_or("--depth needs a number")?
                            .parse()
                            .map_err(|_| "--depth needs a number".to_string())?;
                    }
                    other => return Err(format!("unknown option {other}")),
                }
            }
            let cases: Vec<jotnar_agent::Case> = if corpus_mode {
                jotnar_agent::corpus().into_iter().map(|c| c.case).collect()
            } else {
                jotnar_agent::generated_corpus()
                    .into_iter()
                    .map(|c| c.case)
                    .collect()
            };
            let rows = state_effects(&cases, &Harness::default(), depth);
            let mut text = String::new();
            for row in &rows {
                text.push_str(&serde_json::to_string(row).map_err(|e| e.to_string())?);
                text.push('\n');
            }
            match out {
                Some(path) => {
                    std::fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))?;
                }
                None => print!("{text}"),
            }
            Ok(ExitCode::SUCCESS)
        }
        "fit-state-effect-model" => {
            let mut out = PathBuf::from("runs/state-effect-model");
            let mut corpus_mode = false;
            let mut generated = true;
            let mut depth = 2usize;
            let mut epochs = 10usize;
            let mut lr = 1i64;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--out" => {
                        out = PathBuf::from(args.next().ok_or("--out needs a path")?);
                    }
                    "--corpus" => {
                        corpus_mode = true;
                        generated = false;
                    }
                    "--generated" => {
                        generated = true;
                        corpus_mode = false;
                    }
                    "--depth" => {
                        depth = args
                            .next()
                            .ok_or("--depth needs a number")?
                            .parse()
                            .map_err(|_| "--depth needs a number".to_string())?;
                    }
                    "--epochs" => {
                        epochs = args
                            .next()
                            .ok_or("--epochs needs a number")?
                            .parse()
                            .map_err(|_| "--epochs needs a number".to_string())?;
                    }
                    "--lr" => {
                        lr = args
                            .next()
                            .ok_or("--lr needs a number")?
                            .parse()
                            .map_err(|_| "--lr needs a number".to_string())?;
                    }
                    other => return Err(format!("unknown option {other}")),
                }
            }
            fit_state_effect_model(&out, corpus_mode, generated, depth, epochs, lr)
        }
        "fit-cost-model" => {
            let mut out = PathBuf::from("runs/cost-model");
            let mut corpus_mode = false;
            let mut generated = true;
            let mut depth = 2usize;
            let mut lambda = 1.0f64;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--out" => {
                        out = PathBuf::from(args.next().ok_or("--out needs a path")?);
                    }
                    "--corpus" => {
                        corpus_mode = true;
                        generated = false;
                    }
                    "--generated" => {
                        generated = true;
                        corpus_mode = false;
                    }
                    "--depth" => {
                        depth = args
                            .next()
                            .ok_or("--depth needs a number")?
                            .parse()
                            .map_err(|_| "--depth needs a number".to_string())?;
                    }
                    "--lambda" => {
                        lambda = args
                            .next()
                            .ok_or("--lambda needs a number")?
                            .parse()
                            .map_err(|_| "--lambda needs a number".to_string())?;
                    }
                    other => return Err(format!("unknown option {other}")),
                }
            }
            fit_cost_model(&out, corpus_mode, generated, depth, lambda)
        }
        "fit-joint-model" => {
            let mut out = PathBuf::from("runs/joint-model");
            let mut state_effect_weights: Option<PathBuf> = None;
            let mut backbone: Option<PathBuf> = None;
            let mut corpus_mode = false;
            let mut generated = true;
            let mut depth = 2usize;
            let mut beam = 4usize;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--out" => {
                        out = PathBuf::from(args.next().ok_or("--out needs a path")?);
                    }
                    "--state-effect-weights" => {
                        state_effect_weights = Some(PathBuf::from(
                            args.next().ok_or("--state-effect-weights needs a path")?,
                        ));
                    }
                    "--backbone" => {
                        backbone =
                            Some(PathBuf::from(args.next().ok_or("--backbone needs a path")?));
                    }
                    "--corpus" => {
                        corpus_mode = true;
                        generated = false;
                    }
                    "--generated" => {
                        generated = true;
                        corpus_mode = false;
                    }
                    "--depth" => {
                        depth = args
                            .next()
                            .ok_or("--depth needs a number")?
                            .parse()
                            .map_err(|_| "--depth needs a number".to_string())?;
                    }
                    "--beam" => {
                        beam = args
                            .next()
                            .ok_or("--beam needs a number")?
                            .parse()
                            .map_err(|_| "--beam needs a number".to_string())?;
                    }
                    other => return Err(format!("unknown option {other}")),
                }
            }
            fit_joint_model(
                &out,
                state_effect_weights.as_deref(),
                backbone.as_deref(),
                corpus_mode,
                generated,
                depth,
                beam,
            )
        }
        "fit-looped-encoder" => {
            let mut out = PathBuf::from("runs/encoder");
            let mut corpus_mode = false;
            let mut generated = false;
            let mut max_loops = 3usize;
            let mut encoder_weights: Option<PathBuf> = None;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--out" => {
                        out = PathBuf::from(args.next().ok_or("--out needs a path")?);
                    }
                    "--corpus" => corpus_mode = true,
                    "--generated" => generated = true,
                    "--max-loops" => {
                        max_loops = args
                            .next()
                            .ok_or("--max-loops needs a number")?
                            .parse()
                            .map_err(|_| "--max-loops needs a number".to_string())?;
                    }
                    "--encoder-weights" => {
                        encoder_weights = Some(PathBuf::from(
                            args.next().ok_or("--encoder-weights needs a path")?,
                        ));
                    }
                    other => return Err(format!("unknown option {other}")),
                }
            }
            fit_looped_encoder(
                &out,
                corpus_mode,
                generated,
                max_loops,
                encoder_weights.as_deref(),
            )
        }
        "fit-encoder" => {
            let mut out = PathBuf::from("runs/encoder");
            let mut corpus_mode = false;
            let mut generated = false;
            let mut epochs = 10usize;
            let mut lr = 1i64;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--out" => {
                        out = PathBuf::from(args.next().ok_or("--out needs a path")?);
                    }
                    "--corpus" => corpus_mode = true,
                    "--generated" => generated = true,
                    "--epochs" => {
                        epochs = args
                            .next()
                            .ok_or("--epochs needs a number")?
                            .parse()
                            .map_err(|_| "--epochs needs a number".to_string())?;
                    }
                    "--lr" => {
                        lr = args
                            .next()
                            .ok_or("--lr needs a number")?
                            .parse()
                            .map_err(|_| "--lr needs a number".to_string())?;
                    }
                    other => return Err(format!("unknown option {other}")),
                }
            }
            fit_encoder(&out, corpus_mode, generated, epochs, lr)
        }
        "fit-recurrent" => {
            let mut out = PathBuf::from("runs/agent");
            let mut deep_train = false;
            let mut efficient = false;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--out" => {
                        out = PathBuf::from(args.next().ok_or("--out needs a path")?);
                    }
                    "--deep-train" => deep_train = true,
                    "--efficient" => efficient = true,
                    other => return Err(format!("unknown option {other}")),
                }
            }
            fit_recurrent(&out, deep_train, efficient)
        }
        "archive" => {
            let mut file: Option<PathBuf> = None;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--file" => {
                        file = Some(PathBuf::from(args.next().ok_or("--file needs a path")?));
                    }
                    other => return Err(format!("unknown option {other}")),
                }
            }
            let path = file.ok_or("archive requires --file FILE")?;
            let archive = Archive::load(&path).map_err(|e| e.to_string())?;
            println!(
                "archive: {} entries (capacity {}, min_novelty {})",
                archive.entries.len(),
                archive.capacity,
                archive.min_novelty
            );
            for entry in &archive.entries {
                println!(
                    "  [{:>3}/{:>3} train, {:>3}/{:>3} held, novelty {:>2}] {}",
                    entry.correct,
                    entry.correct + entry.wrong,
                    entry.heldout_correct,
                    entry.heldout_correct + entry.heldout_wrong,
                    entry.novelty,
                    entry.label
                );
            }
            Ok(ExitCode::SUCCESS)
        }
        "feature-audit" => {
            let mut corpus_mode = false;
            let mut generated = true;
            let mut weights: Option<PathBuf> = None;
            let mut top_k = 3usize;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--corpus" => {
                        corpus_mode = true;
                        generated = false;
                    }
                    "--generated" => {
                        generated = true;
                        corpus_mode = false;
                    }
                    "--state-effect-weights" => {
                        weights = Some(PathBuf::from(
                            args.next().ok_or("--state-effect-weights needs a path")?,
                        ));
                    }
                    "--top-k" => {
                        top_k = args
                            .next()
                            .ok_or("--top-k needs a number")?
                            .parse()
                            .map_err(|_| "--top-k needs a number".to_string())?;
                    }
                    other => return Err(format!("unknown option {other}")),
                }
            }
            let (train, heldout) = if generated {
                fit_splits(true, true)
            } else {
                fit_splits(false, corpus_mode)
            };
            feature_audit(
                &train,
                &heldout,
                &jotnar_agent::hard_cases(),
                weights.as_deref(),
                top_k.max(1),
            )?;
            Ok(ExitCode::SUCCESS)
        }
        "run" => {
            let mut out = PathBuf::from("runs/agent");
            let mut rounds = 1usize;
            let mut proposer = "jotnar".to_string();
            let mut loops = 2usize;
            let mut fit = false;
            let mut deep_train = false;
            let mut corpus_mode = false;
            let mut generated = false;
            let mut weights: Option<PathBuf> = None;
            let mut encoder_weights: Option<PathBuf> = None;
            let mut looped_encoder_weights: Option<PathBuf> = None;
            let mut budget_predictor: Option<PathBuf> = None;
            let mut backbone: Option<PathBuf> = None;
            let mut teacher_actions: Option<PathBuf> = None;
            let mut effect_weights: Option<PathBuf> = None;
            let mut state_effect_weights: Option<PathBuf> = None;
            let mut joint_weights: Option<PathBuf> = None;
            let mut cost_weights: Option<PathBuf> = None;
            let mut cost_weight = 4i64;
            let mut beam = 4usize;
            let mut depth = 2usize;
            let mut archive_path: Option<PathBuf> = None;
            let mut archive_novelty = 2usize;
            let mut archive_capacity = 16usize;
            let mut hard = false;
            let mut qd_seeds = 8usize;
            let mut encoder_loops = 0usize;
            let mut include_complete = true;
            let mut include_singles = true;
            let mut include_pairs = true;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--out" => {
                        out = PathBuf::from(args.next().ok_or("--out needs a path")?);
                    }
                    "--rounds" => {
                        rounds = args
                            .next()
                            .ok_or("--rounds needs a number")?
                            .parse()
                            .map_err(|_| "--rounds needs a number".to_string())?;
                    }
                    "--proposer" => {
                        proposer = args.next().ok_or("--proposer needs a value")?;
                    }
                    "--loops" => {
                        loops = args
                            .next()
                            .ok_or("--loops needs a number")?
                            .parse()
                            .map_err(|_| "--loops needs a number".to_string())?;
                    }
                    "--fit" => fit = true,
                    "--deep-train" => deep_train = true,
                    "--corpus" => corpus_mode = true,
                    "--generated" => generated = true,
                    "--weights" => {
                        weights = Some(PathBuf::from(args.next().ok_or("--weights needs a path")?));
                    }
                    "--encoder-weights" => {
                        encoder_weights = Some(PathBuf::from(
                            args.next().ok_or("--encoder-weights needs a path")?,
                        ));
                    }
                    "--encoder-loops" => {
                        encoder_loops = args
                            .next()
                            .ok_or("--encoder-loops needs a number")?
                            .parse()
                            .map_err(|_| "--encoder-loops needs a number".to_string())?;
                    }
                    "--looped-encoder-weights" => {
                        looped_encoder_weights = Some(PathBuf::from(
                            args.next().ok_or("--looped-encoder-weights needs a path")?,
                        ));
                    }
                    "--budget-predictor" => {
                        budget_predictor = Some(PathBuf::from(
                            args.next().ok_or("--budget-predictor needs a path")?,
                        ));
                    }
                    "--backbone" => {
                        backbone =
                            Some(PathBuf::from(args.next().ok_or("--backbone needs a path")?));
                    }
                    "--teacher-actions" => {
                        teacher_actions = Some(PathBuf::from(
                            args.next().ok_or("--teacher-actions needs a path")?,
                        ));
                    }
                    "--effect-weights" => {
                        effect_weights = Some(PathBuf::from(
                            args.next().ok_or("--effect-weights needs a path")?,
                        ));
                    }
                    "--state-effect-weights" => {
                        state_effect_weights = Some(PathBuf::from(
                            args.next().ok_or("--state-effect-weights needs a path")?,
                        ));
                    }
                    "--joint-weights" => {
                        joint_weights = Some(PathBuf::from(
                            args.next().ok_or("--joint-weights needs a path")?,
                        ));
                    }
                    "--cost-weights" => {
                        cost_weights = Some(PathBuf::from(
                            args.next().ok_or("--cost-weights needs a path")?,
                        ));
                    }
                    "--cost-weight" => {
                        cost_weight = args
                            .next()
                            .ok_or("--cost-weight needs a number")?
                            .parse()
                            .map_err(|_| "--cost-weight needs a number".to_string())?;
                    }
                    "--beam" => {
                        beam = args
                            .next()
                            .ok_or("--beam needs a number")?
                            .parse()
                            .map_err(|_| "--beam needs a number".to_string())?;
                    }
                    "--depth" => {
                        depth = args
                            .next()
                            .ok_or("--depth needs a number")?
                            .parse()
                            .map_err(|_| "--depth needs a number".to_string())?;
                    }
                    "--archive" => {
                        archive_path =
                            Some(PathBuf::from(args.next().ok_or("--archive needs a path")?));
                    }
                    "--archive-novelty" => {
                        archive_novelty = args
                            .next()
                            .ok_or("--archive-novelty needs a number")?
                            .parse()
                            .map_err(|_| "--archive-novelty needs a number".to_string())?;
                    }
                    "--archive-capacity" => {
                        archive_capacity = args
                            .next()
                            .ok_or("--archive-capacity needs a number")?
                            .parse()
                            .map_err(|_| "--archive-capacity needs a number".to_string())?;
                    }
                    "--hard" => hard = true,
                    "--no-complete" => include_complete = false,
                    "--no-singles" => include_singles = false,
                    "--no-pairs" => include_pairs = false,
                    "--qd-seeds" => {
                        qd_seeds = args
                            .next()
                            .ok_or("--qd-seeds needs a number")?
                            .parse()
                            .map_err(|_| "--qd-seeds needs a number".to_string())?;
                    }
                    other => return Err(format!("unknown option {other}")),
                }
            }
            let all = if generated {
                jotnar_agent::generated_corpus()
                    .into_iter()
                    .map(|c| c.case)
                    .collect()
            } else {
                cases()
            };
            let (run_cases, heldout_cases) = if generated {
                let mut train = jotnar_agent::generated_split_cases(Split::Train);
                train.extend(jotnar_agent::generated_split_cases(Split::Calibration));
                (train, jotnar_agent::generated_split_cases(Split::Heldout))
            } else if corpus_mode {
                let mut train = split_cases(Split::Train);
                train.extend(split_cases(Split::Calibration));
                (train, split_cases(Split::Heldout))
            } else {
                (all.clone(), Vec::new())
            };
            let corpus_like = corpus_mode || generated;
            let mut harness = Harness::default();
            std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
            let memory = out.join("experience.jsonl");
            let mut jotnar_prop = if corpus_mode {
                JotnarProposer::corpus_prior()
            } else {
                JotnarProposer::prior()
            };
            let mut looped_prop = match &weights {
                Some(path) => LoopedProposer::load(loops, path).map_err(|e| e.to_string())?,
                None if corpus_mode => LoopedProposer::corpus_prior(loops),
                None => LoopedProposer::prior(loops),
            };
            let det_prop = DeterministicProposer;
            let teacher_prop = match &teacher_actions {
                Some(path) => Some(TeacherActionProposer::load(path).map_err(|e| e.to_string())?),
                None => None,
            };
            let world_prop = match &effect_weights {
                Some(path) => Some(WorldModelProposer::new(
                    EffectModel::load(path).map_err(|e| e.to_string())?,
                    4,
                )),
                None => None,
            };
            let beam_prop = match &state_effect_weights {
                Some(path) => {
                    let model = StateEffectModel::load(path).map_err(|e| e.to_string())?;
                    let mut proposer = BeamProposer::new(model, beam, depth);
                    proposer.include_complete = include_complete;
                    proposer.include_singles = include_singles;
                    proposer.include_pairs = include_pairs;
                    if let Some(path) = &cost_weights {
                        let cost = CostModel::load(path).map_err(|e| e.to_string())?;
                        proposer = proposer.with_cost(cost, cost_weight);
                    }
                    Some(proposer)
                }
                None => None,
            };
            let joint_prop = match &joint_weights {
                Some(path) => {
                    let mut proposer = JointProposer::load(path).map_err(|e| e.to_string())?;
                    proposer.include_complete = include_complete;
                    proposer.include_singles = include_singles;
                    proposer.include_pairs = include_pairs;
                    Some(proposer)
                }
                None => None,
            };
            let qd_prop = archive_path.as_ref().map(|path| {
                let archive = Archive::load(path).unwrap_or_else(|_| Archive::new(16, 2));
                ArchiveProposer::new(archive, qd_seeds)
            });
            let task_prop = match &encoder_weights {
                Some(path) => TaskEncoder::load(path).map_err(|e| e.to_string())?,
                None => TaskEncoder::prior(),
            };
            let looped_task_prop = if let Some(path) = &looped_encoder_weights {
                Some(LoopedTaskEncoder::load(path).map_err(|e| e.to_string())?)
            } else if encoder_loops > 0 {
                Some(LoopedTaskEncoder::from_linear(&task_prop, encoder_loops))
            } else {
                None
            };
            let looped_task_prop = if let Some(path) = &budget_predictor {
                let predictor = BudgetPredictor::load(path).map_err(|e| e.to_string())?;
                let max_loops = predictor.max_loops;
                Some(match looped_task_prop {
                    Some(e) => e.with_budget_predictor(predictor),
                    None => LoopedTaskEncoder::from_linear(&task_prop, max_loops)
                        .with_budget_predictor(predictor),
                })
            } else {
                looped_task_prop
            };
            let looped_task_prop = if let Some(path) = &backbone {
                let bb = Backbone::load(path).map_err(|e| e.to_string())?;
                Some(match looped_task_prop {
                    Some(e) => e.with_backbone(bb),
                    None => LoopedTaskEncoder::from_linear(&task_prop, encoder_loops.max(3))
                        .with_backbone(bb),
                })
            } else {
                looped_task_prop
            };
            if fit && proposer == "task" {
                println!(
                    "fit: task-encoder fitting is not implemented in this slice; using the prior"
                );
            } else if fit && proposer == "teacher" {
                println!("fit: teacher actions are fixed; nothing to fit");
            } else if fit && (proposer == "world" || proposer == "beam" || proposer == "joint") {
                println!("fit: world-model weights are fixed; nothing to fit");
            } else if fit {
                let calibration_cases = if corpus_like {
                    run_cases.clone()
                } else if deep_train {
                    jotnar_agent::deep_calibration_cases()
                } else {
                    jotnar_agent::calibration_cases()
                };
                let calibration = run_round_with_memory(
                    &harness,
                    &calibration_cases,
                    &out.join("calibration"),
                    &memory,
                    &det_prop,
                );
                if proposer == "looped" {
                    let changes = looped_prop.fit_from_report(&harness, &calibration, &[], 1);
                    looped_prop
                        .save(&out.join("looped.weights.json"))
                        .map_err(|e| e.to_string())?;
                    println!("fit(looped): {}", changes.join("; "));
                } else {
                    let changes = jotnar_prop.fit_from_report(&harness, &calibration, &[], 1);
                    jotnar_prop
                        .save(&out.join("proposer.weights.json"))
                        .map_err(|e| e.to_string())?;
                    println!("fit: {}", changes.join("; "));
                }
            }
            let prop: &dyn Proposer = if proposer == "deterministic" {
                &det_prop
            } else if proposer == "jotnar" {
                &jotnar_prop
            } else if proposer == "looped" {
                &looped_prop
            } else if proposer == "task" {
                match &looped_task_prop {
                    Some(p) => p,
                    None => &task_prop,
                }
            } else if proposer == "teacher" {
                teacher_prop
                    .as_ref()
                    .ok_or("--proposer teacher requires --teacher-actions FILE")?
            } else if proposer == "world" {
                world_prop
                    .as_ref()
                    .ok_or("--proposer world requires --effect-weights FILE")?
            } else if proposer == "beam" {
                beam_prop
                    .as_ref()
                    .ok_or("--proposer beam requires --state-effect-weights FILE")?
            } else if proposer == "joint" {
                joint_prop
                    .as_ref()
                    .ok_or("--proposer joint requires --joint-weights FILE")?
            } else if proposer == "qd" {
                qd_prop
                    .as_ref()
                    .ok_or("--proposer qd requires --archive FILE")?
            } else {
                return Err(format!("unknown proposer {proposer:?}"));
            };
            if proposer == "jotnar" {
                std::fs::write(out.join("proposer.jotnar"), &jotnar_prop.listing)
                    .map_err(|e| e.to_string())?;
            } else if proposer == "looped" {
                std::fs::write(out.join("proposer.jotnar"), &looped_prop.listing)
                    .map_err(|e| e.to_string())?;
            } else if proposer == "task" {
                let listing = match &looped_task_prop {
                    Some(p) => p.listings_file(),
                    None => &task_prop.listing,
                };
                std::fs::write(out.join("proposer.jotnar"), listing).map_err(|e| e.to_string())?;
            } else if proposer == "teacher" {
                let actions = teacher_prop
                    .as_ref()
                    .map(|p| &p.actions)
                    .ok_or("--proposer teacher requires --teacher-actions FILE")?;
                std::fs::write(
                    out.join("proposer.json"),
                    serde_json::to_string_pretty(actions).map_err(|e| e.to_string())? + "\n",
                )
                .map_err(|e| e.to_string())?;
            } else if proposer == "world" {
                let model = world_prop
                    .as_ref()
                    .map(|p| &p.model)
                    .ok_or("--proposer world requires --effect-weights FILE")?;
                std::fs::write(out.join("proposer.jotnar"), &model.listing)
                    .map_err(|e| e.to_string())?;
                std::fs::write(
                    out.join("proposer.json"),
                    serde_json::to_string_pretty(model).map_err(|e| e.to_string())? + "\n",
                )
                .map_err(|e| e.to_string())?;
            } else if proposer == "beam" {
                let model = beam_prop
                    .as_ref()
                    .map(|p| &p.model)
                    .ok_or("--proposer beam requires --state-effect-weights FILE")?;
                std::fs::write(out.join("proposer.jotnar"), &model.listing)
                    .map_err(|e| e.to_string())?;
                std::fs::write(
                    out.join("proposer.json"),
                    serde_json::to_string_pretty(model).map_err(|e| e.to_string())? + "\n",
                )
                .map_err(|e| e.to_string())?;
            } else if proposer == "joint" {
                let joint = joint_prop
                    .as_ref()
                    .ok_or("--proposer joint requires --joint-weights FILE")?;
                std::fs::write(out.join("proposer.jotnar"), &joint.effect.listing)
                    .map_err(|e| e.to_string())?;
                std::fs::write(
                    out.join("proposer.json"),
                    serde_json::to_string_pretty(joint).map_err(|e| e.to_string())? + "\n",
                )
                .map_err(|e| e.to_string())?;
            }
            let mut archive_state = archive_path.as_ref().map(|path| {
                let mut archive = Archive::load(path)
                    .unwrap_or_else(|_| Archive::new(archive_capacity, archive_novelty));
                archive.capacity = archive_capacity;
                archive.min_novelty = archive_novelty;
                archive
            });
            for round in 0..rounds {
                let dir = out.join(format!("round-{}", round + 1));
                let memory_view = jotnar_agent::load_experience(&memory);
                let report = run_round_with_memory(&harness, &run_cases, &dir, &memory, prop);
                if let Some(archive) = archive_state.as_mut() {
                    for candidate in &report.candidates {
                        if candidate.train.correct() < report.base.correct() {
                            continue;
                        }
                        let held = if heldout_cases.is_empty() {
                            candidate.heldout.clone()
                        } else {
                            evaluate(&candidate.harness, &heldout_cases, false)
                        };
                        let _ = archive.insert(ArchiveEntry {
                            label: candidate.label.clone(),
                            source: proposer.clone(),
                            harness: candidate.harness.clone(),
                            descriptor: candidate.harness.state_features(),
                            correct: candidate.train.correct(),
                            wrong: candidate.train.wrong(),
                            heldout_correct: held.correct(),
                            heldout_wrong: held.wrong(),
                            novelty: 0,
                        });
                    }
                }
                if let Err(e) = jotnar_agent::trace::write_round_traces(
                    &dir.join("traces.jsonl"),
                    &run_cases,
                    &heldout_cases,
                    &report,
                    &proposer,
                    round + 1,
                ) {
                    eprintln!("trace: {e}");
                }
                if round == 0
                    && (proposer == "jotnar" || proposer == "looped" || proposer == "task")
                {
                    let trace = if proposer == "looped" {
                        serde_json::to_value(looped_prop.trace(
                            &harness,
                            &report.base,
                            &memory_view,
                        ))
                        .unwrap()
                    } else if proposer == "task" {
                        serde_json::json!({
                            "kind": "task-encoder",
                            "encoder_loops": encoder_loops,
                            "task_features": jotnar_agent::TASK_FEATURE_NAMES,
                            "latent": jotnar_agent::LATENT_NAMES,
                            "actions": jotnar_agent::ACTIONS,
                        })
                    } else {
                        serde_json::to_value(jotnar_agent::proposer::trace(
                            &jotnar_prop,
                            &harness,
                            &report.base,
                            &memory_view,
                        ))
                        .unwrap()
                    };
                    std::fs::write(
                        out.join("proposer.json"),
                        serde_json::to_string_pretty(&trace).unwrap() + "\n",
                    )
                    .map_err(|e| e.to_string())?;
                }
                if corpus_like {
                    let held = evaluate(&report.adopted_harness, &heldout_cases, false);
                    print_corpus_report(&report, &held, round + 1, rounds);
                } else {
                    print_report(&report, round + 1, rounds);
                }
                harness = report.adopted_harness.clone();
            }
            if hard {
                let hard_cases = jotnar_agent::hard_cases();
                let hard_report = evaluate(&harness, &hard_cases, false);
                println!(
                    "hard audit: {}/{} correct, {} wrong (never used for selection or adoption)",
                    hard_report.correct(),
                    hard_cases.len(),
                    hard_report.wrong()
                );
                for outcome in &hard_report.outcomes {
                    println!(
                        "  {:32} {:14} steps {}",
                        outcome.case,
                        outcome.kind.as_str(),
                        outcome.steps
                    );
                }
                let payload = serde_json::json!({
                    "correct": hard_report.correct(),
                    "wrong": hard_report.wrong(),
                    "total_steps": hard_report.total_steps,
                    "outcomes": hard_report.outcomes.iter().map(|o| serde_json::json!({
                        "case": o.case,
                        "family": o.family,
                        "kind": o.kind.as_str(),
                        "steps": o.steps,
                        "note": o.note,
                    })).collect::<Vec<_>>(),
                });
                let _ = std::fs::write(
                    out.join("hard.json"),
                    serde_json::to_string_pretty(&payload).unwrap() + "\n",
                );
            }
            if let (Some(path), Some(archive)) = (&archive_path, &archive_state) {
                archive.save(path).map_err(|e| e.to_string())?;
                println!(
                    "archive: {} entries -> {}",
                    archive.entries.len(),
                    path.display()
                );
            }
            Ok(ExitCode::SUCCESS)
        }
        other => Err(format!("unknown command {other:?}\n\n{}", usage())),
    }
}

fn argmax(scores: &[i64]) -> usize {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by_key(|i| std::cmp::Reverse(scores[*i]));
    order.first().copied().unwrap_or(0)
}

fn fit_splits(
    generated: bool,
    corpus_mode: bool,
) -> (Vec<jotnar_agent::Case>, Vec<jotnar_agent::Case>) {
    if generated {
        let mut train = jotnar_agent::generated_split_cases(Split::Train);
        train.extend(jotnar_agent::generated_split_cases(Split::Calibration));
        (train, jotnar_agent::generated_split_cases(Split::Heldout))
    } else if corpus_mode {
        let mut train = split_cases(Split::Train);
        train.extend(split_cases(Split::Calibration));
        (train, split_cases(Split::Heldout))
    } else {
        (cases(), Vec::new())
    }
}

type LoopFitKey = (usize, usize, i64, i64, i64, i64);

#[allow(clippy::too_many_arguments)]
fn fit_backbone(
    out: &std::path::Path,
    corpus_mode: bool,
    generated: bool,
    qwen: bool,
    hidden: usize,
    mix: i64,
    dim: usize,
    inter: usize,
    lr: i64,
    max_loops: usize,
) -> Result<ExitCode, String> {
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let (train, heldout) = if generated {
        fit_splits(true, true)
    } else if corpus_mode {
        fit_splits(false, true)
    } else {
        (cases(), Vec::new())
    };
    let base = Harness::default();
    let named = TaskEncoder::prior();
    let views = jotnar_agent::case_views(&train, &evaluate(&base, &train, false));
    let tasks: Vec<Vec<i64>> = views
        .iter()
        .map(|view| jotnar_agent::task_encoder::task_vector(view, &base))
        .collect();
    let targets: Vec<Vec<i64>> = tasks
        .iter()
        .map(|task| named.latent(task))
        .collect::<Result<_, _>>()?;

    let backbone = if qwen {
        let backbone = Backbone::qwen_named(&named.w_enc, &named.b_enc, dim, inter);
        let err = tasks
            .iter()
            .zip(&targets)
            .map(|(task, target)| {
                backbone
                    .latent(task)
                    .iter()
                    .zip(target)
                    .map(|(x, y)| (x - y).abs())
                    .max()
                    .unwrap_or(0)
            })
            .max()
            .unwrap_or(0);
        backbone
            .save(&out.join("backbone.weights.json"))
            .map_err(|e| e.to_string())?;
        let q = backbone.qwen.as_ref().expect("qwen weights");
        println!(
            "backbone fit: {} samples, kind=qwen, dim={}, inter={}, max latent error={err}",
            tasks.len(),
            q.dim,
            q.inter
        );
        backbone
    } else {
        let mut backbone = Backbone::mlp(&named.w_enc, &named.b_enc, hidden, mix);
        let err = backbone.fit(&tasks, &targets, lr);
        backbone
            .save(&out.join("backbone.weights.json"))
            .map_err(|e| e.to_string())?;
        println!(
            "backbone fit: {} samples, kind=mlp, hidden={}, mix={}, lr={}, max latent error={err}",
            tasks.len(),
            backbone.w1.len(),
            mix,
            lr
        );
        backbone
    };

    // Named-selection baseline, then the MLP backbone, both through the full
    // corpus loop (driver + exact gate).
    let named_encoder = LoopedTaskEncoder::prior(max_loops);
    let named_report = run_round_with(&base, &train, &out.join("eval-named"), &named_encoder);
    let named_held = if heldout.is_empty() {
        None
    } else {
        Some(evaluate(&named_report.adopted_harness, &heldout, false))
    };
    let encoder = LoopedTaskEncoder::prior(max_loops).with_backbone(backbone);
    encoder
        .save(&out.join("looped-encoder-backbone.weights.json"))
        .map_err(|e| e.to_string())?;
    let report = run_round_with(&base, &train, &out.join("eval"), &encoder);
    let held = if heldout.is_empty() {
        None
    } else {
        Some(evaluate(&report.adopted_harness, &heldout, false))
    };

    println!(
        "named: {} correct, {} wrong -> {} correct, {} wrong (adopted={})",
        named_report.base.correct(),
        named_report.base.wrong(),
        named_report.adopted.correct(),
        named_report.adopted.wrong(),
        named_report.adopted_flag
    );
    println!(
        "backbone: {} correct, {} wrong -> {} correct, {} wrong (adopted={})",
        report.base.correct(),
        report.base.wrong(),
        report.adopted.correct(),
        report.adopted.wrong(),
        report.adopted_flag
    );
    if let (Some(named_held), Some(held)) = (named_held, held) {
        println!(
            "heldout named={} correct, {} wrong; backbone={} correct, {} wrong",
            named_held.correct(),
            named_held.wrong(),
            held.correct(),
            held.wrong()
        );
    }
    println!("wrote {}", out.join("backbone.weights.json").display());
    println!(
        "wrote {} (backbone + action head)",
        out.join("looped-encoder-backbone.weights.json").display()
    );
    Ok(ExitCode::SUCCESS)
}

#[allow(clippy::too_many_arguments)]
fn fit_qwen(
    out: &std::path::Path,
    generated: bool,
    masks: usize,
    mask_rate: f64,
    ridge: f64,
    rounds: usize,
    seed: u64,
    scale: f64,
    dim: usize,
    inter: usize,
    max_loops: usize,
) -> Result<ExitCode, String> {
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let calibration = if generated {
        jotnar_agent::generated_split_cases(Split::Calibration)
    } else {
        split_cases(Split::Calibration)
    };
    let (train, heldout) = fit_splits(generated, true);
    let base = Harness::default();
    let named = TaskEncoder::prior();
    let report = run_round(&base, &calibration, &out.join("calib-base"));
    let views = jotnar_agent::case_views(&calibration, &report.base);
    let tasks: Vec<Vec<i64>> = views
        .iter()
        .map(|view| jotnar_agent::task_encoder::task_vector(view, &base))
        .collect();
    let targets: Vec<Vec<f64>> = tasks
        .iter()
        .map(|task| {
            named
                .latent(task)
                .expect("named latent")
                .into_iter()
                .map(|value| value as f64)
                .collect()
        })
        .collect();

    let mut state = seed | 1;
    let mut rnd = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut samples = Vec::new();
    for (task, target) in tasks.iter().zip(&targets) {
        for _ in 0..masks {
            let mut masked = task.clone();
            let mut nonzero = 0;
            for value in &mut masked {
                if rnd() < mask_rate {
                    *value = 0;
                } else if *value != 0 {
                    nonzero += 1;
                }
            }
            if nonzero == 0 && !task.is_empty() {
                masked[0] = task[0];
            }
            samples.push((masked, target.clone()));
        }
    }

    let mut backbone = Backbone::qwen_random(&named.w_enc, &named.b_enc, dim, inter, seed, scale);
    let before = samples
        .iter()
        .map(|(task, want)| {
            backbone
                .qwen
                .as_ref()
                .expect("qwen weights")
                .forward(task)
                .iter()
                .zip(want)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f64, f64::max)
        })
        .fold(0.0f64, f64::max);
    let err = backbone
        .qwen
        .as_mut()
        .expect("qwen weights")
        .fit_readouts(&samples, ridge, rounds);
    backbone
        .save(&out.join("backbone-qwen-trained.weights.json"))
        .map_err(|e| e.to_string())?;
    let trained_encoder = LoopedTaskEncoder::prior(max_loops).with_backbone(backbone);
    trained_encoder
        .save(&out.join("looped-encoder-qwen-trained.weights.json"))
        .map_err(|e| e.to_string())?;
    let named_encoder = LoopedTaskEncoder::prior(max_loops);
    let named_report = run_round_with(&base, &train, &out.join("eval-named"), &named_encoder);
    let trained_report = run_round_with(&base, &train, &out.join("eval-trained"), &trained_encoder);
    let named_held = evaluate(&named_report.adopted_harness, &heldout, false);
    let trained_held = evaluate(&trained_report.adopted_harness, &heldout, false);
    let adopted = trained_report.adopted.correct() == train.len()
        && trained_report.adopted.wrong() == 0
        && trained_held.correct() == heldout.len()
        && trained_held.wrong() == 0;

    println!(
        "qwen fit: {} calibration cases, {} masked samples, dim={}, inter={}, seed={}, ridge={}, rounds={}, masked latent error: before={before:.4}, after={err:.4}",
        calibration.len(),
        samples.len(),
        dim,
        inter,
        seed,
        ridge,
        rounds
    );
    println!(
        "named:   {} correct, {} wrong -> {} correct, {} wrong",
        named_report.base.correct(),
        named_report.base.wrong(),
        named_report.adopted.correct(),
        named_report.adopted.wrong()
    );
    println!(
        "trained: {} correct, {} wrong -> {} correct, {} wrong",
        trained_report.base.correct(),
        trained_report.base.wrong(),
        trained_report.adopted.correct(),
        trained_report.adopted.wrong()
    );
    println!(
        "heldout named={} correct, {} wrong; trained={} correct, {} wrong; adopted={adopted}",
        named_held.correct(),
        named_held.wrong(),
        trained_held.correct(),
        trained_held.wrong()
    );
    println!(
        "wrote {}",
        out.join("backbone-qwen-trained.weights.json").display()
    );
    println!(
        "wrote {}",
        out.join("looped-encoder-qwen-trained.weights.json")
            .display()
    );
    Ok(ExitCode::SUCCESS)
}

fn fit_effect_model(
    out: &std::path::Path,
    _corpus_mode: bool,
    generated: bool,
    epochs: usize,
    lr: i64,
) -> Result<ExitCode, String> {
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let (train, _) = if generated {
        fit_splits(true, true)
    } else {
        fit_splits(false, true)
    };
    let effects = action_effects(&train, &Harness::default());
    let mut model = EffectModel::prior();
    let accuracy = model.fit(&effects, epochs, lr);
    model
        .save(&out.join("effect-model.weights.json"))
        .map_err(|e| e.to_string())?;
    std::fs::write(out.join("effect-model.jotnar"), &model.listing).map_err(|e| e.to_string())?;
    println!(
        "effect model: {} cases, {} actions, epochs={}, lr={}, train top-1 accuracy={accuracy:.3}",
        effects.len(),
        jotnar_agent::ACTIONS.len(),
        epochs,
        lr
    );
    println!("wrote {}", out.join("effect-model.weights.json").display());
    println!("wrote {}", out.join("effect-model.jotnar").display());
    Ok(ExitCode::SUCCESS)
}

fn l1_distance(a: &[i64], b: &[i64]) -> usize {
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x - *y).unsigned_abs() as usize)
        .sum()
}

fn minimal_solvers(case: &jotnar_agent::Case, base: &Harness, depth: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    for (_, subset, harness) in jotnar_agent::reachable_states(base, depth) {
        let report = evaluate(&harness, std::slice::from_ref(case), false);
        if report.outcomes[0].kind.correct() {
            out.push(subset);
        }
    }
    out.sort_by_key(|subset| subset.len());
    out
}

fn feature_audit(
    train: &[jotnar_agent::Case],
    heldout: &[jotnar_agent::Case],
    hard: &[jotnar_agent::Case],
    weights: Option<&std::path::Path>,
    top_k: usize,
) -> Result<(), String> {
    let base = Harness::default();
    let model = match weights {
        Some(path) => Some(StateEffectModel::load(path).map_err(|e| e.to_string())?),
        None => None,
    };
    let train_report = evaluate(&base, train, false);
    let train_tasks: Vec<Vec<i64>> = jotnar_agent::case_views(train, &train_report)
        .iter()
        .map(|view| jotnar_agent::task_encoder::task_vector(view, &base))
        .collect();
    println!(
        "feature audit: {} training cases, {} held-out, {} hard",
        train.len(),
        heldout.len(),
        hard.len()
    );
    for (name, cases) in [("heldout", heldout), ("hard", hard)] {
        if cases.is_empty() {
            continue;
        }
        let report = evaluate(&base, cases, false);
        let views = jotnar_agent::case_views(cases, &report);
        println!("--- {name} ---");
        for (case, view) in cases.iter().zip(views.iter()) {
            let task = jotnar_agent::task_encoder::task_vector(view, &base);
            let distance = train_tasks
                .iter()
                .map(|train_task| l1_distance(&task, train_task))
                .min()
                .unwrap_or(0);
            let outcome = report
                .outcomes
                .iter()
                .find(|o| o.case == case.intent.id)
                .map(|o| o.kind.as_str())
                .unwrap_or("missing");
            let solvers = minimal_solvers(case, &base, 3);
            let best = solvers.first();
            let names = best
                .map(|subset| {
                    subset
                        .iter()
                        .map(|action| jotnar_agent::ACTIONS[*action])
                        .collect::<Vec<_>>()
                        .join(" + ")
                })
                .unwrap_or_else(|| "-".to_string());
            println!(
                "  {:32} count-dist {:3}  base {:14}  min-solver-size {}  [{}]",
                case.intent.id,
                distance,
                outcome,
                best.map(|s| s.len()).unwrap_or(0),
                names
            );
            println!(
                "    geom: start rank {} numel {} -> target rank {} numel {}; ops imatmul={} iselect={} iadd={}; reduce_applicable={}",
                view.start_rank,
                view.start_numel,
                view.target_rank,
                view.target_numel,
                view.start_imatmul,
                view.start_iselect,
                view.start_iadd,
                view.start_is_field && view.target_is_scalar && view.start_outputs == 1
            );
            if let Some(model) = &model {
                let scores = model.scores(&task, &base.state_features());
                let mut order: Vec<usize> = (0..scores.len())
                    .filter(|action| jotnar_agent::action_applies(*action, &task))
                    .collect();
                order.sort_by_key(|action| std::cmp::Reverse(scores[*action]));
                let top = order
                    .iter()
                    .take(top_k)
                    .map(|action| {
                        format!("{} ({})", jotnar_agent::ACTIONS[*action], scores[*action])
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                println!("    learned top-{top_k}: {top}");
                if let Some(best) = best {
                    let reachable = best
                        .iter()
                        .all(|action| order.iter().take(top_k).any(|top| top == action));
                    println!("    min-solver in learned top-{top_k}: {reachable}");
                }
            }
        }
    }
    Ok(())
}

fn fit_state_effect_model(
    out: &std::path::Path,
    corpus_mode: bool,
    generated: bool,
    depth: usize,
    epochs: usize,
    lr: i64,
) -> Result<ExitCode, String> {
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let (train, _) = if generated {
        fit_splits(true, true)
    } else {
        fit_splits(false, corpus_mode)
    };
    let base = Harness::default();
    let states = jotnar_agent::effects::reachable_states(&base, depth);
    let rows = state_effects(&train, &base, depth);
    let mut model = StateEffectModel::prior();
    let accuracy = model.fit(&rows, epochs, lr);
    model
        .save(&out.join("state-effect-model.weights.json"))
        .map_err(|e| e.to_string())?;
    std::fs::write(out.join("state-effect-model.jotnar"), &model.listing)
        .map_err(|e| e.to_string())?;
    println!(
        "state effect model: {} cases, {} states, {} transitions, {} actions, depth={}, epochs={}, lr={}, discriminating top-1 accuracy={accuracy:.3}",
        train.len(),
        states.len(),
        rows.len(),
        jotnar_agent::ACTIONS.len(),
        depth,
        epochs,
        lr
    );
    println!(
        "wrote {}",
        out.join("state-effect-model.weights.json").display()
    );
    println!("wrote {}", out.join("state-effect-model.jotnar").display());
    Ok(ExitCode::SUCCESS)
}

fn fit_cost_model(
    out: &std::path::Path,
    corpus_mode: bool,
    generated: bool,
    depth: usize,
    lambda: f64,
) -> Result<ExitCode, String> {
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let (train, _) = if generated {
        fit_splits(true, true)
    } else {
        fit_splits(false, corpus_mode)
    };
    let base = Harness::default();
    let rows = state_effects(&train, &base, depth);
    let mut model = CostModel::prior();
    let mae = model.fit(&rows, lambda);
    model
        .save(&out.join("cost-model.weights.json"))
        .map_err(|e| e.to_string())?;
    std::fs::write(out.join("cost-model.jotnar"), &model.listing).map_err(|e| e.to_string())?;
    println!(
        "cost model: {} cases, {} transitions, depth={}, lambda={}, mae={mae:.3}",
        train.len(),
        rows.len(),
        depth,
        lambda
    );
    println!("wrote {}", out.join("cost-model.weights.json").display());
    println!("wrote {}", out.join("cost-model.jotnar").display());
    Ok(ExitCode::SUCCESS)
}

#[allow(clippy::too_many_arguments)]
fn fit_joint_model(
    out: &std::path::Path,
    state_effect_weights: Option<&std::path::Path>,
    backbone: Option<&std::path::Path>,
    corpus_mode: bool,
    generated: bool,
    depth: usize,
    beam: usize,
) -> Result<ExitCode, String> {
    let state_path = state_effect_weights
        .ok_or("--state-effect-weights FILE is required for fit-joint-model")?;
    let backbone_path = backbone.ok_or("--backbone FILE is required for fit-joint-model")?;
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let (train, _) = if generated {
        fit_splits(true, true)
    } else {
        fit_splits(false, corpus_mode)
    };
    let base = Harness::default();
    let rows = state_effects(&train, &base, depth);
    let effect = StateEffectModel::load(state_path).map_err(|e| e.to_string())?;
    let bb = Backbone::load(backbone_path).map_err(|e| e.to_string())?;
    let joint = fit_joint(effect, bb, &rows, beam, depth);
    joint
        .save(&out.join("joint-model.weights.json"))
        .map_err(|e| e.to_string())?;
    println!(
        "joint model: {} cases, {} transitions, {} action prototypes, effect_weight={}, latent_weight={}, depth={}, beam={}",
        train.len(),
        rows.len(),
        joint.prototypes.iter().filter(|p| p.iter().any(|v| *v != 0)).count(),
        joint.effect_weight,
        joint.latent_weight,
        depth,
        beam
    );
    println!("wrote {}", out.join("joint-model.weights.json").display());
    Ok(ExitCode::SUCCESS)
}

#[allow(clippy::too_many_arguments)]
fn distill_qwen(
    out: &std::path::Path,
    teacher_path: &std::path::Path,
    dim: usize,
    inter: usize,
    seed: u64,
    scale: f64,
    ridge: f64,
    rounds: usize,
    max_loops: usize,
    end_to_end: bool,
    epochs: usize,
    lr: f64,
    corpus_mode: bool,
    generated: bool,
) -> Result<ExitCode, String> {
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let records = jotnar_agent::read_teacher_records(teacher_path)?;
    if records.is_empty() {
        return Err("teacher file has no records".to_string());
    }
    let samples: Vec<(Vec<i64>, Vec<f64>)> = records
        .iter()
        .map(|record| (record.task.clone(), record.latent.clone()))
        .collect();
    let mut sources: BTreeMap<String, usize> = BTreeMap::new();
    for record in &records {
        *sources.entry(record.source.clone()).or_default() += 1;
    }
    let named = TaskEncoder::prior();
    let mut backbone = Backbone::qwen_random(&named.w_enc, &named.b_enc, dim, inter, seed, scale);
    let before = samples
        .iter()
        .map(|(task, target)| {
            backbone
                .qwen
                .as_ref()
                .expect("qwen weights")
                .forward(task)
                .iter()
                .zip(target)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f64, f64::max)
        })
        .fold(0.0f64, f64::max);
    let initial = backbone.clone();
    let mut candidates: Vec<(String, Backbone, f64)> =
        vec![("zero-init".to_string(), initial.clone(), before)];
    let warm_err: Option<f64>;
    let mut e2e_err = None;
    if end_to_end {
        let warm = backbone
            .qwen
            .as_mut()
            .expect("qwen weights")
            .fit_readouts(&samples, ridge, rounds);
        warm_err = Some(warm);
        candidates.push(("readout".to_string(), backbone.clone(), warm));
        let e2e = backbone
            .qwen
            .as_mut()
            .expect("qwen weights")
            .fit_end_to_end(&samples, epochs, lr);
        e2e_err = Some(e2e);
        candidates.push(("end-to-end".to_string(), backbone.clone(), e2e));
    } else {
        let warm = backbone
            .qwen
            .as_mut()
            .expect("qwen weights")
            .fit_readouts(&samples, ridge, rounds);
        warm_err = Some(warm);
        candidates.push(("readout".to_string(), backbone.clone(), warm));
    }

    let mut best_mode = "zero-init".to_string();
    let mut best_err = before;
    let mut best = initial;
    if corpus_mode || generated {
        let (train, heldout) = if generated {
            fit_splits(true, true)
        } else {
            fit_splits(false, true)
        };
        let mut best_score = (0usize, 0usize, 0usize, 0usize);
        for (name, candidate, err) in &candidates {
            let encoder = LoopedTaskEncoder::prior(max_loops).with_backbone(candidate.clone());
            let report = run_round_with(
                &Harness::default(),
                &train,
                &out.join(format!("eval-{name}")),
                &encoder,
            );
            let selected = report.candidates.iter().find(|c| c.selected);
            let (tc, tw) = selected
                .map(|c| (c.train.correct(), c.train.wrong()))
                .unwrap_or((0, 0));
            let held = selected
                .map(|c| evaluate(&c.harness, &heldout, false))
                .unwrap_or_else(|| evaluate(&report.adopted_harness, &heldout, false));
            let score = (
                tc,
                usize::MAX - tw,
                held.correct(),
                usize::MAX - held.wrong(),
            );
            println!(
                "gate {name}: train {tc}/{} ({tw} wrong), held {}/{}",
                train.len(),
                held.correct(),
                heldout.len()
            );
            if score > best_score {
                best_score = score;
                best = candidate.clone();
                best_err = *err;
                best_mode = name.clone();
            }
        }
        let perfect = best_score.0 == train.len()
            && best_score.1 == usize::MAX
            && best_score.2 == heldout.len()
            && best_score.3 == usize::MAX;
        if !perfect {
            if let Some((name, candidate, err)) =
                candidates.iter().find(|(name, _, _)| name == "zero-init")
            {
                best = candidate.clone();
                best_err = *err;
                best_mode = name.clone();
            }
        }
    } else {
        for (name, candidate, err) in &candidates {
            if *err < best_err {
                best = candidate.clone();
                best_err = *err;
                best_mode = name.clone();
            }
        }
    }
    let backbone = best;
    let after = best_err;
    backbone
        .save(&out.join("backbone-qwen-distilled.weights.json"))
        .map_err(|e| e.to_string())?;
    let looped = LoopedTaskEncoder::prior(max_loops).with_backbone(backbone);
    looped
        .save(&out.join("looped-encoder-qwen-distilled.weights.json"))
        .map_err(|e| e.to_string())?;
    println!(
        "distill: {} teacher rows, sources={sources:?}, mode={}, dim={}, inter={}, seed={}, ridge={}, rounds={}, epochs={}, lr={}, teacher latent error: zero={before:.4}, warm={warm_err:?}, e2e={e2e_err:?}, selected={best_mode}, after={after:.4}",
        records.len(),
        if end_to_end { "end-to-end" } else { "readout" },
        dim,
        inter,
        seed,
        ridge,
        rounds,
        epochs,
        lr
    );
    println!(
        "wrote {}",
        out.join("backbone-qwen-distilled.weights.json").display()
    );
    println!(
        "wrote {}",
        out.join("looped-encoder-qwen-distilled.weights.json")
            .display()
    );
    Ok(ExitCode::SUCCESS)
}

fn fit_budget_predictor(
    out: &std::path::Path,
    corpus_mode: bool,
    generated: bool,
    max_loops: usize,
    epochs: usize,
    backbone: Option<&std::path::Path>,
) -> Result<ExitCode, String> {
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let (train, heldout) = fit_splits(generated, corpus_mode);
    let base = Harness::default();
    let backbone = match backbone {
        Some(path) => Some(Backbone::load(path).map_err(|e| e.to_string())?),
        None => None,
    };
    let looped = match &backbone {
        Some(backbone) => LoopedTaskEncoder::prior(max_loops).with_backbone(backbone.clone()),
        None => LoopedTaskEncoder::prior(max_loops),
    };
    let samples = jotnar_agent::calibration_dataset(&train, &base);
    let mut positives: BTreeMap<String, BTreeSet<usize>> = BTreeMap::new();
    for sample in &samples {
        if sample.delta > 0 {
            positives
                .entry(sample.case_id.clone())
                .or_default()
                .insert(sample.action);
        }
    }
    let views = jotnar_agent::case_views(&train, &evaluate(&base, &train, false));

    let mut data: Vec<(Vec<i64>, usize)> = Vec::new();
    let mut distribution = BTreeMap::<usize, usize>::new();
    for view in &views {
        let task = jotnar_agent::task_encoder::task_vector(view, &base);
        let label = if view.outcome != jotnar_agent::OutcomeKind::WrongRefusal {
            1
        } else {
            let pos = positives.get(&view.case_id);
            (1..=max_loops)
                .find(|loops| {
                    let Some(pos) = pos else {
                        return false;
                    };
                    looped
                        .scores(&task, *loops)
                        .map(|s| pos.contains(&argmax(&s)))
                        .unwrap_or(false)
                })
                .unwrap_or(max_loops)
        };
        *distribution.entry(label).or_default() += 1;
        data.push((task, label));
    }
    let mut predictor = BudgetPredictor::prior(max_loops);
    let correct = predictor.fit(&data, epochs, 1)?;
    predictor
        .save(&out.join("budget-predictor.weights.json"))
        .map_err(|e| e.to_string())?;
    println!(
        "budget fit: {} samples, epochs={epochs}, train accuracy {}/{}",
        data.len(),
        correct,
        data.len()
    );
    println!("budget distribution: {distribution:?}");

    if !heldout.is_empty() {
        let hviews = jotnar_agent::case_views(&heldout, &evaluate(&base, &heldout, false));
        let hsamples = jotnar_agent::calibration_dataset(&heldout, &base);
        let mut hpositives: BTreeMap<String, BTreeSet<usize>> = BTreeMap::new();
        for sample in &hsamples {
            if sample.delta > 0 {
                hpositives
                    .entry(sample.case_id.clone())
                    .or_default()
                    .insert(sample.action);
            }
        }
        let mut matches = 0usize;
        for view in &hviews {
            let task = jotnar_agent::task_encoder::task_vector(view, &base);
            let predicted = predictor.predict(&task);
            let expected = if view.outcome != jotnar_agent::OutcomeKind::WrongRefusal {
                1
            } else {
                let pos = hpositives.get(&view.case_id);
                (1..=max_loops)
                    .find(|loops| {
                        let Some(pos) = pos else {
                            return false;
                        };
                        looped
                            .scores(&task, *loops)
                            .map(|s| pos.contains(&argmax(&s)))
                            .unwrap_or(false)
                    })
                    .unwrap_or(max_loops)
            };
            if predicted == expected {
                matches += 1;
            }
            println!(
                "  heldout {}: predicted={predicted} expected={expected}",
                view.case_id
            );
        }
        println!("heldout budget matches: {matches}/{}", hviews.len());
    }
    let mut fused = LoopedTaskEncoder::prior(max_loops);
    if let Some(backbone) = backbone {
        fused = fused.with_backbone(backbone);
    }
    let fused = fused.with_budget_predictor(predictor);
    fused
        .save(&out.join("looped-encoder.weights.json"))
        .map_err(|e| e.to_string())?;
    println!(
        "wrote {}",
        out.join("budget-predictor.weights.json").display()
    );
    println!(
        "wrote {} (fused action + budget heads)",
        out.join("looped-encoder.weights.json").display()
    );
    Ok(ExitCode::SUCCESS)
}

fn fit_looped_encoder(
    out: &std::path::Path,
    corpus_mode: bool,
    generated: bool,
    max_loops: usize,
    encoder_weights: Option<&std::path::Path>,
) -> Result<ExitCode, String> {
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let (train, heldout) = fit_splits(generated, corpus_mode);
    let base = Harness::default();
    let linear = match encoder_weights {
        Some(path) => TaskEncoder::load(path).map_err(|e| e.to_string())?,
        None => TaskEncoder::prior(),
    };
    let samples = jotnar_agent::calibration_dataset(&train, &base);
    let mut positives: BTreeMap<String, BTreeSet<usize>> = BTreeMap::new();
    for sample in &samples {
        if sample.delta > 0 {
            positives
                .entry(sample.case_id.clone())
                .or_default()
                .insert(sample.action);
        }
    }
    let views = jotnar_agent::case_views(&train, &evaluate(&base, &train, false));

    println!(
        "fit-loop: {} calibration cases, {} held-out; max_loops={max_loops}",
        train.len(),
        heldout.len()
    );
    println!(
        "{:<6} {:<7} {:<9} {:<6} {:<8} {:<8}",
        "diag", "cross", "families", "cost", "calib", "heldout"
    );
    let mut families: BTreeMap<String, Vec<&jotnar_agent::CaseView>> = BTreeMap::new();
    for view in &views {
        families.entry(view.family.clone()).or_default().push(view);
    }

    let mut best: Option<(LoopFitKey, BTreeMap<String, usize>)> = None;
    for diag in 2..=6 {
        for cross in -4..=1 {
            let encoder = LoopedTaskEncoder::with_recurrence(&linear, max_loops, diag, cross);
            let mut budgets: BTreeMap<String, usize> = BTreeMap::new();
            let mut total_cost = 0usize;
            let mut covered_families = 0usize;
            let mut valid = true;

            for (family, fam_views) in &families {
                let required: Vec<&jotnar_agent::CaseView> = fam_views
                    .iter()
                    .filter(|v| positives.contains_key(&v.case_id))
                    .copied()
                    .collect();
                if required.is_empty() {
                    continue;
                }
                let covered_at = |loops: usize| -> bool {
                    required.iter().all(|v| {
                        let task = jotnar_agent::task_encoder::task_vector(v, &base);
                        let pos = positives.get(&v.case_id).expect("positive set");
                        encoder
                            .scores(&task, loops)
                            .map(|s| pos.contains(&argmax(&s)))
                            .unwrap_or(false)
                    })
                };
                // Preserve the crossover: a family must fail at one loop.
                if covered_at(1) {
                    valid = false;
                    break;
                }
                let Some(budget) = (2..=max_loops).find(|loops| covered_at(*loops)) else {
                    valid = false;
                    break;
                };
                budgets.insert(family.clone(), budget);
                total_cost += budget;
                covered_families += 1;
            }
            if !valid {
                continue;
            }

            let encoder = encoder.with_budgets(budgets.clone());
            let proposals = encoder.propose_cases(&views, &base, &[]);
            let calib_correct = proposals
                .as_ref()
                .and_then(|p| p.first())
                .map(|m| evaluate(&m.harness, &train, false).correct())
                .unwrap_or(0);
            let held_correct = if heldout.is_empty() {
                0
            } else {
                proposals
                    .as_ref()
                    .and_then(|p| p.first())
                    .map(|m| evaluate(&m.harness, &heldout, false).correct())
                    .unwrap_or(0)
            };
            println!(
                "{diag:<6} {cross:<7} {covered_families:<9} {total_cost:<6} {calib_correct:<8} {held_correct:<8}"
            );
            let dist = (diag - 4).abs() + (cross + 2).abs();
            let key = (
                calib_correct,
                covered_families,
                -(total_cost as i64),
                -dist,
                diag,
                cross,
            );
            if best.as_ref().map(|(b, _)| key > *b).unwrap_or(true) {
                best = Some((key, budgets));
            }
        }
    }
    let ((calib, covered, neg_cost, neg_dist, diag, cross), budgets) =
        best.ok_or("no grid point preserved the one-loop failure and covered every family")?;
    let encoder = LoopedTaskEncoder::with_recurrence(&linear, max_loops, diag, cross)
        .with_budgets(budgets.clone());
    encoder
        .save(&out.join("looped-encoder.weights.json"))
        .map_err(|e| e.to_string())?;
    println!(
        "\nchosen: diag={diag} cross={cross} calib={calib} families={covered} cost={} prior_distance={} budgets={budgets:?}",
        -neg_cost,
        -neg_dist
    );
    if !heldout.is_empty() {
        let proposals = encoder.propose_cases(&views, &base, &[]);
        let held = proposals
            .as_ref()
            .and_then(|p| p.first())
            .map(|m| evaluate(&m.harness, &heldout, false))
            .map(|r| (r.correct(), r.outcomes.len()))
            .unwrap_or((0, heldout.len()));
        println!("heldout: {}/{}", held.0, held.1);
    }
    println!(
        "wrote {}",
        out.join("looped-encoder.weights.json").display()
    );
    Ok(ExitCode::SUCCESS)
}

fn fit_encoder(
    out: &std::path::Path,
    corpus_mode: bool,
    generated: bool,
    epochs: usize,
    lr: i64,
) -> Result<ExitCode, String> {
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let train = fit_splits(generated, corpus_mode).0;
    let base = Harness::default();
    let samples = jotnar_agent::calibration_dataset(&train, &base);
    let mut encoder = TaskEncoder::prior();
    let report = encoder.fit(&samples, epochs, lr)?;
    encoder
        .save(&out.join("task-encoder.weights.json"))
        .map_err(|e| e.to_string())?;
    println!(
        "encoder fit: {} samples ({} positive, {} negative), epochs={}, lr={}",
        report.samples, report.positive, report.negative, report.epochs, lr
    );
    println!(
        "  positive action top-1: {}/{}; negative action avoided: {}/{}",
        report.positive_top1, report.positive, report.negative_safe, report.negative
    );
    println!("wrote {}", out.join("task-encoder.weights.json").display());
    Ok(ExitCode::SUCCESS)
}

fn fit_recurrent(
    out: &std::path::Path,
    deep_train: bool,
    efficient: bool,
) -> Result<ExitCode, String> {
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let base = Harness::default();
    let calibration_cases = if deep_train {
        jotnar_agent::deep_calibration_cases()
    } else {
        jotnar_agent::calibration_cases()
    };
    let calib = run_round(&base, &calibration_cases, &out.join("calib-base"));
    let full = run_round(&base, &jotnar_agent::cases(), &out.join("full-base"));
    let f_calib = jotnar_agent::proposer::features(&base, &calib.base, &[]);
    let f_full = jotnar_agent::proposer::features(&base, &full.base, &[]);
    let ref_idx = jotnar_agent::ACTIONS
        .iter()
        .position(|a| *a == "refinements + 1")
        .ok_or("refinements action missing")?;

    let delta = |label: &str| {
        calib
            .candidates
            .iter()
            .find(|c| c.index != 0 && c.label == label)
            .map(|c| c.train.correct() as i64 - calib.base.correct() as i64)
            .unwrap_or(0)
    };
    let need_reduce_sum = delta("add compose[reduce_sum]") > 0;
    let need_deep = delta("depth + 1") > 0 || delta("refinements + 1") > 0;

    println!(
        "calibration: {} cases; deep_train={deep_train} efficient={efficient} need_reduce_sum={need_reduce_sum} need_deep={need_deep}",
        calibration_cases.len()
    );
    println!(
        "{:<6} {:<7} {:<8} {:<7} {:<5} {:<8} {:<8} {:<8}",
        "diag", "cross", "margin", "loops", "ret", "1loop", "2loop", "4loop"
    );
    let mut best: Option<(i64, i64, i64, i64, i64, i64, i64)> = None;
    for diag in 2..=6 {
        for cross in -4..=1 {
            let mut margin_total = 0i64;
            let mut loop_cost = if need_reduce_sum || need_deep { 8 } else { 0 };
            let mut retained = !(need_reduce_sum || need_deep);
            let mut marks = [false; 3];
            for (k, loops) in [1usize, 2, 4].into_iter().enumerate() {
                let p = LoopedProposer::with_recurrence(loops, diag, cross);
                let s_calib = p.scores(&f_calib).unwrap_or_default();
                margin_total += top2_margin(&s_calib);
                let sum_ok = !need_reduce_sum || top2_has(&s_calib, 0);
                let deep_ok = !need_deep || top2_has(&s_calib, 1) || top2_has(&s_calib, 5);
                if sum_ok && deep_ok {
                    loop_cost = loop_cost.min(loops as i64);
                }
                if loops == 4 {
                    retained = sum_ok && deep_ok;
                }
                let s = p.scores(&f_full).unwrap_or_default();
                marks[k] = top2_has(&s, ref_idx);
            }
            let dist = (diag - 4).abs() + (cross + 2).abs();
            println!(
                "{diag:<6} {cross:<7} {margin_total:<8} {loop_cost:<7} {:<5} {:<8} {:<8} {:<8}",
                retained, marks[0], marks[1], marks[2]
            );
            let covered = i64::from(loop_cost < 8);
            let key = if efficient {
                (
                    covered,
                    i64::from(retained),
                    -loop_cost,
                    margin_total,
                    -dist,
                    diag,
                    cross,
                )
            } else {
                (0, 0, 0, margin_total, -dist, diag, cross)
            };
            if best.as_ref().map(|b| key > *b).unwrap_or(true) {
                best = Some(key);
            }
        }
    }
    let (covered, retained, neg_cost, margin_total, neg_dist, diag, cross) =
        best.ok_or("no grid points")?;
    let loop_cost = -neg_cost;
    let chosen = LoopedProposer::with_recurrence(2, diag, cross);
    chosen
        .save(&out.join("looped.weights.json"))
        .map_err(|e| e.to_string())?;
    println!(
        "\nchosen: diag={diag} cross={cross} covered={covered} retained={retained} loop_cost={loop_cost} margin={margin_total} prior_distance={}",
        -neg_dist
    );
    for loops in [1usize, 2, 4] {
        let p = LoopedProposer::with_recurrence(loops, diag, cross);
        let s = p.scores(&f_full).unwrap_or_default();
        let calib_scores = p.scores(&f_calib).unwrap_or_default();
        let sum_ok = !need_reduce_sum || top2_has(&calib_scores, 0);
        let deep_ok = !need_deep || top2_has(&calib_scores, 1) || top2_has(&calib_scores, 5);
        println!(
            "  loops={loops} calib_covered={} heldout_refinements={} scores={s:?}",
            sum_ok && deep_ok,
            top2_has(&s, ref_idx)
        );
    }
    println!("wrote {}", out.join("looped.weights.json").display());
    Ok(ExitCode::SUCCESS)
}

fn top2_margin(scores: &[i64]) -> i64 {
    if scores.len() < 3 {
        return 0;
    }
    let mut v = scores.to_vec();
    v.sort_unstable_by(|a, b| b.cmp(a));
    v[1] - v[2]
}

fn top2_has(scores: &[i64], idx: usize) -> bool {
    let mut ranked: Vec<usize> = (0..scores.len()).collect();
    ranked.sort_by_key(|i| std::cmp::Reverse(scores[*i]));
    ranked[..2.min(ranked.len())].contains(&idx)
}

fn print_corpus_report(report: &RoundReport, held: &EvalReport, round: usize, rounds: usize) {
    println!("\n== corpus round {round}/{rounds} ==");
    println!(
        "{:<14} {:>16} {:>16} {:>16}",
        "family", "base", "adopted", "heldout"
    );
    for family in [
        "elementwise",
        "reduction",
        "matrix",
        "select",
        "refine",
        "control",
    ] {
        println!(
            "{:<14} {:>16} {:>16} {:>16}",
            family,
            fmt_family(&report.base, family),
            fmt_family(&report.adopted, family),
            fmt_family(held, family)
        );
    }
    println!(
        "selection: {}/{} correct, {} wrong -> {}/{} correct, {} wrong",
        report.base.correct(),
        report.base.outcomes.len(),
        report.base.wrong(),
        report.adopted.correct(),
        report.adopted.outcomes.len(),
        report.adopted.wrong()
    );
    println!(
        "heldout:   {}/{} correct, {} wrong",
        held.correct(),
        held.outcomes.len(),
        held.wrong()
    );
    println!(
        "selected:  {} (blind index {:?}); adopted={}",
        report.chosen_label, report.chosen_index, report.adopted_flag
    );
    if !report.edits.is_empty() {
        println!("edits:     {}", report.edits.join("; "));
    }
}

fn fmt_family(report: &EvalReport, family: &str) -> String {
    let total = report
        .outcomes
        .iter()
        .filter(|o| o.family == family)
        .count();
    let correct = report.correct_in(family);
    format!("{correct}/{total} ok, {} wrong", total - correct)
}

fn print_report(report: &jotnar_agent::RoundReport, round: usize, rounds: usize) {
    println!("\n== round {round}/{rounds} ==");
    println!("{:<24} {:<12} {:<12}", "case", "incumbent", "adopted");
    for outcome in &report.base.outcomes {
        let adopted = report
            .adopted
            .outcomes
            .iter()
            .find(|o| o.case == outcome.case)
            .map(|o| o.kind.as_str())
            .unwrap_or("missing");
        println!(
            "{:<24} {:<12} {:<12}",
            outcome.case,
            outcome.kind.as_str(),
            adopted
        );
    }
    println!(
        "\nincumbent: {}/{} correct, {} wrong, {} steps",
        report.base.correct(),
        report.base.outcomes.len(),
        report.base.wrong(),
        report.base.total_steps
    );
    println!(
        "adopted:   {}/{} correct, {} wrong, {} steps",
        report.adopted.correct(),
        report.adopted.outcomes.len(),
        report.adopted.wrong(),
        report.adopted.total_steps
    );
    println!(
        "held-out:  {}/{} -> {}/{}",
        report.base_heldout.correct(),
        report.base_heldout.outcomes.len(),
        report.adopted_heldout.correct(),
        report.adopted_heldout.outcomes.len()
    );
    println!(
        "selected:  {} (blind index {:?})",
        report.chosen_label, report.chosen_index
    );
    println!("adopted:   {}", report.adopted_flag);
    if !report.edits.is_empty() {
        println!("edits:     {}", report.edits.join("; "));
    }
    if let Some(exp) = &report.experience {
        println!("lesson:    {}", exp.lesson);
        println!("evidence:  {}", exp.evidence);
    }
    println!("artifacts: {}", report.out_dir.display());
}
