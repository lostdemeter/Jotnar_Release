use std::collections::HashMap;

use crate::error::{at, Error, Result};
use crate::value::Kind;

pub const UNKNOWN: &str = "UNKNOWN";

/// A mnemonic's layout signature: one pattern per input, per output.
///
/// Pattern grammar (same as the Python frontend):
/// * `*`      — matches anything, binds nothing;
/// * `$VAR`   — unifies a whole layout string;
/// * `$VAR^T` — the transpose-derived form;
/// * literal  — `KIND:GEOM`, must match exactly.
#[derive(Clone, Debug, Default)]
pub struct LayoutSig {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

impl LayoutSig {
    pub fn new(inputs: &[&str], outputs: &[&str]) -> Self {
        LayoutSig {
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            outputs: outputs.iter().map(|s| s.to_string()).collect(),
        }
    }
}

/// Unify one layout pattern against an actual layout.
pub fn resolve(pat: &str, actual: &str, bindings: &mut HashMap<String, String>) -> Result<String> {
    if pat == "*" {
        return Ok(actual.to_string());
    }
    if actual == UNKNOWN {
        return Ok(if pat.starts_with('$') {
            UNKNOWN.to_string()
        } else {
            pat.to_string()
        });
    }
    if let Some(stripped) = pat.strip_prefix('$') {
        let (base, derived) = match stripped.strip_suffix("^T") {
            Some(b) => (format!("${b}"), true),
            None => (format!("${stripped}"), false),
        };
        let got = match bindings.get(&base) {
            Some(bound) if bound != UNKNOWN && bound != actual => {
                return Err(Error::asm(format!(
                    "layout conflict: {base} is {bound}, got {actual}"
                )));
            }
            Some(bound) if bound != UNKNOWN => bound.clone(),
            _ => actual.to_string(),
        };
        bindings.insert(base, got.clone());
        if got == UNKNOWN {
            return Ok(UNKNOWN.to_string());
        }
        return Ok(if derived { format!("{got}^T") } else { got });
    }
    // `KIND:*` is the documented "kind matches, geometry any" pattern (e.g.
    // `I:*` is an integer stream of any shape). Only a bare `*` is a fully
    // unconstrained wildcard.
    if let (Some((pk, pg)), Some((ak, _ag))) = (pat.split_once(':'), actual.split_once(':')) {
        if pg == "*" {
            if pk != ak {
                return Err(Error::asm(format!(
                    "layout kind mismatch: want {pat}, got {actual}"
                )));
            }
            return Ok(actual.to_string());
        }
    }
    if pat != actual {
        return Err(Error::asm(format!(
            "layout mismatch: want {pat}, got {actual}"
        )));
    }
    Ok(pat.to_string())
}

/// Unify one instruction's stream layouts.
///
/// Returns `(name, layout)` pairs for the outputs, in declaration order.
pub fn check_layouts(
    outs: &[String],
    mn: &str,
    args: &[String],
    line: &str,
    sig: Option<&LayoutSig>,
    layouts: &HashMap<String, String>,
) -> Result<Vec<(String, String)>> {
    let fallback_in: Vec<String>;
    let fallback_out: Vec<String>;
    let (in_pats, out_pats): (&[String], &[String]) = match sig {
        Some(s) => (&s.inputs, &s.outputs),
        None => {
            fallback_in = vec!["*".to_string(); args.len()];
            fallback_out = vec!["*".to_string(); outs.len()];
            (&fallback_in, &fallback_out)
        }
    };
    let mut bindings = HashMap::new();
    let mut attempt = || -> Result<Vec<(String, String)>> {
        for (a, pat) in args.iter().zip(in_pats.iter()) {
            let actual = layouts.get(a).cloned().unwrap_or_else(|| {
                if is_literal(a) {
                    "F:SCALAR".to_string()
                } else {
                    UNKNOWN.to_string()
                }
            });
            resolve(pat, &actual, &mut bindings)?;
        }
        let mut out_lays = Vec::with_capacity(outs.len());
        for pat in out_pats {
            let lay = if pat == "*" {
                UNKNOWN.to_string()
            } else if let Some(stripped) = pat.strip_prefix('$') {
                let (base, derived) = match stripped.strip_suffix("^T") {
                    Some(b) => (format!("${b}"), true),
                    None => (format!("${stripped}"), false),
                };
                match bindings
                    .get(&base)
                    .cloned()
                    .unwrap_or_else(|| UNKNOWN.into())
                {
                    b if b == UNKNOWN => UNKNOWN.to_string(),
                    b if derived => format!("{b}^T"),
                    b => b,
                }
            } else {
                pat.clone()
            };
            out_lays.push(lay);
        }
        Ok(outs.iter().cloned().zip(out_lays).collect())
    };
    attempt().map_err(|e| match e {
        Error::Asm(msg) => at(line, mn, msg),
        other => other,
    })
}

/// `KIND` prefix of a layout, if it has one.
pub fn layout_kind(layout: &str) -> Option<&str> {
    layout.split_once(':').map(|(kind, _)| kind)
}

/// The layout kind required by a declared stream, compared against a value.
pub fn kind_matches(layout: &str, value_kind: Kind) -> bool {
    match layout_kind(layout) {
        Some(want) => want == value_kind.as_str(),
        None => true,
    }
}

/// Python's `_is_literal`: a bare number in argument position.
pub fn is_literal(a: &str) -> bool {
    a.trim().parse::<f64>().is_ok()
}
