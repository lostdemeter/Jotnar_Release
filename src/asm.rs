//! Frontend: IMPORT splicing, DEF/CALL macro expansion, REPEAT unrolling,
//! parsing, registry binding, and the static layout verifier.
//!
//! Behaviour follows `chain/asm.py`; diagnostics name the op and source
//! line (`file:line`, plus `via CALL@site` for expanded lines).

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use regex::Regex;

use crate::error::{Error, Result};
use crate::ir::{BoundInstr, BoundProgram, Config, InDecl, Instr, Program};
use crate::layout::{check_layouts, is_literal, UNKNOWN};
use crate::ops;
use crate::value::Kind;

#[derive(Clone, Debug)]
struct Entry {
    text: String,
    origin: String,
    pre_stripped: bool,
}

#[derive(Clone, Debug)]
struct Def {
    fins: Vec<String>,
    fouts: Vec<String>,
    fin_lay: BTreeMap<String, Option<String>>,
    fout_lay: BTreeMap<String, Option<String>>,
    body: Vec<Entry>,
    origin: String,
}

type Defs = BTreeMap<String, Def>;

/// Search/expansion options.
#[derive(Clone, Debug, Default)]
pub struct AsmOptions {
    pub basedir: PathBuf,
    pub stdlib: Option<PathBuf>,
    pub origin: Option<String>,
}

impl AsmOptions {
    pub fn new(basedir: impl Into<PathBuf>) -> Self {
        AsmOptions {
            basedir: basedir.into(),
            stdlib: None,
            origin: None,
        }
    }
}

fn head_of(line: &str) -> String {
    line.split_whitespace()
        .next()
        .unwrap_or("")
        .to_ascii_uppercase()
}

fn split_head(line: &str) -> (String, Option<&str>) {
    let trimmed = line.trim_start();
    match trimmed.find(char::is_whitespace) {
        Some(idx) => (
            trimmed[..idx].to_ascii_uppercase(),
            Some(trimmed[idx..].trim_start()),
        ),
        None => (trimmed.to_ascii_uppercase(), None),
    }
}

fn strip_comment(line: &str) -> &str {
    line.split('#').next().unwrap_or("").trim()
}

fn def_re() -> &'static Regex {
    static R: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"DEF\s+(\w+)\s*\((.*)\)\s*->\s*\((.*)\)\s*$").expect("valid DEF regex")
    })
}

fn call_re() -> &'static Regex {
    static R: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    R.get_or_init(|| Regex::new(r"(\w+)\s*\((.*)\)\s*$").expect("valid CALL regex"))
}

fn call_assign_re() -> &'static Regex {
    static R: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"(?i)^(CALL)\s+(\w+)\s*\((.*)\)\s*$").expect("valid CALL assignment regex")
    })
}

fn ident_re() -> &'static Regex {
    static R: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    R.get_or_init(|| Regex::new(r"[A-Za-z_]\w*").expect("valid identifier regex"))
}

fn line_of(entry: &Entry) -> String {
    if entry.pre_stripped {
        entry.text.clone()
    } else {
        strip_comment(&entry.text).to_string()
    }
}

fn default_stdlib(basedir: &Path) -> Option<PathBuf> {
    // Python resolves `chain/../stdlib`. For the standalone crate, search the
    // listing directory, then its ancestors (so `programs/x.asm` finds the
    // repo-level `stdlib`), then the current directory.
    for ancestor in basedir.ancestors() {
        let cand = ancestor.join("stdlib");
        if cand.is_dir() {
            return Some(cand);
        }
    }
    let cand = PathBuf::from("stdlib");
    cand.is_dir().then_some(cand)
}

fn resolve_import(target: &str, basedir: &Path, stdlib: Option<&Path>) -> PathBuf {
    if let Some(stdlib) = stdlib {
        if !target.contains('/') && !target.contains('\\') && !target.starts_with('.') {
            let cand = stdlib.join(target);
            if cand.is_file() {
                return cand;
            }
        }
    }
    basedir.join(target)
}

/// Expand `text` into flat lines, collecting DEFs.
fn expand_into(
    text: &str,
    basedir: &Path,
    origin: &str,
    defs: &mut Defs,
    stack: &[PathBuf],
    stdlib: Option<&Path>,
    out: &mut Vec<Entry>,
) -> Result<()> {
    let raw: Vec<(String, String)> = text
        .lines()
        .enumerate()
        .map(|(i, l)| (l.to_string(), format!("{origin}:{}", i + 1)))
        .collect();

    // Pre-scan this file's IN declarations so CALL-site checks see caller
    // layouts (imported INs stay UNKNOWN, like the Python frontend).
    let mut in_lay0: HashMap<String, Option<String>> = HashMap::new();
    for (line, _) in &raw {
        let code = strip_comment(line);
        if code.is_empty() {
            continue;
        }
        let (head, rest) = split_head(code);
        if head == "IN" {
            if let Some(rest) = rest {
                if let Some(idx) = rest.to_ascii_uppercase().find(" AS ") {
                    let names = &rest[..idx];
                    let lay = rest[idx + 4..].trim();
                    let lay = if lay.is_empty() {
                        None
                    } else {
                        Some(lay.to_string())
                    };
                    for nm in names.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                        in_lay0.insert(nm.to_string(), lay.clone());
                    }
                }
            }
        }
    }

    let mut i = 0usize;
    let mut counter = 0usize;
    while i < raw.len() {
        let (line, org) = raw[i].clone();
        i += 1;
        let code = strip_comment(&line);
        if code.is_empty() {
            continue;
        }
        let (head, rest) = split_head(code);
        match head.as_str() {
            "IMPORT" => {
                let target = rest
                    .unwrap_or("")
                    .trim()
                    .trim_matches(|c| c == '"' || c == '\'');
                if target.is_empty() {
                    return Err(Error::asm(format!("{org}: IMPORT needs a path")));
                }
                let path = resolve_import(target, basedir, stdlib);
                let real = std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
                if stack.contains(&real) {
                    return Err(Error::asm(format!(
                        "{org}: IMPORT cycle ({})",
                        real.display()
                    )));
                }
                let sub = std::fs::read_to_string(&path).map_err(|e| {
                    Error::asm(format!("{org}: IMPORT cannot read {}: {e}", path.display()))
                })?;
                let sub_dir = path.parent().unwrap_or(Path::new("."));
                let mut new_stack = stack.to_vec();
                new_stack.push(real);
                expand_into(
                    &sub,
                    sub_dir,
                    &path.display().to_string(),
                    defs,
                    &new_stack,
                    stdlib,
                    out,
                )?;
            }
            "DEF" => {
                let re = def_re();
                let caps = re.captures(code).ok_or_else(|| {
                    Error::asm(format!(
                        "{org}: bad DEF syntax (want DEF name(a, b) -> (c))"
                    ))
                })?;
                let name = caps[1].to_string();
                let fins = parse_formals(&caps[2], "formal", &org)?;
                let fouts = parse_formals(&caps[3], "return", &org)?;
                if defs.contains_key(&name) {
                    return Err(Error::asm(format!("{org}: duplicate DEF '{name}'")));
                }
                let mut body = Vec::new();
                let mut ended = false;
                while i < raw.len() {
                    let (bline, borg) = raw[i].clone();
                    i += 1;
                    let bcode = strip_comment(&bline);
                    if bcode.is_empty() {
                        continue;
                    }
                    let (btok, _) = split_head(bcode);
                    if btok == "END" {
                        ended = true;
                        break;
                    }
                    if btok == "DEF" {
                        return Err(Error::asm(format!("{borg}: nested DEF not allowed")));
                    }
                    if matches!(
                        btok.as_str(),
                        "CONFIG" | "STATE" | "IN" | "IMPORT" | "RANGE"
                    ) {
                        return Err(Error::asm(format!(
                            "{borg}: {btok} not allowed inside DEF (program-global only)"
                        )));
                    }
                    body.push(Entry {
                        text: bline,
                        origin: borg,
                        pre_stripped: false,
                    });
                }
                if !ended {
                    return Err(Error::asm(format!("{org}: DEF '{name}' missing END")));
                }
                let fin_lay = fins.iter().cloned().collect();
                let fout_lay = fouts.iter().cloned().collect();
                defs.insert(
                    name,
                    Def {
                        fins: fins.into_iter().map(|(n, _)| n).collect(),
                        fouts: fouts.into_iter().map(|(n, _)| n).collect(),
                        fin_lay,
                        fout_lay,
                        body,
                        origin: org,
                    },
                );
            }
            "CALL" => {
                return Err(Error::asm(format!(
                    "{org}: CALL needs outputs (want x = CALL name(...))"
                )));
            }
            _ => {
                if let Some(eq) = code.find('=') {
                    let left = &code[..eq];
                    let right = code[eq + 1..].trim();
                    let (rh, rrest) = split_head(right);
                    if rh == "CALL" {
                        let rest = rrest.ok_or_else(|| {
                            Error::asm(format!(
                                "{org}: CALL needs a callee (want x = CALL name(...))"
                            ))
                        })?;
                        let outs: Vec<String> = left
                            .split(',')
                            .map(str::trim)
                            .filter(|s| !s.is_empty())
                            .map(str::to_string)
                            .collect();
                        expand_call(rest, &outs, &org, defs, &mut counter, &in_lay0, &[], out)?;
                        continue;
                    }
                }
                out.push(Entry {
                    text: line,
                    origin: org,
                    pre_stripped: false,
                });
            }
        }
    }
    Ok(())
}

fn parse_formals(spec: &str, what: &str, org: &str) -> Result<Vec<(String, Option<String>)>> {
    let mut out = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let up = part.to_ascii_uppercase();
        let (nm, lay) = if let Some(idx) = up.find(" AS ") {
            let nm = part[..idx].trim();
            let lay = part[idx + 4..].trim();
            (
                nm,
                if lay.is_empty() {
                    None
                } else {
                    Some(lay.to_string())
                },
            )
        } else {
            (part, None)
        };
        if nm.is_empty()
            || !nm
                .chars()
                .next()
                .map(|c| c.is_ascii_alphabetic())
                .unwrap_or(false)
            || !nm.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            return Err(Error::asm(format!("{org}: bad {what} name '{nm}'")));
        }
        out.push((nm.to_string(), lay));
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn expand_call(
    mn_args: &str,
    outs: &[String],
    site: &str,
    defs: &Defs,
    counter: &mut usize,
    in_lay0: &HashMap<String, Option<String>>,
    dstack: &[String],
    out: &mut Vec<Entry>,
) -> Result<()> {
    let re = call_re();
    let caps = re
        .captures(mn_args)
        .ok_or_else(|| Error::asm(format!("{site}: bad CALL syntax (want name(a, b))")))?;
    let name = caps[1].to_string();
    let actuals: Vec<String> = caps[2]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    let def = defs
        .get(&name)
        .ok_or_else(|| Error::asm(format!("{site}: CALL of unknown DEF '{name}'")))?;
    if dstack.iter().any(|n| n == &name) {
        return Err(Error::asm(format!(
            "{site}: recursive CALL '{name}' (def stack {dstack:?})"
        )));
    }
    if actuals.len() != def.fins.len() {
        return Err(Error::asm(format!(
            "{site}: CALL {name} wants {} args, got {}",
            def.fins.len(),
            actuals.len()
        )));
    }
    if outs.len() != def.fouts.len() {
        return Err(Error::asm(format!(
            "{site}: CALL {name} produces {}, got {} outs",
            def.fouts.len(),
            outs.len()
        )));
    }
    for (fml, act) in def.fins.iter().zip(actuals.iter()) {
        let fl = def.fin_lay.get(fml).and_then(|v| v.clone());
        if let Some(fl) = &fl {
            if fl.starts_with('$') {
                return Err(Error::asm(format!(
                    "{site}: DEF '{name}' formal '{fml}' must not use $VAR layouts"
                )));
            }
        }
        let al = in_lay0.get(act).and_then(|v| v.clone());
        if let (Some(fl), Some(al)) = (fl, al) {
            if fl != al {
                return Err(Error::asm(format!(
                    "{site}: CALL {name}('{act}' is {al}) violates formal '{fml}' wants {fl} (DEF at {})",
                    def.origin
                )));
            }
        }
    }

    *counter += 1;
    let pre = format!("{name}#{}.", *counter);
    let mut mapping: HashMap<String, String> = HashMap::new();
    for (f, a) in def.fins.iter().zip(actuals.iter()) {
        mapping.insert(f.clone(), a.clone());
    }
    for (f, a) in def.fouts.iter().zip(outs.iter()) {
        mapping.insert(f.clone(), a.clone());
    }

    let mut next_stack = dstack.to_vec();
    next_stack.push(name.clone());

    let ident_re = ident_re();
    for entry in &def.body {
        let code = strip_comment(&entry.text);
        if code.is_empty() {
            continue;
        }
        let (tok, tok_rest) = split_head(code);
        if tok == "CALL" {
            let rest = tok_rest.unwrap_or("");
            expand_call(
                rest,
                &[],
                &format!("{} via CALL@{site}", entry.origin),
                defs,
                counter,
                in_lay0,
                &next_stack,
                out,
            )?;
            continue;
        }

        if let Some(eq) = code.find('=') {
            let left = &code[..eq];
            let right = code[eq + 1..].trim();
            let lo: Vec<String> = left
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(|o| {
                    mapping
                        .get(o)
                        .cloned()
                        .unwrap_or_else(|| format!("{pre}{o}"))
                })
                .collect();

            let call_re = call_assign_re();
            if let Some(caps) = call_re.captures(right) {
                let inner = format!("{}({})", &caps[2], &caps[3]);
                expand_call(
                    &inner,
                    &lo,
                    &format!("{} via CALL@{site}", entry.origin),
                    defs,
                    counter,
                    in_lay0,
                    &next_stack,
                    out,
                )?;
                continue;
            }

            let (head, mut args) = if let Some(open) = right.find('(') {
                let head = right[..open].trim().to_string();
                let inside = &right[open + 1..];
                let inside = inside.strip_suffix(')').unwrap_or(inside);
                (
                    head,
                    inside
                        .split(',')
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(str::to_string)
                        .collect::<Vec<_>>(),
                )
            } else {
                let (head, rest) = split_head(right);
                let args = rest
                    .map(|r| {
                        r.split(',')
                            .map(str::trim)
                            .filter(|s| !s.is_empty())
                            .map(str::to_string)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                (head, args)
            };

            if head.eq_ignore_ascii_case("CALL") {
                // Assignment-form inner call: `x = CALL name(args)`.
                let inner = args.join(",");
                expand_call(
                    &inner,
                    &lo,
                    &format!("{} via CALL@{site}", entry.origin),
                    defs,
                    counter,
                    in_lay0,
                    &next_stack,
                    out,
                )?;
                continue;
            }

            for a in args.iter_mut() {
                if !is_literal(a) {
                    *a = mapping
                        .get(a)
                        .cloned()
                        .unwrap_or_else(|| format!("{pre}{a}"));
                }
            }
            let new_code = format!("{} = {}({})", lo.join(", "), head, args.join(", "));
            out.push(Entry {
                text: new_code,
                origin: format!("{} via CALL@{site}", entry.origin),
                pre_stripped: true,
            });
        } else {
            let rewritten = ident_re.replace_all(code, |caps: &regex::Captures| {
                let w = &caps[0];
                mapping
                    .get(w)
                    .cloned()
                    .unwrap_or_else(|| format!("{pre}{w}"))
            });
            out.push(Entry {
                text: rewritten.into_owned(),
                origin: format!("{} via CALL@{site}", entry.origin),
                pre_stripped: true,
            });
        }
    }
    Ok(())
}

fn collect_repeat_body(flat: &[Entry], mut i: usize) -> Result<(Vec<Entry>, usize)> {
    let mut depth = 1;
    let mut body = Vec::new();
    while i < flat.len() {
        let line = line_of(&flat[i]);
        let head = head_of(&line);
        if head == "REPEAT" {
            depth += 1;
        } else if head == "END" {
            depth -= 1;
            if depth == 0 {
                return Ok((body, i + 1));
            }
        }
        body.push(flat[i].clone());
        i += 1;
    }
    Err(Error::asm("REPEAT without END"))
}

fn has_break(body: &[Entry]) -> bool {
    let mut depth = 0i32;
    for entry in body {
        let line = line_of(entry);
        let head = head_of(&line);
        if head == "REPEAT" {
            depth += 1;
        } else if head == "END" {
            depth -= 1;
        } else if head == "BREAK" && depth == 0 {
            return true;
        }
    }
    false
}

fn expand_repeats(flat: &[Entry]) -> Result<Vec<Entry>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < flat.len() {
        let line = line_of(&flat[i]);
        let (head, rest) = split_head(&line);
        if head == "REPEAT" {
            let rest = rest.unwrap_or("");
            if rest.split_whitespace().count() != 1 {
                return Err(Error::asm(format!(
                    "line {}: REPEAT needs a count",
                    flat[i].origin
                )));
            }
            let count_tok = rest.split_whitespace().next().unwrap_or("");
            let parsed = count_tok.parse::<f64>().ok();
            let (body, nxt) = collect_repeat_body(flat, i + 1)?;
            match parsed {
                Some(f) => {
                    if !f.is_finite() || f.fract() != 0.0 {
                        return Err(Error::asm(format!(
                            "line {}: REPEAT count must be an integral literal",
                            flat[i].origin
                        )));
                    }
                    let n = f as i64;
                    if n < 0 {
                        return Err(Error::asm(format!(
                            "line {}: REPEAT count must be >= 0",
                            flat[i].origin
                        )));
                    }
                    if has_break(&body) {
                        out.push(Entry {
                            text: format!("__REPEAT__ {count_tok}"),
                            origin: flat[i].origin.clone(),
                            pre_stripped: true,
                        });
                        out.extend(expand_repeats(&body)?);
                        out.push(Entry {
                            text: "__END__".to_string(),
                            origin: flat[i].origin.clone(),
                            pre_stripped: true,
                        });
                    } else {
                        let expanded = expand_repeats(&body)?;
                        for _ in 0..n {
                            out.extend(expanded.clone());
                        }
                    }
                }
                None => {
                    out.push(Entry {
                        text: format!("__REPEAT__ {count_tok}"),
                        origin: flat[i].origin.clone(),
                        pre_stripped: true,
                    });
                    out.extend(expand_repeats(&body)?);
                    out.push(Entry {
                        text: "__END__".to_string(),
                        origin: flat[i].origin.clone(),
                        pre_stripped: true,
                    });
                }
            }
            i = nxt;
            continue;
        }
        if head == "END" {
            return Err(Error::asm(format!(
                "line {}: END without REPEAT",
                flat[i].origin
            )));
        }
        out.push(flat[i].clone());
        i += 1;
    }
    Ok(out)
}

/// Parse a listing into a [`Program`].
pub fn parse(text: &str, opts: &AsmOptions) -> Result<Program> {
    let origin = opts
        .origin
        .clone()
        .unwrap_or_else(|| opts.basedir.display().to_string());
    let stdlib = opts
        .stdlib
        .clone()
        .or_else(|| default_stdlib(&opts.basedir));
    let mut defs = Defs::new();
    let mut flat = Vec::new();
    expand_into(
        text,
        &opts.basedir,
        &origin,
        &mut defs,
        &[],
        stdlib.as_deref(),
        &mut flat,
    )?;
    let flat = expand_repeats(&flat)?;

    let mut config = Config::new();
    let mut inputs: Vec<InDecl> = Vec::new();
    let mut instrs: Vec<Instr> = Vec::new();
    let mut state: Vec<String> = Vec::new();

    for entry in &flat {
        let line = line_of(entry);
        if line.is_empty() {
            continue;
        }
        let ln = entry.origin.clone();
        let (head, rest) = split_head(&line);
        match head.as_str() {
            "__REPEAT__" => {
                let arg = rest.unwrap_or("").trim().to_string();
                instrs.push(Instr {
                    outs: vec![],
                    mn: "__REPEAT__".into(),
                    args: vec![arg],
                    line: ln,
                });
                continue;
            }
            "__END__" => {
                instrs.push(Instr {
                    outs: vec![],
                    mn: "__END__".into(),
                    args: vec![],
                    line: ln,
                });
                continue;
            }
            "BREAK" => {
                let rest = rest.unwrap_or("");
                if rest.split_whitespace().count() != 1 {
                    return Err(Error::asm(format!("line {ln}: BREAK needs a condition")));
                }
                instrs.push(Instr {
                    outs: vec![],
                    mn: "__BREAK__".into(),
                    args: vec![rest.trim().to_string()],
                    line: ln,
                });
                continue;
            }
            "CONFIG" => {
                let kv: Vec<&str> = rest.unwrap_or("").split_whitespace().collect();
                if kv.len() != 2 {
                    return Err(Error::asm(format!("line {ln}: CONFIG needs key+value")));
                }
                config.insert(kv[0], kv[1]);
            }
            "RANGE" => {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() != 4 {
                    return Err(Error::asm(format!("line {ln}: RANGE needs name+lo+hi")));
                }
                if parts[2].parse::<f64>().is_err() || parts[3].parse::<f64>().is_err() {
                    return Err(Error::asm(format!(
                        "line {ln}: RANGE bounds must be numeric"
                    )));
                }
            }
            "STATE" => {
                let names: Vec<String> = rest
                    .unwrap_or("")
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .collect();
                if names.is_empty() {
                    return Err(Error::asm(format!("line {ln}: STATE needs a name")));
                }
                for nm in names {
                    if state.contains(&nm) {
                        return Err(Error::asm(format!("line {ln}: duplicate STATE {nm}")));
                    }
                    state.push(nm);
                }
            }
            "IN" => {
                let rest = rest.unwrap_or("");
                let up = rest.to_ascii_uppercase();
                let (names, lay) = if let Some(idx) = up.find(" AS ") {
                    let names = rest[..idx]
                        .split(',')
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(str::to_string)
                        .collect::<Vec<_>>();
                    let lay = rest[idx + 4..].trim();
                    (
                        names,
                        if lay.is_empty() {
                            None
                        } else {
                            Some(lay.to_string())
                        },
                    )
                } else {
                    (
                        rest.split(',')
                            .map(str::trim)
                            .filter(|s| !s.is_empty())
                            .map(str::to_string)
                            .collect(),
                        None,
                    )
                };
                if names.is_empty() {
                    return Err(Error::asm(format!("line {ln}: IN needs a name")));
                }
                for nm in names {
                    if inputs.iter().any(|d| d.name == nm) {
                        return Err(Error::asm(format!("line {ln}: duplicate IN {nm}")));
                    }
                    inputs.push(InDecl {
                        name: nm,
                        layout: lay.clone(),
                    });
                }
            }
            _ => {
                if let Some(eq) = line.find('=') {
                    let left = &line[..eq];
                    let right = line[eq + 1..].trim();
                    let outs: Vec<String> = left
                        .split(',')
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(str::to_string)
                        .collect();
                    let (mn, args) = if let Some(open) = right.find('(') {
                        let mn = right[..open].trim().to_ascii_uppercase();
                        let rest = &right[open + 1..];
                        if !rest.ends_with(')') {
                            return Err(Error::asm(format!("line {ln}: unbalanced parens")));
                        }
                        let args = rest[..rest.len() - 1]
                            .split(',')
                            .map(str::trim)
                            .filter(|s| !s.is_empty())
                            .map(str::to_string)
                            .collect::<Vec<_>>();
                        (mn, args)
                    } else {
                        let (mn, rest) = split_head(right);
                        let args = rest
                            .map(|r| {
                                r.split(',')
                                    .map(str::trim)
                                    .filter(|s| !s.is_empty())
                                    .map(str::to_string)
                                    .collect()
                            })
                            .unwrap_or_default();
                        (mn, args)
                    };
                    if outs.is_empty() || mn.is_empty() {
                        return Err(Error::asm(format!("line {ln}: bad instruction")));
                    }
                    instrs.push(Instr {
                        outs,
                        mn,
                        args,
                        line: ln,
                    });
                } else {
                    return Err(Error::asm(format!(
                        "line {ln}: unparseable: {:?}",
                        entry.text
                    )));
                }
            }
        }
    }

    if inputs.is_empty() {
        return Err(Error::asm("no IN declared"));
    }
    for nm in &state {
        if !inputs.iter().any(|d| &d.name == nm) {
            return Err(Error::asm(format!(
                "STATE {nm} must also be IN-declared (seed)"
            )));
        }
    }
    let assigned: std::collections::HashSet<&str> = instrs
        .iter()
        .flat_map(|i| i.outs.iter().map(|s| s.as_str()))
        .collect();
    for nm in &state {
        if !assigned.contains(nm.as_str()) {
            return Err(Error::asm(format!(
                "STATE {nm} is never assigned as OUT (no carry)"
            )));
        }
    }

    Ok(Program {
        config,
        inputs,
        instrs,
        state,
    })
}

/// Bind mnemonics and check arity/inputs statically.
pub fn assemble(text: &str, opts: &AsmOptions) -> Result<BoundProgram> {
    let registry = ops::registry();
    let program = parse(text, opts)?;
    let mut bound = Vec::with_capacity(program.instrs.len());
    let mut defined: std::collections::HashSet<String> =
        program.inputs.iter().map(|d| d.name.clone()).collect();
    let mut loop_depth = 0i32;

    for ins in program.instrs {
        match ins.mn.as_str() {
            "__REPEAT__" => {
                if ins.args.len() != 1 {
                    return Err(Error::asm(format!(
                        "line {}: REPEAT needs a count",
                        ins.line
                    )));
                }
                let a = &ins.args[0];
                if !defined.contains(a) && !is_literal(a) {
                    return Err(Error::asm(format!(
                        "line {}: REPEAT count '{a}' not defined",
                        ins.line
                    )));
                }
                loop_depth += 1;
                bound.push(BoundInstr {
                    outs: vec![],
                    mn: ins.mn,
                    args: ins.args,
                    line: ins.line,
                    sig: None,
                });
                continue;
            }
            "__END__" => {
                loop_depth -= 1;
                bound.push(BoundInstr {
                    outs: vec![],
                    mn: ins.mn,
                    args: ins.args,
                    line: ins.line,
                    sig: None,
                });
                continue;
            }
            "__BREAK__" => {
                if loop_depth <= 0 {
                    return Err(Error::asm(format!(
                        "line {}: BREAK outside REPEAT",
                        ins.line
                    )));
                }
                if ins.args.len() != 1 {
                    return Err(Error::asm(format!(
                        "line {}: BREAK needs a condition",
                        ins.line
                    )));
                }
                let a = &ins.args[0];
                if !defined.contains(a) && !is_literal(a) {
                    return Err(Error::asm(format!(
                        "line {}: BREAK condition '{a}' not defined",
                        ins.line
                    )));
                }
                bound.push(BoundInstr {
                    outs: vec![],
                    mn: ins.mn,
                    args: ins.args,
                    line: ins.line,
                    sig: None,
                });
                continue;
            }
            mn => {
                let def = registry.get(mn).ok_or_else(|| {
                    Error::asm(format!(
                        "line {}: unknown mnemonic: {mn} (known: {} ops)",
                        ins.line,
                        registry.len()
                    ))
                })?;
                if ins.args.len() != def.arity_in {
                    return Err(Error::asm(format!(
                        "line {}: {mn} wants {} args, got {}",
                        ins.line,
                        def.arity_in,
                        ins.args.len()
                    )));
                }
                if ins.outs.len() != def.arity_out {
                    return Err(Error::asm(format!(
                        "line {}: {mn} produces {}, got {} outs",
                        ins.line,
                        def.arity_out,
                        ins.outs.len()
                    )));
                }
                for a in &ins.args {
                    if !defined.contains(a) && !is_literal(a) {
                        return Err(Error::asm(format!(
                            "line {}: {mn}: input stream '{a}' not defined yet",
                            ins.line
                        )));
                    }
                }
                for o in &ins.outs {
                    defined.insert(o.clone());
                }
                bound.push(BoundInstr {
                    outs: ins.outs,
                    mn: ins.mn,
                    args: ins.args,
                    line: ins.line,
                    sig: def.sig.clone(),
                });
            }
        }
    }

    Ok(BoundProgram {
        config: program.config,
        inputs: program.inputs,
        instrs: bound,
        state: program.state,
    })
}

/// Static layout verification report.
#[derive(Clone, Debug)]
pub struct VerifyReport {
    pub errors: Vec<String>,
    pub resolved: usize,
    pub total: usize,
    pub layouts: HashMap<String, String>,
    pub defs: BTreeMap<String, DefIface>,
}

#[derive(Clone, Debug)]
pub struct DefIface {
    pub formals: Vec<(String, Option<String>)>,
    pub returns: Vec<(String, Option<String>)>,
    pub origin: String,
}

/// Replay layout unification over declared layouts only.
pub fn verify(text: &str, opts: &AsmOptions) -> Result<VerifyReport> {
    let bound = assemble(text, opts)?;
    let mut layouts: HashMap<String, String> = bound
        .inputs
        .iter()
        .map(|d| {
            (
                d.name.clone(),
                d.layout.clone().unwrap_or_else(|| UNKNOWN.to_string()),
            )
        })
        .collect();
    let mut errors = Vec::new();

    for ins in &bound.instrs {
        match check_layouts(
            &ins.outs,
            &ins.mn,
            &ins.args,
            &ins.line,
            ins.sig.as_ref(),
            &layouts,
        ) {
            Ok(pairs) => {
                for (name, lay) in pairs {
                    layouts.insert(name, lay);
                }
            }
            Err(e) => {
                errors.push(e.to_string());
                for o in &ins.outs {
                    layouts.insert(o.clone(), UNKNOWN.to_string());
                }
            }
        }
    }

    let total = layouts.len();
    let resolved = layouts.values().filter(|v| *v != UNKNOWN).count();

    // DEF interface table (expand a second time, like Python verify()).
    let origin = opts
        .origin
        .clone()
        .unwrap_or_else(|| opts.basedir.display().to_string());
    let stdlib = opts
        .stdlib
        .clone()
        .or_else(|| default_stdlib(&opts.basedir));
    let mut defs = Defs::new();
    let mut flat = Vec::new();
    expand_into(
        text,
        &opts.basedir,
        &origin,
        &mut defs,
        &[],
        stdlib.as_deref(),
        &mut flat,
    )?;
    let iface = defs
        .into_iter()
        .map(|(name, d)| {
            let formals = d
                .fins
                .iter()
                .map(|f| (f.clone(), d.fin_lay.get(f).and_then(|v| v.clone())))
                .collect();
            let returns = d
                .fouts
                .iter()
                .map(|f| (f.clone(), d.fout_lay.get(f).and_then(|v| v.clone())))
                .collect();
            (
                name,
                DefIface {
                    formals,
                    returns,
                    origin: d.origin,
                },
            )
        })
        .collect();

    Ok(VerifyReport {
        errors,
        resolved,
        total,
        layouts,
        defs: iface,
    })
}

/// Structural shape helper shared with STATE checks.
pub fn value_shape(v: &crate::value::Value) -> Option<Vec<usize>> {
    v.shape()
}

/// Kind of a value as used by the declared-layout cross-check.
pub fn value_kind(v: &crate::value::Value) -> Kind {
    v.kind()
}

/// All DEF names in a listing (for CLI output).
pub fn def_names(text: &str, opts: &AsmOptions) -> Result<Vec<String>> {
    let origin = opts
        .origin
        .clone()
        .unwrap_or_else(|| opts.basedir.display().to_string());
    let stdlib = opts
        .stdlib
        .clone()
        .or_else(|| default_stdlib(&opts.basedir));
    let mut defs = Defs::new();
    let mut flat = Vec::new();
    expand_into(
        text,
        &opts.basedir,
        &origin,
        &mut defs,
        &[],
        stdlib.as_deref(),
        &mut flat,
    )?;
    Ok(defs.keys().cloned().collect())
}
