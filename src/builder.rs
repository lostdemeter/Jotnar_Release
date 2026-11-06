//! Programmatic listing construction (port of `chain/builder.py`).
//!
//! The builder emits ordinary Jotnar-LAN text. The assembler remains the
//! authority: generated text is never trusted without assemble/run parity.

use crate::error::{Error, Result};

#[derive(Clone, Debug)]
pub enum Arg {
    Name(String),
    Int(i64),
    Float(f64),
}

impl Arg {
    fn render(&self) -> Result<String> {
        match self {
            Arg::Name(s) => Ok(s.clone()),
            Arg::Int(v) => Ok(v.to_string()),
            Arg::Float(v) => Ok(format!("{v:?}")),
        }
    }
}

impl From<&str> for Arg {
    fn from(v: &str) -> Self {
        Arg::Name(v.to_string())
    }
}

impl From<String> for Arg {
    fn from(v: String) -> Self {
        Arg::Name(v)
    }
}

impl From<&String> for Arg {
    fn from(v: &String) -> Self {
        Arg::Name(v.clone())
    }
}

impl From<i64> for Arg {
    fn from(v: i64) -> Self {
        Arg::Int(v)
    }
}

impl From<f64> for Arg {
    fn from(v: f64) -> Self {
        Arg::Float(v)
    }
}

#[derive(Clone, Debug)]
enum Out {
    One(String),
    Many(Vec<String>),
}

#[derive(Clone, Debug)]
enum Line {
    Op {
        out: Out,
        mn: String,
        args: Vec<String>,
    },
    Call {
        out: Out,
        name: String,
        args: Vec<String>,
    },
    Def {
        name: String,
        formals: Vec<String>,
        body: Vec<Line>,
        outs: Vec<String>,
    },
}

/// A builder program. Mirrors the Python `Prog` surface.
#[derive(Clone, Debug)]
pub struct Prog {
    pub name: String,
    config: Vec<(String, String)>,
    imports: Vec<String>,
    inputs: Vec<String>,
    lines: Vec<Line>,
    auto: usize,
}

impl Prog {
    pub fn new(name: impl Into<String>) -> Self {
        Prog {
            name: name.into(),
            config: Vec::new(),
            imports: Vec::new(),
            inputs: Vec::new(),
            lines: Vec::new(),
            auto: 0,
        }
    }

    pub fn config(&mut self, key: impl Into<String>, value: impl Into<String>) -> &mut Self {
        self.config.push((key.into(), value.into()));
        self
    }

    pub fn import(&mut self, path: impl Into<String>) -> &mut Self {
        self.imports.push(path.into());
        self
    }

    /// Declare input streams.
    pub fn inp<I, S>(&mut self, names: I) -> Result<&mut Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for name in names {
            let name = name.into();
            if self.inputs.contains(&name) {
                return Err(Error::asm(format!("builder: duplicate IN '{name}'")));
            }
            self.inputs.push(name);
        }
        Ok(self)
    }

    fn fresh(&mut self, hint: &str) -> String {
        self.auto += 1;
        format!("{hint}{}", self.auto)
    }

    fn render(args: &[Arg]) -> Result<Vec<String>> {
        args.iter().map(Arg::render).collect()
    }

    /// Emit one op. `out=None` allocates `t1`, `t2`, ...
    pub fn op(&mut self, mn: &str, args: &[Arg], out: Option<&str>) -> Result<String> {
        let o = match out {
            Some(o) => o.to_string(),
            None => self.fresh("t"),
        };
        self.lines.push(Line::Op {
            out: Out::One(o.clone()),
            mn: mn.to_uppercase(),
            args: Self::render(args)?,
        });
        Ok(o)
    }

    /// Convenience for the common all-stream-name form.
    pub fn op_names(&mut self, mn: &str, args: &[&str], out: Option<&str>) -> Result<String> {
        let args: Vec<Arg> = args.iter().map(|x| Arg::from(*x)).collect();
        self.op(mn, &args, out)
    }

    /// Emit one op with multiple outputs.
    pub fn op_many(
        &mut self,
        mn: &str,
        args: &[Arg],
        outs: Option<&[&str]>,
    ) -> Result<Vec<String>> {
        let o = match outs {
            Some(o) => o.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            None => {
                let n = args.len().max(1);
                (0..n).map(|_| self.fresh("t")).collect()
            }
        };
        self.lines.push(Line::Op {
            out: Out::Many(o.clone()),
            mn: mn.to_uppercase(),
            args: Self::render(args)?,
        });
        Ok(o)
    }

    /// Emit one CALL. `nout=1` allocates a single auto output.
    pub fn call(
        &mut self,
        name: &str,
        args: &[Arg],
        out: Option<&str>,
        nout: usize,
    ) -> Result<OutNames> {
        let rendered = Self::render(args)?;
        let o = if nout == 1 {
            OutNames::One(match out {
                Some(o) => o.to_string(),
                None => self.fresh("t"),
            })
        } else {
            OutNames::Many((0..nout).map(|_| self.fresh("t")).collect())
        };
        self.lines.push(Line::Call {
            out: match &o {
                OutNames::One(x) => Out::One(x.clone()),
                OutNames::Many(x) => Out::Many(x.clone()),
            },
            name: name.to_string(),
            args: rendered,
        });
        Ok(o)
    }

    /// Define a DEF block. `body` builds the local program and returns the
    /// declared output stream(s).
    pub fn defn<F>(&mut self, name: &str, formals: &[&str], body: F) -> Result<&mut Self>
    where
        F: FnOnce(&mut Prog) -> Result<Vec<String>>,
    {
        let mut sub = Prog::new(format!("{}:{name}", self.name));
        sub.inp(formals.iter().copied())?;
        let outs = body(&mut sub)?;
        self.lines.push(Line::Def {
            name: name.to_string(),
            formals: formals.iter().map(|s| s.to_string()).collect(),
            body: sub.lines,
            outs,
        });
        Ok(self)
    }

    fn emit_line(&self, line: &Line, indent: usize, out: &mut Vec<String>) -> Result<()> {
        let ind = "  ".repeat(indent);
        match line {
            Line::Op { out: o, mn, args } => {
                let o = match o {
                    Out::One(s) => s.clone(),
                    Out::Many(v) => format!("({})", v.join(", ")),
                };
                out.push(format!("{ind}{o} = {mn}({})", args.join(", ")));
            }
            Line::Call { out: o, name, args } => {
                let o = match o {
                    Out::One(s) => s.clone(),
                    Out::Many(v) => format!("({})", v.join(", ")),
                };
                out.push(format!("{ind}{o} = CALL {name}({})", args.join(", ")));
            }
            Line::Def {
                name,
                formals,
                body,
                outs,
            } => {
                out.push(format!(
                    "{ind}DEF {name}({}) -> ({})",
                    formals.join(", "),
                    outs.join(", ")
                ));
                for l in body {
                    self.emit_line(l, indent + 1, out)?;
                }
                out.push(format!("{ind}END"));
            }
        }
        Ok(())
    }

    /// Render the listing.
    pub fn text(&self) -> Result<String> {
        let mut lines = Vec::new();
        for (k, v) in &self.config {
            lines.push(format!("CONFIG {k} {v}"));
        }
        for p in &self.imports {
            lines.push(format!("IMPORT \"{p}\""));
        }
        for n in &self.inputs {
            lines.push(format!("IN {n}"));
        }
        for l in &self.lines {
            self.emit_line(l, 0, &mut lines)?;
        }
        Ok(lines.join("\n") + "\n")
    }
}

#[derive(Clone, Debug)]
pub enum OutNames {
    One(String),
    Many(Vec<String>),
}

impl OutNames {
    pub fn one(&self) -> Result<&str> {
        match self {
            OutNames::One(s) => Ok(s),
            OutNames::Many(_) => Err(Error::asm("builder: expected one output")),
        }
    }
}

// ---------------------------------------------------------------------------
// Stdlib seed composites (the small, stable subset; the Qwen-specific ones
// stay in the application layer for now).
// ---------------------------------------------------------------------------

pub fn attn_scores(
    p: &mut Prog,
    qh: &str,
    kh: &str,
    pos: &str,
    scale_head2: bool,
) -> Result<String> {
    let qr = p.op_names("ROTARY", &[qh, pos], None)?;
    let kr = p.op_names("ROTARY", &[kh, pos], None)?;
    let kt = p.op_names("TRANSPOSE", &[&kr], None)?;
    let mut sc = p.op_names("BATCH_MATMUL", &[&qr, &kt], None)?;
    if scale_head2 {
        let ht = p.op_names("TBETA", &[&sc], None)?;
        sc = p.op_names("MUL", &[&sc, &ht], None)?;
    }
    Ok(sc)
}

pub fn attn_select(
    p: &mut Prog,
    sc: &str,
    vh: &str,
    cmask: &str,
    neg: &str,
    out: &str,
) -> Result<String> {
    let ms = p.op_names("SELECT", &[cmask, sc, neg], None)?;
    let s = p.op_names("TSHIFT", &[&ms], None)?;
    let pr = p.op_names("SOFTMAX_WIDE", &[&s], None)?;
    p.op_names("BATCH_MATMUL", &[&pr, vh], Some(out))
}

pub fn attn_head(
    p: &mut Prog,
    x: &str,
    wq: &str,
    wk: &str,
    wv: &str,
    pos: &str,
    cmask: &str,
    out: &str,
    scale_head2: bool,
) -> Result<String> {
    let q = p.op_names("MATMUL", &[x, wq], None)?;
    let k = p.op_names("MATMUL", &[x, wk], None)?;
    let v = p.op_names("MATMUL", &[x, wv], None)?;
    let sc = attn_scores(p, &q, &k, pos, scale_head2)?;
    let neg = p.op_names("BETA", &[&sc], None)?;
    attn_select(p, &sc, &v, cmask, &neg, out)
}
