//! A circuit state is an ordinary assembled Jotnar listing plus provenance.

use std::collections::{BTreeMap, BTreeSet};

use jotnar::asm::{self, AsmOptions};
use jotnar::BoundProgram;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum StateKind {
    Map,
    Fitted,
    Composed,
    Specialized,
    Rewired,
    Refined,
}

impl StateKind {
    pub fn as_str(self) -> &'static str {
        match self {
            StateKind::Map => "map",
            StateKind::Fitted => "fitted",
            StateKind::Composed => "composed",
            StateKind::Specialized => "specialized",
            StateKind::Rewired => "rewired",
            StateKind::Refined => "refined",
        }
    }
}

#[derive(Clone, Debug)]
pub struct State {
    pub name: String,
    pub text: String,
    pub bound: BoundProgram,
    pub kind: StateKind,
    pub family: Option<String>,
    pub params: BTreeMap<String, String>,
}

fn opts() -> AsmOptions {
    AsmOptions::new(".")
}

impl State {
    pub fn new(name: impl Into<String>, text: impl Into<String>) -> Result<Self, String> {
        let text = text.into();
        let bound = asm::assemble(&text, &opts()).map_err(|e| e.to_string())?;
        Ok(Self {
            name: name.into(),
            text,
            bound,
            kind: StateKind::Map,
            family: None,
            params: BTreeMap::new(),
        })
    }

    pub fn with_kind(mut self, kind: StateKind) -> Self {
        self.kind = kind;
        self
    }

    pub fn with_family(
        mut self,
        family: impl Into<String>,
        params: BTreeMap<String, String>,
    ) -> Self {
        self.family = Some(family.into());
        self.params = params;
        self
    }

    pub fn input_names(&self) -> Vec<String> {
        self.bound.inputs.iter().map(|d| d.name.clone()).collect()
    }

    pub fn ops(&self) -> Vec<String> {
        self.bound
            .instrs
            .iter()
            .filter(|i| !i.is_repeat() && !i.is_end() && !i.is_break())
            .map(|i| i.mn.clone())
            .collect()
    }

    /// Streams produced but never consumed. These are the listing's outputs.
    pub fn outputs(&self) -> Vec<String> {
        let inputs: BTreeSet<String> = self.input_names().into_iter().collect();
        let mut produced: Vec<String> = Vec::new();
        let mut seen = BTreeSet::new();
        let mut consumed = BTreeSet::new();
        for ins in &self.bound.instrs {
            for out in &ins.outs {
                if seen.insert(out.clone()) {
                    produced.push(out.clone());
                }
            }
            for arg in &ins.args {
                if !is_literal(arg)
                    && (inputs.contains(arg) || seen.contains(arg) || !is_numeric(arg))
                {
                    consumed.insert(arg.clone());
                }
            }
        }
        produced
            .into_iter()
            .filter(|name| !consumed.contains(name))
            .collect()
    }

    pub fn structural_signature(&self) -> (usize, usize, Vec<String>) {
        let mut ops = self.ops();
        ops.sort();
        (self.input_names().len(), self.outputs().len(), ops)
    }

    pub fn key(&self) -> String {
        let (nin, nout, ops) = self.structural_signature();
        format!(
            "{}|{:?}|{}|{}|{}|{:?}",
            self.kind.as_str(),
            self.family,
            nin,
            nout,
            ops.join(","),
            self.params
        )
    }

    pub fn recipe(&self) -> String {
        format!(
            "state[{}] kind={} family={:?} params={:?}",
            self.name,
            self.kind.as_str(),
            self.family,
            self.params
        )
    }
}

fn is_numeric(s: &str) -> bool {
    s.parse::<i64>().is_ok() || s.parse::<f64>().is_ok()
}

fn is_literal(s: &str) -> bool {
    is_numeric(s) || s.starts_with('"')
}
