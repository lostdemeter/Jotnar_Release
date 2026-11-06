use std::collections::BTreeMap;

use crate::layout::LayoutSig;

/// Program configuration (`CONFIG key value`).
///
/// Values are kept as raw strings, matching the Python frontend. Ops read
/// them through the typed helpers below; absent keys fall back at the call
/// site exactly as documented by each op.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Config {
    map: BTreeMap<String, String>,
}

impl Config {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.map.insert(key.into(), value.into());
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.map.get(key).map(|s| s.as_str())
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.map.contains_key(key)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &String)> {
        self.map.iter()
    }

    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.get(key).and_then(|v| v.parse::<f64>().ok())
    }

    pub fn get_i64(&self, key: &str) -> Option<i64> {
        self.get_f64(key).map(|v| v as i64)
    }

    pub fn get_f64_or(&self, key: &str, default: f64) -> f64 {
        self.get_f64(key).unwrap_or(default)
    }

    pub fn get_i64_or(&self, key: &str, default: i64) -> i64 {
        self.get_i64(key).unwrap_or(default)
    }
}

/// One `IN name [AS layout]` declaration.
#[derive(Clone, Debug)]
pub struct InDecl {
    pub name: String,
    pub layout: Option<String>,
}

/// One parsed instruction.
#[derive(Clone, Debug)]
pub struct Instr {
    pub outs: Vec<String>,
    pub mn: String,
    pub args: Vec<String>,
    pub line: String,
}

impl Instr {
    /// Pseudo-instructions used for loops.
    pub fn is_repeat(&self) -> bool {
        self.mn == "__REPEAT__"
    }

    pub fn is_end(&self) -> bool {
        self.mn == "__END__"
    }

    pub fn is_break(&self) -> bool {
        self.mn == "__BREAK__"
    }
}

/// Parsed program (before registry binding).
#[derive(Clone, Debug)]
pub struct Program {
    pub config: Config,
    pub inputs: Vec<InDecl>,
    pub instrs: Vec<Instr>,
    pub state: Vec<String>,
}

/// Registry-bound instruction: arity checked, layout signature attached.
#[derive(Clone, Debug)]
pub struct BoundInstr {
    pub outs: Vec<String>,
    pub mn: String,
    pub args: Vec<String>,
    pub line: String,
    pub sig: Option<LayoutSig>,
}

impl BoundInstr {
    pub fn is_repeat(&self) -> bool {
        self.mn == "__REPEAT__"
    }

    pub fn is_end(&self) -> bool {
        self.mn == "__END__"
    }

    pub fn is_break(&self) -> bool {
        self.mn == "__BREAK__"
    }
}

/// A program after assembling against the mnemonic registry.
#[derive(Clone, Debug)]
pub struct BoundProgram {
    pub config: Config,
    pub inputs: Vec<InDecl>,
    pub instrs: Vec<BoundInstr>,
    pub state: Vec<String>,
}

impl BoundProgram {
    pub fn input_names(&self) -> Vec<&str> {
        self.inputs.iter().map(|i| i.name.as_str()).collect()
    }
}
