/// The one error type used across the assembler and the runtime.
///
/// `Asm` mirrors Python's `AsmError` (parse/assemble/layout/shape failures).
/// `Unimplemented` is the Rust equivalent of `NoPattern`: an op or backend
/// exists in the registry but has no implementation yet. It is deliberately
/// loud, never a silent fallback.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Asm(String),

    #[error("{0}")]
    Unimplemented(String),

    #[error("{0}")]
    Runtime(String),

    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

impl Error {
    pub fn asm(msg: impl Into<String>) -> Self {
        Error::Asm(msg.into())
    }

    pub fn runtime(msg: impl Into<String>) -> Self {
        Error::Runtime(msg.into())
    }

    pub fn unimplemented(msg: impl Into<String>) -> Self {
        Error::Unimplemented(msg.into())
    }
}

impl From<&str> for Error {
    fn from(value: &str) -> Self {
        Error::Asm(value.to_string())
    }
}

impl From<String> for Error {
    fn from(value: String) -> Self {
        Error::Asm(value)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// Small helper for the many "line {ln} ({mn}): ..." diagnostics.
pub fn at(line: &str, mn: &str, msg: impl std::fmt::Display) -> Error {
    Error::Asm(format!("line {line} ({mn}): {msg}"))
}
