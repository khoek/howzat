//! Lexicographic reverse search (LRS) traversal.

mod tableau;
mod util;

mod enumerator;
mod input;
mod ops;
mod output;

pub use calculo::num::{Int, IntError};
pub use enumerator::Traversal;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cursor {
    Scan { next_cobasis_pos: usize },
    Backtrack,
}

/// Restart configuration for a traversal.
#[derive(Clone, Debug)]
pub enum Start {
    Root,
    Cobasis { cobasis: Vec<u32> },
    Checkpoint(Checkpoint),
}

/// A compact restart checkpoint expressed in terms of variable indices and traversal cursor state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Checkpoint {
    pub cobasis: Vec<u32>,
    pub depth: usize,
    pub cursor: Cursor,
}

#[derive(Clone, Debug)]
pub struct Options {
    pub cache_limit: usize,
    pub emit_all_bases: bool,
    pub max_depth: Option<usize>,
    pub start: Start,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            cache_limit: 0,
            emit_all_bases: false,
            max_depth: None,
            start: Start::Root,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Error {
    DimensionTooLarge,
    Infeasible,
    InvalidWarmStart,
    Arithmetic(IntError),
    InvariantViolation,
}

pub type Result<T> = std::result::Result<T, Error>;

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Arithmetic(err) => Some(err),
            _ => None,
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionTooLarge => write!(f, "dimension too large"),
            Self::Infeasible => write!(f, "infeasible"),
            Self::InvalidWarmStart => write!(f, "invalid warm start"),
            Self::Arithmetic(err) => write!(f, "arithmetic error: {err}"),
            Self::InvariantViolation => write!(f, "invariant violation"),
        }
    }
}

impl From<IntError> for Error {
    fn from(value: IntError) -> Self {
        Self::Arithmetic(value)
    }
}
