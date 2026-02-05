/// Errors that can occur during polyhedral computations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HowzatError {
    /// The input dimensions exceed the maximum supported size.
    DimensionTooLarge,
    /// The caller provided invalid arguments (e.g. out-of-bounds indices).
    InvalidArgument { context: &'static str },
    /// The input contains linearity rows that the requested operation cannot handle.
    CannotHandleLinearity,
    /// A numeric conversion failed (e.g., coercing between number types).
    ConversionFailure,
    /// The LP solver detected cycling during pivoting.
    LpCycling,
    /// Numerical inconsistency was detected (e.g., rounding errors caused invalid state).
    NumericallyInconsistent,
}

impl std::error::Error for HowzatError {}

impl std::fmt::Display for HowzatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionTooLarge => write!(f, "dimension too large"),
            Self::InvalidArgument { context } => write!(f, "invalid argument: {context}"),
            Self::CannotHandleLinearity => write!(f, "cannot handle linearity rows"),
            Self::ConversionFailure => write!(f, "numeric conversion failed"),
            Self::LpCycling => write!(f, "LP solver cycling detected"),
            Self::NumericallyInconsistent => write!(f, "numerical inconsistency detected"),
        }
    }
}
