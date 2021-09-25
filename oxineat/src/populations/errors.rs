use std::error::Error;
use std::fmt;

/// An error type indicating unsuccessful
/// offspring allocation.
#[derive(Debug)]
pub(crate) enum OffspringAllotmentError {
    /// The population was fully degenerate,
    /// i.e. no offspring were assigned.
    DegeneratePopulation,
}

impl fmt::Display for OffspringAllotmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DegeneratePopulation => write!(f, "attempted evolution on degenerate population"),
        }
    }
}

impl Error for OffspringAllotmentError {}
