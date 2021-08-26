use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub(crate) enum OffspringAllotmentError {
    DegeneratePopulation,
}

impl fmt::Display for OffspringAllotmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DegeneratePopulation => write!(f, "attempted evolution on degenerate population")
        }
    }
}

impl Error for OffspringAllotmentError {}
