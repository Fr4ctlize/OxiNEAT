use std::error::Error;
use std::fmt;

use crate::Innovation;

#[derive(Debug)]
pub(crate) enum GeneViabilityError {
    DuplicateGeneID(Innovation),
    NonexistantEndpoints(Innovation, Innovation),
    DuplicateGeneWithEndpoints(Innovation, (Innovation, Innovation)),
}

#[derive(Debug)]
pub(crate) enum NodeViabilityError {
    DuplicateNodeID(Innovation),
}

#[derive(Debug)]
pub(crate) enum GeneMutationError {
    AllInputsFullyConnected,
    NoInputOutputPairFound,
}

#[derive(Debug)]
pub(crate) enum NodeMutationError {
    EmptyGenome,
}

impl fmt::Display for GeneViabilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateGeneID(id) => write!(f, "duplicate gene insertion with id {}", id),
            Self::NonexistantEndpoints(input, output) => write!(
                f,
                "gene insertion between nonexistant endpoint(s) {} -> {}",
                input, output
            ),
            Self::DuplicateGeneWithEndpoints(duplicate_id, (input, output)) => write!(
                f,
                "gene insertion with endpoints {} -> {} and id {} shadows gene with same endpoints",
                input, output, duplicate_id,
            ),
        }
    }
}

impl fmt::Display for NodeViabilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateNodeID(id) => write!(f, "duplicate node insertion with id {}", id),
        }
    }
}

impl fmt::Display for GeneMutationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AllInputsFullyConnected => write!(f, "gene mutation on fully-connected genome"),
            Self::NoInputOutputPairFound => {
                write!(f, "no viable input-output pair found for gene mutation")
            }
        }
    }
}

impl fmt::Display for NodeMutationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyGenome => write!(f, "node mutation on empty genome"),
        }
    }
}

impl Error for GeneViabilityError {}
impl Error for NodeViabilityError {}
impl Error for GeneMutationError {}
impl Error for NodeMutationError {}
