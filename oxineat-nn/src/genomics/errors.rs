use crate::Innovation;

use std::error::Error;
use std::fmt;

/// An error type indicating the attempted
/// removal of an item that is absent in the genome.
#[derive(Debug)]
pub(crate) enum AbsentEntryRemoval {
    /// The item was a gene.
    Gene(Innovation),
}

/// An error type indicating the gene being created
/// or added is invalid.
#[derive(Debug)]
pub(crate) enum GeneValidityError {
    /// The gene's ID is a duplicate.
    /// Optionally contains the endpoints of the gene.
    DuplicateGeneID(Innovation, Option<(Innovation, Innovation)>),
    /// The gene's endpoints do not exist.
    NonexistantEndpoints(Innovation, Innovation),
    /// The gene has the same endpoints as another with a different ID.
    DuplicateGeneWithEndpoints(Innovation, (Innovation, Innovation)),
    /// The endpoint of the gene is a SENSOR node, which is not allowed.
    SensorEndpoint(Innovation),
}

/// An error type indicating the node being created
/// or added is invalid.
#[derive(Debug)]
pub(crate) enum NodeValidityError {
    /// The node's ID is a duplicate.
    DuplicateNodeID(Innovation),
}

/// An error type indicating a failure
/// to carry out a gene addition mutation.
#[derive(Debug)]
pub(crate) enum GeneAdditionMutationError {
    /// All nodes in the genome are fully connected.
    GenomeFullyConnected,
    /// No pair of nodes was found to connect.
    NoInputOutputPairFound,
}

/// An error type indicating a failure
/// to carry out a node addition mutation.
#[derive(Debug)]
pub(crate) enum NodeAdditionMutationError {
    /// The genome was empty.
    EmptyGenome,
}

impl fmt::Display for AbsentEntryRemoval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gene(id) => write!(f, "attempted removal of nonexistant gene with id {}", id),
        }
    }
}

impl fmt::Display for GeneValidityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateGeneID(gene_id, endpoints) => match endpoints {
                Some((input_id, output_id)) => write!(
                    f,
                    "duplicate gene insertion with id {} between endpoints {} -> {}",
                    gene_id, input_id, output_id
                ),
                None => write!(f, "duplicate gene insertion with id {}", gene_id),
            },
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
            Self::SensorEndpoint(id) => write!(
                f,
                "gene insertion with sensor node as endpoint with id {}",
                id
            ),
        }
    }
}

impl fmt::Display for NodeValidityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateNodeID(id) => write!(f, "duplicate node insertion with id {}", id),
        }
    }
}

impl fmt::Display for GeneAdditionMutationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GenomeFullyConnected => write!(f, "gene mutation on fully-connected genome"),
            Self::NoInputOutputPairFound => {
                write!(f, "no viable input-output pair found for gene mutation")
            }
        }
    }
}

impl fmt::Display for NodeAdditionMutationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyGenome => write!(f, "node mutation on empty genome"),
        }
    }
}

impl Error for AbsentEntryRemoval {}
impl Error for GeneValidityError {}
impl Error for NodeValidityError {}
impl Error for GeneAdditionMutationError {}
impl Error for NodeAdditionMutationError {}
