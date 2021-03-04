use crate::genomes::GeneticConfig;
use crate::Innovation;

use std::collections::HashMap;

/// A `History` keeps track of gene and node innovations in a
/// population, in order to make sure identical mutations
/// are assigned the same innovation numbers.
/// 
/// Clearing the history's mutation lists once per generation
/// is usually enough to keep innovation numbers while exploding
/// without making the history too large.
#[derive(Debug)]
pub struct History {
    next_gene_innovation: Innovation,
    next_node_innovation: Innovation,
    gene_innovations: HashMap<Innovation, Vec<[Innovation; 3]>>,
    node_innovations: HashMap<[Innovation; 2], Innovation>,
}

impl History {
    /// Creates a new History using the specified configuration.
    pub fn new(config: GeneticConfig) -> History {
        todo!()
    }

    /// Returns the next gene innovation number, or the
    /// previously assigned number to the same gene mutation.
    pub fn next_gene_innovation(&self, input_id: Innovation, output_id: Innovation) -> Innovation {
        todo!()
    }

    /// Returns the next node and gene innovation numbers,
    /// or the previously assigned numbers to the same node mutation.
    pub fn next_node_innovation(&self, split_gene: Innovation) -> [Innovation; 3] {
        todo!()
    }

    /// Adds a gene mutation to the history and returns
    /// the corresponding new gene innovation number, or the
    /// previously assigned number for the same gene mutation.
    pub fn add_gene_innovation(
        &mut self,
        input_id: Innovation,
        output_id: Innovation,
    ) -> Innovation {
        todo!()
    }

    /// Adds a node mutation to the history and returns the
    /// innovation numbers of the corresponding new node and
    /// genes, or the previously assigned numbers for the
    /// same node mutation.
    /// 
    /// If `duplicate` is `true` and the node mutation is
    /// already registered, the returned innovation numbers
    /// will be computed as if it were a new mutation. This
    /// is used in situations in which the mutating genome
    /// already split the same gene in a previous mutation,
    /// which would result in duplicate genes and nodes within
    /// the same genome. This can be detected if the numbers
    /// returned by this function without setting `duplicate`
    /// to `true` refer to genes/nodes already present in the
    /// genome.
    pub fn add_node_innovation(&mut self, split_gene: Innovation, duplicate: bool) -> [Innovation; 3] {
        todo!()
    }

    /// Clears the history's lists of mutations, but keeps 
    /// its innovation number counts.
    pub fn clear(&mut self) {
        todo!()
    }
}

#[cfg(test)]
mod tests {}
