use crate::genomes::GeneticConfig;
use crate::Innovation;

use std::collections::hash_map::{Entry, HashMap};

/// A `History` keeps track of gene and node innovations in a
/// population, in order to make sure identical mutations
/// are assigned the same innovation numbers.
///
/// Clearing the history's mutation lists once per generation
/// is usually enough to keep innovation numbers while exploding
/// without making the history too large.
///
/// For gene innovations the input and output nodes are used to
/// identify identical mutations, and the corresponding innovation
/// number is recorded.
///
/// For node innovations the split gene is used to identify
/// identical mutatinos, and the innovation numbers for the
/// corresponding input gene, new node, and output gene are
/// recorded, in that order.
#[derive(Debug)]
pub struct History {
    next_gene_innovation: Innovation,
    next_node_innovation: Innovation,
    gene_innovations: HashMap<(Innovation, Innovation), Innovation>,
    node_innovations: HashMap<Innovation, (Innovation, Innovation, Innovation)>,
}

impl History {
    /// Creates a new History using the specified configuration.
    pub fn new(config: &GeneticConfig) -> History {
        History {
            // Pre-allocate innovation numbers for all possible initial
            // genes, and the input and output nodes.
            next_gene_innovation: config.input_count.get() * config.output_count.get(),
            next_node_innovation: config.input_count.get() + config.output_count.get(),
            gene_innovations: HashMap::new(),
            node_innovations: HashMap::new(),
        }
    }

    /// Returns the next gene innovation number, or the
    /// previously assigned number to the same gene mutation.
    pub fn next_gene_innovation(&self, input_id: Innovation, output_id: Innovation) -> Innovation {
        *self
            .gene_innovations
            .get(&(input_id, output_id))
            .unwrap_or(&self.next_gene_innovation)
    }

    /// Returns the next node and gene innovation numbers,
    /// or the previously assigned numbers to the same node mutation.
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
    pub fn next_node_innovation(
        &self,
        split_gene: Innovation,
        duplicate: bool,
    ) -> (Innovation, Innovation, Innovation) {
        if !self.node_innovations.contains_key(&split_gene) || duplicate {
            (
                self.next_gene_innovation,
                self.next_node_innovation,
                self.next_gene_innovation + 1,
            )
        } else {
            self.node_innovations[&split_gene]
        }
    }

    /// Adds a gene mutation to the history and returns
    /// the corresponding new gene innovation number, or the
    /// previously assigned number for the same gene mutation.
    pub fn add_gene_innovation(
        &mut self,
        input_id: Innovation,
        output_id: Innovation,
    ) -> Innovation {
        match self.gene_innovations.entry((input_id, output_id)) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                let entry = *entry.insert(self.next_gene_innovation);
                self.next_gene_innovation += 1;
                entry
            }
        }
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
    pub fn add_node_innovation(
        &mut self,
        split_gene: Innovation,
        duplicate: bool,
    ) -> (Innovation, Innovation, Innovation) {
        if !self.node_innovations.contains_key(&split_gene) || duplicate {
            let innovation_record = (
                self.next_gene_innovation,
                self.next_node_innovation,
                self.next_gene_innovation + 1,
            );
            self.node_innovations.insert(split_gene, innovation_record);
            self.next_gene_innovation += 2;
            self.next_node_innovation += 1;
            innovation_record
        } else {
            self.node_innovations[&split_gene]
        }
    }

    /// Clears the history's lists of mutations, but keeps
    /// its innovation number counts.
    pub fn clear(&mut self) {
        self.gene_innovations.clear();
        self.node_innovations.clear();
    }

    /// Returns the highest gene innovation number generated.
    pub fn gene_innovation_count(&self) -> Innovation {
        self.next_gene_innovation - 1
    }

    /// Returns the hightest node innovation number generated.
    pub fn node_innovation_count(&self) -> Innovation {
        self.next_node_innovation - 1
    }
}

#[cfg(test)]
mod tests {}
