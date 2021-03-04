mod config;
mod genes;
mod history;
mod nodes;

pub use config::GeneticConfig;
pub use genes::Gene;
pub use history::History;
pub use nodes::{ActivationType, Node, NodeType};

use crate::{Innovation, Result};

use std::collections::HashMap;

/// Genomes are the focus of evolution in NEAT.
/// They are a collection of genes and nodes that can be instantiated
/// as a phenotype (a neural network) and evaluated
/// for performance in a task, which results numerically in
/// their fitness score. Genomes can be progressively mutated,
/// thus adding complexity and functionality.
#[derive(Clone, Debug, PartialEq)]
pub struct Genome {
    genes: HashMap<Innovation, Gene>,
    nodes: HashMap<Innovation, Node>,
    pub fitness: f64,
}

impl Genome {
    /// Create a new genome with the specified configuration.
    pub fn new(config: GeneticConfig) -> Genome {
        todo!()
    }

    /// Add a new gene to the genome.
    /// Returns a reference to the new gene.
    ///
    /// # Errors
    ///
    /// This function will return an error if a gene with the same
    /// `gene_id` already existed in the genome, or if either `input_id`
    /// or `output_id` do not correspond to nodes present in the genome.
    pub fn add_gene(
        &mut self,
        gene_id: Innovation,
        input_id: Innovation,
        output_id: Innovation,
        weight: f64,
    ) -> Result<&Gene> {
        todo!()
    }

    /// Add a new node to the genome.
    /// Returns a reference  to the newly created node.
    ///
    /// # Errors
    ///
    /// Returns an reference to the newly created node,
    /// or an Error if a node of the same ID already existed
    /// in the genome.
    pub fn add_node(
        &mut self,
        node_id: Innovation,
        activation_type: ActivationType,
    ) -> Result<&Node> {
        todo!()
    }

    /// Induces a _weight mutation_ in the genome.
    pub fn mutate_weights(&mut self, config: GeneticConfig) {
        todo!()
    }

    /// Induces a _gene mutation_ in the genome.
    /// If successful, returns the newly added gene.
    ///
    /// # Errors
    ///
    /// Returns an error if no viable pair of nodes
    /// exists or [too many] attempts have failed.
    ///
    /// [too many]: crate::genomes::GeneticConfig::max_gene_mutation_attempts
    pub fn mutate_genes(&mut self, config: GeneticConfig, history: &mut History) -> Result<&Gene> {
        todo!()
    }

    /// Induces a _node mutation_ in the genome.
    /// If succesful, returns the triplet (_in gene_, _new node_, _out gene_)
    /// as a tuple of references.
    ///
    /// # Errors
    ///
    /// This function returns an error if there are no genes in the genome
    /// that could be split.
    pub fn mutate_nodes(
        &mut self,
        config: GeneticConfig,
        history: &mut History,
    ) -> Result<(&Gene, &Node, &Gene)> {
        todo!()
    }

    /// Combines the callee genome with an `other` genome and
    /// returns their _child_ genome.
    ///
    /// Depending on [`config.mutate_only_chance`] and
    /// [`config.mate_only_chance`], the child may simply be a
    /// clone of the parent with higher fitness, and may or may not
    /// undergo any mutations. Mutation chances are defined by their
    /// corresponding entries in `config`.
    ///
    /// [`config.mutate_only_chance`]: crate::genomes::GeneticConfig::mutate_only_chance
    /// [`config.mate_only_chance`]: crate::genomes::GeneticConfig::mate_only_chance
    pub fn mate_with(&self, other: &Genome, config: GeneticConfig) -> Genome {
        todo!()
    }

    /// Calculates the _genetic distance_ between the callee and `other`,
    /// weighting node and weight differences as specified in `config`.
    pub fn genetic_distance_to(&mut self, other: &Genome, config: GeneticConfig) -> f64 {
        todo!()
    }

    /// Returns a reference to the genome's gene map.
    pub fn genes(&self) -> &HashMap<Innovation, Gene> {
        &self.genes
    }

    /// Returns a reference to the genome's node map.
    pub fn nodes(&self) -> &HashMap<Innovation, Node> {
        &self.nodes
    }
}

#[cfg(test)]
mod tests {}
