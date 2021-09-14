use crate::genomics::ActivationType;

use std::num::NonZeroUsize;

/// Configuration data for genome generation
/// and inter-genome operations.
#[derive(Clone, Debug)]
pub struct GeneticConfig {
    /// Number of inputs in a genome.
    pub input_count: NonZeroUsize,
    /// Number of outputs in a genome.
    pub output_count: NonZeroUsize,
    /// Possible activation types for nodes in a genome.
    /// If an empty vector is given, nodes will default
    /// to [`Sigmoid`].
    /// 
    /// [`Sigmoid`]: crate::genomics::ActivationType
    pub activation_types: Vec<ActivationType>,
    /// Activation types of output nodes in a genome.
    /// If fewer than [`output_count`] are specified,
    /// the default is [`Sigmoid`].
    /// 
    /// [`output_count`]: GeneticConfig::output_count
    /// [`Sigmoid`]: crate::genomics::ActivationType
    pub output_activation_types: Vec<ActivationType>,
    /// Chance of child mutation during mating.
    pub child_mutation_chance: f32,
    /// Chance that common gene weights are averaged during mating,
    /// instead of copying the weight from a randomly chosen parent.
    pub mate_by_averaging_chance: f32,
    /// Chance a suppressed gene is unsuppressed if it was
    /// suppressed in either parent.
    pub suppression_reset_chance: f32,
    /// Chance that a gene between an input-output node pair
    /// is created during initial genome generation.
    pub initial_expression_chance: f32,
    /// Maximum magnitude of a gene's weight.
    pub weight_bound: f32,
    /// Chance of a gene weight being reset during mating.
    pub weight_reset_chance: f32,
    /// Chance of a gene weight being nudged during mating, if not reset.
    pub weight_nudge_chance: f32,
    /// Magnitude of bound on weight mutation uniform distribution.
    /// It is assumed to be lesser than [`weight_bound`]
    ///
    /// [`weight_bound`]: GeneticConfig::weight_bound
    pub weight_mutation_power: f32,
    /// Chance of a node addition mutation taking place during mating.
    pub node_addition_mutation_chance: f32,
    /// Chance of a gene addition mutation taking place during mating.
    pub gene_addition_mutation_chance: f32,
    /// Chance of a node deletion mutation taking place during mating.
    pub node_deletion_mutation_chance: f32,
    /// Chance of a gene deletion mutation taking place during mating.
    pub gene_deletion_mutation_chance: f32,
    /// Maximum number of gene mutation attempts before
    /// mutation returns with failure.
    pub max_gene_addition_mutation_attempts: usize,
    /// Chance that a recursive gene will be created during
    /// gene mutation if possible.
    pub recursion_chance: f32,
    /// Weight of excess genes in genetic distance.
    pub excess_gene_factor: f32,
    /// Weight of disjoint genes in genetic distance.
    pub disjoint_gene_factor: f32,
    /// Weight of the common gene weight average in genetic distance.
    pub common_weight_factor: f32,
}

impl GeneticConfig {
    /// Returns a "zero-valued" default configuration.
    /// All values are 0, empty, or in the case of
    /// `NonZeroUsize`s, 1.
    pub fn zero() -> GeneticConfig {
        GeneticConfig {
            input_count: NonZeroUsize::new(1).unwrap(),
            output_count: NonZeroUsize::new(1).unwrap(),
            activation_types: vec![],
            output_activation_types: vec![],
            child_mutation_chance: 0.0,
            mate_by_averaging_chance: 0.0,
            suppression_reset_chance: 0.0,
            initial_expression_chance: 0.0,
            weight_bound: 0.0,
            weight_reset_chance: 0.0,
            weight_nudge_chance: 0.0,
            weight_mutation_power: 0.0,
            node_addition_mutation_chance: 0.0,
            gene_addition_mutation_chance: 0.0,
            node_deletion_mutation_chance: 0.0,
            gene_deletion_mutation_chance: 0.0,
            max_gene_addition_mutation_attempts: 0,
            recursion_chance: 0.0,
            excess_gene_factor: 0.0,
            disjoint_gene_factor: 0.0,
            common_weight_factor: 0.0,
        }
    }
}
