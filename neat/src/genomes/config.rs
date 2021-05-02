use crate::genomes::ActivationType;

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
    pub activation_types: Vec<ActivationType>,
    /// Chance of producing a mutated clone of a parent
    /// during mating.
    pub mutate_only_chance: f32,
    /// Chance of producing a non-mutated child during mating.
    pub mate_only_chance: f32,
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
    /// [`weight_bound`]: crate::GeneticConfig::weight_bound
    pub weight_mutation_power: f32,
    /// Chance of a node mutation taking place during mating.
    pub node_mutation_chance: f32,
    /// Chance of a gene mutation taking place during mating.
    pub gene_mutation_chance: f32,
    /// Maximum number of gene mutation attempts before
    /// mutation returns with failure.
    pub max_gene_mutation_attempts: usize,
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

impl Default for GeneticConfig {
    fn default() -> GeneticConfig {
        GeneticConfig {
            input_count: NonZeroUsize::new(1).unwrap(),
            output_count: NonZeroUsize::new(1).unwrap(),
            activation_types: vec![],
            mutate_only_chance: 0.0,
            mate_only_chance: 0.0,
            mate_by_averaging_chance: 0.0,
            suppression_reset_chance: 0.0,
            initial_expression_chance: 0.0,
            weight_bound: 0.0,
            weight_reset_chance: 0.0,
            weight_nudge_chance: 0.0,
            weight_mutation_power: 0.0,
            node_mutation_chance: 0.0,
            gene_mutation_chance: 0.0,
            max_gene_mutation_attempts: 0,
            recursion_chance: 0.0,
            excess_gene_factor: 0.0,
            disjoint_gene_factor: 0.0,
            common_weight_factor: 0.0,
        }
    }
}
