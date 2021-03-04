use crate::genomes::ActivationType;

/// Configuration data for genome generation
/// and inter-genome operations.
#[derive(Clone)]
pub struct GeneticConfig {
    /// Number of inputs in a genome.
    pub input_count: usize,
    /// Number of outputs in a genome.
    pub output_count: usize,
    /// Possible activation types for nodes in a genome.
    pub activation_types: Vec<ActivationType>,
    /// Chance of producing a mutated clone of a parent
    /// during mating.
    pub mutate_only_chance: f64,
    /// Chance of producing a non-mutated child during mating.
    pub mate_only_chance: f64,
    /// Chance that common gene weights are averaged during mating,
    /// instead of copying the weight from a randomly chosen parent.
    pub mate_by_averaging_chance: f64,
    /// Chance a suppressed gene is unsuppressed if it was
    /// suppressed in either parent.
    pub suppression_reset_chance: f64,
    /// Chance that a gene between an input-output node pair
    /// is created during initial genome generation.
    pub initial_expression_chance: f64,
    /// Maximum magnitude of a gene's weight.
    pub weight_bound: f64,
    /// Chance of a gene weight being reset during mating.
    pub weight_reset_chance: f64,
    /// Chance of a gene weight being nudged during mating, if not reset.
    pub weight_nudge_chance: f64,
    /// Magnitude of bound on weight mutation uniform distribution.
    pub weight_mutation_power: f64,
    /// Chance of a node mutation taking place during mating.
    pub node_mutation_chance: f64,
    /// Chance of a gene mutation taking place during mating.
    pub gene_mutation_chance: f64,
    /// Maximum number of gene mutation attempts before
    /// mutation returns with failure.
    pub max_gene_mutation_attempts: usize,
    /// Chance that a recursive gene will be created during
    /// gene mutation if possible.
    pub recursion_chance: f64,
    /// Weight of excess genes in genetic distance.
    pub excess_gene_factor: f64,
    /// Weight of disjoint genes in genetic distance.
    pub disjoint_gene_factor: f64,
    /// Weight of the common gene weight average in genetic distance.
    pub common_weight_factor: f64,
}
