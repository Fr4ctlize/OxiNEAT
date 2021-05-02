use crate::genomes::GeneticConfig;
use crate::Innovation;

use rand::{thread_rng, Rng};

/// Genes are the principal components of genomes.
/// They are created between two nodes, and become
/// network connections in the genome's phenotype.
#[derive(Clone, Debug, PartialEq)]
pub struct Gene {
    id: Innovation,
    input: Innovation,
    output: Innovation,
    pub(super) weight: f32,
    pub(super) suppressed: bool,
}

impl Gene {
    /// Returns a new _unsuppressed_ gene with the specified parameters.
    pub fn new(id: Innovation, input: Innovation, output: Innovation, weight: f32) -> Gene {
        Gene {
            id,
            input,
            output,
            weight,
            suppressed: false,
        }
    }

    /// Returns a random weight. Uses a uniform distribution
    /// over the range ±config.weight_mutation_power.
    pub(super) fn random_weight(config: &GeneticConfig) -> f32 {
        thread_rng().gen_range(-config.weight_mutation_power..=config.weight_mutation_power)
    }

    /// Randomizes the gene's weight. Uses a uniform
    /// distribution over the range ±[`weight_mutation_power`].
    ///
    /// [`weight_mutation_power`]: crate::genomes::GeneticConfig::weight_mutation_power
    pub fn randomize_weight(&mut self, config: &GeneticConfig) {
        self.weight = Self::random_weight(config);
    }

    /// Nudges the gene's weight by a random amount. Uses
    /// a uniform distribution over the range ±[`weight_mutation_power`].
    /// If the weight's magnitude would exceed the [`weight_bound`],
    /// the weight is set to the maximum magnitude with the same
    /// sign.
    ///
    /// [`weight_mutation_power`]: crate::genomes::GeneticConfig::weight_mutation_power
    /// [`weight_bound`]: crate::genomes::GeneticConfig::weight_bound
    pub fn nudge_weight(&mut self, config: &GeneticConfig) {
        self.weight += Self::random_weight(config);
        self.weight = self
            .weight
            .max(-config.weight_mutation_power)
            .min(config.weight_mutation_power);
    }

    /// Returns the gene's innovation number.
    pub fn innovation(&self) -> Innovation {
        self.id
    }

    /// Returns the gene's input node's innovation number.
    pub fn input(&self) -> Innovation {
        self.input
    }

    /// Returns the gene's output node's innovation number.
    pub fn output(&self) -> Innovation {
        self.output
    }

    /// Returns the gene's weight.
    pub fn weight(&self) -> f32 {
        self.weight
    }

    /// Returns the gene's suppression status.
    pub fn suppressed(&self) -> bool {
        self.suppressed
    }
}

#[cfg(test)]
mod tests {}
