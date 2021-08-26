use crate::genomes::GeneticConfig;
use crate::Innovation;

use std::fmt;

use rand::{thread_rng, Rng};
use serde::{Serialize, Deserialize};

/// Genes are the principal components of genomes.
/// They are created between two nodes, and become
/// network connections in the genome's phenotype.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct Gene {
    id: Innovation,
    input: Innovation,
    output: Innovation,
    weight: f32,
    suppressed: bool,
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
        thread_rng().gen_range(-config.weight_bound..=config.weight_bound)
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
        self.weight +=
            thread_rng().gen_range(-config.weight_mutation_power..=config.weight_mutation_power);
        self.weight = self.weight.clamp(-config.weight_bound, config.weight_bound);
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

    /// Sets the gene's weight
    pub fn set_weight(&mut self, w: f32) {
        self.weight = w;
    }

    /// Returns the gene's suppression status.
    pub fn suppressed(&self) -> bool {
        self.suppressed
    }

    /// Sets the gene's suppression status.
    pub fn set_suppressed(&mut self, suppression: bool) {
        self.suppressed = suppression;
    }

    /// Returns the gene's innput and output's innovation numbers.
    pub(super) fn endpoints(&self) -> (Innovation, Innovation) {
        (self.input, self.output)
    }
}

impl fmt::Display for Gene {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}{:?}[{:?}->{:?}, {:.3}]{}",
            if self.suppressed { "(" } else { "" },
            self.id,
            self.input,
            self.output,
            self.weight,
            if self.suppressed { ")" } else { "" },
        )
    }
}

#[cfg(test)]
mod tests {}
