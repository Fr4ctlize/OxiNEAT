use crate::genomes::GeneticConfig;
use crate::Innovation;

/// Genes are the principal components of genomes.
/// They are created between two nodes, and become
/// network connections in the genome's phenotype.
#[derive(Clone, Debug, PartialEq)]
pub struct Gene {
    id: Innovation,
    input: Innovation,
    output: Innovation,
    pub(super) weight: f64,
    pub(super) suppressed: bool,
}

impl Gene {
    /// Returns a new _unsuppressed_ gene with the specified parameters.
    pub fn new(id: Innovation, input: Innovation, output: Innovation, weight: f64) -> Gene {
        todo!()
    }

    /// Randomizes the gene's weight. Uses a uniform
    /// distribution over the range ±[`weight_mutation_power`].
    ///
    /// [`weight_mutation_power`]: crate::genomes::GeneticConfig::weight_mutation_power
    pub fn randomize_weight(&mut self, config: GeneticConfig) {
        todo!()
    }

    /// Nudges the gene's weight by a random amount. Uses
    /// a uniform distribution over the range ±[`weight_mutation_power`].
    /// If the weight's magnitude would exceed the [`weight_bound`],
    /// the weight is set to the maximum magnitude with the same
    /// sign.
    ///
    /// [`weight_mutation_power`]: crate::genomes::GeneticConfig::weight_mutation_power
    /// [`weight_bound`]: crate::genomes::GeneticConfig::weight_bound
    pub fn nudge_weight(&mut self, config: GeneticConfig) {
        todo!()
    }

    /// Returns the gene's innovation number.
    pub fn innovation(&self) -> Innovation {
        todo!()
    }

    /// Returns the gene's input node's innovation number.
    pub fn input(&self) -> Innovation {
        todo!()
    }

    /// Returns the gene's output node's innovation number.
    pub fn output(&self) -> Innovation {
        todo!()
    }

    /// Returns the gene's weight.
    pub fn weight(&self) -> f64 {
        todo!()
    }

    /// Returns the gene's suppression status.
    pub fn suppressed(&self) -> bool {
        todo!()
    }
}

#[cfg(test)]
mod tests {}
