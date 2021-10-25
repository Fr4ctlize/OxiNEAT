use crate::genomics::GeneticConfig;
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
    /// 
    /// # Examples
    /// ```
    /// use oxineat_nn::genomics::Gene;
    /// 
    /// let gene = Gene::new(42, 3, 9, 2.0);
    /// ```
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
    /// [`weight_mutation_power`]: crate::genomics::GeneticConfig::weight_mutation_power
    /// 
    /// # Examples
    /// ```
    /// use oxineat_nn::genomics::{Gene, GeneticConfig};
    /// 
    /// let mut gene = Gene::new(42, 3, 9, 2.0);
    /// 
    /// # assert_eq!(gene.weight(), 2.0);
    /// 
    /// gene.randomize_weight(&GeneticConfig {
    ///     weight_bound: 5.0,
    ///     ..GeneticConfig::zero()
    /// });
    /// 
    /// # // The weight should change to a random value
    /// # // in the [-5.0, 5.0] range.
    /// # // NOTE: it is possible for the new value to
    /// # // be the same as the old, depending on RNG seed.
    /// # assert_ne!(gene.weight(), 2.0);
    /// # assert!(gene.weight().abs() <= 5.0);
    /// ```
    pub fn randomize_weight(&mut self, config: &GeneticConfig) {
        self.weight = Self::random_weight(config);
    }

    /// Nudges the gene's weight by a random amount. Uses
    /// a uniform distribution over the range ±[`weight_mutation_power`].
    /// If the weight's magnitude would exceed the [`weight_bound`],
    /// the weight is set to the maximum magnitude with the same
    /// sign.
    ///
    /// [`weight_mutation_power`]: crate::genomics::GeneticConfig::weight_mutation_power
    /// [`weight_bound`]: crate::genomics::GeneticConfig::weight_bound
    /// 
    /// # Examples
    /// ```
    /// use oxineat_nn::genomics::{Gene, GeneticConfig};
    /// 
    /// let mut gene = Gene::new(42, 3, 9, 3.0);
    /// 
    /// # assert_eq!(gene.weight(), 3.0);
    /// 
    /// gene.nudge_weight(&GeneticConfig {
    ///     weight_mutation_power: 2.5,
    ///     weight_bound: 5.0,
    ///     ..GeneticConfig::zero()
    /// });
    /// 
    /// # // The weight should be nudged by a random value
    /// # // in the [-2.5, 2.5] range, and clamped into [-5.0, 5.0].
    /// # // NOTE: it is possible for the new value to be the
    /// # // same as the old, depending on RNG seed.
    /// # assert_ne!(gene.weight(), 3.0);
    /// # assert!((gene.weight() - 3.0).abs() <= 2.5);
    /// # assert!(gene.weight().abs() <= 5.0);
    /// ```
    pub fn nudge_weight(&mut self, config: &GeneticConfig) {
        self.weight +=
            thread_rng().gen_range(-config.weight_mutation_power..=config.weight_mutation_power);
        self.weight = self.weight.clamp(-config.weight_bound, config.weight_bound);
    }

    /// Returns the gene's innovation number.
    /// 
    /// # Examples
    /// ```
    /// use oxineat_nn::genomics::Gene;
    /// 
    /// let gene = Gene::new(42, 3, 9, 2.0);
    /// 
    /// assert_eq!(gene.innovation(), 42);
    /// ```
    pub fn innovation(&self) -> Innovation {
        self.id
    }

    /// Returns the gene's input node's innovation number.
    /// 
    /// # Examples
    /// ```
    /// use oxineat_nn::genomics::Gene;
    /// 
    /// let gene = Gene::new(42, 3, 9, 2.0);
    /// 
    /// assert_eq!(gene.input(), 3);
    /// ```
    pub fn input(&self) -> Innovation {
        self.input
    }

    /// Returns the gene's output node's innovation number.
    /// 
    /// # Examples
    /// ```
    /// use oxineat_nn::genomics::Gene;
    /// 
    /// let gene = Gene::new(42, 3, 9, 2.0);
    /// 
    /// assert_eq!(gene.output(), 9);
    /// ```
    pub fn output(&self) -> Innovation {
        self.output
    }

    /// Returns the gene's weight.
    /// 
    /// # Examples
    /// ```
    /// use oxineat_nn::genomics::Gene;
    /// 
    /// let gene = Gene::new(42, 3, 9, 2.0);
    /// 
    /// assert_eq!(gene.weight(), 2.0);
    /// ```
    pub fn weight(&self) -> f32 {
        self.weight
    }

    /// Sets the gene's weight
    /// 
    /// # Examples
    /// ```
    /// use oxineat_nn::genomics::Gene;
    /// 
    /// let mut gene = Gene::new(42, 3, 9, 2.0);
    /// 
    /// assert_eq!(gene.weight(), 2.0);
    /// 
    /// gene.set_weight(-5.0);
    /// 
    /// assert_eq!(gene.weight(), -5.0);
    /// ```
    pub fn set_weight(&mut self, w: f32) {
        self.weight = w;
    }

    /// Returns the gene's suppression status.
    /// 
    /// # Examples
    /// ```
    /// use oxineat_nn::genomics::Gene;
    /// 
    /// let gene = Gene::new(42, 3, 9, 2.0);
    /// 
    /// assert_eq!(gene.suppressed(), false);
    /// ```
    pub fn suppressed(&self) -> bool {
        self.suppressed
    }

    /// Sets the gene's suppression status.
    /// 
    /// # Examples
    /// ```
    /// use oxineat_nn::genomics::Gene;
    /// 
    /// let mut gene = Gene::new(42, 3, 9, 2.0);
    /// 
    /// assert_eq!(gene.suppressed(), false);
    /// 
    /// gene.set_suppressed(true);
    /// 
    /// assert_eq!(gene.suppressed(), true);
    /// ```
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
