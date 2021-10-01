/// An interface for genomes that can be used by NEAT.
pub trait Genome {
    type Config;
    type InnovationHistory: InnovationHistory<Config = Self::Config>;

    /// Returns a randomized genome.
    fn new(config: &Self::Config) -> Self;

    /// Returns the genetic distance between two genomes.
    fn genetic_distance(first: &Self, second: &Self, config: &Self::Config) -> f32;

    /// Combines two genomes and returns a "child" genome.
    fn mate(
        parent1: &Self,
        parent2: &Self,
        history: &mut Self::InnovationHistory,
        config: &Self::Config,
    ) -> Self;

    /// Sets the genome's fitness value.
    ///
    /// Should make sure that the fitness value is ≥0;
    /// otherwise NEAT will probably break.
    fn set_fitness(&mut self, fitness: f32);

    /// Returns the genome's fitness value.
    fn fitness(&self) -> f32;
}

/// An Innovation History is used to keep track
/// of genetic innovations throught successive
/// generations of genomes.
///
/// The exact function and utility of the
/// InnovationHistory is left to the implementor.
pub trait InnovationHistory {
    type Config;

    fn new(config: &Self::Config) -> Self;
}
