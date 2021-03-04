mod config;
mod species;

pub use config::PopConfig;
pub use species::Species;

use crate::genomes::{GeneticConfig, Genome, History};

/// A Population is a collection of genomes grouped
/// into species that can be evolved using a genome
/// evaluation function as the source of selective
/// pressure.
pub struct Population<'a> {
    genomes: Vec<Genome>,
    species: Vec<Species<'a>>,
    history: History,
    generation: usize,
    historical_species_count: usize,
    pop_config: PopConfig,
    gen_config: GeneticConfig,
}

impl<'a> Population<'a> {
    /// Creates a new population using the passed configurations.
    /// 
    /// These configurations shouldn't be modified once evolution
    /// begins, thus they are copied and kept by the population for
    /// the duration of its lifetime.
    pub fn new(pop_config: PopConfig, gen_config: GeneticConfig) -> Population<'a> {
        todo!()
    }

    /// Evaluates the fitness of each genome in the
    /// population using the passed evaluator.
    pub fn evaluate_fitness(
        &mut self,
        evaluator: fn(&Genome) -> f64,
    ) {
        todo!()
    }

    /// Evolves the population by mating the best performing
    /// genomes of each species, and re-speciating genomes
    /// as appropiate.
    /// 
    /// If the [adoption rate] is different from 1, offspring
    /// will have a chance of being placed into their parent's
    /// species without speciation, which seems to help NEAT
    /// find solutions faster. (See [[Nodine, T., 2010]].)
    /// 
    /// [Nodine, T., 2010]: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.175.2884&rep=rep1&type=pdf
    /// [adoption rate]: crate::populations::PopConfig::adoption_rate
    pub fn evolve(&mut self) {
        todo!()
    }

    /// Returns the currently best-performing genome.
    pub fn champion(&self) -> &Genome {
        todo!()
    }

    /// Returns a view of all current genomes.
    pub fn genomes(&self) -> &[Genome] {
        todo!()
    }

    /// Returns a random sample of size `n` from the
    /// population.
    pub fn random_sample(&self, n: usize) -> Vec<&Genome> {
        todo!()
    }

    /// Returns a view of all current species.
    pub fn species(&self) -> &[Species] {
        todo!()
    }

    /// Returns the current generation number.
    pub fn generation(&self) -> usize {
        todo!()
    }
}

#[cfg(test)]
mod tests {}
