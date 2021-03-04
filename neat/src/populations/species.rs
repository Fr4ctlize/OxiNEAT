use super::PopConfig;
use super::Population;
use crate::genomes::GeneticConfig;
use crate::genomes::{Genome, History};

/// Species identifier. Specifies
/// the generation in which the species
/// was born, and the count of other species
/// generated in the _same generation_ before
/// the one identified (i.e, if it was the
/// third species born in generation 5, it
/// will be species [5, 2]).
#[derive(Debug)]
pub struct SpeciesID(usize, usize);

/// Species are collections of reproductively
/// compatible (within a certain [genetic distance])
/// genomes. Membership is determined by calculating
/// the genetic distance to a _representative_,
/// which could be the first genome of the species
/// to exist (as in this implementation), or a
/// randomly chosen member of the species each
/// generation.
/// 
/// Species will stagnate after [`stagnation_threshold`]
/// generations without improving the species' fitness,
/// and will thereafter be penalized when mating.
/// 
/// [genetic distance]: crate::populations::PopConfig::distance_threshold
/// [`stagnation_threshold`]: crate::populations::PopConfig::survival_threshold
#[derive(Debug)]
pub struct Species<'a> {
    id: SpeciesID,
    genomes: Vec<&'a Genome>,
    representative: Genome,
    stagnation: usize,
    shared_fitness: f64,
}

impl<'a> Species<'a> {
    /// Creates a new species with the specified ID and
    /// representative.
    pub fn new(id: SpeciesID, representative: &Genome) -> Species<'a> {
        todo!()
    }

    /// Returns the species' ID.
    pub fn id(&self) -> SpeciesID {
        todo!()
    }

    /// Returns the species' representative.
    pub fn representative(&self) -> &Genome {
        todo!()
    }

    /// Adds a genome to the species.
    pub fn add_genome(&mut self, genome: &'a Genome) {
        todo!()
    }

    /// Generates the species' assigned offspring,
    /// keeping the [species' elite] and mating the
    /// [top performers].
    /// 
    /// [species' elite]: crate::populations::PopConfig::elitism;
    /// [top performers]: crate::populations::PopConfig::survival_threshold;
    pub fn generate_offspring(&self, assigned_offspring: usize, pop: Population) -> Vec<Genome> {
        todo!()
    }

    /// Returns the species' _member-count adjusted_
    /// fitness. I.e., the average of the species'
    /// genome's fitnesses.
    pub fn adjusted_fitness(&self) -> f64 {
        todo!()
    }

    /// Returns the number of generations the species
    /// has been stagnated. 
    pub fn time_stagnated(&self) -> usize {
        todo!()
    }

    /// Returns a random sample of the species' members.
    pub fn random_sample(&self, n: usize) -> Vec<&'a Genome> {
        todo!()
    }
}

#[cfg(test)]
mod tests {}
