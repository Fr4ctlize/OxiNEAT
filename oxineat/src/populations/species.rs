use super::PopulationConfig;
use crate::genomes::{GeneticConfig, Genome};

/// Species identifier. Specifies
/// the generation in which the species
/// was born, and the count of other species
/// generated in the _same generation_ before
/// the one identified (i.e, if it was the
/// third species born in generation 5, it
/// will be species [5, 2]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SpeciesID(pub usize, pub usize);

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
/// [genetic distance]: PopulationConfig::distance_threshold
/// [`stagnation_threshold`]: PopulationConfig::survival_threshold
#[derive(Debug, Clone)]
pub struct Species {
    id: SpeciesID,
    pub(super) genomes: Vec<Genome>,
    representative: Genome,
    stagnation: usize,
    max_fitness: f32,
}

impl Species {
    /// Creates a new species with the specified ID and
    /// representative.
    pub fn new(id: SpeciesID, representative: Genome) -> Species {
        Species {
            id,
            genomes: vec![representative.clone()],
            representative,
            stagnation: 0,
            max_fitness: 0.0,
        }
    }

    /// Returns the species' ID.
    pub fn id(&self) -> SpeciesID {
        self.id
    }

    /// Returns the species' representative.
    pub fn representative(&self) -> &Genome {
        &self.representative
    }

    pub fn genetic_distance(&self, other: &Genome, config: &GeneticConfig) -> f32 {
        self.representative.genetic_distance_to(other, config)
    }

    /// Adds a genome to the species.
    pub fn add_genome(&mut self, genome: Genome) {
        self.genomes.push(genome);
    }

    /// Updates the species' record of maximum
    /// fitness, to keep track of stagnation.
    pub fn update_fitness(&mut self) {
        let max_fitness = self
            .genomes
            .iter()
            .map(|g| g.fitness)
            .max_by(|a, b| {
                a.partial_cmp(b)
                    .unwrap_or_else(|| panic!("uncomparable fitness value detected"))
            })
            .unwrap_or(0.0);
        if max_fitness <= self.max_fitness {
            self.stagnation += 1;
        }
        self.max_fitness = max_fitness;
    }

    /// Returns the species' _member-count adjusted_
    /// fitness. I.e., the average of the species'
    /// genome's fitnesses.
    pub fn adjusted_fitness(&self) -> f32 {
        self.genomes.iter().map(|g| g.fitness).sum::<f32>() / self.genomes.len() as f32
    }

    /// Returns the number of generations the species
    /// has been stagnated.
    pub fn time_stagnated(&self) -> usize {
        self.stagnation
    }

    /// Returns the species' members.
    pub fn genomes(&self) -> &[Genome] {
        &self.genomes
    }

    /// Returns the currently best-performing genome.
    pub fn champion(&self) -> &Genome {
        self.genomes
            .iter()
            .max_by(|g1, g2| {
                g1.fitness
                    .partial_cmp(&g2.fitness)
                    .unwrap_or_else(|| panic!("uncomparable fitness value detected"))
            })
            .expect("empty species has no champion")
    }

    pub(super) fn count_elite(&self, config: &PopulationConfig) -> usize {
        self.genomes.len().min(config.elitism)
    }

    pub(super) fn count_survivors(&self, config: &PopulationConfig) -> usize {
        (self.genomes.len() as f32 * config.survival_threshold).ceil() as usize
    }
}

#[cfg(test)]
mod tests {}
