//! A Population is a collection of genomes grouped
//! into species that can be evolved using a genome
//! evaluation function as the source of selective
//! pressure.
mod config;
mod errors;
mod log;
mod offspring_factory;
mod species;

pub use config::PopulationConfig;
use errors::*;
pub use log::*;
use offspring_factory::OffspringFactory;
pub use species::{Species, SpeciesID};

use crate::genomics::{GeneticConfig, Genome, History};

use std::collections::HashMap;

use rand::prelude::{IteratorRandom, Rng, SliceRandom};

/// A population of genomes.
pub struct Population {
    species: Vec<Species>,
    history: History,
    generation: usize,
    historical_species_count: usize,
    population_config: PopulationConfig,
    genetic_config: GeneticConfig,
}

impl Population {
    /// Creates a new population using the passed configurations.
    ///
    /// These configurations shouldn't be modified once evolution
    /// begins, thus they are copied and kept by the population for
    /// the duration of its lifetime.
    pub fn new(population_config: PopulationConfig, genetic_config: GeneticConfig) -> Population {
        Population {
            species: {
                let mut s0 = Species::new(SpeciesID(0, 0), Genome::new(&genetic_config));
                s0.genomes.extend(
                    (1..population_config.population_size.get())
                        .map(|_| Genome::new(&genetic_config)),
                );
                vec![s0]
            },
            history: History::new(&genetic_config),
            generation: 0,
            historical_species_count: 1,
            population_config,
            genetic_config,
        }
    }

    /// Evaluates the fitness of each genome in the
    /// population using the passed evaluator.
    ///
    /// The return value of the evaluation function
    /// should be positive.
    pub fn evaluate_fitness<E>(&mut self, mut evaluator: E)
    where
        E: FnMut(&Genome) -> f32,
    {
        for genome in self.species.iter_mut().flat_map(|s| &mut s.genomes) {
            let fitness = evaluator(genome);
            assert!(fitness >= 0.0, "fitness function return a negative value");
            genome.fitness = fitness;
        }
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
    /// # Errors
    /// Returns an error if the population has become degenerate
    /// (zero maximum fitness or all genomes are culled due to
    /// stagnation).
    ///
    /// [adoption rate]: PopulationConfig::adoption_rate
    /// [Nodine, T., 2010]: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.175.2884&rep=rep1&type=pdf
    pub fn evolve(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        match self.allot_offspring() {
            Ok(allotted_offspring) => {
                self.species.iter_mut().for_each(Species::update_fitness);
                self.generate_offspring(&allotted_offspring);
                self.respeciate_all();
                self.remove_extinct_species();
                self.generation += 1;
                // self.history.clear();
                Ok(())
            }
            Err(e) => Err(e.into()),
        }
    }

    /// Allot the number of offspring for each species,
    /// based on proportional adjusted species fitness
    /// and stagnation status.
    ///
    /// # Errors
    ///
    /// Returns an error if all genome's fitnesses are 0.
    fn allot_offspring(&self) -> Result<Vec<usize>, OffspringAllotmentError> {
        match self.get_species_adjusted_fitness() {
            Some(adjusted_fitnesses) => Ok(round_retain_sum(&adjusted_fitnesses)),
            None => Err(OffspringAllotmentError::DegeneratePopulation),
        }
    }

    fn get_species_adjusted_fitness(&self) -> Option<Vec<f32>> {
        let fitnesses = self.species_fitness_with_stagnation_penalty();
        let fitness_sum: f32 = fitnesses.iter().copied().sum();
        if fitness_sum == 0.0 {
            return None;
        }
        Some(
            fitnesses
                .iter()
                .map(|f| *f / fitness_sum * self.population_config.population_size.get() as f32)
                .collect(),
        )
    }

    fn species_fitness_with_stagnation_penalty(&self) -> Vec<f32> {
        self.species
            .iter()
            .map(|s| {
                if s.time_stagnated() >= self.population_config.stagnation_threshold.get() {
                    s.adjusted_fitness() * (1.0 - self.population_config.stagnation_penalty)
                } else {
                    s.adjusted_fitness()
                }
            })
            .collect()
    }

    /// Generates each species' assigned offspring,
    /// keeping the [species' elite] and mating the
    /// [top performers].
    ///
    /// Has a [chance] of selecting a partner
    /// from another species.
    ///
    /// Offspring are assigned randomly to the species
    /// of either one of the parents.
    ///
    /// [species' elite]: PopulationConfig::elitism
    /// [top performers]: PopulationConfig::survival_threshold
    /// [chance]: PopulationConfig::interspecies_mating_chance
    fn generate_offspring(&mut self, allotted_offspring: &[usize]) {
        self.sort_species_members_by_decreasing_fitness();

        let mut species_offspring = OffspringFactory::new(
            &self.species,
            &mut self.history,
            &self.genetic_config,
            &self.population_config,
        )
        .generate_offspring(allotted_offspring);

        for species in &mut self.species {
            species.genomes = species_offspring.remove(&species.id()).unwrap();
        }
    }

    fn sort_species_members_by_decreasing_fitness(&mut self) {
        for species in &mut self.species {
            species.genomes.sort_unstable_by(|g1, g2| {
                g2.fitness
                    .partial_cmp(&g1.fitness)
                    .unwrap_or_else(|| panic!("invalid genome fitnesses detected (NaN)"))
            });
        }
    }

    /// Reassigns each genome to a species based on genetic
    /// distance to species representatives. Has a 1-[adoption rate]
    /// chance of not modifying a genome's assigned species.
    fn respeciate_all(&mut self) {
        let mut new_species_count = 0;
        // Reassign removed genomes.
        for genome in self.drain_incompatible_genomes_from_species() {
            if self.respeciate(
                genome,
                SpeciesID(self.historical_species_count, new_species_count),
            ) {
                new_species_count += 1;
            }
        }
        if new_species_count > 0 {
            self.historical_species_count += 1;
        }
    }

    /// Assigns a genome to a speces based on genetic distance
    /// to species representatives. Returns whether a new species
    /// was created to house the genome.
    fn respeciate(&mut self, genome: Genome, new_species_id: SpeciesID) -> bool {
        // Assign if possible to a currently existing species.
        for species in &mut self.species {
            if Genome::genetic_distance(&genome, species.representative(), &self.genetic_config)
                < self.population_config.distance_threshold
            {
                species.add_genome(genome);
                return false;
            }
        }
        // Create a new species if a compatible one has not been found.
        self.species.push(Species::new(new_species_id, genome));
        true
    }

    /// Removes and returns all genomes incompatible with their
    /// species, iff they are to be adopted.
    fn drain_incompatible_genomes_from_species(&mut self) -> impl Iterator<Item = Genome> {
        let mut incompatibles = vec![];
        let mut rng = rand::thread_rng();
        // Remove all genomes that are incompatible with
        // their current species, iff they are to be adopted.
        for species in &mut self.species {
            let mut i = 0;
            while i < species.genomes.len() {
                if rng.gen::<f32>() < self.population_config.adoption_rate
                    && Genome::genetic_distance(
                        &species.genomes[i],
                        species.representative(),
                        &self.genetic_config,
                    ) >= self.population_config.distance_threshold
                {
                    incompatibles.push(species.genomes.swap_remove(i));
                } else {
                    i += 1;
                }
            }
        }
        incompatibles.into_iter()
    }

    fn remove_extinct_species(&mut self) {
        let mut i = 0;
        while i < self.species.len() {
            if self.species[i].genomes.is_empty() {
                self.species.swap_remove(i);
            } else {
                i += 1;
            }
        }
        self.species.sort_unstable_by_key(|s| s.id());
    }

    /// Resets the population to an initial randomized state.
    /// Used primarily in case of population degeneration, e.g.
    /// when all genomes have a fitness score of 0.
    pub fn reset(&mut self) {
        *self = Population::new(self.population_config.clone(), self.genetic_config.clone());
    }

    /// Returns the currently best-performing genome.
    pub fn champion(&self) -> &Genome {
        self.species
            .iter()
            .flat_map(|s| &s.genomes)
            .max_by(|g1, g2| {
                g1.fitness
                    .partial_cmp(&g2.fitness)
                    .unwrap_or_else(|| panic!("invalid genome fitnesses detected (NaN)"))
            })
            .expect("empty population has no champion")
    }

    /// Returns an iterator over all current genomes.
    pub fn genomes(&self) -> impl Iterator<Item = &Genome> {
        self.species.iter().flat_map(|s| &s.genomes)
    }

    /// Returns an iterator over all current species.
    pub fn species(&self) -> impl Iterator<Item = &Species> {
        self.species.iter()
    }

    /// Returns the current generation number.
    pub fn generation(&self) -> usize {
        self.generation
    }

    pub fn history(&self) -> &History {
        &self.history
    }
}

/// Rounds all values to positive whole numbers
/// while preserving their order and sum, assuming it is also whole.
/// Rounding is done in the manner that minimizes
/// the average error to the original set of values.
fn round_retain_sum(values: &[f32]) -> Vec<usize> {
    let total_sum = values.iter().sum::<f32>().round() as usize;
    let mut truncated: Vec<(usize, usize, f32)> = values
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let u = f.floor();
            let e = f - u;
            (i, u as usize, e)
        })
        .collect();
    let truncated_sum: usize = truncated.iter().map(|(_, u, _)| *u).sum();
    let remainder: usize = total_sum - truncated_sum;
    // Sort in decreasing order of error
    truncated.sort_unstable_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    for (_, u, _) in &mut truncated[..remainder] {
        *u += 1;
    }
    truncated.sort_by_key(|(i, ..)| *i);
    truncated.iter().map(|(_, u, _)| *u).collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn round_retain_sum() {
        let v = [5.2, 9.5, 2.8, 1.3, 2.2, 2.7, 6.3];
        let w = super::round_retain_sum(&v);
        assert_eq!(v.iter().sum::<f32>(), w.iter().sum::<usize>() as f32);
        assert_eq!(w, [5, 10, 3, 1, 2, 3, 6]);
    }
}
