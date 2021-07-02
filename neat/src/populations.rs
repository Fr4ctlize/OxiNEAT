mod config;
mod log;
mod species;

pub use config::PopulationConfig;
pub use log::{EvolutionLogger, ReportingLevel};
pub use species::{Species, SpeciesID};

use crate::genomes::{GeneticConfig, Genome, History};

use std::collections::HashMap;

use rand::prelude::{IteratorRandom, Rng, SliceRandom};

/// A Population is a collection of genomes grouped
/// into species that can be evolved using a genome
/// evaluation function as the source of selective
/// pressure.
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
            historical_species_count: 0,
            population_config,
            genetic_config,
        }
    }

    /// Evaluates the fitness of each genome in the
    /// population using the passed evaluator.
    ///
    /// The return value of the evaluation function
    /// should be positive.
    pub fn evaluate_fitness(&mut self, evaluator: fn(&Genome) -> f32) {
        for genome in self.species.iter_mut().flat_map(|s| &mut s.genomes) {
            genome.fitness = evaluator(&genome);
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
    /// [adoption rate]: crate::populations::PopConfig::adoption_rate
    /// [Nodine, T., 2010]: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.175.2884&rep=rep1&type=pdf
    pub fn evolve(&mut self) {
        match self.allot_offspring() {
            Ok(allotted_offspring) => {
                self.generate_offspring(&allotted_offspring);
                self.respeciate_all();
            }
            Err(_) => self.reset(),
        }
    }

    /// Allot the number of offspring for each species,
    /// based on proportional adjusted species fitness
    /// and stagnation status.
    ///
    /// # Errors
    ///
    /// Returns an error if all genome's fitnesses are 0.
    fn allot_offspring(&self) -> Result<Vec<usize>, ()> {
        let fitnesses: Vec<f32> = self
            .species
            .iter()
            .map(|s| {
                if s.time_stagnated() >= self.population_config.stagnation_threshold.get() {
                    s.adjusted_fitness() * (1.0 - self.population_config.stagnation_penalty)
                } else {
                    s.adjusted_fitness()
                }
            })
            .collect();
        let fitness_sum: f32 = fitnesses.iter().copied().sum();
        if fitness_sum == 0.0 {
            return Err(());
        }
        let fractional_allotments: Vec<f32> = fitnesses
            .iter()
            .map(|f| *f / fitness_sum * self.population_config.population_size.get() as f32)
            .collect();
        Ok(round_retain_sum(&fractional_allotments))
    }

    /// Generates each species' assigned offspring,
    /// keeping the [species' elite] and mating the
    /// [top performers].
    ///
    /// Has a [small chance] of selecting a partner
    /// from another species.
    ///
    /// Offspring are assigned randomly to the species
    /// of either one of the parents.
    ///
    /// [species' elite]: crate::populations::PopConfig::elitism
    /// [top performers]: crate::populations::PopConfig::survival_threshold
    /// [small chance]: crate::populations::PopConfig::interspecies_mating_chance
    fn generate_offspring(&mut self, allotted_offspring: &[usize]) {
        for species in &mut self.species {
            species
                .genomes
                .sort_unstable_by(|g1, g2| g1.fitness.partial_cmp(&g2.fitness).unwrap());
        }

        let mut species_offspring: HashMap<SpeciesID, Vec<Genome>> =
            self.species.iter().map(|s| (s.id(), vec![])).collect();

        for (allotted_offspring, current_species) in allotted_offspring.iter().zip(&self.species) {
            // Number of genomes directly copied to next generation.
            let elite = ((self.population_config.elitism * current_species.genomes.len() as f32)
                .ceil() as usize)
                .max(*allotted_offspring);
            // Number of genomes eligible for mating.
            let survivors = (self.population_config.survival_threshold
                * current_species.genomes.len() as f32)
                .ceil() as usize;
            // Number of genomes derived from mating.
            let offspring = (*allotted_offspring - elite).max(0);
            // Genomes eligible for mating.
            let eligible_parents: Vec<&Genome> =
                current_species.genomes[..survivors].iter().collect();

            // Copy elites.
            for g in &current_species.genomes[0..elite] {
                species_offspring
                    .get_mut(&current_species.id())
                    .unwrap()
                    .push(g.clone());
            }
            let mut rng = rand::thread_rng();
            // Mate parents
            for _ in 0..offspring {
                let mut parents = eligible_parents.choose_multiple(&mut rng, 2);
                let (parent1, mut parent2) = (*parents.next().unwrap(), *parents.next().unwrap());
                let mut parent2_species = current_species.id();
                // Possibly choose other parent from other species
                if self.species.len() > 1
                    && rng.gen::<f32>() < self.population_config.interspecies_mating_chance
                {
                    let other_species = self
                        .species
                        .iter()
                        .filter(|s| s.id() != current_species.id())
                        .choose(&mut rng)
                        .unwrap();
                    parent2 = other_species.genomes.choose(&mut rng).unwrap();
                    parent2_species = other_species.id();
                }
                let child_species = if rng.gen::<usize>() % 2 == 0 {
                    current_species.id()
                } else {
                    parent2_species
                };
                species_offspring.get_mut(&child_species).unwrap().push(
                    parent1
                        .mate_with(parent2, &mut self.history, &self.genetic_config)
                        .unwrap(),
                );
            }
        }

        for species in &mut self.species {
            species.genomes = species_offspring.remove(&species.id()).unwrap();
        }
    }

    /// Reassigns each genome to a species based on genetic
    /// distance to species representatives. Has a 1-[adoption rate]
    /// chance of not modifying a genome's assigned species.
    fn respeciate_all(&mut self) {
        let mut new_species_count = 0;
        // Reassign removed genomes.
        for genome in self.drain_incompatibles() {
            if self.respeciate(
                genome,
                SpeciesID(self.historical_species_count, new_species_count),
            ) {
                new_species_count += 1;
            }
        }
    }

    /// Assigns a genome to a speces based on genetic distance
    /// to species representatives. Returns whether a new species
    /// was created to house the genome.
    fn respeciate(&mut self, genome: Genome, new_species_id: SpeciesID) -> bool {
        // Assign if possible to a currently existing species.
        for species in &mut self.species {
            if genome.genetic_distance_to(species.representative(), &self.genetic_config)
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
    fn drain_incompatibles(&mut self) -> impl Iterator<Item = Genome> {
        let mut incompatibles = vec![];
        let mut rng = rand::thread_rng();
        // Remove all genomes that are incompatible with
        // their current species, iff they are to be adopted.
        for species in &mut self.species {
            let mut i = 0;
            while i < species.genomes.len() {
                if rng.gen::<f32>() < self.population_config.adoption_rate
                    && species.genomes[i]
                        .genetic_distance_to(species.representative(), &self.genetic_config)
                        >= self.population_config.distance_threshold
                {
                    incompatibles.push(species.genomes.swap_remove(i));
                } else {
                    i += 1;
                }
            }
        }
        incompatibles.into_iter()
    }

    /// Resets the population to an initial randomized state.
    /// Used primarily in case of population degeneration, e.g.
    /// when all genomes have a fitness score of 0.
    fn reset(&mut self) {
        *self = Population::new(self.population_config.clone(), self.genetic_config.clone());
    }

    /// Returns the currently best-performing genome.
    pub fn champion(&self) -> &Genome {
        &self
            .species
            .iter()
            .flat_map(|s| &s.genomes)
            .max_by(|g1, g2| g1.fitness.partial_cmp(&g2.fitness).unwrap())
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
