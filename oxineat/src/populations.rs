//! A Population is a collection of genomes.
//! These are grouped into species, which can
//! be evolved using a genome evaluation function
//! as the source of selective pressure.
mod config;
mod errors;
pub mod logging;
mod offspring_factory;
mod rt_population;
mod species;

use crate::{Genome, InnovationHistory};
pub use config::PopulationConfig;
use errors::*;
use offspring_factory::OffspringFactory;
pub use species::{Species, SpeciesID};

use rand::prelude::Rng;
use serde::{Deserialize, Serialize};

/// A population of genomes.
#[derive(Serialize, Deserialize)]
pub struct Population<C, H, G> {
    species: Vec<Species<G>>,
    history: H,
    generation: usize,
    historical_species_count: usize,
    population_config: PopulationConfig,
    genetic_config: C,
}

impl<C, H, G> Population<C, H, G>
where
    G: Genome<InnovationHistory = H, Config = C> + Clone,
{
    /// Creates a new population using the passed configurations.
    ///
    /// The type of `genetic_config` depends on the implementation
    /// of [`Genome`], and is effectively opaque to the population.
    ///
    /// [`Genome`]: crate::Genome
    ///
    /// # Examples
    /// ```
    /// # use oxineat_nn::genomics::{GeneticConfig, NNGenome as G};
    /// use oxineat::{Population, PopulationConfig};
    ///
    /// let pop_config = PopulationConfig {
    ///     // Set desired configuration
    ///     ..PopulationConfig::zero()
    /// };
    /// # let genetic_config = GeneticConfig::zero();
    ///
    /// // With `G` a suitable type implementing `Genome`...
    /// let population = Population::<_, _, G>::new(pop_config, genetic_config);
    /// ```
    pub fn new(population_config: PopulationConfig, genetic_config: C) -> Population<C, H, G>
    where
        H: InnovationHistory<Config = C>,
    {
        Population {
            species: {
                let mut s0 = Species::new(SpeciesID(0, 0), G::new(&genetic_config));
                s0.genomes
                    .extend((1..population_config.size.get()).map(|_| G::new(&genetic_config)));
                vec![s0]
            },
            history: H::new(&genetic_config),
            generation: 0,
            historical_species_count: 1,
            population_config,
            genetic_config,
        }
    }

    /// Creates a new population using the passed configurations,
    /// and seeds it with the specified genomes. Each array of genomes
    /// is assigned to its own species, with the first as the
    /// species representative. If the number of seed genomes is not
    /// as large as the configured population size, the remaining
    /// space is filled as during normal population
    ///
    /// Returns `None` if either the configured population size is
    /// lesser than the number of seed genomes, or any of the genomes
    /// are incompatible with the specified genetic config, as established
    /// by [`Genome::conforms_to`].
    ///
    /// The type of `genetic_config` depends on the implementation
    /// of [`Genome`], and is effectively opaque to the population.
    ///
    /// [`Genome`]: crate::Genome
    /// [`Genome::conforms_to`]: crate::Genome::conforms_to
    ///
    /// # Examples
    /// ```
    /// # use oxineat_nn::genomics::{GeneticConfig, NNGenome};
    /// use oxineat::{Population, PopulationConfig};
    ///
    /// let pop_config = PopulationConfig {
    ///     // Set desired configuration
    ///     size: std::num::NonZeroUsize::new(100).unwrap(),
    ///     ..PopulationConfig::zero()
    /// };
    /// # let genetic_config = GeneticConfig {
    /// #     weight_bound: 1.0,
    /// #     ..GeneticConfig::zero()
    /// # };
    /// # let g1 = NNGenome::new(&genetic_config);
    /// # let g2 = NNGenome::new(&genetic_config);
    /// # let g3 = NNGenome::new(&genetic_config);
    ///
    /// // With `seed` a slice of a suitable type implementing `Genome`...
    /// let population = Population::new_seeded(vec![vec![g1, g2], vec![g3]], pop_config, genetic_config).unwrap();
    ///
    /// # assert_eq!(population.species().count(), 3);
    /// # assert_eq!(population.species().map(|s| s.genomes().count()).collect::<Vec<_>>(), vec![97, 2, 1]);
    /// ```
    pub fn new_seeded(
        genomes: Vec<Vec<G>>,
        population_config: PopulationConfig,
        genetic_config: C,
    ) -> Option<Population<C, H, G>>
    where
        H: InnovationHistory<Config = C>,
    {
        let seed_count = genomes.iter().map(|v| v.len()).sum();
        if population_config.size.get() < seed_count {
            return None;
        }

        if !genomes
            .iter()
            .map(|v| v.iter())
            .flatten()
            .all(|g| g.conforms_to(&genetic_config))
        {
            return None;
        }

        let species: Vec<Species<G>> = std::iter::once({
            let mut s0 = Species::new(SpeciesID(0, 0), G::new(&genetic_config));
            s0.genomes.extend(
                (1..population_config.size.get() - seed_count).map(|_| G::new(&genetic_config)),
            );
            s0
        })
        .chain(
            genomes
                .into_iter()
                .filter(|v| !v.is_empty())
                .enumerate()
                .map(|(i, mut genomes)| {
                    let mut s = Species::new(SpeciesID(0, i + 1), genomes.swap_remove(0));
                    for g in genomes {
                        s.add_genome(g.clone())
                    }
                    s
                }),
        )
        .collect();
        let species_count = species.len();

        Some(Population {
            species: species,
            history: H::new(&genetic_config),
            generation: 0,
            historical_species_count: species_count,
            population_config,
            genetic_config,
        })
    }

    /// Evaluates the fitness of each genome in the
    /// population using the passed evaluator.
    ///
    /// The return value of the evaluation function
    /// should be positive.
    ///
    /// # Examples
    /// ```
    /// # use oxineat_nn::genomics::GeneticConfig;
    /// # use oxineat_nn::networks::FunctionApproximatorNetwork;
    /// use oxineat::{Population, PopulationConfig};
    ///
    /// # let genetic_config = GeneticConfig::zero();
    /// let mut population = Population::new(
    ///     PopulationConfig::zero(),
    ///     genetic_config,
    /// );
    ///
    /// population.evaluate_fitness(|g| {
    ///     # let mut network = FunctionApproximatorNetwork::from::<1>(g);
    ///     # // Networks with outputs closer to 0 are given higher scores.
    ///     # let fitness = (1.0 - (network.evaluate_at(&[1.0])[0] - 0.0)).powf(2.0);
    ///     // Compute genome's fitness...
    ///     return fitness;
    /// });
    /// ```
    pub fn evaluate_fitness<E>(&mut self, mut evaluator: E)
    where
        E: FnMut(&G) -> f32,
    {
        for genome in self.species.iter_mut().flat_map(|s| &mut s.genomes) {
            let fitness = evaluator(genome);
            assert!(fitness >= 0.0, "fitness function return a negative value");
            genome.set_fitness(fitness);
        }
    }

    /// Evolves the population by mating the best performing
    /// genomes of each species, and re-speciating genomes
    /// as appropiate.
    ///
    /// If the [adoption rate] is less than 1, offspring
    /// will have a chance of being placed into their parent's
    /// species without speciation, which seems to help NEAT
    /// find solutions faster. (See [[Nodine, T., 2010]].)
    ///
    /// # Panics
    /// This function will panic if
    /// `config.survival_threshold == 0.0` and
    /// `config.elitism` isn't high enough to cover
    /// the number of offspring assigned to a species,
    /// as there would be no parents from which to generate
    /// offspring.
    ///
    /// # Errors
    /// Returns an error if the population has become degenerate
    /// (zero maximum fitness or all genomes are culled due to
    /// stagnation).
    ///
    /// [adoption rate]: PopulationConfig::adoption_rate
    /// [Nodine, T., 2010]: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.175.2884&rep=rep1&type=pdf
    ///
    /// # Examples
    /// ```
    /// # use oxineat_nn::genomics::{GeneticConfig, NNGenome as G};
    /// # use oxineat_nn::networks::FunctionApproximatorNetwork;
    /// use oxineat::{Population, PopulationConfig};
    ///
    /// # let genetic_config = GeneticConfig::zero();
    /// // With `G` a suitable type implementing `Genome`...
    /// let mut population = Population::new(
    ///     PopulationConfig {
    ///         survival_threshold: 1.0,
    ///         ..PopulationConfig::zero()
    ///     },
    ///     genetic_config,
    /// );
    ///
    /// population.evaluate_fitness(|g| {
    ///     # let mut network = FunctionApproximatorNetwork::from::<1>(g);
    ///     # // Networks with outputs closer to 0 are given higher scores.
    ///     # let fitness = (1.0 - (network.evaluate_at(&[1.0])[0] - 0.0)).powf(2.0);
    ///     // Compute genome's fitness...
    ///     return fitness;
    /// });
    ///
    /// if let Err(e) = population.evolve() {
    ///     eprintln!("{}", e);
    /// }
    /// ```
    pub fn evolve(&mut self) -> Result<(), Box<dyn std::error::Error>>
    where
        H: InnovationHistory<Config = C>,
    {
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

    /// Collects all species adjusted fitnesses.
    /// Returns `None` if population fitness sum is 0.
    fn get_species_adjusted_fitness(&self) -> Option<Vec<f32>> {
        let fitnesses = self.species_fitness_with_stagnation_penalty();
        let fitness_sum: f32 = fitnesses.iter().copied().sum();
        if fitness_sum == 0.0 {
            return None;
        }
        Some(
            fitnesses
                .iter()
                .map(|f| *f / fitness_sum * self.population_config.size.get() as f32)
                .collect(),
        )
    }

    /// Returns each species' adjusted fitness,
    /// with stagnation penalties applied.
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
    fn generate_offspring(&mut self, allotted_offspring: &[usize])
    where
        H: InnovationHistory<Config = C>,
    {
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

    /// Sorts each species' members by fitness in descending order.
    fn sort_species_members_by_decreasing_fitness(&mut self) {
        for species in &mut self.species {
            species.genomes.sort_unstable_by(|g1, g2| {
                g2.fitness()
                    .partial_cmp(&g1.fitness())
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
    fn respeciate(&mut self, genome: G, new_species_id: SpeciesID) -> bool {
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
    fn drain_incompatible_genomes_from_species(&mut self) -> impl Iterator<Item = G> {
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

    /// Removes all extinct (0 assigned offspring)
    /// species from the population.
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
    ///
    /// # Examples
    /// ```
    /// # use oxineat_nn::genomics::{GeneticConfig, NNGenome as G};
    /// use oxineat::{Population, PopulationConfig};
    ///
    /// # let genetic_config = GeneticConfig::zero();
    /// // With `G` a suitable type implementing `Genome`...
    /// let mut population = Population::<_, _, G>::new(
    ///     PopulationConfig::zero(),
    ///     genetic_config,
    /// );
    ///
    /// // Evolve the population on some task, until
    /// // population.evolve() returns an Err.
    /// population.reset();
    /// ```
    pub fn reset(&mut self)
    where
        C: Clone,
        H: InnovationHistory<Config = C>,
    {
        *self = Population::new(self.population_config.clone(), self.genetic_config.clone());
    }

    /// Returns the currently best-performing genome.
    ///
    /// # Examples
    /// ```
    /// # use oxineat_nn::genomics::{GeneticConfig, NNGenome as G};
    /// use oxineat::{Population, PopulationConfig};
    ///
    /// # let genetic_config = GeneticConfig::zero();
    /// // With `G` a suitable type implementing `Genome`...
    /// let mut population = Population::<_, _, G>::new(
    ///     PopulationConfig {
    ///         size: std::num::NonZeroUsize::new(20).unwrap(),
    ///         ..PopulationConfig::zero()
    ///     },
    ///     genetic_config,
    /// );
    ///
    /// let mut fitness = 0.0;
    /// population.evaluate_fitness(move |g| {
    ///     fitness += 10.0;
    ///     fitness
    /// });
    ///
    /// assert_eq!(population.champion().fitness(), 20.0 * 10.0);
    /// ```
    pub fn champion(&self) -> &G {
        self.species
            .iter()
            .flat_map(|s| &s.genomes)
            .max_by(|g1, g2| {
                g1.fitness()
                    .partial_cmp(&g2.fitness())
                    .unwrap_or_else(|| panic!("invalid genome fitnesses detected (NaN)"))
            })
            .expect("empty population has no champion")
    }

    /// Returns an iterator over all current genomes.
    ///
    /// # Examples
    /// ```
    /// # use oxineat_nn::genomics::{GeneticConfig, NNGenome as G};
    /// use oxineat::{Population, PopulationConfig};
    ///
    /// # let genetic_config = GeneticConfig::zero();
    /// // With `G` a suitable type implementing `Genome`...
    /// let population = Population::<_, _, G>::new(PopulationConfig::zero(), genetic_config);
    ///
    /// for genome in population.genomes() {
    ///     println!("{}", genome);
    /// }
    /// ```
    pub fn genomes(&self) -> impl Iterator<Item = &G> {
        self.species.iter().flat_map(|s| &s.genomes)
    }

    /// Returns an iterator over all current species.
    ///
    /// # Examples
    /// ```
    /// # use oxineat_nn::genomics::{GeneticConfig, NNGenome as G};
    /// use oxineat::{Population, PopulationConfig};
    ///
    /// # let genetic_config = GeneticConfig::zero();
    /// // With `G` a suitable type implementing `Genome`...
    /// let population = Population::<_, _, G>::new(PopulationConfig::zero(), genetic_config);
    ///
    /// for species in population.species() {
    ///     println!(
    ///         "Species {:?} contains the following genomes: {:?}",
    ///         species.id(),
    ///         species.genomes().cloned().collect::<Vec<_>>()
    ///     );
    /// }
    /// ```
    pub fn species(&self) -> impl Iterator<Item = &Species<G>> {
        self.species.iter()
    }

    /// Returns the current generation number.
    ///
    /// # Examples
    /// ```
    /// # use oxineat_nn::genomics::{GeneticConfig, NNGenome as G};
    /// use oxineat::{Population, PopulationConfig};
    ///
    /// # let genetic_config = GeneticConfig::zero();
    /// // With `G` a suitable type implementing `Genome`...
    /// let population = Population::<_, _, G>::new(PopulationConfig::zero(), genetic_config);
    ///
    /// assert_eq!(population.generation(), 0);
    /// ```
    pub fn generation(&self) -> usize {
        self.generation
    }

    /// Returns the population's innovation history.
    ///
    /// # Examples
    /// ```
    /// # use oxineat_nn::genomics::{GeneticConfig, NNGenome as G};
    /// use oxineat::{Population, PopulationConfig};
    ///
    /// # let genetic_config = GeneticConfig::zero();
    /// // With `G` a suitable type implementing `Genome`...
    /// let population = Population::<_, _, G>::new(PopulationConfig::zero(), genetic_config);
    /// let history = population.history();
    /// ```
    pub fn history(&self) -> &H {
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
        let v = [
            5.2,
            9.5,
            2.8,
            1.3,
            2.2,
            2.7,
            6.3,
            1.0000000000001,
            0.9999999999999,
        ];
        let w = super::round_retain_sum(&v);
        assert_eq!(v.iter().sum::<f32>(), w.iter().sum::<usize>() as f32);
        assert_eq!(w, [5, 10, 3, 1, 2, 3, 6, 1, 1]);
    }
}
