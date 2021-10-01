use crate::populations::PopulationConfig;
use crate::Genome;

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
/// and will thereafter be penalized during reproduction.
///
/// [genetic distance]: PopulationConfig::distance_threshold
/// [`stagnation_threshold`]: PopulationConfig::survival_threshold
#[derive(Debug, Clone)]
pub struct Species<G> {
    id: SpeciesID,
    pub(super) genomes: Vec<G>,
    representative: G,
    stagnation: usize,
    max_fitness: f32,
}

impl<G: Genome + Clone> Species<G> {
    /// Creates a new species with the specified ID and
    /// representative. The representative is also added
    /// to the species' genome pool.
    ///
    /// # Examples
    /// ```
    /// use oxineat::genomics::{GeneticConfig, Genome};
    /// use oxineat::populations::{SpeciesID, Species};
    ///
    /// let species = Species::new(
    ///     SpeciesID(1, 0),
    ///     Genome::new(&GeneticConfig::zero()),
    /// );
    /// ```
    pub fn new(id: SpeciesID, representative: G) -> Species<G> {
        Species {
            id,
            genomes: vec![representative.clone()],
            representative,
            stagnation: 0,
            max_fitness: 0.0,
        }
    }

    /// Returns the species' ID.
    ///
    /// # Examples
    /// ```
    /// use oxineat::genomics::{GeneticConfig, Genome};
    /// use oxineat::populations::{SpeciesID, Species};
    ///
    /// let species = Species::new(
    ///     SpeciesID(1, 0),
    ///     Genome::new(&GeneticConfig::zero()),
    /// );
    ///
    /// assert_eq!(species.id(), SpeciesID(1, 0));
    /// ```
    pub fn id(&self) -> SpeciesID {
        self.id
    }

    /// Returns the species' representative.
    ///
    /// # Examples
    /// ```
    /// use oxineat::genomics::{GeneticConfig, Genome};
    /// use oxineat::populations::{SpeciesID, Species};
    ///
    /// let representative = Genome::new(&GeneticConfig::zero());
    /// let species = Species::new(
    ///     SpeciesID(1, 0),
    ///     representative.clone(),
    /// );
    ///
    /// assert_eq!(species.representative(), &representative);
    /// ```
    pub fn representative(&self) -> &G {
        &self.representative
    }

    /// Returns the genetic distance between the species'
    /// representative and `other`.
    ///
    /// # Examples
    /// ```
    /// use oxineat::genomics::{GeneticConfig, Genome};
    /// use oxineat::populations::{SpeciesID, Species};
    ///
    /// let config = GeneticConfig {
    ///     disjoint_gene_factor: 1.0,
    ///     excess_gene_factor: 1.0,
    ///     common_weight_factor: 0.4,
    ///     ..GeneticConfig::zero()
    /// };
    /// let representative = Genome::new(&config);
    /// let species = Species::new(
    ///     SpeciesID(1, 0),
    ///     representative.clone(),
    /// );
    ///
    /// assert_eq!(species.genetic_distance(&representative, &config), 0.0);
    /// ```
    pub fn genetic_distance<C>(&self, other: &G, config: &C) -> f32
    where
        G: Genome<Config = C>,
    {
        G::genetic_distance(&self.representative, other, config)
    }

    /// Adds a genome to the species.
    /// # Examples
    /// ```
    /// use oxineat::genomics::{GeneticConfig, Genome};
    /// use oxineat::populations::{SpeciesID, Species};
    ///
    /// let config = GeneticConfig::zero();
    /// let mut species = Species::new(
    ///     SpeciesID(1, 0),
    ///     Genome::new(&config),
    /// );
    ///
    /// let genome = Genome::new(&config);
    /// species.add_genome(genome.clone());
    ///
    /// assert!(species.genomes().find(|g| *g == &genome).is_some());
    /// ```
    pub fn add_genome(&mut self, genome: G) {
        self.genomes.push(genome);
    }

    /// Updates the species' record of maximum
    /// fitness, to keep track of stagnation.
    pub(super) fn update_fitness(&mut self) {
        let max_fitness = self
            .genomes
            .iter()
            .map(|g| g.fitness())
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
    ///
    /// # Examples
    /// ```
    /// use oxineat::genomics::{GeneticConfig, Genome};
    /// use oxineat::populations::{SpeciesID, Species};
    ///
    /// let config = GeneticConfig::zero();
    /// let mut species = Species::new(
    ///     SpeciesID(1, 0),
    ///     Genome::new(&config),
    /// );
    ///
    /// let mut g1 = Genome::new(&config);
    /// let mut g2 = Genome::new(&config);
    /// g1.set_fitness(20.0);
    /// g2.set_fitness(30.0);
    /// species.add_genome(g1);
    /// species.add_genome(g2);
    ///
    /// // The species representative + `g1` and `g2`.
    /// assert_eq!(species.adjusted_fitness(), (0.0 + 20.0 + 30.0) / 3.0);
    /// ```
    pub fn adjusted_fitness(&self) -> f32 {
        self.genomes.iter().map(|g| g.fitness()).sum::<f32>() / self.genomes.len() as f32
    }

    /// Returns the number of generations the species
    /// has been stagnated.
    ///
    /// # Examples
    /// ```
    /// use oxineat::genomics::{GeneticConfig, Genome};
    /// use oxineat::populations::{SpeciesID, Species};
    ///
    /// let config = GeneticConfig::zero();
    /// let species = Species::new(
    ///     SpeciesID(1, 0),
    ///     Genome::new(&config),
    /// );
    ///
    /// println!("{}", species.time_stagnated());
    /// ```
    pub fn time_stagnated(&self) -> usize {
        self.stagnation
    }

    /// Returns an iterator over the species' members.
    ///
    /// # Examples
    /// ```
    /// use oxineat::genomics::{GeneticConfig, Genome};
    /// use oxineat::populations::{SpeciesID, Species};
    ///
    /// let config = GeneticConfig::zero();
    /// let mut species = Species::new(
    ///     SpeciesID(1, 0),
    ///     Genome::new(&config),
    /// );
    ///
    /// for g in species.genomes() {
    ///     println!("{}", g);
    /// }
    /// ```
    pub fn genomes(&self) -> impl Iterator<Item = &G> {
        self.genomes.iter()
    }

    /// Returns the currently best-performing genome.
    ///
    /// # Examples
    /// ```
    /// use oxineat::genomics::{GeneticConfig, Genome};
    /// use oxineat::populations::{SpeciesID, Species};
    ///
    /// let config = GeneticConfig::zero();
    ///
    /// let mut g1 = Genome::new(&config);
    /// let mut g2 = Genome::new(&config);
    /// let mut g3 = Genome::new(&config);
    /// g1.set_fitness(5.0);
    /// g2.set_fitness(20.0);
    /// g3.set_fitness(10.0);
    ///
    /// let mut species = Species::new(
    ///     SpeciesID(1, 0),
    ///     g1,
    /// );
    /// species.add_genome(g2.clone());
    /// species.add_genome(g3);
    ///
    /// assert_eq!(species.champion(), &g2);
    /// ```
    pub fn champion(&self) -> &G {
        self.genomes
            .iter()
            .max_by(|g1, g2| {
                g1.fitness()
                    .partial_cmp(&g2.fitness())
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
