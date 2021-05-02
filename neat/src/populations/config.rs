/// Configuration data for population generation
/// and evolution.
#[derive(Clone, Debug)]
pub struct PopConfig {
    /// Size of the population.
    pub population_size: usize,
    /// Genetic distance threshold, beyond which
    /// genomes are considered as belonging to
    /// different species.
    pub distance_threshold: f32,
    /// Top % of each species which is copied
    /// as-is to the next generation.
    pub elitism: f32,
    /// Top % of each species which can participate
    /// in mating.
    pub survival_threshold: f32,
    /// Chance that a child will be speciated
    /// instead of being directly assigned to its
    /// parent's species.
    pub adoption_rate: f32,
    /// Chance that genomes from different species
    /// will be selected to mate.
    pub interspecies_mating_chance: f32,
    /// Number of generations without a fitness increase
    /// before a species is considered _stagnated_.
    pub stagnation_threshold: usize,
    /// Desired amount of species in the population.
    /// If zero, no species control will take effect.
    pub target_species: usize,
}
