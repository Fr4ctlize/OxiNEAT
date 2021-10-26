use std::num::NonZeroUsize;

/// Configuration data for population generation
/// and evolution.
/// 
/// # Note
/// All quantities expressing probabilities
/// should be in the range [0.0, 1.0]. Using
/// values that are not in this bound may result
/// in odd behaviours and/or incorrect programs.
#[derive(Clone, Debug)]
pub struct PopulationConfig {
    /// Size of the population.
    pub size: NonZeroUsize,
    /// Genetic distance threshold, beyond which
    /// genomes are considered as belonging to
    /// different species.
    pub distance_threshold: f32,
    /// Top n of each species which is copied
    /// as-is to the next generation.
    pub elitism: usize,
    /// Top % of each species which can participate
    /// in mating.
    pub survival_threshold: f32,
    /// Chance that a child will be speciated
    /// instead of being directly assigned to its
    /// parent's species.
    pub adoption_rate: f32,
    /// Chance that offspring will be the result
    /// of sexual reproduction (as opposed to asexual).
    pub sexual_reproduction_chance: f32,
    /// Chance that genomes from different species
    /// will be selected to mate.
    pub interspecies_mating_chance: f32,
    /// Number of generations without a fitness increase
    /// before a species is considered _stagnated_.
    pub stagnation_threshold: NonZeroUsize,
    /// Offspring allotment penalty for stagnation.
    /// Stagnated species will receive this percentage
    /// fewer offspring.
    pub stagnation_penalty: f32,
    // /// Desired amount of species in the population.
    // /// If zero, no species control will take effect.
    // pub target_species: usize,
}

impl PopulationConfig {
    /// Returns a "zero-valued" default configuration.
    /// All values are 0, empty, or in the case of
    /// `NonZeroUsize`s, 1.
    ///
    /// # Note
    /// This value is not suitable for use in most experiments.
    /// It is meant as a way to abbreviate configuration
    /// instantiation, or to fill in unused values.
    ///
    /// # Examples
    /// ```
    /// use oxineat::PopulationConfig;
    ///
    /// let cfg1 = PopulationConfig::zero();
    ///
    /// let cfg2 = PopulationConfig {
    ///     // Specify some values here...
    ///     stagnation_penalty: 0.5,
    ///     // Default the rest...
    ///     ..PopulationConfig::zero()
    /// };
    /// ```
    pub const fn zero() -> PopulationConfig {
        PopulationConfig {
            // SAFETY: 1 is a valid NonZeroUsize. Replace this with
            // NonZeroUsize::new(1).unwrap() once const Option::unwrap 
            // becomes stable.
            size: unsafe { NonZeroUsize::new_unchecked(1) },
            distance_threshold: 0.0,
            elitism: 0,
            survival_threshold: 0.0,
            adoption_rate: 0.0,
            sexual_reproduction_chance: 0.0,
            interspecies_mating_chance: 0.0,
            // SAFETY: 1 is a valid NonZeroUsize. Replace this with
            // NonZeroUsize::new(1).unwrap() once const Option::unwrap 
            // becomes stable.
            stagnation_threshold: unsafe { NonZeroUsize::new_unchecked(1) },
            stagnation_penalty: 0.0,
        }
    }
}
