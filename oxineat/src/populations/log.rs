use super::{Population, SpeciesID};

use crate::genome::{Genome, InnovationHistory};

use std::fmt;

/// Defines different possible reporting levels for logging.
#[derive(Clone, Copy, Debug)]
pub enum ReportingLevel {
    /// Clones the entire population.
    AllGenomes,
    /// Clones species and their champions.
    SpeciesChampions,
    /// Clones only the population champion.
    PopulationChampion,
    /// Clones no genomes.
    NoGenomes,
}

/// A snapshot of a population.
#[derive(Clone, Debug)]
pub struct Log<G> {
    pub generation_number: usize,
    pub generation_sample: GenerationMemberRecord<G>,
    pub species_count: usize,
    pub fitness: Stats,
    pub genome_stats: Vec<(String, Stats)>,
}

impl<G: Genome> fmt::Display for Log<G> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Log {{\n\
            \tgeneration_number: {:?}\n\
            \tspecies_count: {:?}\n\
            \tfitness: {:?}\n\
            {}
            }}",
            &self.generation_number,
            &self.species_count,
            &self.fitness,
            self.genome_stats
                .iter()
                .map(|(name, stats)| format!("\t{}: {:?}\n", name, stats))
                .collect::<Vec<_>>()
                .join("")
        )
    }
}

/// A struct for reporting basic statistical data.
#[derive(Clone, Debug)]
pub struct Stats {
    pub maximum: f32,
    pub minimum: f32,
    pub mean: f32,
    pub median: f32,
}

impl Stats {
    /// Returns statistics about numbers in a sequence.
    ///
    /// # Examples
    /// ```
    /// use oxineat::populations::Stats;
    ///
    ///
    /// let stats = Stats::from([-2.0, -1.0, 0.5, 1.0, 1.5].iter().copied());
    /// assert_eq!(stats.maximum, 1.5);
    /// assert_eq!(stats.minimum, -2.0);
    /// assert_eq!(stats.mean, 0.0);
    /// assert_eq!(stats.median, 0.5);
    /// ```
    pub fn from(data: impl Iterator<Item = f32>) -> Stats {
        let mut data: Vec<f32> = data.collect();
        let mid = data.len() / 2;
        let (mut max, mut min, mut sum) = (f32::MIN, f32::MAX, 0.0);
        for d in &data {
            max = d.max(max);
            min = d.min(min);
            sum += d;
        }
        let mean = sum / data.len() as f32;
        let mut median = *data
            .select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap())
            .1;
        if data.len() % 2 == 0 {
            median = (median
                + *data
                    .select_nth_unstable_by(mid + 1, |a, b| a.partial_cmp(b).unwrap())
                    .1)
                / 2.0;
        }
        Stats {
            maximum: max,
            minimum: min,
            mean,
            median,
        }
    }
}

/// A reporting-level dependant store
/// of genomes from a population.
#[derive(Clone, Debug)]
pub enum GenerationMemberRecord<G> {
    /// Species IDs, genomes and stagnation level.
    Species(Vec<(SpeciesID, Vec<G>, usize)>),
    /// Only species IDs, species champions, and stagnation level.
    SpeciesChampions(Vec<(SpeciesID, G, usize)>),
    /// Only population champion.
    PopulationChampion(G),
    /// Empty.
    None,
}

/// A log of the evolution of a population over time.
#[derive(Clone, Debug)]
pub struct EvolutionLogger<G> {
    reporting_level: ReportingLevel,
    logs: Vec<Log<G>>,
}

impl<G: Genome + Clone> EvolutionLogger<G> {
    /// Returns a logger with the appropiate reporting level.
    ///
    /// # Examples
    /// ```
    /// use oxineat::populations::{EvolutionLogger, ReportingLevel};
    ///
    /// let logger = EvolutionLogger::new(ReportingLevel::NoGenomes);
    /// ```
    pub fn new(reporting_level: ReportingLevel) -> EvolutionLogger<G> {
        EvolutionLogger {
            reporting_level,
            logs: vec![],
        }
    }

    /// Store a snapshot of a population.
    ///
    /// # Examples
    /// ```
    /// use oxineat::genomics::GeneticConfig;
    /// use oxineat::populations::{Population, PopulationConfig};
    /// use oxineat::populations::{EvolutionLogger, ReportingLevel};
    ///
    /// let mut logger = EvolutionLogger::new(ReportingLevel::NoGenomes);
    /// let mut population = Population::new(PopulationConfig::zero(), GeneticConfig::zero());
    ///
    /// // Do something with the population...
    /// population.evolve();
    ///
    /// // Then log a snapshot of the population.
    /// logger.log(&population);
    /// ```
    pub fn log<C, H, GSE, const N: usize>(
        &mut self,
        population: &Population<C, H, G>,
        genome_stat_extractor: &GSE,
        stat_names: [&str; N],
    ) where
        H: InnovationHistory<Config = C>,
        G: Genome<InnovationHistory = H, Config = C>,
        GSE: Fn(&G) -> [f32; N],
    {
        let (fitnesses, stats): (Vec<f32>, Vec<[f32; N]>) = population
            .species
            .iter()
            .flat_map(|s| s.genomes.iter())
            .map(|g| (g.fitness(), genome_stat_extractor(g)))
            .unzip();
        let stats = stat_names
            .iter()
            .cloned()
            .map(String::from)
            .zip(unzip_n_vecs(stats.into_iter()))
            .map(|(name, data)| (name, Stats::from(data.into_iter())))
            .collect();
        self.logs.push(Log {
            generation_number: population.generation(),
            generation_sample: match self.reporting_level {
                ReportingLevel::AllGenomes => GenerationMemberRecord::Species(
                    population
                        .species()
                        .map(|s| (s.id(), s.genomes().cloned().collect(), s.time_stagnated()))
                        .collect(),
                ),
                ReportingLevel::SpeciesChampions => GenerationMemberRecord::SpeciesChampions(
                    population
                        .species()
                        .map(|s| (s.id(), s.champion().clone(), s.time_stagnated()))
                        .collect(),
                ),
                ReportingLevel::PopulationChampion => {
                    GenerationMemberRecord::PopulationChampion(population.champion().clone())
                }
                ReportingLevel::NoGenomes => GenerationMemberRecord::None,
            },
            species_count: population.species().count(),
            fitness: Stats::from(fitnesses.into_iter()),
            genome_stats: stats,
        })
    }

    // gene_count: Stats::from(&mut stats.iter().map(|(g, _, _)| *g)),
    // node_count: Stats::from(&mut stats.iter().map(|(_, n, _)| *n)),
    // max_gene_innovation: population.history().max_gene_innovation(),
    // max_node_innovation: population.history().max_node_innovation()

    /// Iterate over all logged snapshots.
    ///
    /// # Examples
    /// ```
    /// use oxineat::populations::{EvolutionLogger, ReportingLevel};
    ///
    /// let logger = EvolutionLogger::new(ReportingLevel::AllGenomes);
    /// // Log some stuff... then
    /// for log in logger.iter() {
    ///     println!("{}", log);
    /// }
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = &Log<G>> {
        self.logs.iter()
    }
}

fn unzip_n_vecs<T: Clone, const N: usize>(mut iter: impl Iterator<Item = [T; N]>) -> Vec<Vec<T>> {
    let mut vecs = vec![Vec::default(); N];
    while let Some(items) = iter.next() {
        let mut i = 0;
        for item in items {
            vecs[i].push(item);
            i += 1;
        }
    }
    vecs
}
