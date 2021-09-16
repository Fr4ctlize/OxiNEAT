use super::{Population, SpeciesID};
use crate::genomics::Genome;
use crate::Innovation;

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
pub struct Log {
    pub generation_number: usize,
    pub generation_sample: GenerationMemberRecord,
    pub species_count: usize,
    pub fitness: Stats,
    pub gene_count: Stats,
    pub node_count: Stats,
    pub max_gene_innovation: Innovation,
    pub max_node_innovation: Innovation,
}

impl fmt::Display for Log {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Log {{\n\
            \tgeneration_number: {:?}\n\
            \tspecies_count: {:?}\n\
            \tfitness: {:?}\n\
            \tgene_count: {:?}\n\
            \tnode_count: {:?}\n\
            \tmax_gene_innovation: {:?}\n\
            \tmax_node_innovation: {:?}\n\
            }}",
            &self.generation_number,
            &self.species_count,
            &self.fitness,
            &self.gene_count,
            &self.node_count,
            &self.max_gene_innovation,
            &self.max_node_innovation
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
pub enum GenerationMemberRecord {
    /// Species IDs, genomes and stagnation level.
    Species(Vec<(SpeciesID, Vec<Genome>, usize)>),
    /// Only species IDs, species champions, and stagnation level.
    SpeciesChampions(Vec<(SpeciesID, Genome, usize)>),
    /// Only population champion.
    PopulationChampion(Genome),
    /// Empty.
    None,
}

/// A log of the evolution of a population over time.
#[derive(Clone, Debug)]
pub struct EvolutionLogger {
    reporting_level: ReportingLevel,
    logs: Vec<Log>,
}

impl EvolutionLogger {
    /// Returns a logger with the appropiate reporting level.
    /// 
    /// # Examples
    /// ```
    /// use oxineat::populations::{EvolutionLogger, ReportingLevel};
    /// 
    /// let logger = EvolutionLogger::new(ReportingLevel::NoGenomes);
    /// ```
    pub fn new(reporting_level: ReportingLevel) -> EvolutionLogger {
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
    pub fn log(&mut self, population: &Population) {
        let stats: Vec<(f32, f32, f32)> = population
            .species
            .iter()
            .flat_map(|s| s.genomes.iter())
            .map(|g| {
                (
                    g.genes().count() as f32,
                    g.nodes().count() as f32,
                    g.fitness,
                )
            })
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
            fitness: Stats::from(&mut stats.iter().map(|(_, _, f)| *f)),
            gene_count: Stats::from(&mut stats.iter().map(|(g, _, _)| *g)),
            node_count: Stats::from(&mut stats.iter().map(|(_, n, _)| *n)),
            max_gene_innovation: population.history().max_gene_innovation(),
            max_node_innovation: population.history().max_node_innovation(),
        })
    }

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
    pub fn iter(&self) -> impl Iterator<Item = &Log> {
        self.logs.iter()
    }
}
