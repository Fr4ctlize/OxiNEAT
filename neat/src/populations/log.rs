use super::{Population, SpeciesID};
use crate::genomes::Genome;
use crate::Innovation;

use std::fmt;

#[derive(Clone, Copy, Debug)]
pub enum ReportingLevel {
    AllGenomes,
    SpeciesChampions,
    PopulationChampion,
    NoGenomes,
}

#[derive(Clone, Debug)]
pub struct Log {
    pub generation_number: usize,
    pub generation_sample: Generation,
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

#[derive(Clone, Debug)]
pub struct Stats {
    pub maximum: f32,
    pub minimum: f32,
    pub mean: f32,
    pub median: f32,
}

impl Stats {
    pub fn from(data: &mut dyn Iterator<Item = f32>) -> Stats {
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

#[derive(Clone, Debug)]
pub enum Generation {
    Species(Vec<(SpeciesID, Vec<Genome>, usize)>),
    SpeciesChampions(Vec<(SpeciesID, Genome, usize)>),
    PopulationChampion(Genome),
    None,
}

#[derive(Clone, Debug)]
pub struct EvolutionLogger {
    reporting_level: ReportingLevel,
    logs: Vec<Log>,
}

impl EvolutionLogger {
    pub fn new(reporting_level: ReportingLevel) -> EvolutionLogger {
        EvolutionLogger {
            reporting_level,
            logs: vec![],
        }
    }

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
                ReportingLevel::AllGenomes => Generation::Species(
                    population
                        .species()
                        .map(|s| (s.id(), s.genomes().to_vec(), s.time_stagnated()))
                        .collect(),
                ),
                ReportingLevel::SpeciesChampions => Generation::SpeciesChampions(
                    population
                        .species()
                        .map(|s| (s.id(), s.champion().clone(), s.time_stagnated()))
                        .collect(),
                ),
                ReportingLevel::PopulationChampion => {
                    Generation::PopulationChampion(population.champion().clone())
                }
                ReportingLevel::NoGenomes => Generation::None,
            },
            species_count: population.species().count(),
            fitness: Stats::from(&mut stats.iter().map(|(_, _, f)| *f)),
            gene_count: Stats::from(&mut stats.iter().map(|(g, _, _)| *g)),
            node_count: Stats::from(&mut stats.iter().map(|(_, n, _)| *n)),
            max_gene_innovation: population.history().gene_innovation_count(),
            max_node_innovation: population.history().node_innovation_count(),
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = &Log> {
        self.logs.iter()
    }
}
