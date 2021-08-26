//! An implementation of NeuroEvolution of Augmenting Topologies,
//! following the 2002 paper: <http://nn.cs.utexas.edu/keyword?stanley:ec02>
//! 
//! It is designed to be highly-configurable, and supports population 
//! logging throughout the evolutionary process.
//! 
//! This crate was implemented as both a learning exercise in using Rust
//! and as a tool for my own experimentation. Critiques and contributions
//! are welcome.
//! 
//! # Usage example
//! The following is an implementation of the classic XOR problem:
//! 
//! ```
//! use oxineat::{
//!     PopulationConfig,
//!     GeneticConfig,
//!     genomes::{Genome, ActivationType},
//!     networks::Network,
//!     populations::Population,
//! };
//! use std::num::NonZeroUsize;
//! 
//! // Allowed error margin for neural net answers.
//! const ERROR_MARGIN: f32 = 0.3;
//! 
//! fn evaluate_xor(genome: &Genome) -> f32 {
//!     let mut network = Network::from(genome);
//!
//!     let values = [
//!         ([1.0, 0.0, 0.0], 0.0),
//!         ([1.0, 0.0, 1.0], 1.0),
//!         ([1.0, 1.0, 0.0], 1.0),
//!         ([1.0, 1.0, 1.0], 0.0),
//!     ];
//!
//!     let mut errors = [0.0, 0.0, 0.0, 0.0];
//!     for (i, (input, output)) in values.iter().enumerate() {
//!         network.clear_state();
//!         for _ in 0..3 {
//!             network.set_inputs(input);
//!             network.activate();
//!         }
//!         errors[i] = (network.outputs()[0] - output).abs();
//!         if errors[i] < ERROR_MARGIN {
//!             errors[i] = 0.0;
//!         }
//!     }
//!
//!     (4.0 - errors.iter().copied().sum::<f32>()).powf(2.0)
//! }
//!
//! fn main() {
//!     let genetic_config = GeneticConfig {
//!         input_count: NonZeroUsize::new(3).unwrap(),
//!         output_count: NonZeroUsize::new(1).unwrap(),
//!         activation_types: vec![ActivationType::Sigmoid],
//!         output_activation_types: vec![ActivationType::Sigmoid],
//!         mutate_only_chance: 0.25,
//!         mate_only_chance: 0.2,
//!         mate_by_averaging_chance: 0.4,
//!         suppression_reset_chance: 1.0,
//!         initial_expression_chance: 1.0,
//!         weight_bound: 5.0,
//!         weight_reset_chance: 0.2,
//!         weight_nudge_chance: 0.9,
//!         weight_mutation_power: 2.5,
//!         node_mutation_chance: 0.03,
//!         gene_mutation_chance: 0.05,
//!         max_gene_mutation_attempts: 20,
//!         recursion_chance: 0.0,
//!         excess_gene_factor: 1.0,
//!         disjoint_gene_factor: 1.0,
//!         common_weight_factor: 0.4,
//!     };
//!
//!     let population_config = PopulationConfig {
//!         population_size: NonZeroUsize::new(150).unwrap(),
//!         distance_threshold: 3.0,
//!         elitism: 1,
//!         survival_threshold: 0.2,
//!         adoption_rate: 1.0,
//!         interspecies_mating_chance: 0.001,
//!         stagnation_threshold: NonZeroUsize::new(15).unwrap(),
//!         stagnation_penalty: 1.0,
//!     };
//! 
//!     let mut population = Population::new(population_config, genetic_config);
//!     for _ in 0..100 {
//!         population.evaluate_fitness(evaluate_xor);
//!         if (population.champion().fitness() - 16.0).abs() < f32::EPSILON {
//!             println!("Solution found: {}", population.champion());
//!             break;
//!         }
//!         if let Err(e) = population.evolve() {
//!             eprintln!("{}", e);
//!             break;
//!         }
//!     }
//! }
//! ```

pub mod genomes;
pub mod networks;
pub mod populations;

pub use genomes::GeneticConfig;
pub use populations::PopulationConfig;

/// Identifier type used to designate historically
/// identical mutations for the purposes of
/// genome comparison and genetic tracking.
pub type Innovation = usize;
