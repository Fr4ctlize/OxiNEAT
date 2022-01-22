use oxineat_nn::genomics::{ActivationType, NNGenome, GeneticConfig};
use oxineat_nn::networks::FunctionApproximatorNetwork;
use oxineat::{Population, PopulationConfig, logging::Stats};
// use neat::populations::{EvolutionLogger, ReportingLevel};

use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use rayon::prelude::*;
use ron;

const ERROR_MARGIN: f32 = 0.3;

fn evaluate_xor(genome: &NNGenome) -> f32 {
    let mut network = FunctionApproximatorNetwork::<1>::from(genome);

    let values = [
        ([1.0, 0.0, 0.0], 0.0),
        ([1.0, 0.0, 1.0], 1.0),
        ([1.0, 1.0, 0.0], 1.0),
        ([1.0, 1.0, 1.0], 0.0),
    ];

    let mut errors = [0.0, 0.0, 0.0, 0.0];
    for (i, (input, output)) in values.iter().enumerate() {
        errors[i] = (network.evaluate_at(input)[0] - output).abs();
        if errors[i] < ERROR_MARGIN {
            errors[i] = 0.0;
        }
    }

    (4.0 - errors.iter().copied().sum::<f32>()).powf(2.0)
}

fn main() {
    let genetic_config = GeneticConfig {
        input_count: NonZeroUsize::new(3).unwrap(),
        output_count: NonZeroUsize::new(1).unwrap(),
        activation_types: vec![ActivationType::Sigmoid],
        output_activation_types: vec![ActivationType::Sigmoid],
        child_mutation_chance: 0.65,
        mate_by_averaging_chance: 0.4,
        suppression_reset_chance: 1.0,
        initial_expression_chance: 1.0,
        weight_bound: 5.0,
        weight_reset_chance: 0.2,
        weight_nudge_chance: 0.9,
        weight_mutation_power: 2.5,
        node_addition_mutation_chance: 0.03,
        gene_addition_mutation_chance: 0.05,
        node_deletion_mutation_chance: 0.001,
        gene_deletion_mutation_chance: 0.002,
        max_gene_addition_mutation_attempts: 20,
        recursion_chance: 0.0,
        excess_gene_factor: 1.0,
        disjoint_gene_factor: 1.0,
        common_weight_factor: 0.4,
        ..GeneticConfig::zero()
    };
    let population_config = PopulationConfig {
        size: NonZeroUsize::new(150).unwrap(),
        distance_threshold: 3.0,
        elitism: 1,
        survival_threshold: 0.2,
        adoption_rate: 1.0,
        sexual_reproduction_chance: 0.6,
        interspecies_mating_chance: 0.001,
        stagnation_threshold: NonZeroUsize::new(15).unwrap(),
        stagnation_penalty: 1.0,
    };

    stress_test(&genetic_config, &population_config);
    // serde_test(&genetic_config, &population_config);
    // memory_test(&genetic_config, &population_config);
}

fn stress_test(genetic_config: &GeneticConfig, population_config: &PopulationConfig) {
    let generations = Arc::new(Mutex::new(vec![]));

    const ITERATIONS: usize = 2000;
    (0..ITERATIONS).for_each(|_| {
        let mut population = Population::new(population_config.clone(), genetic_config.clone());
        for _ in 0..100 {
            population.evaluate_fitness(evaluate_xor);
            // logger.lock().unwrap().log(&population);
            {
                if (population.champion().fitness() - 16.0).abs() < f32::EPSILON {
                    break;
                }
            }
            if let Err(e) = population.evolve() {
                eprintln!("{}", e);
                // population.reset()
                break;
            }
        }
        if (population.champion().fitness() - 16.0).abs() < f32::EPSILON {
            generations
                .lock()
                .unwrap()
                .push(Some(population.generation()));
        } else {
            generations.lock().unwrap().push(None);
        }
    });

    let generations = generations.lock().unwrap();

    println!(
        "Successful run generation count {:?}, {}% failure rate over {} iterations",
        Stats::from(
            &mut generations
                .iter()
                .filter_map(|g| g.as_ref().map(|g| *g as f32))
        ),
        generations.iter().filter(|g| g.is_none()).count() as f32 * 100.0 / ITERATIONS as f32,
        ITERATIONS
    );
}

fn serde_test(genetic_config: &GeneticConfig, population_config: &PopulationConfig) {
    let mut pop = "".into();
    let mut population = Population::new(population_config.clone(), genetic_config.clone());
    for _ in 0..100 {
        population.evaluate_fitness(evaluate_xor);
        if (population.champion().fitness() - 16.0).abs() < f32::EPSILON {
            println!("{}", ron::to_string(&population.champion()).unwrap());
            pop = ron::to_string(&population).unwrap();
            break;
        }
        if let Err(e) = population.evolve() {
            eprintln!("{}", e);
            break;
        }
    }
    let mut population: Population<_, _, _> = ron::from_str(&pop).unwrap();
    for _ in 0..100 {
        population.evaluate_fitness(evaluate_xor);
        if (population.champion().fitness() - 16.0).abs() < f32::EPSILON {
            println!("{}", population.champion());
            break;
        }
        if let Err(e) = population.evolve() {
            eprintln!("{}", e);
            break;
        }
    }
}

fn memory_test(genetic_config: &GeneticConfig, population_config: &PopulationConfig) {
    let mut population = Population::new(population_config.clone(), genetic_config.clone());
    for _ in 0..10000 {
        population.evaluate_fitness(evaluate_xor);
        if let Err(e) = population.evolve() {
            eprintln!("{}", e);
            break;
        }
    }
}