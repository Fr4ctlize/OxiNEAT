//! This implementation of the `Genome` trait is a collection of genes and
//! nodes that can be instantiated as a phenotype (a neural network). [`NNGenomes`]
//! can be progressively mutated, thus adding complexity and functionality.
//!
//! [`NNGenomes`]: crate::genomics::NNGenome

mod config;
mod errors;
mod genes;
mod history;
mod nodes;

pub use config::GeneticConfig;
use errors::*;
pub use genes::Gene;
pub use history::History;
pub use nodes::{ActivationType, Node, NodeType};

use crate::Innovation;

use std::collections::HashMap;
use std::collections::HashSet;
use std::error::Error;
use std::fmt;

use ahash::RandomState;
use oxineat::Genome;
use rand::prelude::{IteratorRandom, Rng, SliceRandom};
use serde::{Deserialize, Serialize};

/// A mutable collection of genes and nodes.
///
/// Supports Serde for convenient genome saving and loading.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct NNGenome {
    genes: HashMap<Innovation, Gene, RandomState>,
    nodes: HashMap<Innovation, Node, RandomState>,
    node_pairings: HashSet<(Innovation, Innovation), RandomState>,
    pub(in crate) fitness: f32,
}

impl Genome for NNGenome {
    type InnovationHistory = History;
    type Config = GeneticConfig;

    fn new(config: &GeneticConfig) -> NNGenome {
        NNGenome::new(config)
    }

    fn genetic_distance(first: &NNGenome, second: &NNGenome, config: &GeneticConfig) -> f32 {
        NNGenome::genetic_distance(first, second, config)
    }

    fn mate(
        parent1: &NNGenome,
        parent2: &NNGenome,
        history: &mut History,
        config: &GeneticConfig,
    ) -> NNGenome {
        NNGenome::mate(parent1, parent2, history, config)
    }

    fn set_fitness(&mut self, fitness: f32) {
        self.set_fitness(fitness)
    }

    fn fitness(&self) -> f32 {
        self.fitness()
    }

    fn conforms_to(&self, config: &GeneticConfig) -> bool {
        let mut input_count = 0;
        let mut output_count = 0;
        let mut output_activation_types = vec![];

        for node in self.nodes.values() {
            match node.node_type() {
                NodeType::Sensor => {
                    input_count += 1;
                }
                NodeType::Actuator => {
                    output_count += 1;
                    output_activation_types.push((node.innovation(), node.activation_type()));
                }
                _ => {
                    if !config.activation_types.contains(&node.activation_type()) {
                        return false;
                    }
                }
            }
        }

        output_activation_types.sort_unstable_by_key(|(id, _)| *id);
        let output_activation_types = output_activation_types
            .into_iter()
            .map(|(_, t)| t)
            .collect::<Vec<_>>();

        let recursion_disabled = config.recursion_chance == 0.0;
        for gene in self.genes.values() {
            if gene.weight().abs() >= config.weight_bound {
                return false;
            }
            if recursion_disabled && gene.input() == gene.output() {
                return false;
            }
        }

        input_count == config.input_count.get()
            && output_count == config.output_count.get()
            && output_activation_types[..config.output_activation_types.len()]
                == config.output_activation_types
    }
}

impl NNGenome {
    /// Create a new genome with the specified configuration.
    ///
    /// Initially generated genes are given the innovation number
    /// `o + i ⨯ output_count`, where `i` is the innovation number
    /// of their input node and `o` is that of their output node.
    /// Thus, genes created through mutation start at innovation
    /// number `input_count ⨯ output_count`.
    ///
    /// # Examples
    /// ```
    /// use oxineat_nn::genomics::{ActivationType, GeneticConfig, NNGenome, NodeType};
    /// # use std::num::NonZeroUsize;
    ///
    /// let config = GeneticConfig {
    ///     // Some config...
    ///     input_count: NonZeroUsize::new(3).unwrap(),
    ///     output_count: NonZeroUsize::new(2).unwrap(),
    ///     initial_expression_chance: 1.0,
    ///     weight_bound: 5.0,
    ///     ..GeneticConfig::zero()
    /// };
    ///
    /// let genome = NNGenome::new(&config);
    ///
    /// // As configured, the genome should have 3 sensors + 2 actuators.
    /// assert_eq!(genome.nodes().count(), 3 + 2);
    /// assert_eq!(genome.nodes().filter(|n| n.node_type() == NodeType::Sensor).count(), 3);
    /// assert_eq!(genome.nodes().filter(|n| n.node_type() == NodeType::Actuator).count(), 2);
    ///
    /// // And with an initial_expression_chance of 1, there is a gene for every pair of nodes.
    /// assert_eq!(genome.genes().count(), 3 * 2);
    ///
    /// // All genes should have weights within the established bound.
    /// assert!(genome.genes().all(|g| g.weight().abs() <= config.weight_bound));
    ///
    /// // All genes should have innovation numbers in the range (0..6)
    /// assert!(genome.genes().all(|g| (0..3 * 2).contains(&g.innovation())));
    /// ```
    pub fn new(config: &GeneticConfig) -> NNGenome {
        let mut nodes = Self::generate_input_output_nodes(config);
        let (genes, node_pairings) = Self::generate_initial_genes(&mut nodes, config);

        NNGenome {
            genes,
            nodes,
            fitness: 0.0,
            node_pairings,
        }
    }

    /// Generate the input and output nodes
    /// corresponding to the passed configuration.
    fn generate_input_output_nodes(
        config: &GeneticConfig,
    ) -> HashMap<Innovation, Node, RandomState> {
        let input_count = config.input_count.get();
        let output_count = config.output_count.get();

        let mut nodes =
            HashMap::with_capacity_and_hasher(input_count + output_count, RandomState::new());

        for i in 0..input_count {
            nodes.insert(i, Node::new(i, NodeType::Sensor, ActivationType::Identity));
        }

        for o in 0..output_count {
            nodes.insert(
                o + input_count,
                Node::new(
                    o + input_count,
                    NodeType::Actuator,
                    *config
                        .output_activation_types
                        .get(o)
                        .unwrap_or(&ActivationType::Sigmoid),
                ),
            );
        }

        nodes
    }

    /// Generate the initial genes for the passed
    /// set of I/O nodes, each with [`initial_expression_chance`]
    /// probability of being expressed.
    ///
    /// [`initial_expression_chance`]: crate::genomics::GeneticConfig::initial_expression_chance
    fn generate_initial_genes(
        nodes: &mut HashMap<Innovation, Node, RandomState>,
        config: &GeneticConfig,
    ) -> (
        HashMap<Innovation, Gene, RandomState>,
        HashSet<(Innovation, Innovation), RandomState>,
    ) {
        let input_count = config.input_count.get();
        let output_count = config.output_count.get();
        let mut genes = HashMap::default();
        let mut node_pairings = HashSet::default();

        if config.initial_expression_chance != 0.0 {
            let mut rng = rand::thread_rng();
            for i in 0..input_count {
                for o in 0..output_count {
                    if rng.gen::<f32>() < config.initial_expression_chance {
                        let id = o + i * output_count;
                        genes.insert(
                            id,
                            Gene::new(id, i, o + input_count, Gene::random_weight(config)),
                        );
                        node_pairings.insert((i, o + input_count));
                        nodes.get_mut(&i).unwrap().add_output_gene(id).unwrap();
                        nodes
                            .get_mut(&(o + input_count))
                            .unwrap()
                            .add_input_gene(id)
                            .unwrap();
                    }
                }
            }
        }

        (genes, node_pairings)
    }

    /// Add a new gene to the genome.
    /// Returns a reference to the new gene.
    ///
    /// # Errors
    ///
    /// This function will return an error if a gene with the same
    /// `gene_id` already existed in the genome, if either `input_id`
    /// or `output_id` do not correspond to nodes present in the genome,
    /// or if `output_id` corresponds to a sensor node.
    ///
    /// # Examples
    /// ```
    /// use oxineat_nn::genomics::{ActivationType, GeneticConfig, NNGenome};
    /// use std::num::NonZeroUsize;
    ///
    /// let config = GeneticConfig {
    ///     input_count: NonZeroUsize::new(3).unwrap(),
    ///     output_count: NonZeroUsize::new(2).unwrap(),
    ///     initial_expression_chance: 0.0,
    ///     ..GeneticConfig::zero()
    /// };
    ///
    /// let mut genome = NNGenome::new(&config);
    ///
    /// // The genome is initially empty.
    /// assert_eq!(genome.genes().count(), 0);
    ///
    /// let inserted_gene = genome.add_gene(42, 2, 4, 2.5).unwrap().clone();
    ///
    /// // The genome now contains a neuron with the specified characteristics.
    /// assert_eq!(&inserted_gene, genome.genes().next().unwrap());
    /// assert_eq!(inserted_gene.innovation(), 42);
    /// assert_eq!(inserted_gene.input(), 2);
    /// assert_eq!(inserted_gene.output(), 4);
    /// assert_eq!(inserted_gene.weight(), 2.5);
    ///
    /// // Make a cycle (gene 43 goes 3 -> 4, gene 44 goes 4 -> 3).
    /// genome.add_gene(43, 3, 4, -3.0).unwrap();
    /// genome.add_gene(44, 4, 3, 1.0).unwrap();
    ///
    /// // Recursive gene.
    /// genome.add_gene(45, 4, 4, -1.0).unwrap();
    /// ```
    pub fn add_gene(
        &mut self,
        gene_id: Innovation,
        input_id: Innovation,
        output_id: Innovation,
        weight: f32,
    ) -> Result<&mut Gene, impl Error> {
        self.check_gene_viability(gene_id, input_id, output_id)
            .map(move |_| self.add_gene_unchecked(gene_id, input_id, output_id, weight))
    }

    /// Add a new gene to the genome.
    /// Returns a reference to the new gene.
    /// Assumes that the gene is not a duplicate
    /// or invalid gene for the genome.
    fn add_gene_unchecked(
        &mut self,
        gene_id: Innovation,
        input_id: Innovation,
        output_id: Innovation,
        weight: f32,
    ) -> &mut Gene {
        self.nodes
            .get_mut(&input_id)
            .unwrap()
            .add_output_gene(gene_id)
            .unwrap();
        self.nodes
            .get_mut(&output_id)
            .unwrap()
            .add_input_gene(gene_id)
            .unwrap();
        self.node_pairings.insert((input_id, output_id));
        self.genes
            .entry(gene_id)
            .or_insert_with(|| Gene::new(gene_id, input_id, output_id, weight))
    }

    /// Checks whether a gene is a duplicate or
    /// is invalid for the genome.
    ///
    /// # Errors
    ///
    /// Returns an error if the gene is not viable.
    fn check_gene_viability(
        &self,
        gene_id: Innovation,
        input_id: Innovation,
        output_id: Innovation,
    ) -> Result<(), GeneValidityError> {
        use GeneValidityError::*;
        if self.genes.contains_key(&gene_id) {
            Err(DuplicateGeneID(gene_id, Some((input_id, output_id))))
        } else if !(self.nodes.contains_key(&input_id) && self.nodes.contains_key(&output_id)) {
            Err(NonexistantEndpoints(input_id, output_id))
        } else if self.node_pairings.contains(&(input_id, output_id)) {
            Err(DuplicateGeneWithEndpoints(gene_id, (input_id, output_id)))
        } else if self.nodes[&output_id].node_type() == NodeType::Sensor {
            Err(SensorEndpoint(output_id))
        } else {
            Ok(())
        }
    }

    /// Add a new node to the genome.
    /// Returns a reference  to the newly created node.
    ///
    /// # Panics
    ///
    /// This function panics if a node of the
    /// same ID already existed in the genome.
    ///
    /// # Examples
    /// ```
    /// use oxineat_nn::genomics::{ActivationType, GeneticConfig, NNGenome, NodeType};
    /// use std::num::NonZeroUsize;
    ///
    /// let config = GeneticConfig {
    ///     input_count: NonZeroUsize::new(3).unwrap(),
    ///     output_count: NonZeroUsize::new(2).unwrap(),
    ///     initial_expression_chance: 1.0,
    ///     ..GeneticConfig::zero()
    /// };
    ///
    /// let mut genome = NNGenome::new(&config);
    ///
    /// // The new genome should have 3 sensors and 2 actuators only.
    /// assert_eq!(genome.nodes().count(), 3 + 2);
    ///
    /// let inserted_node = genome.add_node(42, ActivationType::Sigmoid).unwrap().clone();
    ///
    /// // The genome now has 1 additional neuron, as returned.
    /// assert_eq!(genome.nodes().count(), 3 + 2 + 1);
    /// assert_eq!(genome.nodes().filter(|n| *n == &inserted_node).count(), 1);
    ///
    /// // The inserted node has the specified characteristics.
    /// assert_eq!(inserted_node.innovation(), 42);
    /// assert_eq!(inserted_node.activation_type(), ActivationType::Sigmoid);
    /// assert_eq!(inserted_node.node_type(), NodeType::Neuron);
    /// ```
    pub fn add_node(
        &mut self,
        node_id: Innovation,
        activation_type: ActivationType,
    ) -> Result<&mut Node, impl Error> {
        self.check_node_viability(node_id)
            .map(move |_| self.add_node_unchecked(node_id, activation_type))
    }

    /// Add a new node to the genome.
    /// Returns a reference  to the newly created node.
    /// Assumes the node is not a duplicate.
    fn add_node_unchecked(
        &mut self,
        node_id: Innovation,
        activation_type: ActivationType,
    ) -> &mut Node {
        self.nodes
            .entry(node_id)
            .or_insert_with(|| Node::new(node_id, NodeType::Neuron, activation_type))
    }

    /// Checks whether a node is a duplicate
    /// for the genome.
    ///
    /// # Errors
    ///
    /// Rertuns an error in case of a duplicate.
    fn check_node_viability(&self, node_id: Innovation) -> Result<(), NodeValidityError> {
        if self.nodes.contains_key(&node_id) {
            Err(NodeValidityError::DuplicateNodeID(node_id))
        } else {
            Ok(())
        }
    }

    /// Induces a _weight mutation_ in the genome.
    ///
    /// # Examples
    /// ### Weight reset
    /// The weight is set to a random value in the range
    /// `[-weight_bound, weight_bound]`.
    /// ```
    /// use oxineat_nn::genomics::{GeneticConfig, NNGenome};
    /// use std::num::NonZeroUsize;
    ///
    /// let config = GeneticConfig {
    ///     input_count: NonZeroUsize::new(1).unwrap(),
    ///     output_count: NonZeroUsize::new(1).unwrap(),
    ///     initial_expression_chance: 1.0,
    ///     weight_bound: 5.0,
    ///     weight_reset_chance: 1.0,
    ///     ..GeneticConfig::zero()
    /// };
    ///
    /// // The genome has a single gene, whose weight we can retrieve.
    /// let mut genome = NNGenome::new(&config);
    /// let initial_weight = genome.genes().next().unwrap().weight();
    ///
    /// genome.mutate_weights(&config);
    /// let new_weight = genome.genes().next().unwrap().weight();
    ///
    /// // The gene should now have a different weight.
    /// // This *could* potentially fail, depending on RNG seed.
    /// assert_ne!(initial_weight, new_weight);
    ///
    /// // The gene's weight is still be within the weight bounds.
    /// assert!(new_weight.abs() <= config.weight_bound);
    /// ```
    ///
    /// ### Weight nudge
    /// A random value from the range `[-weight_mutation_power, weight_mutation_power]`
    /// is added to the weight, which is then clamped to `[-weight_bound, weight_bound]`.
    /// ```
    /// use oxineat_nn::genomics::{GeneticConfig, NNGenome};
    /// use std::num::NonZeroUsize;
    ///
    /// let config = GeneticConfig {
    ///     input_count: NonZeroUsize::new(1).unwrap(),
    ///     output_count: NonZeroUsize::new(1).unwrap(),
    ///     initial_expression_chance: 1.0,
    ///     weight_bound: 5.0,
    ///     weight_mutation_power: 2.5,
    ///     weight_nudge_chance: 1.0,
    ///     ..GeneticConfig::zero()
    /// };
    ///
    /// let mut genome = NNGenome::new(&config);
    /// let initial_weight = genome.genes().next().unwrap().weight();
    ///
    /// genome.mutate_weights(&config);
    /// let new_weight = genome.genes().next().unwrap().weight();
    ///
    /// // The gene should now have a different weight.
    /// // This *could* potentially fail, depending on RNG seed.
    /// assert_ne!(initial_weight, new_weight);
    ///
    /// // The gene's weight is still be within the weight bounds, and the nudge distance.
    /// assert!(new_weight.abs() <= config.weight_bound);
    /// assert!((new_weight - initial_weight).abs() <= config.weight_mutation_power);
    /// ```
    pub fn mutate_weights(&mut self, config: &GeneticConfig) {
        let mut rng = rand::thread_rng();
        let max_innovation = self.genes.keys().copied().max().unwrap_or_default().max(1) as f32;
        for gene in self.genes.values_mut() {
            // Older genes have a lower chance of being reset,
            // with the assumption being they've had more time
            // to settle into an optimised value.
            if rng.gen::<f32>()
                < config.weight_reset_chance
                    * ((gene.innovation() + 1) as f32 / max_innovation).powf(2.0)
            {
                gene.randomize_weight(config);
            } else if rng.gen::<f32>() < config.weight_nudge_chance {
                gene.nudge_weight(config);
            }
        }
    }

    /// Induces a _gene mutation_ in the genome.
    /// If successful, returns the newly added gene.
    ///
    /// # Note
    /// This function will not succeed if
    /// `config.max_gene_addition_mutation_attempts == 0.0`.
    ///
    /// # Errors
    /// Returns an error if no viable pair of nodes
    /// exists or [too many] attempts have failed.
    ///
    /// [too many]: crate::genomics::GeneticConfig::max_gene_addition_mutation_attempts
    ///
    /// # Examples
    /// ```
    /// use oxineat_nn::genomics::{GeneticConfig, NNGenome, History};
    /// use std::num::NonZeroUsize;
    ///
    /// let config = GeneticConfig {
    ///     initial_expression_chance: 0.0,
    ///     weight_bound: 5.0,
    ///     gene_addition_mutation_chance: 1.0,
    ///     max_gene_addition_mutation_attempts: 1,
    ///     recursion_chance: 1.0,
    ///     ..GeneticConfig::zero()
    /// };
    ///
    /// let mut genome = NNGenome::new(&config);
    ///
    /// // The genome is initially empty.
    /// assert_eq!(genome.genes().count(), 0);
    ///
    /// genome.mutate_add_gene(&mut History::new(&config), &config).unwrap();
    ///
    /// // The genome now has a new gene.
    /// assert_eq!(genome.genes().count(), 1);
    /// ```
    pub fn mutate_add_gene(
        &mut self,
        history: &mut History,
        config: &GeneticConfig,
    ) -> Result<&Gene, Box<dyn Error>> {
        let non_sensor_nodes = self.select_non_sensor_nodes();
        let mut potential_inputs = self.select_potential_input_nodes(non_sensor_nodes.len());

        if potential_inputs.is_empty() {
            return Err(GeneAdditionMutationError::GenomeFullyConnected.into());
        }

        let mut rng = rand::thread_rng();
        potential_inputs.shuffle(&mut rng);

        match self.find_node_pair(&potential_inputs, &non_sensor_nodes, config) {
            Some((source_node, dest_node)) => {
                Ok(self.add_gene_mutation(source_node, dest_node, history, config))
            }
            None => Err(GeneAdditionMutationError::NoInputOutputPairFound.into()),
        }
    }

    /// Returns all non-sensor nodes in the genome.
    fn select_non_sensor_nodes(&self) -> HashSet<Innovation, RandomState> {
        self.nodes
            .iter()
            .filter_map(|(id, n)| {
                if n.node_type() != NodeType::Sensor {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Returns all nodes that are not already
    /// fully-connected.
    fn select_potential_input_nodes(&self, non_sensor_node_count: usize) -> Vec<Innovation> {
        self.nodes
            .iter()
            .filter_map(|(id, n)| {
                if n.output_genes().count() < non_sensor_node_count {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Returns the first member of `potential_inputs ⨯ potential_outputs`
    /// that is computed.
    fn find_node_pair(
        &self,
        potential_inputs: &[Innovation],
        potential_outputs: &HashSet<Innovation, RandomState>,
        config: &GeneticConfig,
    ) -> Option<(Innovation, Innovation)> {
        potential_inputs
            .iter()
            .take(config.max_gene_addition_mutation_attempts)
            .filter_map(|i| {
                self.choose_output_node_for(i, potential_outputs, config)
                    .map(|output| (*i, output))
            })
            .next()
    }

    /// Randomly chooses a node from `potential_outputs` that
    /// `candidate_input` could be linked to by a new gene.
    fn choose_output_node_for(
        &self,
        candidate_input: &Innovation,
        potential_outputs: &HashSet<Innovation, RandomState>,
        config: &GeneticConfig,
    ) -> Option<Innovation> {
        let mut rng = rand::thread_rng();

        let candidate_input = &self.nodes[candidate_input];
        let mut candidate_outputs = potential_outputs - &self.output_nodes_of(candidate_input);

        if candidate_input.node_type() != NodeType::Sensor
            && candidate_outputs.contains(&candidate_input.innovation())
            && rng.gen::<f32>() < config.recursion_chance
        {
            Some(candidate_input.innovation())
        } else {
            candidate_outputs.remove(&candidate_input.innovation());
            candidate_outputs.iter().choose(&mut rng).copied()
        }
    }

    /// Returns the set of all output nodes of
    /// output genes of `node`.
    fn output_nodes_of(&self, node: &Node) -> HashSet<Innovation, RandomState> {
        node.output_genes()
            .map(|id| self.genes[id].output())
            .collect()
    }

    /// Adds the result of a gene mutation
    /// to the genome.
    fn add_gene_mutation(
        &mut self,
        input_node: Innovation,
        output_node: Innovation,
        history: &mut History,
        config: &GeneticConfig,
    ) -> &Gene {
        let gene_id = history.next_gene_innovation(input_node, output_node);
        let gene = self
            .add_gene(
                gene_id,
                input_node,
                output_node,
                Gene::random_weight(config),
            )
            .unwrap();
        history.add_gene_innovation(input_node, output_node);
        gene
    }

    /// Induces a _node mutation_ in the genome.
    /// If succesful, returns the triplet (_in gene_, _new node_, _out gene_)
    /// as a tuple of references.
    ///
    /// # Errors
    ///
    /// This function returns an error if there are no genes in the genome
    /// that could be split.
    ///
    /// # Examples
    /// ```
    /// use oxineat_nn::genomics::{ActivationType, GeneticConfig, History, NNGenome, NodeType};
    /// use std::num::NonZeroUsize;
    ///
    /// let config = GeneticConfig {
    ///     input_count: NonZeroUsize::new(1).unwrap(),
    ///     output_count: NonZeroUsize::new(1).unwrap(),
    ///     initial_expression_chance: 1.0,
    ///     node_addition_mutation_chance: 1.0,
    ///     activation_types: vec![ActivationType::Sigmoid],
    ///     ..GeneticConfig::zero()
    /// };
    ///
    /// let mut genome = NNGenome::new(&config);
    ///
    /// // The genome starts with a single neuron gene,
    /// // and a sensor and actuator node.
    /// assert_eq!(genome.genes().count(), 1);
    /// assert_eq!(genome.nodes().count(), 1 + 1);
    ///
    /// let prev_gene = genome.genes().next().unwrap().clone();
    ///
    /// let (new_input_gene, new_node, new_output_gene) =
    ///     genome.mutate_add_node(&mut History::new(&config), &config).unwrap();
    ///
    /// assert_eq!(new_input_gene.output(), new_node.innovation());
    /// assert_eq!(new_input_gene.weight(), 1.0);
    ///
    /// assert_eq!(new_output_gene.input(), new_node.innovation());
    /// assert_eq!(new_output_gene.weight(), prev_gene.weight());
    ///
    /// assert_eq!(new_node.activation_type(), ActivationType::Sigmoid);
    /// assert_eq!(new_node.node_type(), NodeType::Neuron);
    ///
    /// assert_eq!(genome.genes().count(), 1 + 2);
    /// assert_eq!(genome.nodes().count(), 1 + 1 + 1);
    ///
    /// // Old gene is suppressed.
    /// assert!(genome
    ///     .genes()
    ///     .filter(|g| g.innovation() == prev_gene.innovation())
    ///     .all(|g| g.suppressed())
    /// )
    /// ```
    pub fn mutate_add_node(
        &mut self,
        history: &mut History,
        config: &GeneticConfig,
    ) -> Result<(&Gene, &Node, &Gene), Box<dyn Error>> {
        let gene_to_split = match self.choose_random_gene() {
            Some(gene_to_split) => gene_to_split,
            None => return Err(NodeAdditionMutationError::EmptyGenome.into()),
        };

        let (mutation, duplicate) =
            self.get_node_mutation_innovation_triplet(gene_to_split, history);
        Ok(self.add_node_mutation(gene_to_split, mutation, duplicate, history, config))
    }

    /// Randomly selects a gene from the genome.
    /// Returns `None` if the genome is empty.
    fn choose_random_gene(&self) -> Option<Innovation> {
        let mut rng = rand::thread_rng();
        self.genes.keys().choose(&mut rng).cloned()
    }

    /// Retrieves the next node mutation triplet
    /// from `history`, accounting for the possibility
    /// that a node mutation on `gene_to_split` has
    /// previoulsy occurred in the genome or its
    /// ancestors. If this is the case, the returned
    /// `bool` is `true`, otherwise it's false.
    fn get_node_mutation_innovation_triplet(
        &mut self,
        gene_to_split: Innovation,
        history: &History,
    ) -> ((Innovation, Innovation, Innovation), bool) {
        let (input_gene, new_node, output_gene) =
            history.next_node_innovation(gene_to_split, false);

        if self.nodes.contains_key(&new_node) {
            (history.next_node_innovation(gene_to_split, true), true)
        } else {
            ((input_gene, new_node, output_gene), false)
        }
    }

    /// Adds the result of a node mutation
    /// to the genome.
    fn add_node_mutation(
        &mut self,
        gene_to_split: Innovation,
        mutation: (Innovation, Innovation, Innovation),
        duplicate: bool,
        history: &mut History,
        config: &GeneticConfig,
    ) -> (&Gene, &Node, &Gene) {
        let (input_gene, new_node, output_gene) = mutation;
        let (input_node, output_node) = self.genes[&gene_to_split].endpoints();

        // This used to be a check to return Err,
        // but it doesn't seem like this could
        // ever occurr. The calling function
        // should have checked for this already.
        debug_assert!(!self.nodes.contains_key(&new_node));

        history.add_node_innovation(gene_to_split, duplicate);

        self.suppress_gene_unchecked(gene_to_split);
        self.add_node_unchecked(
            new_node,
            *config
                .activation_types
                .choose(&mut rand::thread_rng())
                .unwrap_or(&ActivationType::Sigmoid),
        );
        self.add_gene_unchecked(input_gene, input_node, new_node, 1.0);
        self.add_gene_unchecked(
            output_gene,
            new_node,
            output_node,
            Gene::random_weight(config),
        );

        (
            &self.genes[&input_gene],
            &self.nodes[&new_node],
            &self.genes[&output_gene],
        )
    }

    /// Suppresses the passed gene without checking if
    /// it is part of the genome.
    fn suppress_gene_unchecked(&mut self, gene: Innovation) {
        self.genes.get_mut(&gene).unwrap().set_suppressed(true);
    }

    /// Deletes a randomly-chosen gene from the genome.
    ///
    /// Returns `None` if the network is empty, or `Some(gene)`
    /// otherwise.
    ///
    /// # Examples
    /// ```
    /// use oxineat_nn::genomics::{GeneticConfig, NNGenome};
    ///
    /// let config = GeneticConfig {
    ///     initial_expression_chance: 1.0,
    ///     gene_deletion_mutation_chance: 1.0,
    ///     ..GeneticConfig::zero()
    /// };
    /// let mut genome = NNGenome::new(&config);
    ///
    /// let initial_gene_count = genome.genes().count();
    ///
    /// genome.mutate_delete_gene();
    ///
    /// assert_eq!(genome.genes().count(), initial_gene_count - 1);
    /// ```
    pub fn mutate_delete_gene(&mut self) -> Option<Gene> {
        self.genes
            .keys()
            .copied()
            .choose(&mut rand::thread_rng())
            .map(|innovation| self.remove_gene_unchecked(innovation))
    }

    /// Deletes a randomly-chosen (non-I/O) node, and all
    /// incident genes, from the genome.
    ///
    /// Returns `None` if the network is empty, or
    /// `Some((node, incident_genes))` otherwise.
    ///
    /// # Examples
    /// ```
    /// use oxineat_nn::genomics::{ActivationType, GeneticConfig, Gene, NNGenome, Node, NodeType};
    ///
    /// let config = GeneticConfig {
    ///     node_deletion_mutation_chance: 1.0,
    ///     ..GeneticConfig::zero()
    /// };
    /// let mut genome = NNGenome::new(&config);
    /// genome.add_node(42, ActivationType::Sigmoid).unwrap();
    /// genome.add_gene(16, 0, 42, 1.0).unwrap();
    /// genome.add_gene(17, 42, 1, 1.0).unwrap();
    ///
    /// let initial_node_count = genome.nodes().count();
    /// let initial_gene_count = genome.genes().count();
    ///
    /// let (removed_node, removed_input_genes, removed_output_genes) = genome.mutate_delete_node().unwrap();
    ///
    /// let test_node = {
    ///     let mut node = Node::new(42, NodeType::Neuron, ActivationType::Sigmoid);
    ///     node.add_input_gene(16);
    ///     node.add_output_gene(17);
    ///     node
    /// };
    ///
    /// assert_eq!(removed_node, test_node);
    /// assert_eq!(removed_input_genes, vec![Gene::new(16, 0, 42, 1.0)]);
    /// assert_eq!(removed_output_genes, vec![Gene::new(17, 42, 1, 1.0)]);
    /// ```
    pub fn mutate_delete_node(&mut self) -> Option<(Node, Vec<Gene>, Vec<Gene>)> {
        self.nodes
            .values()
            .filter_map(|n| match n.node_type() {
                NodeType::Neuron => Some(n.innovation()),
                _ => None,
            })
            .choose(&mut rand::thread_rng())
            .map(|node_id| self.remove_node_unchecked(node_id))
    }

    /// Removes the specified gene from the genome. Returns `None`
    /// if the gene is not present in the genome.
    ///
    /// # Examples
    /// ```
    /// use oxineat_nn::genomics::{ActivationType, GeneticConfig, Gene, NNGenome};
    ///
    /// let config = GeneticConfig::zero();
    /// let mut genome = NNGenome::new(&config);
    /// genome.add_node(42, ActivationType::Sigmoid).unwrap();
    /// genome.add_gene(16, 0, 42, 1.0).unwrap();
    ///
    /// assert_eq!(genome.remove_gene(16), Some(Gene::new(16, 0, 42, 1.0)));
    /// assert_eq!(genome.remove_gene(16), None);
    /// ```
    pub fn remove_gene(&mut self, gene_id: Innovation) -> Option<Gene> {
        if self.genes.contains_key(&gene_id) {
            Some(self.remove_gene_unchecked(gene_id))
        } else {
            None
        }
    }

    /// Removes the specified gene from the genome, without checking
    /// for its presence.
    fn remove_gene_unchecked(&mut self, gene_id: Innovation) -> Gene {
        let deleted_gene = self.genes.remove(&gene_id).unwrap();
        // If this is called from a node deletion mutation,
        // either the input or output node has already been
        // deleted, so we only remove the gene from the other's
        // I/O set.
        if let Some(node) = self.nodes.get_mut(&deleted_gene.input()) {
            node.remove_output_gene(gene_id).unwrap();
        }
        if let Some(node) = self.nodes.get_mut(&deleted_gene.output()) {
            node.remove_input_gene(gene_id).unwrap();
        }
        self.node_pairings
            .remove(&(deleted_gene.input(), deleted_gene.output()));
        deleted_gene
    }

    /// Removes the specified node from the genome,
    /// and all connected genes. Returns `None`
    /// if the node is not present in the genome.
    ///
    /// # Examples
    /// ```
    /// use oxineat_nn::genomics::{ActivationType, GeneticConfig, Gene, NNGenome, Node, NodeType};
    ///
    /// let config = GeneticConfig {
    ///     node_deletion_mutation_chance: 1.0,
    ///     ..GeneticConfig::zero()
    /// };
    /// let mut genome = NNGenome::new(&config);
    /// genome.add_node(42, ActivationType::Sigmoid).unwrap();
    /// genome.add_gene(16, 0, 42, 1.0).unwrap();
    /// genome.add_gene(17, 42, 1, 1.0).unwrap();
    ///
    /// let initial_node_count = genome.nodes().count();
    /// let initial_gene_count = genome.genes().count();
    ///
    /// let (removed_node, removed_input_genes, removed_output_genes) = genome.remove_node(42).unwrap();
    ///
    /// let test_node = {
    ///     let mut node = Node::new(42, NodeType::Neuron, ActivationType::Sigmoid);
    ///     node.add_input_gene(16);
    ///     node.add_output_gene(17);
    ///     node
    /// };
    ///
    /// assert_eq!(removed_node, test_node);
    /// assert_eq!(removed_input_genes, vec![Gene::new(16, 0, 42, 1.0)]);
    /// assert_eq!(removed_output_genes, vec![Gene::new(17, 42, 1, 1.0)]);
    /// ```
    pub fn remove_node(&mut self, node_id: Innovation) -> Option<(Node, Vec<Gene>, Vec<Gene>)> {
        if self.nodes.contains_key(&node_id) {
            Some(self.remove_node_unchecked(node_id))
        } else {
            None
        }
    }

    /// Removes the specified node from the genome, without checking
    /// for its presence.
    fn remove_node_unchecked(&mut self, node_id: Innovation) -> (Node, Vec<Gene>, Vec<Gene>) {
        let node = self.nodes.remove(&node_id).unwrap();
        let input_genes: Vec<Gene> = node
            .input_genes()
            .copied()
            .map(|gene_id| self.remove_gene_unchecked(gene_id))
            .collect();
        let recursive_gene = input_genes
            .iter()
            .find(|gene| gene.input() == gene.output());
        let output_genes = node
            .output_genes()
            .copied()
            .map(|gene_id| match recursive_gene {
                None => self.remove_gene_unchecked(gene_id),
                Some(gene) => {
                    if gene.innovation() == gene_id {
                        gene.clone()
                    } else {
                        self.remove_gene_unchecked(gene_id)
                    }
                }
            })
            .collect();

        (node, input_genes, output_genes)
    }

    /// Combines both genomes and returns their _child_ genome.
    ///
    /// Depending on [`config.child_mutation_chance`], the child may
    /// undergo mutations. Mutation chances are defined by their
    /// corresponding entries in `config`.
    ///
    /// [`config.sexual_reproduction_chance`]: crate::genomics::GeneticConfig::sexual_reproduction_chance
    /// [`config.child_mutation_chance`]: crate::genomics::GeneticConfig::child_mutation_chance
    ///
    /// # Examples
    /// ### Asexual reproduction, no mutations
    /// ```
    /// use oxineat_nn::genomics::{ActivationType, GeneticConfig, History, NNGenome};
    /// use std::num::NonZeroUsize;
    ///
    /// let config = GeneticConfig {
    ///     input_count: NonZeroUsize::new(3).unwrap(),
    ///     output_count: NonZeroUsize::new(2).unwrap(),
    ///     activation_types: vec![ActivationType::Sigmoid],
    ///     initial_expression_chance: 1.0,
    ///     child_mutation_chance: 0.0,
    ///     ..GeneticConfig::zero()
    /// };
    ///
    /// let parent = NNGenome::new(&config);
    /// let parent_genes: Vec<_> = parent.genes().cloned().collect();
    ///
    /// // A genome can be "mated" with itself,
    /// // which implies asexual reproduction:
    /// let child = NNGenome::mate(&parent, &parent, &mut History::new(&config), &config);
    /// assert!(child.genes().all(|g| parent_genes.contains(g)));
    /// ```
    ///
    /// ### Sexual reproduction, with mutations
    /// ```
    /// use oxineat_nn::genomics::{ActivationType, GeneticConfig, History, NNGenome};
    /// use std::num::NonZeroUsize;
    ///
    /// let config = GeneticConfig {
    ///     input_count: NonZeroUsize::new(3).unwrap(),
    ///     output_count: NonZeroUsize::new(2).unwrap(),
    ///     activation_types: vec![ActivationType::Sigmoid],
    ///     initial_expression_chance: 0.5,
    ///     child_mutation_chance: 1.0,
    ///     weight_nudge_chance: 1.0,
    ///     gene_addition_mutation_chance: 1.0,
    ///     node_addition_mutation_chance: 1.0,
    ///     ..GeneticConfig::zero()
    /// };
    ///
    /// // All genomes have fitness 0 when generated.
    /// let genome1 = NNGenome::new(&config);
    /// let genome2 = NNGenome::new(&config);
    /// let genome1_gene_ids: Vec<_> = genome1.genes().map(|g| g.innovation()).collect();
    /// let genome2_gene_ids: Vec<_> = genome2.genes().map(|g| g.innovation()).collect();
    ///
    /// let child = NNGenome::mate(&genome1, &genome2, &mut History::new(&config), &config);
    /// let child_gene_ids: Vec<_> = child.genes().map(|g| g.innovation()).collect();
    ///
    /// // The child has at least the same number of genes as each parent.
    /// assert!(child_gene_ids.len() >= genome1_gene_ids.len());
    /// assert!(child_gene_ids.len() >= genome2_gene_ids.len());
    ///
    /// // The child will have at most the sum of the parent's genes,
    /// // plust at most one from a gene addition mutation and two
    /// // from a node addition mutation.
    /// assert!(child_gene_ids.len() <= genome1_gene_ids.len() + genome2_gene_ids.len() + 1 + 2);
    /// assert!(child_gene_ids.iter().filter(|id| !genome1_gene_ids.contains(id) && !genome2_gene_ids.contains(id)).count() <= 3);
    /// ```
    pub fn mate(
        parent1: &NNGenome,
        parent2: &NNGenome,
        history: &mut History,
        config: &GeneticConfig,
    ) -> NNGenome {
        let (parent1, parent2) = if parent1.fitness <= parent2.fitness {
            (parent1, parent2)
        } else {
            (parent2, parent1)
        };

        let mut child = parent1.clone();

        child.combine(parent2, config);
        if rand::thread_rng().gen::<f32>() < config.child_mutation_chance {
            child.mutate_all(history, config);
        }

        child.reset_suppresseds(config);

        child
    }

    /// Performs all mutations on self.
    fn mutate_all(&mut self, history: &mut History, config: &GeneticConfig) {
        let mut rng = rand::thread_rng();
        if rng.gen::<f32>() < config.node_deletion_mutation_chance {
            let _ = self.mutate_delete_node();
        }
        if rng.gen::<f32>() < config.gene_deletion_mutation_chance {
            let _ = self.mutate_delete_gene();
        }
        self.mutate_weights(config);
        if rng.gen::<f32>() < config.node_addition_mutation_chance {
            let _ = self.mutate_add_node(history, config);
        }
        if rng.gen::<f32>() < config.gene_addition_mutation_chance {
            let _ = self.mutate_add_gene(history, config);
        }
    }

    /// Adds all uncommon structure and combines genes to `self`.
    fn combine(&mut self, other: &NNGenome, config: &GeneticConfig) {
        if (self.fitness - other.fitness).abs() < f32::EPSILON {
            self.add_noncommon_structure(other);
        }
        if rand::thread_rng().gen::<f32>() < config.mate_by_averaging_chance {
            self.average_common_genes(other);
        } else {
            self.randomly_choose_common_genes(other);
        }
    }

    /// Adds all genes and nodes not common to both genomes to `self`.
    fn add_noncommon_structure(&mut self, other: &NNGenome) {
        for (id, node) in &other.nodes {
            if !self.nodes.contains_key(id) {
                self.add_node(*id, node.activation_type()).unwrap();
            }
        }

        for (id, gene) in &other.genes {
            if !self.node_pairings.contains(&(gene.input(), gene.output())) {
                self.add_gene(*id, gene.input(), gene.output(), gene.weight())
                    .unwrap();
            }
        }
    }

    /// Combines all common genes by averaging weights, and suppresses
    /// genes that are suppresssed in either genome.
    fn average_common_genes(&mut self, other: &NNGenome) {
        for (id, others_gene) in &other.genes {
            if let Some(own_gene) = self.genes.get_mut(id) {
                own_gene.set_weight((own_gene.weight() + others_gene.weight()) / 2.0);
            }
        }
    }

    /// Combines all common genes by chosing weights randomly between genomes, and suppresses
    /// genes that are suppresssed in either genome.
    fn randomly_choose_common_genes(&mut self, other: &NNGenome) {
        let mut rng = rand::thread_rng();
        for (id, gene) in &other.genes {
            if let Some(own) = self.genes.get_mut(id) {
                if rng.gen::<bool>() {
                    own.set_weight(gene.weight());
                }
            }
        }
    }

    /// Unsuppresses suppressed genes with probability `config.suppression_reset_chance`;
    fn reset_suppresseds(&mut self, config: &GeneticConfig) {
        let mut rng = rand::thread_rng();
        for gene in self.genes.values_mut() {
            if gene.suppressed() && rng.gen::<f32>() < config.suppression_reset_chance {
                gene.set_suppressed(false);
            }
        }
    }

    /// Calculates the _genetic distance_ between `self` and `other`,
    /// weighting node and weight differences as specified in `config`.
    ///
    /// # Examples
    /// ```
    /// use oxineat_nn::genomics::{ActivationType, GeneticConfig, History, NNGenome};
    /// use std::num::NonZeroUsize;
    ///
    /// // Completely arbitrary quantities.
    /// const EXCESS_FACTOR: f32 = 1.5;
    /// const DISJOINT_FACTOR: f32 = 0.5;
    /// const WEIGHT_FACTOR: f32 = 0.333;
    ///
    /// let config = GeneticConfig {
    ///     input_count: NonZeroUsize::new(2).unwrap(),
    ///     output_count: NonZeroUsize::new(1).unwrap(),
    ///     activation_types: vec![ActivationType::Sigmoid],
    ///     initial_expression_chance: 0.0,
    ///     excess_gene_factor: EXCESS_FACTOR,
    ///     disjoint_gene_factor: DISJOINT_FACTOR,
    ///     common_weight_factor: WEIGHT_FACTOR,
    ///     ..GeneticConfig::zero()
    /// };
    ///
    /// let mut genome1 = NNGenome::new(&config);
    /// let mut genome2 = NNGenome::new(&config);
    ///
    /// genome1.add_node(3, ActivationType::Sigmoid).unwrap();
    /// genome2.add_node(3, ActivationType::Sigmoid).unwrap();
    ///
    /// // Common gene, weight difference of 2.0.
    /// genome1.add_gene(0, 0, 2, 1.0).unwrap();
    /// genome2.add_gene(0, 0, 2, -1.0).unwrap();
    ///
    /// // Disjoint genes.
    /// genome1.add_gene(1, 1, 2, 3.0).unwrap();
    /// genome2.add_gene(2, 1, 3, 1.0).unwrap();
    ///
    /// // Common gene, weight_difference of 0.0.
    /// genome1.add_gene(3, 2, 3, 1.0).unwrap();
    /// genome2.add_gene(3, 2, 3, 1.0).unwrap();
    ///
    /// // Excess gene.
    /// genome1.add_gene(4, 2, 2, 3.0).unwrap();
    ///
    /// assert_eq!(
    ///     NNGenome::genetic_distance(&genome1, &genome2, &config),
    ///     DISJOINT_FACTOR * (1 + 1) as f32 +
    ///         EXCESS_FACTOR * (1) as f32 +
    ///         WEIGHT_FACTOR * (2.0 + 0.0) / 2.0  // Divide by number of common genes to get average.
    /// );
    /// ```
    pub fn genetic_distance(first: &NNGenome, second: &NNGenome, config: &GeneticConfig) -> f32 {
        let (common_innovations, common_weight_pairs) =
            Self::common_innovations_and_weights(first, second);

        let common_weight_diff = Self::weight_diff_average(&common_weight_pairs);

        let disjoint_gene_count_self = first.count_disjoint_genes(&common_innovations);
        let disjoint_gene_count_other = second.count_disjoint_genes(&common_innovations);
        let disjoint_gene_count = disjoint_gene_count_self + disjoint_gene_count_other;

        let excess_gene_count =
            (first.genes.len() - common_innovations.len() - disjoint_gene_count_self)
                + (second.genes.len() - common_innovations.len() - disjoint_gene_count_other);

        config.disjoint_gene_factor * disjoint_gene_count as f32
            + config.excess_gene_factor * excess_gene_count as f32
            + config.common_weight_factor * common_weight_diff
    }

    /// Returns the set of genes common to both genomes,
    /// and the corresponding pairs of weights.
    fn common_innovations_and_weights(
        g1: &NNGenome,
        g2: &NNGenome,
    ) -> (HashSet<Innovation, RandomState>, Vec<(f32, f32)>) {
        let gene_set_g1: HashSet<Innovation, RandomState> = g1.genes.keys().copied().collect();
        let gene_set_g2: HashSet<Innovation, RandomState> = g2.genes.keys().copied().collect();
        gene_set_g1
            .intersection(&gene_set_g2)
            .copied()
            .map(|id| (id, (g1.genes[&id].weight(), g2.genes[&id].weight())))
            .unzip()
    }

    /// Returns the average difference of all pairs
    /// of weights.
    fn weight_diff_average(weight_pairs: &[(f32, f32)]) -> f32 {
        let count = weight_pairs.len();
        if count != 0 {
            weight_pairs
                .iter()
                .map(|(w1, w2)| (w1 - w2).abs())
                .sum::<f32>()
                / count as f32
        } else {
            0.0
        }
    }

    /// Counts the number of disjoint genes in the genome
    /// relative to the supplied set of common genes.
    fn count_disjoint_genes(&self, common_innovations: &HashSet<Innovation, RandomState>) -> usize {
        let common_innovation_max = common_innovations.iter().max().cloned().unwrap_or_default();
        self.genes
            .keys()
            .cloned()
            .filter(|id| !common_innovations.contains(id) && *id < common_innovation_max)
            .count()
    }

    /// Sets the genome's fitness to the value passed.
    /// Fitness should be a positive quantity.
    ///
    /// # Examples
    /// ```
    /// use oxineat_nn::genomics::{GeneticConfig, NNGenome};
    ///
    /// let mut genome = NNGenome::new(&GeneticConfig::zero());
    ///
    /// assert_eq!(genome.fitness(), 0.0);
    ///
    /// genome.set_fitness(32.0);
    ///
    /// assert_eq!(genome.fitness(), 32.0);
    ///
    /// // Will panic:
    /// // genome.set_fitness(-1.0);
    /// ```
    pub fn set_fitness(&mut self, fitness: f32) {
        assert!(fitness >= 0.0, "fitness function return a negative value");
        self.fitness = fitness;
    }

    /// Returns a the genome's current fitness.
    ///
    /// # Examples
    /// ```
    /// use oxineat_nn::genomics::{GeneticConfig, NNGenome};
    ///
    /// let genome = NNGenome::new(&GeneticConfig::zero());
    ///
    /// assert_eq!(genome.fitness(), 0.0);
    /// ```
    pub fn fitness(&self) -> f32 {
        self.fitness
    }

    /// Returns a n iterator over the set of the genome's genes.
    ///
    /// # Notes
    /// No ordering is guaranteed.
    ///
    /// # Examples
    /// ```
    /// use oxineat_nn::genomics::{GeneticConfig, NNGenome};
    /// use std::num::NonZeroUsize;
    ///
    /// let config = GeneticConfig {
    ///     input_count: NonZeroUsize::new(3).unwrap(),
    ///     output_count: NonZeroUsize::new(2).unwrap(),
    ///     initial_expression_chance: 1.0,
    ///     ..GeneticConfig::zero()
    /// };
    ///
    /// let genome = NNGenome::new(&config);
    ///
    /// for gene in genome.genes() {
    ///     println!("gene: {}", gene);
    /// }
    /// ```
    pub fn genes(&self) -> impl Iterator<Item = &Gene> {
        self.genes.values()
    }

    /// Returns a n iterator over the set of the genome's nodes.
    ///
    /// # Notes
    /// No ordering is guaranteed.
    ///
    /// # Examples
    /// ```
    /// use oxineat_nn::genomics::{GeneticConfig, NNGenome};
    /// use std::num::NonZeroUsize;
    ///
    /// let config = GeneticConfig {
    ///     input_count: NonZeroUsize::new(3).unwrap(),
    ///     output_count: NonZeroUsize::new(2).unwrap(),
    ///     initial_expression_chance: 1.0,
    ///     ..GeneticConfig::zero()
    /// };
    ///
    /// let genome = NNGenome::new(&config);
    ///
    /// for node in genome.nodes() {
    ///     println!("node: {}", node);
    /// }
    /// ```
    pub fn nodes(&self) -> impl Iterator<Item = &Node> {
        self.nodes.values()
    }
}

impl fmt::Display for NNGenome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut genes: Vec<&Gene> = self.genes.values().collect();
        let mut nodes: Vec<&Node> = self.nodes.values().collect();
        genes.sort_unstable_by_key(|g| g.innovation());
        nodes.sort_unstable_by_key(|n| n.innovation());
        f.debug_struct("NNGenome")
            .field("Genes", &genes)
            .field("Nodes", &nodes)
            .field("Fitness", &self.fitness)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;

    #[test]
    fn new_fully_connected() {
        for input_count in 1..10 {
            for output_count in 1..10 {
                let mut config = GeneticConfig::zero();
                config.initial_expression_chance = 1.0;
                config.input_count = NonZeroUsize::new(input_count).unwrap();
                config.output_count = NonZeroUsize::new(output_count).unwrap();
                config.output_activation_types = vec![
                    ActivationType::Sigmoid,
                    ActivationType::Gaussian,
                    ActivationType::Identity,
                    ActivationType::ReLU,
                    ActivationType::Sinusoidal,
                ];

                let full_genome = NNGenome::new(&config);
                assert_eq!(full_genome.genes.len(), input_count * output_count);
                assert_eq!(
                    full_genome
                        .nodes
                        .values()
                        .filter(|n| n.node_type() == NodeType::Sensor
                            && n.activation_type() == ActivationType::Identity)
                        .count(),
                    input_count
                );
                assert_eq!(
                    full_genome
                        .nodes
                        .values()
                        .filter(|n| n.node_type() == NodeType::Actuator
                            && n.activation_type()
                                == *config
                                    .output_activation_types
                                    .get(n.innovation() - input_count)
                                    .unwrap_or(&ActivationType::Sigmoid))
                        .count(),
                    output_count
                );
                assert_eq!(
                    *full_genome.nodes.keys().max().unwrap(),
                    input_count + output_count - 1
                );
                for g in full_genome.genes.values() {
                    assert_eq!(
                        g.innovation(),
                        g.input() * output_count + (g.output() - input_count),
                        "gene: {:?}, total I/O, {}/{}",
                        g,
                        input_count,
                        output_count
                    );
                    assert!(full_genome
                        .nodes
                        .get(&g.input())
                        .unwrap()
                        .output_genes()
                        .collect::<HashSet<_>>()
                        .contains(&g.innovation()));
                    assert!(full_genome
                        .nodes
                        .get(&g.output())
                        .unwrap()
                        .input_genes()
                        .collect::<HashSet<_>>()
                        .contains(&g.innovation()));
                }
            }
        }
    }

    #[test]
    fn new_unconnected() {
        let mut config = GeneticConfig::zero();
        config.initial_expression_chance = 0.0;

        let empty_genome = NNGenome::new(&config);
        assert_eq!(empty_genome.genes.len(), 0);
    }

    #[test]
    fn add_gene() {
        const INNOVATION: Innovation = 631;
        const INPUT: Innovation = 0;
        const OUTPUT: Innovation = 1;
        const WEIGHT: f32 = 3.0;

        let mut config = GeneticConfig::zero();
        config.initial_expression_chance = 0.0;

        let mut genome = NNGenome::new(&config);
        let gene = genome.add_gene(INNOVATION, INPUT, OUTPUT, WEIGHT).unwrap();

        assert_eq!(gene.innovation(), INNOVATION);
        assert_eq!(gene.input(), INPUT);
        assert_eq!(gene.output(), OUTPUT);
        assert_eq!(gene.weight(), WEIGHT);

        let gene = gene.clone();

        assert_eq!(genome.genes.len(), 1);
        assert_eq!(&genome.genes[&INNOVATION], &gene);
    }

    #[test]
    #[should_panic]
    fn add_gene_duplicate_gene_innovation() {
        const INNOVATION: Innovation = 0;
        const INPUT: Innovation = 0;
        const OUTPUT: Innovation = 1;
        const WEIGHT: f32 = 3.0;

        let mut config = GeneticConfig::zero();
        config.initial_expression_chance = 1.0;

        let mut genome = NNGenome::new(&config);
        genome.add_gene(INNOVATION, INPUT, OUTPUT, WEIGHT).unwrap();
    }

    #[test]
    #[should_panic]
    fn add_gene_duplicate_io() {
        const INNOVATION: Innovation = 555;
        const WEIGHT: f32 = 3.0;

        let mut config = GeneticConfig::zero();
        config.initial_expression_chance = 1.0;

        let mut genome = NNGenome::new(&config);
        let input = genome
            .nodes
            .values()
            .filter(|n| n.node_type() == NodeType::Sensor)
            .nth(0)
            .unwrap()
            .innovation();
        let output = genome
            .nodes
            .values()
            .filter(|n| n.node_type() == NodeType::Actuator)
            .nth(0)
            .unwrap()
            .innovation();

        genome.add_gene(INNOVATION, input, output, WEIGHT).unwrap();
    }

    #[test]
    #[should_panic]
    fn add_gene_invalid_input() {
        const INNOVATION: Innovation = 631;
        const INPUT: Innovation = 500;
        const OUTPUT: Innovation = 1;
        const WEIGHT: f32 = 3.0;

        let mut config = GeneticConfig::zero();
        config.initial_expression_chance = 0.0;

        let mut genome = NNGenome::new(&config);
        genome.add_gene(INNOVATION, INPUT, OUTPUT, WEIGHT).unwrap();
    }

    #[test]
    #[should_panic]
    fn add_gene_invalid_output() {
        const INNOVATION: Innovation = 631;
        const INPUT: Innovation = 0;
        const OUTPUT: Innovation = 500;
        const WEIGHT: f32 = 3.0;

        let mut config = GeneticConfig::zero();
        config.initial_expression_chance = 0.0;

        let mut genome = NNGenome::new(&config);
        genome.add_gene(INNOVATION, INPUT, OUTPUT, WEIGHT).unwrap();
    }

    #[test]
    fn add_node() {
        const INPUTS: usize = 1;
        const OUTPUTS: usize = 1;
        const INNOVATION: Innovation = 42;
        const ACTIVATION_TYPE: ActivationType = ActivationType::Gaussian;

        let mut config = GeneticConfig::zero();
        config.initial_expression_chance = 0.0;
        config.input_count = NonZeroUsize::new(INPUTS).unwrap();
        config.output_count = NonZeroUsize::new(OUTPUTS).unwrap();

        let mut genome = NNGenome::new(&config);
        let node = genome.add_node(INNOVATION, ACTIVATION_TYPE).unwrap();

        assert_eq!(node.innovation(), INNOVATION);
        assert_eq!(node.node_type(), NodeType::Neuron);
        assert_eq!(node.activation_type(), ACTIVATION_TYPE);
        assert_eq!(node.input_genes().count(), 0);
        assert_eq!(node.output_genes().count(), 0);

        let node = node.clone();

        assert_eq!(genome.nodes.len(), INPUTS + OUTPUTS + 1);
        assert_eq!(&genome.nodes[&INNOVATION], &node);
    }

    #[test]
    #[should_panic]
    fn add_node_duplicate() {
        const INNOVATION: Innovation = 0;
        const ACTIVATION_TYPE: ActivationType = ActivationType::Gaussian;

        let mut config = GeneticConfig::zero();
        config.initial_expression_chance = 0.0;

        let mut genome = NNGenome::new(&config);
        genome.add_node(INNOVATION, ACTIVATION_TYPE).unwrap();
    }

    /// It is possible this test will fail
    /// due to the new weight returned by the
    /// rng being identical to the previous one,
    /// but the chances of this are minimal.
    #[test]
    fn mutate_weights_reset() {
        let mut config = GeneticConfig::zero();
        config.initial_expression_chance = 1.0;
        config.weight_reset_chance = 1.0;
        config.weight_bound = 3.0;

        let mut genome = NNGenome::new(&config);
        let initial_weight = genome.genes[&0].weight();
        println!("{}", genome);
        genome.mutate_weights(&config);
        println!("{}", genome);
        assert_ne!(initial_weight, genome.genes[&0].weight());
    }

    /// It is possible this test will fail
    /// due to the new weight returned by the
    /// rng being identical to the previous one,
    /// but the chances of this are minimal.
    #[test]
    fn mutate_weights_nudge() {
        let mut config = GeneticConfig::zero();
        config.initial_expression_chance = 1.0;
        config.weight_reset_chance = 0.0;
        config.weight_nudge_chance = 1.0;
        config.weight_mutation_power = 3.0;
        config.weight_bound = 5.0;

        let mut genome = NNGenome::new(&config);
        genome.mutate_weights(&config);
        let initial_weight = genome.genes[&0].weight();
        genome.mutate_weights(&config);
        assert_ne!(initial_weight, genome.genes[&0].weight());
    }

    #[test]
    fn mutate_weights_none() {
        let mut config = GeneticConfig::zero();
        config.initial_expression_chance = 1.0;
        config.weight_reset_chance = 0.0;
        config.weight_nudge_chance = 0.0;
        config.weight_bound = 5.0;

        let mut genome = NNGenome::new(&config);
        genome.mutate_weights(&config);
        let initial_weight = genome.genes[&0].weight();
        genome.mutate_weights(&config);
        assert_eq!(initial_weight, genome.genes[&0].weight());
    }

    #[test]
    fn mutate_gene_addition() {
        let mut config = GeneticConfig::zero();
        config.initial_expression_chance = 1.0;
        config.gene_addition_mutation_chance = 1.0;
        config.max_gene_addition_mutation_attempts = 20;
        config.recursion_chance = 0.0;
        config.weight_mutation_power = 3.0;

        let mut history = History::new(&config);

        let mut genome = NNGenome::new(&config);
        genome.add_node(2, ActivationType::Sigmoid).unwrap();
        let gene = genome.mutate_add_gene(&mut history, &config).unwrap();

        assert_eq!(
            gene.innovation(),
            history.next_gene_innovation(gene.input(), gene.output())
        );
        assert!((0..=2).contains(&gene.input()));
        assert!((1..=2).contains(&gene.output()));
    }

    #[test]
    fn mutate_gene_addition_recursive() {
        let mut config = GeneticConfig::zero();
        config.initial_expression_chance = 1.0;
        config.gene_addition_mutation_chance = 1.0;
        config.max_gene_addition_mutation_attempts = 20;
        config.recursion_chance = 1.0;
        config.weight_mutation_power = 3.0;

        let mut history = History::new(&config);

        let mut genome = NNGenome::new(&config);
        genome.add_node(50, ActivationType::Sigmoid).unwrap();
        genome.add_gene(42, 0, 50, 2.0).unwrap();
        genome.add_gene(43, 50, 1, 2.0).unwrap();
        genome.add_gene(44, 1, 50, 2.0).unwrap();
        let gene = genome.mutate_add_gene(&mut history, &config).unwrap();

        assert_eq!(gene.input(), gene.output());
    }

    #[test]
    #[should_panic]
    fn mutate_gene_addition_no_pairs_found() {
        let mut config = GeneticConfig::zero();
        config.initial_expression_chance = 1.0;
        config.gene_addition_mutation_chance = 1.0;
        config.max_gene_addition_mutation_attempts = 20;
        config.recursion_chance = 1.0;
        config.weight_mutation_power = 3.0;

        let mut history = History::new(&config);

        let mut genome = NNGenome::new(&config);
        // Add a recursive gene to make sure they aren't
        // chosen even though it is permitted.
        genome.add_gene(42, 1, 1, 5.0).unwrap();

        genome.mutate_add_gene(&mut history, &config).unwrap();
    }

    #[test]
    fn mutate_nodes_addition() {
        let mut config = GeneticConfig::zero();
        config.activation_types = vec![ActivationType::Sigmoid, ActivationType::ReLU];
        config.initial_expression_chance = 1.0;
        config.node_addition_mutation_chance = 1.0;

        let mut history = History::new(&config);

        let mut genome = NNGenome::new(&config);
        let (input, node, output) = genome.mutate_add_node(&mut history, &config).unwrap();

        assert_eq!(input.innovation(), *node.input_genes().next().unwrap());
        assert_eq!(output.innovation(), *node.output_genes().next().unwrap());
        assert!(config.activation_types.contains(&node.activation_type()));
        assert_eq!(node.innovation(), genome.nodes[&2].innovation());
        assert_eq!(genome.genes.len(), 3);
        assert_eq!(genome.nodes.len(), 3);
        assert_eq!(genome.genes[&0].suppressed(), true);
    }

    #[test]
    #[should_panic]
    fn mutate_node_addition_no_gene_found() {
        let mut config = GeneticConfig::zero();
        config.initial_expression_chance = 0.0;
        config.node_addition_mutation_chance = 1.0;

        let mut history = History::new(&config);

        let mut genome = NNGenome::new(&config);
        genome.mutate_add_node(&mut history, &config).unwrap();
    }

    #[test]
    fn mutate_gene_deletion() {
        let config = GeneticConfig {
            input_count: NonZeroUsize::new(5).unwrap(),
            output_count: NonZeroUsize::new(5).unwrap(),
            initial_expression_chance: 1.0,
            gene_deletion_mutation_chance: 1.0,
            ..GeneticConfig::zero()
        };
        let mut genome = NNGenome::new(&config);
        let initial_gene_count = genome.genes().count();
        genome.mutate_delete_gene();
        assert_eq!(genome.genes().count(), initial_gene_count - 1);
    }

    #[test]
    fn mutate_node_deletion() {
        let config = GeneticConfig {
            node_deletion_mutation_chance: 1.0,
            ..GeneticConfig::zero()
        };
        let mut genome = NNGenome::new(&config);
        genome.add_node(42, ActivationType::Sigmoid).unwrap();
        genome.add_gene(16, 0, 42, 1.0).unwrap();
        genome.add_gene(17, 42, 1, 1.0).unwrap();
        let initial_node_count = genome.nodes().count();
        let initial_gene_count = genome.genes().count();
        genome.mutate_delete_node();
        assert_eq!(genome.nodes().count(), initial_node_count - 1);
        assert_eq!(genome.genes().count(), initial_gene_count - 2);
    }

    #[test]
    fn add_noncommon_structure() {
        let mut config = GeneticConfig::zero();
        config.input_count = NonZeroUsize::new(2).unwrap();
        config.output_count = NonZeroUsize::new(1).unwrap();
        config.initial_expression_chance = 1.0;

        let mut genome1 = NNGenome::new(&config);
        let mut genome2 = NNGenome::new(&config);
        genome1.add_node(3, ActivationType::Sigmoid).unwrap();
        genome2.add_node(3, ActivationType::Sigmoid).unwrap();
        genome2.add_node(4, ActivationType::ReLU).unwrap();
        genome1.add_gene(2, 0, 3, 1.0).unwrap();
        genome2.add_gene(3, 0, 3, 1.0).unwrap();
        genome2.add_gene(4, 0, 4, 2.0).unwrap();
        genome2.add_gene(5, 3, 4, 3.0).unwrap();
        genome2.add_gene(6, 4, 2, 4.0).unwrap();
        genome1.add_noncommon_structure(&genome2);

        let mut node4 = Node::new(4, NodeType::Neuron, ActivationType::ReLU);
        node4.add_input_gene(4).unwrap();
        node4.add_input_gene(5).unwrap();
        node4.add_output_gene(6).unwrap();
        assert_eq!(&genome1.nodes[&4], &node4);
        assert_eq!(&genome1.genes[&2], &Gene::new(2, 0, 3, 1.0));
        assert_eq!(&genome1.genes[&4], &Gene::new(4, 0, 4, 2.0));
        assert_eq!(&genome1.genes[&5], &Gene::new(5, 3, 4, 3.0));
        assert_eq!(&genome1.genes[&6], &Gene::new(6, 4, 2, 4.0));
    }

    #[test]
    fn combines_genes_average() {
        let mut config = GeneticConfig::zero();
        config.input_count = NonZeroUsize::new(2).unwrap();
        config.output_count = NonZeroUsize::new(1).unwrap();
        config.initial_expression_chance = 0.0;

        let mut genome1 = NNGenome::new(&config);
        genome1.add_node(3, ActivationType::Sigmoid).unwrap();

        let mut genome2 = genome1.clone();

        genome1.add_gene(0, 0, 2, 1.0).unwrap();
        genome2.add_gene(0, 0, 2, 3.0).unwrap();
        genome1.add_gene(1, 0, 3, 2.0).unwrap();
        genome2.add_gene(1, 0, 3, -2.0).unwrap();
        genome1.add_gene(2, 3, 2, 3.0).unwrap();
        genome2.add_gene(2, 3, 2, 3.5).unwrap();
        genome1.add_gene(3, 1, 2, 4.0).unwrap();
        genome2.add_gene(3, 1, 2, 4.0).unwrap();

        genome1.average_common_genes(&genome2);

        assert_eq!(genome1.genes[&0].weight(), 2.0);
        assert_eq!(genome1.genes[&1].weight(), 0.0);
        assert_eq!(genome1.genes[&2].weight(), 3.25);
        assert_eq!(genome1.genes[&3].weight(), 4.0);
    }

    #[test]
    fn combines_genes_random_choice() {
        let mut config = GeneticConfig::zero();
        config.input_count = NonZeroUsize::new(2).unwrap();
        config.output_count = NonZeroUsize::new(1).unwrap();
        config.initial_expression_chance = 0.0;

        let mut genome1 = NNGenome::new(&config);
        genome1.add_node(3, ActivationType::Sigmoid).unwrap();

        let mut genome2 = genome1.clone();

        genome1.add_gene(0, 0, 2, 1.0).unwrap();
        genome2.add_gene(0, 0, 2, 3.0).unwrap();
        genome1.add_gene(1, 0, 3, 2.0).unwrap();
        genome2.add_gene(1, 0, 3, -2.0).unwrap();
        genome1.add_gene(2, 3, 2, 3.0).unwrap();
        genome2.add_gene(2, 3, 2, 3.5).unwrap();
        genome1.add_gene(3, 1, 2, 4.0).unwrap();
        genome2.add_gene(3, 1, 2, 4.0).unwrap();

        genome1.randomly_choose_common_genes(&genome2);

        assert!([1.0, 3.0].contains(&genome1.genes[&0].weight()));
        assert!([2.0, -2.0].contains(&genome1.genes[&1].weight()));
        assert!([3.0, 3.5].contains(&genome1.genes[&2].weight()));
        assert!([4.0].contains(&genome1.genes[&3].weight()));
    }

    #[test]
    fn reset_suppresseds() {
        let mut config = GeneticConfig::zero();
        config.input_count = NonZeroUsize::new(2).unwrap();
        config.output_count = NonZeroUsize::new(1).unwrap();
        config.initial_expression_chance = 0.0;
        config.suppression_reset_chance = 1.0;

        let mut genome = NNGenome::new(&config);
        genome.add_gene(0, 0, 2, 3.0).unwrap().set_suppressed(true);
        genome.add_gene(1, 1, 2, 4.0).unwrap().set_suppressed(true);
        genome.add_gene(2, 2, 2, 5.0).unwrap().set_suppressed(true);

        genome.reset_suppresseds(&config);

        assert!(genome.genes.values().all(|g| !g.suppressed()));
    }

    #[test]
    fn genetic_distance_to() {
        const WEIGHT_FACTOR: f32 = 0.8;
        const DISJOINT_FACTOR: f32 = 0.6;
        const EXCESS_FACTOR: f32 = 0.4;
        let mut config = GeneticConfig::zero();
        config.input_count = NonZeroUsize::new(2).unwrap();
        config.output_count = NonZeroUsize::new(1).unwrap();
        config.initial_expression_chance = 0.0;
        config.common_weight_factor = WEIGHT_FACTOR;
        config.disjoint_gene_factor = DISJOINT_FACTOR;
        config.excess_gene_factor = EXCESS_FACTOR;

        let mut genome1 = NNGenome::new(&config);
        let mut genome2 = NNGenome::new(&config);

        genome1.add_node(3, ActivationType::Sigmoid).unwrap();
        genome2.add_node(4, ActivationType::Sigmoid).unwrap();

        genome1.add_gene(1, 0, 2, -2.0).unwrap();
        genome2.add_gene(1, 0, 2, 2.0).unwrap();

        genome1.add_gene(2, 1, 3, 5.0).unwrap();
        genome2.add_gene(3, 2, 4, 5.0).unwrap();

        genome1.add_gene(4, 1, 2, 3.0).unwrap();
        genome2.add_gene(4, 1, 2, 6.0).unwrap();

        genome1.add_gene(5, 2, 2, 5.0).unwrap();
        genome2.add_gene(6, 4, 4, 1.0).unwrap();

        assert_eq!(
            NNGenome::genetic_distance(&genome1, &genome2, &config),
            DISJOINT_FACTOR * 2.0 + EXCESS_FACTOR * 2.0 + WEIGHT_FACTOR * ((4.0 + 3.0) / 2.0)
        );
    }

    #[test]
    fn genetic_distance_to_equal() {
        let mut config = GeneticConfig::zero();
        config.input_count = NonZeroUsize::new(2).unwrap();
        config.output_count = NonZeroUsize::new(1).unwrap();
        config.initial_expression_chance = 1.0;
        config.common_weight_factor = 1.0;
        config.disjoint_gene_factor = 0.5;
        config.excess_gene_factor = 0.5;

        let mut genome = NNGenome::new(&config);

        genome.add_node(3, ActivationType::Sigmoid).unwrap();
        genome.add_node(4, ActivationType::Sigmoid).unwrap();
        genome.add_gene(3, 2, 4, 5.0).unwrap();
        genome.add_gene(5, 2, 2, 5.0).unwrap();
        genome.add_gene(6, 4, 4, 1.0).unwrap();

        assert_eq!(
            NNGenome::genetic_distance(&genome, &genome.clone(), &config),
            0.0
        );
    }
}
