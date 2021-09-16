//! A Network is a simple near-isomorphism of a Genome
//! generated as the phenotypes of said Genome,
//! with suppressed genes being ignored. Genes are
//! converted into connections, and genome nodes
//! into network nodes.
//! 
//! The `RealTimeNetwork` type is best suited for real-time
//! control tasks, with new inputs set for each activation,
//! and multiple time-steps involved.
//! 
//! For a more instantaneous input-result use-case, the
//! `FunctionApproximatorNetwork` type is more appropiate.
mod connection;
mod function_approximator;

pub use function_approximator::FunctionApproximatorNetwork;

use crate::genomics::{ActivationType, Genome, NodeType};
use crate::Innovation;
use connection::Connection;

use ahash::RandomState;

use std::collections::HashMap;
use std::fmt;

/// An arbitrarily-structured neural network.
#[derive(Clone, Debug)]
pub struct RealTimeNetwork {
    input_count: usize,
    output_count: usize,
    node_ids: Box<[Innovation]>,
    input_sums: Box<[f32]>,
    activation_levels: Box<[f32]>,
    activation_functions: Box<[ActivationType]>,
    connections: Box<[Box<[Connection]>]>,
}

impl RealTimeNetwork {
    /// Generates a new network from the passed genome.
    /// 
    /// # Examples
    /// ```
    /// use oxineat::{
    ///     genomics::{ActivationType, GeneticConfig, Genome},
    ///     networks::RealTimeNetwork,
    /// };
    /// use std::num::NonZeroUsize;
    ///
    /// let genome = Genome::new(&GeneticConfig {
    ///     input_count: NonZeroUsize::new(3).unwrap(),
    ///     output_count: NonZeroUsize::new(2).unwrap(),
    ///     initial_expression_chance: 1.0,
    ///     weight_bound: 5.0,
    ///     ..GeneticConfig::zero()
    /// });
    /// 
    /// let network = RealTimeNetwork::new(&genome);
    /// ```
    pub fn new(genome: &Genome) -> RealTimeNetwork {
        let mut input_nodes = vec![];
        let mut output_nodes = vec![];
        let mut hidden_nodes = vec![];

        for node in genome.nodes() {
            match node.node_type() {
                NodeType::Sensor => &mut input_nodes,
                NodeType::Actuator => &mut output_nodes,
                NodeType::Neuron => &mut hidden_nodes,
            }
            .push((node.innovation(), node.activation_type()));
        }
        // Sorting by id makes the resulting network
        // deterministic, independantly of node iteration order.
        input_nodes.sort_unstable_by_key(|(id, _)| *id);
        output_nodes.sort_unstable_by_key(|(id, _)| *id);
        hidden_nodes.sort_unstable_by_key(|(id, _)| *id);
        let (node_ids, activation_functions): (Vec<_>, Vec<_>) = input_nodes
            .iter()
            .chain(&output_nodes)
            .chain(&hidden_nodes)
            .copied()
            .unzip();
        let total_node_count = input_nodes.len() + output_nodes.len() + hidden_nodes.len();

        let node_index_from_id: HashMap<_, _, RandomState> = node_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (*id, i))
            .collect();
        let mut connections = vec![vec![]; total_node_count];

        for gene in genome.genes().filter(|g| !g.suppressed()) {
            let gene_input_index = node_index_from_id[&gene.input()];
            let gene_output_index = node_index_from_id[&gene.output()];
            let connection = Connection::new(gene_output_index, gene.weight());
            connections[gene_input_index].push(connection);
        }

        RealTimeNetwork {
            input_count: input_nodes.len(),
            output_count: output_nodes.len(),
            node_ids: node_ids.into(),
            input_sums: vec![0.0; total_node_count].into(),
            activation_levels: vec![0.0; total_node_count].into(),
            activation_functions: activation_functions.into(),
            connections: connections.into_iter().map(|v| v.into()).collect(),
        }
    }

    /// Fires all nodes, propagating all activations
    /// (including set inputs), and then computing
    /// new activation levels.
    /// 
    /// # Examples
    /// ```
    /// use oxineat::{
    ///     genomics::{ActivationType, GeneticConfig, Genome},
    ///     networks::RealTimeNetwork,
    /// };
    /// use std::num::NonZeroUsize;
    ///
    /// let mut genome = Genome::new(&GeneticConfig {
    ///     input_count: NonZeroUsize::new(2).unwrap(),
    ///     output_count: NonZeroUsize::new(1).unwrap(),
    ///     output_activation_types: vec![ActivationType::ReLU],
    ///     ..GeneticConfig::zero()
    /// });
    /// genome.add_gene(0, 0, 2, 2.5);
    /// genome.add_gene(1, 1, 2, -2.5);
    /// 
    /// let mut network = RealTimeNetwork::new(&genome);
    /// network.set_inputs(&[0.5, 1.0]);
    /// 
    /// network.activate();
    /// 
    /// assert_eq!(network.outputs()[0], ((0.5 * 2.5 + 1.0 * (-2.5)) as f32).max(0.0));
    /// ```
    pub fn activate(&mut self) {
        self.fire_nodes();
        self.compute_activations();
    }

    /// Propagates each node's signal through all its
    /// outgoing connections.
    fn fire_nodes(&mut self) {
        for (activation, output_connections) in self
            .activation_levels
            .iter_mut()
            .zip(self.connections.iter())
        {
            for connection in output_connections.iter() {
                self.input_sums[connection.output] += *activation * connection.weight;
            }
        }
    }

    /// Computes each node's activation level,
    /// based on input sum.
    fn compute_activations(&mut self) {
        for ((input_sum, activation_level), activation_function) in self.input_sums
            [self.input_count..]
            .iter_mut()
            .zip(&mut self.activation_levels[self.input_count..])
            .zip(&self.activation_functions[self.input_count..])
        {
            *activation_level = compute_activation(*input_sum, *activation_function);
            *input_sum = 0.0;
        }
    }

    /// Clears the activation state of all nodes.
    /// 
    /// # Examples
    /// ```
    /// use oxineat::{
    ///     genomics::{GeneticConfig, Genome},
    ///     networks::RealTimeNetwork,
    /// };
    /// 
    /// let genome = Genome::new(&GeneticConfig{
    ///     initial_expression_chance: 1.0,
    ///     ..GeneticConfig::zero()
    /// });
    /// 
    /// let mut network = RealTimeNetwork::new(&genome);
    /// network.set_inputs(&[1.0]);
    /// network.activate();
    /// assert_ne!(network.outputs()[0], 0.0);
    /// 
    /// network.clear_state();
    /// 
    /// assert_eq!(network.outputs()[0], 0.0);
    /// ```
    pub fn clear_state(&mut self) {
        for (input_sum, activation) in self
            .input_sums
            .iter_mut()
            .zip(self.activation_levels.iter_mut())
        {
            *input_sum = 0.0;
            *activation = 0.0;
        }
    }

    /// Sets the activation level of each input node
    /// to the corresponding value in the passed slice.
    ///
    /// # Errors
    /// This function panics if the length of the passed
    /// slice is not equal to the number of inputs in the network.
    /// 
    /// /// # Examples
    /// ```
    /// use oxineat::{
    ///     genomics::{GeneticConfig, Genome},
    ///     networks::RealTimeNetwork,
    /// };
    /// 
    /// let genome = Genome::new(&GeneticConfig{
    ///     initial_expression_chance: 1.0,
    ///     ..GeneticConfig::zero()
    /// });
    /// 
    /// let mut network = RealTimeNetwork::new(&genome);
    /// network.set_inputs(&[1.0]);
    /// ```
    pub fn set_inputs(&mut self, values: &[f32]) {
        self.activation_levels[..self.input_count].copy_from_slice(values);
    }

    /// Returns the current output node activation levels
    /// as a vector.
    /// 
    /// /// # Examples
    /// ```
    /// use oxineat::{
    ///     genomics::{GeneticConfig, Genome},
    ///     networks::RealTimeNetwork,
    /// };
    /// 
    /// let genome = Genome::new(&GeneticConfig{
    ///     initial_expression_chance: 1.0,
    ///     ..GeneticConfig::zero()
    /// });
    /// 
    /// let mut network = RealTimeNetwork::new(&genome);
    /// assert_eq!(network.outputs()[0], 0.0);
    /// 
    /// network.set_inputs(&[1.0]);
    /// network.activate();
    /// assert_ne!(network.outputs()[0], 0.0);
    /// ```
    pub fn outputs(&self) -> Vec<f32> {
        self.activation_levels[self.input_count..self.input_count + self.output_count].to_vec()
    }
}

// Applies one of the available functions to the input and returns the output as the result
fn compute_activation(input_sum: f32, activation_function: ActivationType) -> f32 {
    match activation_function {
        ActivationType::Sigmoid => 1.0 / (1.0 + (-4.9 * input_sum).exp()),
        ActivationType::Identity => input_sum,
        ActivationType::ReLU => input_sum.max(0.0),
        ActivationType::Gaussian => (-input_sum.powf(2.0)).exp(),
        ActivationType::Sinusoidal => (input_sum * std::f32::consts::PI).sin(),
    }
}

impl fmt::Display for RealTimeNetwork {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (self as &dyn fmt::Debug).fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genomics::{ActivationType, GeneticConfig};
    use std::num::NonZeroUsize;

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-4.9 * x).exp())
    }

    #[test]
    fn from() {
        let mut config = GeneticConfig::zero();
        config.input_count = NonZeroUsize::new(2).unwrap();
        config.output_count = NonZeroUsize::new(2).unwrap();
        config.output_activation_types = vec![ActivationType::Sigmoid, ActivationType::Gaussian];
        let mut genome = Genome::new(&config);
        genome.add_node(4, ActivationType::Sigmoid);

        let ids = [0, 2, 6, 7, 3, 5, 4];
        let inputs = [0, 0, 1, 3, 4, 4, 4];
        let outputs = [2, 4, 4, 3, 3, 2, 4];
        let weights = [1.0, 1.0, 2.5, -2.0, -1.0, -1.5, 3.2];

        for i in 0..7 {
            genome.add_gene(ids[i], inputs[i], outputs[i], weights[i]);
        }
        // Suppressed gene shouldn't be expressed in network.
        genome.add_gene(1, 0, 3, -1.0).set_suppressed(true);

        let network = RealTimeNetwork::new(&genome);
        assert_eq!(network.input_count, 2);
        assert_eq!(network.output_count, 2);
        assert_eq!(
            network.node_ids.len() - network.input_count - network.output_count,
            1
        ); // Hidden nodes
        assert_eq!(
            network.activation_functions[network.input_count],
            ActivationType::Sigmoid
        );
        assert_eq!(
            network.activation_functions[network.input_count + 1],
            ActivationType::Gaussian
        );
        dbg!(&network);
        // Check for suppressed gene.
        assert!(!network.connections[0].contains(&Connection::new(3, -1.0)));
        for node_idx in 0..network.node_ids.len() {
            let (node_id, node_outputs) =
                (network.node_ids[node_idx], &network.connections[node_idx]);
            for idx in (0..7).filter(|i| inputs[*i] == node_id) {
                assert!(node_outputs.contains(&Connection::new(outputs[idx], weights[idx])));
            }
        }
    }

    #[test]
    fn activate_empty() {
        let genome = Genome::new(&GeneticConfig::zero());
        let mut network = RealTimeNetwork::new(&genome);
        assert!((0..100).all(|_| {
            network.activate();
            network.outputs()[0] == sigmoid(0.0)
        }));
    }

    #[test]
    fn activate_single() {
        let mut genome = Genome::new(&GeneticConfig::zero());
        genome.add_gene(0, 0, 1, 1.0);
        let mut network = RealTimeNetwork::new(&genome);
        for input in -20..=20 {
            let input = input as f32 / 10.0;
            network.clear_state();
            network.set_inputs(&[input]);
            network.activate();
            assert_eq!(network.outputs()[0], sigmoid(input))
        }
    }

    #[test]
    fn activate_single_recursive() {
        let mut genome = Genome::new(&GeneticConfig::zero());
        genome.add_gene(0, 0, 1, 1.0);
        genome.add_gene(1, 1, 1, -1.0); // Recursive connection
        let mut network = RealTimeNetwork::new(&genome);
        let mut prev_output = 0.0;
        for input in -20..=20 {
            let input = input as f32 / 10.0;
            network.set_inputs(&[input]);
            network.activate();
            assert_eq!(network.outputs()[0], sigmoid(input - prev_output));
            prev_output = network.outputs()[0];
        }
    }

    #[test]
    fn activate_double() {
        let mut genome = Genome::new(&GeneticConfig::zero());
        genome.add_node(2, ActivationType::Sigmoid);
        genome.add_gene(0, 0, 2, 1.0);
        genome.add_gene(1, 2, 1, 1.0);
        let mut network = RealTimeNetwork::new(&genome);
        for input in -20..=20 {
            let input = input as f32 / 10.0;
            network.clear_state();
            network.set_inputs(&[input]);
            network.activate();
            network.activate();
            assert_eq!(network.outputs()[0], sigmoid(sigmoid(input)))
        }
    }

    #[test]
    fn activate_multiple_inputs() {
        let mut config = GeneticConfig::zero();
        config.input_count = NonZeroUsize::new(3).unwrap();
        let mut genome = Genome::new(&config);
        genome.add_gene(0, 0, 3, -1.0);
        genome.add_gene(1, 1, 3, 1.0);
        genome.add_gene(2, 2, 3, 0.5);
        let mut network = RealTimeNetwork::new(&genome);
        for ((x, y), z) in (-20..=20).zip(-20..=20).zip(-20..=20) {
            let (x, y, z) = (x as f32 / 10.0, y as f32 / 10.0, z as f32 / 10.0);
            network.clear_state();
            network.set_inputs(&[x, y, z]);
            network.activate();
            assert_eq!(
                network.outputs()[0],
                sigmoid(-x + y + 0.5 * z),
                "{} {} {}",
                x,
                y,
                z
            );
        }
    }
}
