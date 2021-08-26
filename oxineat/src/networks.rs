//! A Network is a simple near-isomorphism of a Genome
//! generated as the phenotypes of said Genome,
//! with suppressed genes being ignored. Genes are
//! converted into connections, and genome nodes
//! into network nodes.
mod connections;
mod nodes;
use connections::Connection;
use nodes::Node;

use crate::genomes::{Genome, NodeType};

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

/// A arbitrarily-structured neural network.
#[derive(Clone)]
pub struct Network {
    inputs: Vec<Rc<RefCell<Node>>>,
    hidden: Vec<Rc<RefCell<Node>>>,
    outputs: Vec<Rc<RefCell<Node>>>,
}

impl Network {
    /// Generates a new network from the passed genome.
    pub fn from(genome: &Genome) -> Network {
        let mut network = Network {
            inputs: vec![],
            hidden: vec![],
            outputs: vec![],
        };

        let mut node_map = HashMap::new();

        for node in genome.nodes() {
            let network_node = Rc::new(RefCell::new(Node::new(
                node.innovation(),
                node.activation_type(),
            )));
            node_map.insert(node.innovation(), network_node.clone());
            match node.node_type() {
                NodeType::Sensor => network.inputs.push(network_node),
                NodeType::Neuron => network.hidden.push(network_node),
                NodeType::Actuator => network.outputs.push(network_node),
            }
        }

        for gene in genome.genes().filter(|g| !g.suppressed()) {
            if gene.input() == gene.output() {
                let node = node_map[&gene.input()].clone();
                node.borrow_mut().recursive_connection = Some(Connection::new(
                    gene.innovation(),
                    Rc::downgrade(&node),
                    gene.weight(),
                ));
            } else {
                let connection_output = node_map[&gene.output()].clone();
                let connection = Connection::new(
                    gene.innovation(),
                    Rc::downgrade(&connection_output),
                    gene.weight(),
                );
                node_map[&gene.input()]
                    .borrow_mut()
                    .outputs
                    .push(connection);
            }
        }

        // Sorting by ID makes activation and I/O predictable,
        // as iteration through the genome leads to random orderings.
        network.inputs.sort_unstable_by_key(|n| n.borrow().id());
        network.hidden.sort_unstable_by_key(|n| n.borrow().id());
        network.outputs.sort_unstable_by_key(|n| n.borrow().id());
        for node in network
            .inputs
            .iter()
            .chain(&network.hidden)
            .chain(&network.outputs)
        {
            node.borrow_mut().outputs.sort_unstable_by_key(|c| c.id());
        }

        network
    }

    /// Computes the input sum of each node in the network,
    /// and propagates node outputs via all connections.
    /// Set inputs are propagated directly to all connected
    /// nodes before activation levels are computed, to
    /// accelerate network reaction times.
    pub fn activate(&mut self) {
        for node in &self.inputs {
            node.borrow_mut().fire();
        }
        for node in self.hidden.iter().chain(&self.outputs) {
            node.borrow_mut().compute_activation();
        }
        for node in self.hidden.iter().chain(&self.outputs) {
            node.borrow_mut().fire();
        }
    }

    /// NOTE: Does this even have a sensible definition?
    /// It seems that not even solving the longest-path
    /// problem would work in networks with cyclical
    /// paths, as the network may not have an asymptotically
    /// stable output in such a case. This method
    /// may not make sense for this kind of network.
    ///
    /// Applies [`activate`] repeatedly until all inputs
    /// have affected all outputs. Calling [`clear_state`]
    /// beforehand is recommended.
    ///
    /// [`activate`]: crate::networks::Network::activate
    /// [`clear_state`]: crate::networks::Network::clear_state
    pub fn activate_fully(&mut self) {
        todo!("reconsidering viablity")
    }

    /// Clears the activation state of all nodes.
    pub fn clear_state(&mut self) {
        for node in self.inputs.iter().chain(&self.outputs).chain(&self.hidden) {
            node.borrow_mut().reset();
        }
    }

    /// Sets the activation level of each input node
    /// to the corresponding value in the passed slice.
    ///
    /// # Errors
    /// This function panics if the length of the passed
    /// slice is not equal to the number of inputs in the network.
    pub fn set_inputs(&mut self, values: &[f32]) {
        for (input, v) in self.inputs.iter().zip(values) {
            let mut input = input.borrow_mut();
            input.activation = *v;
            input.activated = true;
        }
    }

    /// Returns the current output node activation levels
    /// as a vector.
    pub fn outputs(&self) -> Vec<f32> {
        self.outputs
            .iter()
            .map(|n| n.borrow().activation())
            .collect()
    }
}

impl fmt::Debug for Network {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inputs: Vec<_> = self.inputs.iter().map(|n| n.borrow()).collect();
        let hidden: Vec<_> = self.hidden.iter().map(|n| n.borrow()).collect();
        let outputs: Vec<_> = self.outputs.iter().map(|n| n.borrow()).collect();
        f.debug_struct("Network")
            .field("inputs", &inputs)
            .field("hidden", &hidden)
            .field("outputs", &outputs)
            .finish()
    }
}

impl fmt::Display for Network {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (self as &dyn fmt::Debug).fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{genomes::ActivationType, GeneticConfig};
    use std::num::NonZeroUsize;

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-4.9 * x).exp())
    }

    #[test]
    fn from() {
        let mut config = GeneticConfig::default();
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

        let network = Network::from(&genome);
        assert_eq!(network.inputs.len(), 2);
        assert_eq!(network.outputs.len(), 2);
        assert_eq!(network.hidden.len(), 1);
        assert_eq!(
            network.outputs[0].borrow().function,
            ActivationType::Sigmoid
        );
        assert_eq!(
            network.outputs[1].borrow().function,
            ActivationType::Gaussian
        );
        dbg!(&network);
        // Check for suppressed gene.
        assert!(network.inputs[0]
            .borrow()
            .outputs
            .iter()
            .all(|c| c.id() != 1));
        for node in network
            .inputs
            .iter()
            .chain(&network.outputs)
            .chain(&network.hidden)
        {
            let node = node.borrow();
            for (i, idx) in (0..7).filter(|i| inputs[*i] == node.id()).enumerate() {
                if outputs[idx] == inputs[idx] {
                    assert_eq!(node.recursive_connection.as_ref().unwrap().id(), ids[idx]);
                    assert_eq!(
                        node.recursive_connection
                            .as_ref()
                            .unwrap()
                            .output
                            .upgrade()
                            .unwrap()
                            .borrow()
                            .id(),
                        outputs[idx]
                    );
                    assert_eq!(
                        node.recursive_connection.as_ref().unwrap().weight,
                        weights[idx]
                    );
                    println!(
                        "{} {} {} {}",
                        ids[idx], inputs[idx], outputs[idx], weights[idx]
                    );
                } else {
                    assert_eq!(node.outputs[i].id(), ids[idx]);
                    assert_eq!(
                        node.outputs[i].output.upgrade().unwrap().borrow().id(),
                        outputs[idx]
                    );
                    assert_eq!(node.outputs[i].weight, weights[idx]);
                    println!(
                        "{} {} {} {}",
                        ids[idx], inputs[idx], outputs[idx], weights[idx]
                    );
                }
            }
        }
    }

    #[test]
    fn activate_empty() {
        let genome = Genome::new(&GeneticConfig::default());
        let mut network = Network::from(&genome);
        assert!((0..100).all(|_| {
            network.activate();
            network.outputs()[0] == 0.0
        }));
    }

    #[test]
    fn activate_single() {
        let mut genome = Genome::new(&GeneticConfig::default());
        genome.add_gene(0, 0, 1, 1.0);
        let mut network = Network::from(&genome);
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
        let mut genome = Genome::new(&GeneticConfig::default());
        genome.add_gene(0, 0, 1, 1.0);
        genome.add_gene(1, 1, 1, -1.0); // Recursive connection
        let mut network = Network::from(&genome);
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
        let mut genome = Genome::new(&GeneticConfig::default());
        genome.add_node(2, ActivationType::Sigmoid);
        genome.add_gene(0, 0, 2, 1.0);
        genome.add_gene(1, 2, 1, 1.0);
        let mut network = Network::from(&genome);
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
        let mut config = GeneticConfig::default();
        config.input_count = NonZeroUsize::new(3).unwrap();
        let mut genome = Genome::new(&config);
        genome.add_gene(0, 0, 3, -1.0);
        genome.add_gene(1, 1, 3, 1.0);
        genome.add_gene(2, 2, 3, 0.5);
        let mut network = Network::from(&genome);
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

    #[test]
    fn activate_fully() {
        let mut genome = Genome::new(&GeneticConfig::default());
        genome.add_node(2, ActivationType::Sigmoid);
        genome.add_gene(0, 0, 2, 1.0);
        genome.add_gene(1, 2, 1, 1.0);
        let mut network = Network::from(&genome);
        for input in -20..=20 {
            let input = input as f32 / 10.0;
            network.clear_state();
            network.set_inputs(&[input]);
            network.activate_fully();
            assert_eq!(network.outputs()[0], sigmoid(sigmoid(input)));
        }
    }
}
