mod connections;
mod nodes;

use connections::Connection;
use nodes::Node;

use crate::genomes::Genome;
use crate::Result;

use std::cell::RefCell;
use std::rc::Rc;

/// A Network is a simple near-isomorphism of a Genome
/// generated as the phenotypes of said Genome,
/// with suppressed genes being ignored. Genes are
/// converted into connections, and genome nodes
/// into network nodes.
#[derive(Clone, Debug)]
pub struct Network<'a> {
    inputs: Vec<Rc<RefCell<Node<'a>>>>,
    hidden: Vec<Rc<RefCell<Node<'a>>>>,
    outputs: Vec<Rc<RefCell<Node<'a>>>>,
    connections: Vec<Connection<'a>>,
}

impl<'a> Network<'a> {
    /// Generates a new network from the passed genome.
    pub fn from(genome: &Genome) -> Result<Network> {
        todo!()
    }

    /// Computes the input sum of each node in the network,
    /// and propagates node outputs via all connections.
    pub fn activate(&mut self) {
        todo!()
    }

    /// Applies [`activate`] repeatedly until all inputs
    /// have affected all outputs. Calling [`clear_state`]
    /// beforehand is recommended.
    ///
    /// [`activate`]: crate::networks::Network::activate
    /// [`clear_state`]: crate::networks::Network::clear_state
    pub fn activate_fully(&mut self) {
        todo!()
    }

    /// Clears the activation state of all nodes.
    pub fn clear_state(&mut self) {
        todo!()
    }

    /// Sets the activation level of each input node
    /// to the corresponding value in the passed slice.
    ///
    /// # Errors
    /// This function panics if the length of the passed
    /// slice is not equal to the number of inputs in the network.
    pub fn set_inputs(&mut self, values: &[f32]) {
        todo!()
    }

    /// Returns the current output node activation levels
    /// as a vector.
    pub fn outputs(&self) -> Vec<f32> {
        todo!()
    }
}

#[cfg(test)]
mod tests {}
