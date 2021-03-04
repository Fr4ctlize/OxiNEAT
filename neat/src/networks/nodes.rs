use super::Connection;
use crate::Innovation;
use crate::genomes::ActivationType;

/// Network equivalent of nodes in genomes.
#[derive(Clone, Debug)]
pub struct Node<'a> {
    id: Innovation,
    pub(super) inputs: &'a Connection<'a>,
    pub(super) outputs: &'a Connection<'a>,
    input_sum: f64,
    activation: f64,
    function: ActivationType,
}

impl<'a> Node<'a> {
    /// Creates a new node with the passed innovation
    /// number and activation type.
    pub fn new(id: Innovation, function: ActivationType) -> Node<'a> {
        todo!()
    }

    /// Returns the node's innovation number.
    pub fn innovation(&self) -> Innovation {
        todo!()
    }

    /// Propagates the node's input sum to all output connections.
    pub fn fire(&mut self) {
        todo!()
    }

    /// Applies the node's activation function to its input sum,
    /// resulting in the node's activation level.
    pub fn compute_activation(&mut self) {
        todo!()
    }

    /// Adds `x` to the node's input sum.
    pub fn add_to_input_sum(&mut self, x: f64) {
        todo!()
    }

    /// Sets the node's input sum and activation
    /// level to 0.
    pub fn reset(&mut self) {
        todo!()
    }
}

#[cfg(test)]
mod tests {}
