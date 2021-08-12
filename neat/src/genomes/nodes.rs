use crate::Innovation;

use std::collections::HashSet;
use std::fmt;

/// An ActivationType represents the type
/// of activation function the node's network
/// equivalent will use.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ActivationType {
    Sigmoid,
    Identity,
    ReLU,
    Gaussian,
    Sinusoidal,
}

/// A NodeType indicates the function of
/// the node's network equivalent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NodeType {
    /// Input nodes.
    Sensor,
    /// Hidden nodes.
    Neuron,
    /// Output nodes.
    Actuator,
}

/// Nodes are the structural elements of genomes
/// between which genes are created.
#[derive(Clone, PartialEq, Debug)]
pub struct Node {
    id: Innovation,
    inputs: HashSet<Innovation>,
    outputs: HashSet<Innovation>,
    node_type: NodeType,
    activation_type: ActivationType,
}

impl Node {
    /// Generate a new node with the passed parameters.
    pub fn new(id: Innovation, node_type: NodeType, activation_type: ActivationType) -> Node {
        Node {
            id,
            inputs: HashSet::new(),
            outputs: HashSet::new(),
            node_type,
            activation_type,
        }
    }

    /// Adds the passed innovation number to the node's
    /// list of input genes.
    ///
    /// # Panics
    /// This function panics if the gene is already
    /// in the node's inputs.
    pub fn add_input_gene(&mut self, input_id: Innovation) {
        if !self.inputs.contains(&input_id) {
            self.inputs.insert(input_id);
        } else {
            panic!("attempted to add duplicate input with ID {}", input_id)
        }
    }

    /// Adds the passed innovation number to the node's
    /// list of output genes.
    ///
    /// # Panics
    /// This function panics if the gene is already
    /// in the node's outputs.
    pub fn add_output_gene(&mut self, output_id: Innovation) {
        if !self.outputs.contains(&output_id) {
            self.outputs.insert(output_id);
        } else {
            panic!("attempted to add duplicate output with ID {}", output_id)
        }
    }

    /// Returns the node's innovation number.
    pub fn innovation(&self) -> Innovation {
        self.id
    }

    /// Returns the list of the node's input genes.
    pub fn input_genes(&self) -> impl Iterator<Item = &Innovation> {
        self.inputs.iter()
    }

    /// Returns the list of the node's output genes.
    pub fn output_genes(&self) -> impl Iterator<Item = &Innovation> {
        self.outputs.iter()
    }

    /// Returns the node's node type.
    pub fn node_type(&self) -> NodeType {
        self.node_type
    }

    /// Returns the node's activation type.
    pub fn activation_type(&self) -> ActivationType {
        self.activation_type
    }
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?}[{:?}, {:?}, IN: {:?}, OUT: {:?}]",
            self.id, self.node_type, self.activation_type, self.inputs, self.outputs,
        )
    }
}

#[cfg(test)]
mod tests {}
