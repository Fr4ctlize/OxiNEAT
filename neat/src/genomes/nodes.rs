use crate::Innovation;

/// An ActivationType represents the type 
/// of activation function the node's network
/// equivalent will use.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ActivationType {
    Logistic,
    Linear,
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
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    id: Innovation,
    inputs: Vec<Innovation>,
    outputs: Vec<Innovation>,
    node_type: NodeType,
    activation_type: ActivationType,
}

impl Node {
    /// Generate a new node with the passed parameters.
    pub fn new(id: Innovation, node_type: NodeType, activation_type: ActivationType) -> Node {
        todo!()
    }

    /// Adds the passed innovation number to the node's
    /// list of input genes.
    pub fn add_input_gene(&mut self, input_id: Innovation) {
        todo!()
    }

    /// Adds the passed innovation number to the node's
    /// list of output genes.
    pub fn add_output_gene(&mut self, output_id: Innovation) {
        todo!()
    }

    /// Returns the list of the node's input genes.
    pub fn input_genes(&self) -> Vec<Innovation> {
        todo!()
    }

    /// Returns the list of the node's output genes.
    pub fn output_genes(&self) -> Vec<Innovation> {
        todo!()
    }

    /// Returns the node's node type.
    pub fn node_type(&self) -> NodeType {
        todo!()
    }
}

#[cfg(test)]
mod tests {}
