use super::{AbsentEntryRemoval, GeneValidityError};
use crate::Innovation;

use ahash::RandomState;
use serde::{Deserialize, Serialize};

use std::collections::HashSet;
use std::error::Error;
use std::fmt;

/// An ActivationType represents the type
/// of activation function the node's network
/// equivalent will use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ActivationType {
    // 1 / (1 + exp(-4.9x))
    Sigmoid,
    // x
    Identity,
    // 0   if x < 0
    // x   if x ≤ 0
    ReLU,
    // exp(-x²)
    Gaussian,
    // sin(πx)
    Sinusoidal,
}

/// A NodeType indicates the function of
/// the node's network equivalent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct Node {
    id: Innovation,
    inputs: HashSet<Innovation, RandomState>,
    outputs: HashSet<Innovation, RandomState>,
    node_type: NodeType,
    activation_type: ActivationType,
}

impl Node {
    /// Generate a new node with the passed parameters.
    ///
    /// # Examples
    /// ```
    /// use oxineat::genomics::{Node, NodeType, ActivationType};
    ///
    /// let node = Node::new(5, NodeType::Neuron, ActivationType::Sigmoid);
    /// ```
    pub fn new(id: Innovation, node_type: NodeType, activation_type: ActivationType) -> Node {
        Node {
            id,
            inputs: HashSet::default(),
            outputs: HashSet::default(),
            node_type,
            activation_type,
        }
    }

    /// Adds the passed innovation number to the node's
    /// list of input genes.
    ///
    /// # Errors
    /// This function returns an error if the gene is already
    /// in the node's inputs.
    ///
    /// # Examples
    /// ```
    /// use oxineat::genomics::{Node, NodeType, ActivationType};
    ///
    /// let mut node = Node::new(5, NodeType::Neuron, ActivationType::Sigmoid);
    ///
    /// node.add_input_gene(9).unwrap();
    /// assert_eq!(*node.input_genes().next().unwrap(), 9);
    ///
    /// assert!(node.add_input_gene(9).is_err());
    /// ```
    pub fn add_input_gene(&mut self, gene_id: Innovation) -> Result<(), impl Error> {
        if !self.inputs.contains(&gene_id) {
            self.inputs.insert(gene_id);
            Ok(())
        } else {
            Err(GeneValidityError::DuplicateGeneID(gene_id, None))
        }
    }

    /// Removes the input gene matching the specified
    /// innovation number.
    ///
    /// # Errors
    /// This function returns an error if the node
    /// has no input gene that matches the passed ID.
    ///
    /// # Examples
    /// ```
    /// use oxineat::genomics::{Node, NodeType, ActivationType};
    ///
    /// let mut node = Node::new(5, NodeType::Neuron, ActivationType::Sigmoid);
    ///
    /// assert!(node.remove_input_gene(9).is_err());
    ///
    /// node.add_input_gene(9).unwrap();
    ///
    /// assert_eq!(node.input_genes().count(), 1);
    ///
    /// assert!(node.remove_input_gene(9).is_ok());
    /// assert_eq!(node.input_genes().count(), 0);
    /// ```
    pub fn remove_input_gene(&mut self, gene_id: Innovation) -> Result<(), impl Error> {
        if !self.inputs.remove(&gene_id) {
            Err(AbsentEntryRemoval::Gene(gene_id))
        } else {
            Ok(())
        }
    }

    /// Adds the passed innovation number to the node's
    /// list of output genes.
    ///
    /// # Errors
    /// This function returns an error if the gene is already
    /// in the node's outputs.
    ///
    /// # Examples
    /// ```
    /// use oxineat::genomics::{Node, NodeType, ActivationType};
    ///
    /// let mut node = Node::new(5, NodeType::Neuron, ActivationType::Sigmoid);
    ///
    /// node.add_output_gene(9).unwrap();
    /// assert_eq!(*node.output_genes().next().unwrap(), 9);
    ///
    /// assert!(node.add_output_gene(9).is_err());
    /// ```
    pub fn add_output_gene(&mut self, gene_id: Innovation) -> Result<(), impl Error> {
        if !self.outputs.contains(&gene_id) {
            self.outputs.insert(gene_id);
            Ok(())
        } else {
            Err(GeneValidityError::DuplicateGeneID(gene_id, None))
        }
    }

    /// Removes the output gene matching the specified
    /// innovation number.
    ///
    /// # Errors
    /// This function returns an error if the node
    /// has no output gene that matches the passed ID.
    ///
    /// # Examples
    /// ```
    /// use oxineat::genomics::{Node, NodeType, ActivationType};
    ///
    /// let mut node = Node::new(5, NodeType::Neuron, ActivationType::Sigmoid);
    ///
    /// assert!(node.remove_output_gene(9).is_err());
    ///
    /// node.add_output_gene(9).unwrap();
    ///
    /// assert_eq!(node.output_genes().count(), 1);
    ///
    /// assert!(node.remove_output_gene(9).is_ok());
    /// assert_eq!(node.output_genes().count(), 0);
    /// ```
    pub fn remove_output_gene(&mut self, gene_id: Innovation) -> Result<(), impl Error> {
        if !self.outputs.remove(&gene_id) {
            Err(AbsentEntryRemoval::Gene(gene_id))
        } else {
            Ok(())
        }
    }

    /// Returns the node's innovation number.
    ///
    /// # Examples
    /// ```
    /// use oxineat::genomics::{Node, NodeType, ActivationType};
    ///
    /// let node = Node::new(5, NodeType::Neuron, ActivationType::Sigmoid);
    ///
    /// assert_eq!(node.innovation(), 5);
    /// ```
    pub fn innovation(&self) -> Innovation {
        self.id
    }

    /// Returns an iterator over the node's input genes.
    ///
    /// # Examples
    /// ```
    /// use oxineat::genomics::{Node, NodeType, ActivationType};
    ///
    /// let mut node = Node::new(5, NodeType::Neuron, ActivationType::Sigmoid);
    /// node.add_input_gene(0);
    /// node.add_input_gene(1);
    /// node.add_input_gene(2);
    ///
    /// for gene in node.input_genes() {
    ///     println!("Node {} has an input gene with id {}", node.innovation(), *gene);
    /// }
    /// ```
    pub fn input_genes(&self) -> impl Iterator<Item = &Innovation> {
        self.inputs.iter()
    }

    /// Returns an iterator over the node's output genes.
    ///
    /// # Examples
    /// ```
    /// use oxineat::genomics::{Node, NodeType, ActivationType};
    ///
    /// let mut node = Node::new(5, NodeType::Neuron, ActivationType::Sigmoid);
    /// node.add_output_gene(0);
    /// node.add_output_gene(1);
    /// node.add_output_gene(2);
    ///
    /// for gene in node.output_genes() {
    ///     println!("Node {} has an output gene with id {}", node.innovation(), *gene);
    /// }
    /// ```
    pub fn output_genes(&self) -> impl Iterator<Item = &Innovation> {
        self.outputs.iter()
    }

    /// Returns the node's node type.
    ///
    /// # Examples
    /// ```
    /// use oxineat::genomics::{Node, NodeType, ActivationType};
    ///
    /// let node = Node::new(5, NodeType::Neuron, ActivationType::Sigmoid);
    ///
    /// assert_eq!(node.node_type(), NodeType::Neuron);
    /// ```
    pub fn node_type(&self) -> NodeType {
        self.node_type
    }

    /// Returns the node's activation type.
    ///
    /// # Examples
    /// ```
    /// use oxineat::genomics::{Node, NodeType, ActivationType};
    ///
    /// let node = Node::new(5, NodeType::Neuron, ActivationType::Sigmoid);
    ///
    /// assert_eq!(node.activation_type(), ActivationType::Sigmoid);
    /// ```
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
