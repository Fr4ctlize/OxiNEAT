use crate::Innovation;

use super::Node;

use std::cell::RefCell;
use std::rc::Rc;

/// Network equivalent of genes in genomes.
#[derive(Clone, Debug)]
pub struct Connection<'a> {
    id: Innovation,
    input: Rc<RefCell<Node<'a>>>,
    output: Rc<RefCell<Node<'a>>>,
    weight: f64,
}

impl<'a> Connection<'a> {
    /// Create a new connection between the specified
    /// input and output nodes.
    pub fn new(input_id: Rc<RefCell<Node<'a>>>, output_id: Rc<RefCell<Node<'a>>>, weight: f64) {
        todo!()
    }

    /// Propagate the input node's input sum, scaled
    /// by the connection's weight.
    pub fn propagate_activation(&self) {
        todo!()
    }
}

#[cfg(test)]
mod tests {}
