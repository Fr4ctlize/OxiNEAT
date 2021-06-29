use crate::Innovation;

use super::Node;

use std::cell::RefCell;
use std::rc::Weak;
use std::fmt;

/// Network equivalent of genes in genomes.
#[derive(Clone)]
pub struct Connection {
    id: Innovation,
    pub(super) output: Weak<RefCell<Node>>,
    pub(super) weight: f32,
}

impl Connection {
    /// Create a new connection between the specified
    /// input and output nodes.
    pub fn new(id: Innovation, output: Weak<RefCell<Node>>, weight: f32) -> Connection {
        Connection {
            id,
            output,
            weight,
        }
    }

    /// Returns the connections's innovation number.
    pub fn id(&self) -> Innovation {
        self.id
    }

    /// Propagate the input node's activation, scaled
    /// by the connection's weight.
    pub fn propagate_activation(&self, activation: f32) {
        self.output
            .upgrade()
            .expect("upgraded Weak pointer to deallocated Node")
            .borrow_mut()
            .add_to_input_sum(activation * self.weight);
    }
}

impl fmt::Debug for Connection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Connection")
         .field("id", &self.id)
         .field("output", &self.output.upgrade().expect("upgraded Weak pointer to deallocated Node").borrow().id())
         .field("weight", &self.weight)
         .finish()
    }
}

#[cfg(test)]
mod tests {}
