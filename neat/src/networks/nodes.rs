use super::Connection;
use crate::genomes::ActivationType;
use crate::Innovation;

/// Network equivalent of nodes in genomes.
#[derive(Clone, Debug)]
pub struct Node {
    id: Innovation,
    pub(super) outputs: Vec<Connection>,
    pub(super) recursive_connection: Option<Connection>,
    input_sum: f32,
    pub(super) activation: f32,
    pub(super) activated: bool,
    will_activate: bool,
    pub(super) function: ActivationType,
}

impl Node {
    /// Creates a new node with the passed innovation
    /// number and activation type.
    pub fn new(id: Innovation, function: ActivationType) -> Node {
        Node {
            id,
            outputs: vec![],
            recursive_connection: None,
            input_sum: 0.0,
            activation: 0.0,
            activated: false,
            will_activate: false,
            function,
        }
    }

    /// Returns the node's innovation number.
    pub fn id(&self) -> Innovation {
        self.id
    }

    /// Returns the node's activation value.
    pub fn activation(&self) -> f32 {
        self.activation
    }

    /// Propagates the node's input sum to all output connections.
    pub fn fire(&mut self) {
        if self.activated {
            for connection in &mut self.outputs {
                connection.propagate_activation(self.activation);
            }
            if let Some(Connection{weight, ..}) = &self.recursive_connection {
                self.input_sum += self.activation * weight;
                self.will_activate = true;
            }
            self.activated = false;
        }
    }

    /// Applies the node's activation function to its input sum,
    /// resulting in the node's activation level.
    pub fn compute_activation(&mut self) {
        self.activated = self.will_activate;
        self.will_activate = false;
        if self.activated {
            self.activation = match self.function {
                ActivationType::Sigmoid => 1.0 / (1.0 + (-4.9 * self.input_sum).exp()),
                ActivationType::Identity => self.input_sum,
                ActivationType::ReLU => {
                    if self.input_sum < 0.0 {
                        0.0
                    } else {
                        self.input_sum
                    }
                }
                ActivationType::Gaussian => (-self.input_sum * self.input_sum).exp(),
                ActivationType::Sinusoidal => (self.input_sum / std::f32::consts::PI).sin(),
            };
            self.input_sum = 0.0;
        } else {
            self.activation = 0.0;
        }
    }

    /// Adds `x` to the node's input sum.
    pub fn add_to_input_sum(&mut self, x: f32) {
        self.input_sum += x;
        self.will_activate = true;
    }

    /// Sets the node's input sum and activation
    /// level to 0.
    pub fn reset(&mut self) {
        self.input_sum = 0.0;
        self.activation = 0.0;
        self.activated = false;
        self.will_activate = false;
    }
}

#[cfg(test)]
mod tests {}
