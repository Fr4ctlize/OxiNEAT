use std::fmt;

#[derive(Clone, Copy, PartialEq)]
pub struct Connection {
    pub output: usize,
    pub weight: f32,
}

impl Connection {
    /// Creates a new Connection with the specified
    /// output node and weight.
    pub fn new(output: usize, weight: f32) -> Connection {
        Connection { output, weight }
    }
}

impl fmt::Debug for Connection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {:.9}", self.output, self.weight)
    }
}
