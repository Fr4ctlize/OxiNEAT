use super::RealTimeNetwork;
use crate::genomics::Genome;

/// A neural network best suited for function
/// approximation.
pub struct FunctionApproximatorNetwork {
    network: RealTimeNetwork,
    depth: usize,
}

impl FunctionApproximatorNetwork {
    /// Generates a new network from the passed genome.
    ///
    /// # Generic parameters
    /// `MAX_NODE_VISITS`: the maximum number of times a node
    /// can be visited in a path through the network before
    /// the network's activation freezes. Setting it to 0 will
    /// effectively disable the entire network, 1 will dissallow
    /// any cycles, 2 will allow single pass through the longest
    /// cycle in the network, etc.
    /// 
    /// # Complexity
    /// This function has `O(d^(n × MAX_NODE_VISITS))` time complexity,
    /// and `O(n × MAX_NODE_VISITS)` space complexity,
    /// where `d` is the highest output count in the genome's nodes.
    ///
    /// # Examples
    /// ```
    /// use oxineat::genomics::{GeneticConfig, Genome};
    /// use oxineat::networks::FunctionApproximatorNetwork;
    ///
    /// let genome = Genome::new(&GeneticConfig::zero());
    /// let network = FunctionApproximatorNetwork::new::<1>(&genome);
    /// ````
    pub fn new<const MAX_NODE_VISITS: u8>(genome: &Genome) -> FunctionApproximatorNetwork {
        let network = RealTimeNetwork::new(genome);
        let depth = (0..network.input_count)
            .map(|root| {
                Self::calculate_depth::<MAX_NODE_VISITS>(
                    &network,
                    root,
                    &mut vec![0; network.connections.len()],
                    0,
                )
            })
            .max()
            .unwrap();

        FunctionApproximatorNetwork { network, depth }
    }

    /// Calculates the length of the longest path
    /// from the `root` node that doesn't pass through
    /// any node more than `MAX_NODE_VISITS` times.
    /// 
    /// # Complexity
    /// This function has `O(d^(n × MAX_NODE_VISITS))` time complexity,
    /// and `O(n × MAX_NODE_VISITS)` space complexity,
    /// where `d` is the highest output count in the genome's nodes.
    fn calculate_depth<const MAX_NODE_VISITS: u8>(
        network: &RealTimeNetwork,
        root: usize,
        visited: &mut [u8],
        current_depth: usize,
    ) -> usize {
        let mut max_depth = 0;

        for c in network.connections[root].iter() {
            if visited[c.output] < MAX_NODE_VISITS {
                visited[c.output] += 1;
                max_depth = max_depth.max(Self::calculate_depth::<MAX_NODE_VISITS>(
                    network,
                    c.output,
                    visited,
                    current_depth + 1,
                ));
                visited[c.output] -= 1;
            }
        }

        if max_depth == 0
            && (network.input_count..network.input_count + network.output_count).contains(&root)
        {
            current_depth
        } else {
            max_depth
        }
    }

    /// Returns the approximated function's value
    /// at the N-dimensional point given by `inputs`.
    ///
    /// # Examples
    /// ```
    /// use oxineat::genomics::{ActivationType, GeneticConfig, Genome};
    /// use oxineat::networks::FunctionApproximatorNetwork;
    /// 
    /// fn sigmoid(x: f32) -> f32 {
    ///     1.0 / (1.0 + (-4.9 * x).exp())
    /// }
    ///
    /// let mut genome = Genome::new(&GeneticConfig::zero());
    /// genome.add_node(2, ActivationType::Sigmoid).unwrap();
    /// genome.add_gene(0, 0, 2, 1.0).unwrap();
    /// genome.add_gene(1, 2, 1, 1.0).unwrap();
    /// let mut network = FunctionApproximatorNetwork::new::<1>(&genome);
    /// 
    /// for input in -20..=20 {
    ///     let input = input as f32 / 10.0;
    ///     assert_eq!(network.evaluate_at(&[input])[0], sigmoid(sigmoid(input)));
    /// }
    /// ```
    pub fn evaluate_at(&mut self, inputs: &[f32]) -> Vec<f32> {
        self.network.clear_state();
        self.network.set_inputs(inputs);
        for _ in 0..self.depth {
            self.network.activate();
        }
        self.network.outputs()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::genomics::{ActivationType, GeneticConfig};

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-4.9 * x).exp())
    }

    fn test_genome() -> Genome {
        let mut genome = Genome::new(&GeneticConfig::zero());
        genome.add_node(2, ActivationType::Sigmoid).unwrap();
        genome.add_gene(0, 0, 1, 0.0).unwrap();
        genome.add_gene(1, 0, 2, 0.0).unwrap();
        genome.add_gene(2, 1, 1, 0.0).unwrap();
        genome.add_gene(3, 1, 2, 0.0).unwrap();
        genome.add_gene(4, 2, 1, 0.0).unwrap();
        genome.add_gene(5, 2, 2, 0.0).unwrap();
        genome
    }

    #[test]
    fn max_visits_0() {
        let network = FunctionApproximatorNetwork::new::<0>(&test_genome());
        assert_eq!(network.depth, 0)
    }

    #[test]
    fn max_visits_1() {
        let network = FunctionApproximatorNetwork::new::<1>(&test_genome());
        assert_eq!(network.depth, 2)
    }

    #[test]
    fn max_visits_2() {
        let network = FunctionApproximatorNetwork::new::<2>(&test_genome());
        assert_eq!(network.depth, 4)
    }

    #[test]
    fn max_visits_3() {
        let network = FunctionApproximatorNetwork::new::<3>(&test_genome());
        assert_eq!(network.depth, 6)
    }

    #[test]
    fn evaluate_at() {
        let mut genome = Genome::new(&GeneticConfig::zero());
        genome.add_node(2, ActivationType::Sigmoid).unwrap();
        genome.add_gene(0, 0, 2, 1.0).unwrap();
        genome.add_gene(1, 2, 1, 1.0).unwrap();
        let mut network = FunctionApproximatorNetwork::new::<1>(&genome);
        for input in -20..=20 {
            let input = input as f32 / 10.0;
            assert_eq!(network.evaluate_at(&[input])[0], sigmoid(sigmoid(input)));
        }
    }
}
