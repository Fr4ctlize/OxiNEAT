mod config;
mod genes;
mod history;
mod nodes;

pub use config::GeneticConfig;
pub use genes::Gene;
pub use history::History;
pub use nodes::{ActivationType, Node, NodeType};

use crate::Innovation;

use rand::prelude::{IteratorRandom, Rng, SliceRandom};
use std::collections::hash_map::HashMap;
use std::collections::HashSet;
use std::fmt;

/// Genomes are the focus of evolution in NEAT.
/// They are a collection of genes and nodes that can be instantiated
/// as a phenotype (a neural network) and evaluated
/// for performance in a task, which results numerically in
/// their fitness score. Genomes can be progressively mutated,
/// thus adding complexity and functionality.
#[derive(Clone, PartialEq)]
pub struct Genome {
    genes: HashMap<Innovation, Gene>,
    nodes: HashMap<Innovation, Node>,
    node_pairings: HashSet<(Innovation, Innovation)>,
    pub(in crate) fitness: f32,
}

impl Genome {
    /// Create a new genome with the specified configuration.
    pub fn new(config: &GeneticConfig) -> Genome {
        let mut nodes = Self::generate_nodes(config);
        let (genes, node_pairings) = Self::generate_initial_genes(&mut nodes, config);

        Genome {
            genes,
            nodes,
            fitness: 0.0,
            node_pairings,
        }
    }

    fn generate_nodes(config: &GeneticConfig) -> HashMap<Innovation, Node> {
        let input_count = config.input_count.get();
        let output_count = config.output_count.get();

        let mut nodes = HashMap::with_capacity(input_count + output_count);

        for i in 0..input_count {
            nodes.insert(i, Node::new(i, NodeType::Sensor, ActivationType::Identity));
        }

        for o in 0..output_count {
            nodes.insert(
                o + input_count,
                Node::new(
                    o + input_count,
                    NodeType::Actuator,
                    *config
                        .output_activation_types
                        .get(o)
                        .unwrap_or(&ActivationType::Sigmoid),
                ),
            );
        }

        nodes
    }

    fn generate_initial_genes(
        nodes: &mut HashMap<Innovation, Node>,
        config: &GeneticConfig,
    ) -> (HashMap<Innovation, Gene>, HashSet<(Innovation, Innovation)>) {
        let input_count = config.input_count.get();
        let output_count = config.output_count.get();
        let mut genes = HashMap::new();
        let mut node_pairings = HashSet::new();

        if config.initial_expression_chance != 0.0 {
            let mut rng = rand::thread_rng();
            for i in 0..input_count {
                for o in 0..output_count {
                    if rng.gen::<f32>() < config.initial_expression_chance {
                        let id = o + i * output_count;
                        genes.insert(
                            id,
                            Gene::new(id, i, o + input_count, Gene::random_weight(config)),
                        );
                        node_pairings.insert((i, o + input_count));
                        nodes.get_mut(&i).unwrap().add_output_gene(id);
                        nodes
                            .get_mut(&(o + input_count))
                            .unwrap()
                            .add_input_gene(id);
                    }
                }
            }
        }

        (genes, node_pairings)
    }

    /// Add a new gene to the genome.
    /// Returns a reference to the new gene.
    ///
    /// # Panics
    ///
    /// This function will panic if a gene with the same
    /// `gene_id` already existed in the genome, or if either `input_id`
    /// or `output_id` do not correspond to nodes present in the genome.
    pub fn add_gene(
        &mut self,
        gene_id: Innovation,
        input_id: Innovation,
        output_id: Innovation,
        weight: f32,
    ) -> &mut Gene {
        self.check_gene_viability(gene_id, input_id, output_id)
            .unwrap();
        unsafe { self.add_gene_unchecked(gene_id, input_id, output_id, weight) }
    }

    /// Add a new gene to the genome.
    /// Returns a reference to the new gene.
    /// Assumes that the gene is not a duplicate
    /// or invalid gene for the genome.
    unsafe fn add_gene_unchecked(
        &mut self,
        gene_id: Innovation,
        input_id: Innovation,
        output_id: Innovation,
        weight: f32,
    ) -> &mut Gene {
        self.nodes
            .get_mut(&input_id)
            .unwrap()
            .add_output_gene(gene_id);
        self.nodes
            .get_mut(&output_id)
            .unwrap()
            .add_input_gene(gene_id);
        self.node_pairings.insert((input_id, output_id));
        self.genes
            .entry(gene_id)
            .or_insert_with(|| Gene::new(gene_id, input_id, output_id, weight))
    }

    /// Checks whether a gene is a duplicate or
    /// is invalid for the genome.
    ///
    /// # Errors
    ///
    /// Returns an error if the gene is not viable.
    fn check_gene_viability(
        &self,
        gene_id: Innovation,
        input_id: Innovation,
        output_id: Innovation,
    ) -> Result<(), String> {
        if self.genes.contains_key(&gene_id) {
            Err(format!("duplicate gene insertion with ID {}", gene_id))
        } else if !(self.nodes.contains_key(&input_id) && self.nodes.contains_key(&output_id)) {
            Err(format!(
                "gene insertion with nonexistant endpoint(s) ({}, {})",
                input_id, output_id
            ))
        } else if self.node_pairings.contains(&(input_id, output_id)) {
            Err(format!(
                "duplicate gene insertion between nodes {} and {} with alternate ID {}",
                input_id, output_id, gene_id
            ))
        } else {
            Ok(())
        }
    }

    /// Add a new node to the genome.
    /// Returns a reference  to the newly created node.
    ///
    /// # Panics
    ///
    /// This function panics if a node of the
    /// same ID already existed in the genome.
    pub fn add_node(&mut self, node_id: Innovation, activation_type: ActivationType) -> &mut Node {
        self.check_node_viability(node_id).unwrap();
        unsafe { self.add_node_unchecked(node_id, activation_type) }
    }

    /// Add a new node to the genome.
    /// Returns a reference  to the newly created node.
    /// Assumes the node is not a duplicate.
    unsafe fn add_node_unchecked(
        &mut self,
        node_id: Innovation,
        activation_type: ActivationType,
    ) -> &mut Node {
        self.nodes
            .entry(node_id)
            .or_insert_with(|| Node::new(node_id, NodeType::Neuron, activation_type))
    }

    /// Checks whether a node is a duplicate
    /// for the genome.
    ///
    /// # Errors
    ///
    /// Rertuns an error in case of a duplicate.
    fn check_node_viability(&self, node_id: Innovation) -> Result<(), String> {
        if self.nodes.contains_key(&node_id) {
            Err(format!("duplicate node insertion with ID {}", node_id))
        } else {
            Ok(())
        }
    }

    /// Induces a _weight mutation_ in the genome.
    pub fn mutate_weights(&mut self, config: &GeneticConfig) {
        let mut rng = rand::thread_rng();
        for gene in self.genes.values_mut() {
            if rng.gen::<f32>() < config.weight_reset_chance {
                gene.randomize_weight(config);
            } else if rng.gen::<f32>() < config.weight_nudge_chance {
                gene.nudge_weight(config);
            }
        }
    }

    /// Induces a _gene mutation_ in the genome.
    /// If successful, returns the newly added gene.
    ///
    /// # Errors
    ///
    /// Returns an error if no viable pair of nodes
    /// exists or [too many] attempts have failed.
    ///
    /// [too many]: crate::genomes::GeneticConfig::max_gene_mutation_attempts
    pub fn mutate_genes(
        &mut self,
        history: &mut History,
        config: &GeneticConfig,
    ) -> Result<&Gene, String> {
        let non_sensor_nodes = self.select_non_sensor_nodes();
        let mut potential_inputs = self.select_potential_input_nodes(&non_sensor_nodes);

        if potential_inputs.is_empty() {
            return Err("no viable input found".to_string());
        }

        let mut rng = rand::thread_rng();
        potential_inputs.shuffle(&mut rng);

        match self.find_node_pair(&potential_inputs, &non_sensor_nodes, config) {
            Some((source_node, dest_node)) => {
                self.add_gene_mutation(source_node, dest_node, history, config)
            }
            None => Err("no valid input-output pair found".to_string()),
        }
    }

    fn select_non_sensor_nodes(&self) -> HashSet<Innovation> {
        self.nodes
            .iter()
            .filter_map(|(id, n)| {
                if n.node_type() != NodeType::Sensor {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect()
    }

    fn select_potential_input_nodes(
        &self,
        non_sensor_nodes: &HashSet<Innovation>,
    ) -> Vec<Innovation> {
        self.nodes
            .iter()
            .filter_map(|(id, n)| {
                if n.output_genes().len() < non_sensor_nodes.len() {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect()
    }

    fn choose_output_node_for(
        &self,
        candidate_input: &Innovation,
        potential_outputs: &HashSet<Innovation>,
        config: &GeneticConfig,
    ) -> Option<Innovation> {
        let mut rng = rand::thread_rng();

        let candidate_input = &self.nodes[candidate_input];
        if candidate_input.node_type() != NodeType::Sensor
            && rng.gen::<f32>() < config.recursion_chance
        {
            Some(candidate_input.innovation())
        } else {
            let mut candidate_outputs = potential_outputs - &self.output_nodes_of(&candidate_input);
            candidate_outputs.remove(&candidate_input.innovation());
            candidate_outputs.iter().choose(&mut rng).copied()
        }
    }

    fn output_nodes_of(&self, node: &Node) -> HashSet<Innovation> {
        node.output_genes()
            .iter()
            .map(|id| self.genes[id].output())
            .collect()
    }

    fn find_node_pair(
        &self,
        potential_inputs: &[Innovation],
        potential_outputs: &HashSet<Innovation>,
        config: &GeneticConfig,
    ) -> Option<(Innovation, Innovation)> {
        potential_inputs
            .iter()
            .take(config.max_gene_mutation_attempts)
            .filter_map(|i| {
                self.choose_output_node_for(i, potential_outputs, config)
                    .map(|output| (*i, output))
            })
            .next()
    }

    fn add_gene_mutation(
        &mut self,
        input_node: Innovation,
        output_node: Innovation,
        history: &mut History,
        config: &GeneticConfig,
    ) -> Result<&Gene, String> {
        let gene_id = history.next_gene_innovation(input_node, output_node);
        let gene = self.add_gene(
            gene_id,
            input_node,
            output_node,
            Gene::random_weight(config),
        );
        history.add_gene_innovation(input_node, output_node);
        Ok(gene)
    }

    /// Induces a _node mutation_ in the genome.
    /// If succesful, returns the triplet (_in gene_, _new node_, _out gene_)
    /// as a tuple of references.
    ///
    /// # Errors
    ///
    /// This function returns an error if there are no genes in the genome
    /// that could be split.
    pub fn mutate_nodes(
        &mut self,
        history: &mut History,
        config: &GeneticConfig,
    ) -> Result<(&Gene, &Node, &Gene), String> {
        let gene_to_split = match self.choose_random_gene() {
            Some(gene_to_split) => gene_to_split,
            None => {
                return Err(
                    "attempted node mutation on empty or completely suppressed genome".to_string(),
                )
            }
        };

        let mutation = self.get_node_mutation_innovation(gene_to_split, history);
        self.add_node_mutation(gene_to_split, mutation, history, config)
    }

    fn choose_random_gene(&self) -> Option<Innovation> {
        let candidate_genes: Vec<&Gene> = self.genes.values().filter(|g| !g.suppressed).collect();

        let mut rng = rand::thread_rng();
        candidate_genes
            .choose(&mut rng)
            .map(|gene| gene.innovation())
    }

    fn get_node_mutation_innovation(
        &mut self,
        gene_to_split: Innovation,
        history: &mut History,
    ) -> (Innovation, Innovation, Innovation) {
        let (input_gene, new_node, output_gene) =
            history.next_node_innovation(gene_to_split, false);

        if self.nodes.contains_key(&new_node) {
            history.next_node_innovation(gene_to_split, true)
        } else {
            (input_gene, new_node, output_gene)
        }
    }

    fn add_node_mutation(
        &mut self,
        gene_to_split: Innovation,
        mutation: (Innovation, Innovation, Innovation),
        history: &mut History,
        config: &GeneticConfig,
    ) -> Result<(&Gene, &Node, &Gene), String> {
        let (input_gene, new_node, output_gene) = mutation;
        let (input_node, output_node) = self.genes[&gene_to_split].endpoints();

        if let Err(e) = self.check_node_viability(new_node) {
            return Err(e);
        }

        history.add_node_innovation(gene_to_split, self.nodes.contains_key(&new_node));

        unsafe {
            self.suppress_gene_unchecked(gene_to_split);
            self.add_node_unchecked(
                new_node,
                *config
                    .activation_types
                    .choose(&mut rand::thread_rng())
                    .unwrap(),
            );
            self.add_gene_unchecked(
                input_gene,
                input_node,
                new_node,
                Gene::random_weight(config),
            );
            self.add_gene_unchecked(
                output_gene,
                new_node,
                output_node,
                Gene::random_weight(config),
            );
        }

        Ok((
            &self.genes[&input_gene],
            &self.nodes[&new_node],
            &self.genes[&output_gene],
        ))
    }

    unsafe fn suppress_gene_unchecked(&mut self, gene: Innovation) {
        self.genes.get_mut(&gene).unwrap().suppressed = true;
    }

    /// Combines the genome with an `other` genome and
    /// returns their _child_ genome.
    ///
    /// Depending on [`config.mutate_only_chance`] and
    /// [`config.mate_only_chance`], the child may simply be a
    /// clone of the parent with higher fitness, and may or may not
    /// undergo any mutations. Mutation chances are defined by their
    /// corresponding entries in `config`.
    ///
    /// [`config.mutate_only_chance`]: crate::genomes::GeneticConfig::mutate_only_chance
    /// [`config.mate_only_chance`]: crate::genomes::GeneticConfig::mate_only_chance
    pub fn mate_with(
        &self,
        other: &Genome,
        history: &mut History,
        config: &GeneticConfig,
    ) -> Genome {
        let (parent1, parent2) = if self.fitness >= other.fitness {
            (self, other)
        } else {
            (other, self)
        };

        let mut child = parent1.clone();

        let mut rng = rand::thread_rng();
        if rng.gen::<f32>() < config.mate_only_chance {
            child.combine(parent2, config);
        } else if rng.gen::<f32>() < config.mutate_only_chance {
            child.mutate_all(history, config);
        } else {
            child.combine(parent2, config);
            child.mutate_all(history, config);
        }

        child.reset_suppressions(config);

        child
    }

    /// Performs all mutations on self.
    fn mutate_all(&mut self, history: &mut History, config: &GeneticConfig) {
        self.mutate_weights(config);
        let mut rng = rand::thread_rng();
        if rng.gen::<f32>() < config.node_mutation_chance {
            let _ = self.mutate_nodes(history, config);
        }
        if rng.gen::<f32>() < config.gene_mutation_chance {
            let _ = self.mutate_genes(history, config);
        }
    }

    /// Adds all uncommon structure and combines genes to `self`.
    fn combine(&mut self, other: &Genome, config: &GeneticConfig) {
        if (self.fitness - other.fitness).abs() < f32::EPSILON {
            self.add_noncommon_structure(other);
        }
        if rand::thread_rng().gen::<f32>() < config.mate_by_averaging_chance {
            self.average_common_genes(other);
        } else {
            self.randomly_choose_common_genes(other);
        }
    }

    /// Adds all genes and nodes not common to both genomes to `self`.
    fn add_noncommon_structure(&mut self, other: &Genome) {
        for (id, node) in &other.nodes {
            if !self.nodes.contains_key(&id) {
                self.add_node(*id, node.activation_type());
            }
        }

        for (id, gene) in &other.genes {
            if !self.node_pairings.contains(&(gene.input(), gene.output())) {
                self.add_gene(*id, gene.input(), gene.output(), gene.weight);
            }
        }
    }

    /// Combines all common genes by averaging weights, and suppresses
    /// genes that are suppresssed in either genome.
    fn average_common_genes(&mut self, other: &Genome) {
        for (id, others_gene) in &other.genes {
            if let Some(own_gene) = self.genes.get_mut(&id) {
                own_gene.weight = (own_gene.weight + others_gene.weight) / 2.0;
            }
        }
    }

    /// Combines all common genes by chosing weights randomly between genomes, and suppresses
    /// genes that are suppresssed in either genome.
    fn randomly_choose_common_genes(&mut self, other: &Genome) {
        let mut rng = rand::thread_rng();
        for (id, gene) in &other.genes {
            if let Some(own) = self.genes.get_mut(&id) {
                if rng.gen::<f32>() < 0.5 {
                    own.weight = gene.weight;
                }
            }
        }
    }

    /// Unsuppresses suppressed genes with probability `config.suppression_reset_chance`;
    fn reset_suppressions(&mut self, config: &GeneticConfig) {
        let mut rng = rand::thread_rng();
        for gene in self.genes.values_mut() {
            if gene.suppressed && rng.gen::<f32>() < config.suppression_reset_chance {
                gene.suppressed = false;
            }
        }
    }

    /// Calculates the _genetic distance_ between `self` and `other`,
    /// weighting node and weight differences as specified in `config`.
    pub fn genetic_distance_to(&self, other: &Genome, config: &GeneticConfig) -> f32 {
        let (common_innovations, common_weight_pairs) =
            Self::common_innovations_and_weights(self, other);

        let common_weight_diff = Self::weight_diff_average(&common_weight_pairs);

        let disjoint_gene_count_self = self.count_disjoint_genes(&common_innovations);
        let disjoint_gene_count_other = other.count_disjoint_genes(&common_innovations);
        let disjoint_gene_count = disjoint_gene_count_self + disjoint_gene_count_other;

        let excess_gene_count =
            (self.genes.len() - common_innovations.len() - disjoint_gene_count_self)
                + (other.genes.len() - common_innovations.len() - disjoint_gene_count_other);

        config.disjoint_gene_factor * disjoint_gene_count as f32
            + config.excess_gene_factor * excess_gene_count as f32
            + config.common_weight_factor * common_weight_diff
    }

    fn common_innovations_and_weights(
        g1: &Genome,
        g2: &Genome,
    ) -> (HashSet<Innovation>, Vec<(f32, f32)>) {
        let gene_set_g1: HashSet<Innovation> = g1.genes.keys().copied().collect();
        let gene_set_g2: HashSet<Innovation> = g2.genes.keys().copied().collect();
        gene_set_g1
            .intersection(&gene_set_g2)
            .copied()
            .map(|id| (id, (g1.genes[&id].weight, g2.genes[&id].weight)))
            .unzip()
    }

    fn weight_diff_average(weight_pairs: &[(f32, f32)]) -> f32 {
        let count = weight_pairs.len();
        weight_pairs
            .iter()
            .map(|(w1, w2)| (w1 - w2).abs())
            .sum::<f32>()
            / count as f32
    }

    fn count_disjoint_genes(&self, common_innovations: &HashSet<Innovation>) -> usize {
        let common_innovation_max = *common_innovations.iter().max().unwrap();
        self.genes()
            .keys()
            .cloned()
            .filter(|id| !common_innovations.contains(id) && *id < common_innovation_max)
            .count()
    }

    /// Returns a reference to the genome's gene map.
    pub fn genes(&self) -> &HashMap<Innovation, Gene> {
        &self.genes
    }

    /// Returns a reference to the genome's node map.
    pub fn nodes(&self) -> &HashMap<Innovation, Node> {
        &self.nodes
    }
}

impl fmt::Debug for Genome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut genes: Vec<&Gene> = self.genes.values().collect();
        let mut nodes: Vec<&Node> = self.nodes.values().collect();
        genes.sort_unstable_by_key(|g| g.innovation());
        nodes.sort_unstable_by_key(|n| n.innovation());
        f.debug_struct("Genome")
            .field("Genes", &genes)
            .field("Nodes", &nodes)
            .field("Fitness", &self.fitness)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;

    #[test]
    fn new_fully_connected() {
        for input_count in 1..10 {
            for output_count in 1..10 {
                let mut config = GeneticConfig::default();
                config.initial_expression_chance = 1.0;
                config.input_count = NonZeroUsize::new(input_count).unwrap();
                config.output_count = NonZeroUsize::new(output_count).unwrap();
                config.output_activation_types = vec![
                    ActivationType::Sigmoid,
                    ActivationType::Gaussian,
                    ActivationType::Identity,
                    ActivationType::ReLU,
                    ActivationType::Sinusoidal,
                ];

                let full_genome = Genome::new(&config);
                assert_eq!(full_genome.genes.len(), input_count * output_count);
                assert_eq!(
                    full_genome
                        .nodes
                        .values()
                        .filter(|n| n.node_type() == NodeType::Sensor
                            && n.activation_type() == ActivationType::Identity)
                        .count(),
                    input_count
                );
                assert_eq!(
                    full_genome
                        .nodes
                        .values()
                        .filter(|n| n.node_type() == NodeType::Actuator
                            && n.activation_type()
                                == *config
                                    .output_activation_types
                                    .get(n.innovation() - input_count)
                                    .unwrap_or(&ActivationType::Sigmoid))
                        .count(),
                    output_count
                );
                assert_eq!(
                    *full_genome.nodes.keys().max().unwrap(),
                    input_count + output_count - 1
                );
                for g in full_genome.genes.values() {
                    assert_eq!(
                        g.innovation(),
                        g.input() * output_count + (g.output() - input_count),
                        "gene: {:?}, total I/O, {}/{}",
                        g,
                        input_count,
                        output_count
                    );
                    assert!(full_genome
                        .nodes
                        .get(&g.input())
                        .unwrap()
                        .output_genes()
                        .contains(&g.innovation()));
                    assert!(full_genome
                        .nodes
                        .get(&g.output())
                        .unwrap()
                        .input_genes()
                        .contains(&g.innovation()));
                }
            }
        }
    }

    #[test]
    fn new_unconnected() {
        let mut config = GeneticConfig::default();
        config.initial_expression_chance = 0.0;

        let empty_genome = Genome::new(&config);
        assert_eq!(empty_genome.genes.len(), 0);
    }

    #[test]
    fn add_gene() {
        const INNOVATION: Innovation = 631;
        const INPUT: Innovation = 0;
        const OUTPUT: Innovation = 1;
        const WEIGHT: f32 = 3.0;

        let mut config = GeneticConfig::default();
        config.initial_expression_chance = 0.0;

        let mut genome = Genome::new(&config);
        let gene = genome.add_gene(INNOVATION, INPUT, OUTPUT, WEIGHT);

        assert_eq!(gene.innovation(), INNOVATION);
        assert_eq!(gene.input(), INPUT);
        assert_eq!(gene.output(), OUTPUT);
        assert_eq!(gene.weight, WEIGHT);

        let gene = gene.clone();

        assert_eq!(genome.genes().len(), 1);
        assert_eq!(&genome.genes[&INNOVATION], &gene);
    }

    #[test]
    #[should_panic]
    fn add_gene_duplicate_gene_innovation() {
        const INNOVATION: Innovation = 0;
        const INPUT: Innovation = 0;
        const OUTPUT: Innovation = 1;
        const WEIGHT: f32 = 3.0;

        let mut config = GeneticConfig::default();
        config.initial_expression_chance = 1.0;

        let mut genome = Genome::new(&config);
        genome.add_gene(INNOVATION, INPUT, OUTPUT, WEIGHT);
    }

    #[test]
    #[should_panic]
    fn add_gene_duplicate_io() {
        const INNOVATION: Innovation = 555;
        const WEIGHT: f32 = 3.0;

        let mut config = GeneticConfig::default();
        config.initial_expression_chance = 1.0;

        let mut genome = Genome::new(&config);
        let input = genome
            .nodes
            .values()
            .filter(|n| n.node_type() == NodeType::Sensor)
            .nth(0)
            .unwrap()
            .innovation();
        let output = genome
            .nodes
            .values()
            .filter(|n| n.node_type() == NodeType::Actuator)
            .nth(0)
            .unwrap()
            .innovation();

        genome.add_gene(INNOVATION, input, output, WEIGHT);
    }

    #[test]
    #[should_panic]
    fn add_gene_invalid_input() {
        const INNOVATION: Innovation = 631;
        const INPUT: Innovation = 500;
        const OUTPUT: Innovation = 1;
        const WEIGHT: f32 = 3.0;

        let mut config = GeneticConfig::default();
        config.initial_expression_chance = 0.0;

        let mut genome = Genome::new(&config);
        genome.add_gene(INNOVATION, INPUT, OUTPUT, WEIGHT);
    }

    #[test]
    #[should_panic]
    fn add_gene_invalid_output() {
        const INNOVATION: Innovation = 631;
        const INPUT: Innovation = 0;
        const OUTPUT: Innovation = 500;
        const WEIGHT: f32 = 3.0;

        let mut config = GeneticConfig::default();
        config.initial_expression_chance = 0.0;

        let mut genome = Genome::new(&config);
        genome.add_gene(INNOVATION, INPUT, OUTPUT, WEIGHT);
    }

    #[test]
    fn add_node() {
        const INPUTS: usize = 1;
        const OUTPUTS: usize = 1;
        const INNOVATION: Innovation = 42;
        const ACTIVATION_TYPE: ActivationType = ActivationType::Gaussian;

        let mut config = GeneticConfig::default();
        config.initial_expression_chance = 0.0;
        config.input_count = NonZeroUsize::new(INPUTS).unwrap();
        config.output_count = NonZeroUsize::new(OUTPUTS).unwrap();

        let mut genome = Genome::new(&config);
        let node = genome.add_node(INNOVATION, ACTIVATION_TYPE);

        assert_eq!(node.innovation(), INNOVATION);
        assert_eq!(node.node_type(), NodeType::Neuron);
        assert_eq!(node.activation_type(), ACTIVATION_TYPE);
        assert_eq!(node.input_genes().len(), 0);
        assert_eq!(node.output_genes().len(), 0);

        let node = node.clone();

        assert_eq!(genome.nodes.len(), INPUTS + OUTPUTS + 1);
        assert_eq!(&genome.nodes[&INNOVATION], &node);
    }

    #[test]
    #[should_panic]
    fn add_node_duplicate() {
        const INNOVATION: Innovation = 0;
        const ACTIVATION_TYPE: ActivationType = ActivationType::Gaussian;

        let mut config = GeneticConfig::default();
        config.initial_expression_chance = 0.0;

        let mut genome = Genome::new(&config);
        genome.add_node(INNOVATION, ACTIVATION_TYPE);
    }

    /// It is possible this test will fail
    /// due to the new weight returned by the
    /// rng being identical to the previous one,
    /// but the chances of this are minimal.
    #[test]
    fn mutate_weights_reset() {
        let mut config = GeneticConfig::default();
        config.initial_expression_chance = 1.0;
        config.weight_reset_chance = 1.0;
        config.weight_mutation_power = 3.0;

        let mut genome = Genome::new(&config);
        let initial_weight = genome.genes[&0].weight;
        genome.mutate_weights(&config);
        assert_ne!(initial_weight, genome.genes[&0].weight);
    }

    /// It is possible this test will fail
    /// due to the new weight returned by the
    /// rng being identical to the previous one,
    /// but the chances of this are minimal.
    #[test]
    fn mutate_weights_nudge() {
        let mut config = GeneticConfig::default();
        config.initial_expression_chance = 1.0;
        config.weight_reset_chance = 0.0;
        config.weight_nudge_chance = 1.0;
        config.weight_mutation_power = 3.0;
        config.weight_bound = 5.0;

        let mut genome = Genome::new(&config);
        genome.mutate_weights(&config);
        let initial_weight = genome.genes[&0].weight;
        genome.mutate_weights(&config);
        assert_ne!(initial_weight, genome.genes[&0].weight);
    }

    #[test]
    fn mutate_weights_none() {
        let mut config = GeneticConfig::default();
        config.initial_expression_chance = 1.0;
        config.weight_reset_chance = 0.0;
        config.weight_nudge_chance = 0.0;
        config.weight_bound = 5.0;

        let mut genome = Genome::new(&config);
        genome.mutate_weights(&config);
        let initial_weight = genome.genes[&0].weight;
        genome.mutate_weights(&config);
        assert_eq!(initial_weight, genome.genes[&0].weight);
    }

    #[test]
    fn mutate_genes() {
        let mut config = GeneticConfig::default();
        config.initial_expression_chance = 1.0;
        config.gene_mutation_chance = 1.0;
        config.max_gene_mutation_attempts = 20;
        config.recursion_chance = 0.0;
        config.weight_mutation_power = 3.0;

        let mut history = History::new(&config);

        let mut genome = Genome::new(&config);
        genome.add_node(2, ActivationType::Sigmoid);
        let gene = genome.mutate_genes(&mut history, &config).unwrap();

        assert_eq!(
            gene.innovation(),
            history.next_gene_innovation(gene.input(), gene.output())
        );
        assert!((0..=2).contains(&gene.input()));
        assert!((1..=2).contains(&gene.output()));
    }

    #[test]
    fn mutate_genes_recursive() {
        let mut config = GeneticConfig::default();
        config.initial_expression_chance = 1.0;
        config.gene_mutation_chance = 1.0;
        config.max_gene_mutation_attempts = 20;
        config.recursion_chance = 1.0;
        config.weight_mutation_power = 3.0;

        let mut history = History::new(&config);

        let mut genome = Genome::new(&config);
        genome.add_node(50, ActivationType::Sigmoid);
        genome.add_gene(42, 0, 50, 2.0);
        genome.add_gene(43, 50, 1, 2.0);
        genome.add_gene(44, 1, 50, 2.0);
        let gene = genome.mutate_genes(&mut history, &config).unwrap();

        assert_eq!(gene.input(), gene.output());
    }

    #[test]
    #[should_panic]
    fn mutate_genes_no_pairs_found() {
        let mut config = GeneticConfig::default();
        config.initial_expression_chance = 1.0;
        config.gene_mutation_chance = 1.0;
        config.max_gene_mutation_attempts = 20;
        config.recursion_chance = 1.0;
        config.weight_mutation_power = 3.0;

        let mut history = History::new(&config);

        let mut genome = Genome::new(&config);
        // Add a recursive gene to make sure they aren't
        // chosen even though it is permitted.
        genome.add_gene(42, 1, 1, 5.0);

        genome.mutate_genes(&mut history, &config);
    }

    #[test]
    fn mutate_nodes() {
        let mut config = GeneticConfig::default();
        config.activation_types = vec![ActivationType::Sigmoid, ActivationType::ReLU];
        config.initial_expression_chance = 1.0;
        config.node_mutation_chance = 1.0;

        let mut history = History::new(&config);

        let mut genome = Genome::new(&config);
        let (input, node, output) = genome.mutate_nodes(&mut history, &config).unwrap();

        assert_eq!(
            input.innovation(),
            *node.input_genes().iter().nth(0).unwrap()
        );
        assert_eq!(
            output.innovation(),
            *node.output_genes().iter().nth(0).unwrap()
        );
        assert!(config.activation_types.contains(&node.activation_type()));
        assert_eq!(node.innovation(), genome.nodes[&2].innovation());
        assert_eq!(genome.genes.len(), 3);
        assert_eq!(genome.nodes.len(), 3);
        assert_eq!(genome.genes[&0].suppressed, true);
    }

    #[test]
    #[should_panic]
    fn mutate_nodes_no_gene_found() {
        let mut config = GeneticConfig::default();
        config.initial_expression_chance = 0.0;
        config.node_mutation_chance = 1.0;

        let mut history = History::new(&config);

        let mut genome = Genome::new(&config);
        genome.mutate_nodes(&mut history, &config).unwrap();
    }

    #[test]
    fn add_noncommon_structure() {
        let mut config = GeneticConfig::default();
        config.input_count = NonZeroUsize::new(2).unwrap();
        config.output_count = NonZeroUsize::new(1).unwrap();
        config.initial_expression_chance = 1.0;

        let mut genome1 = Genome::new(&config);
        let mut genome2 = Genome::new(&config);
        genome1.add_node(3, ActivationType::Sigmoid);
        genome2.add_node(3, ActivationType::Sigmoid);
        genome2.add_node(4, ActivationType::ReLU);
        genome1.add_gene(2, 0, 3, 1.0);
        genome2.add_gene(3, 0, 3, 1.0);
        genome2.add_gene(4, 0, 4, 2.0);
        genome2.add_gene(5, 3, 4, 3.0);
        genome2.add_gene(6, 4, 2, 4.0);
        genome1.add_noncommon_structure(&genome2);

        let mut node4 = Node::new(4, NodeType::Neuron, ActivationType::ReLU);
        node4.add_input_gene(4);
        node4.add_input_gene(5);
        node4.add_output_gene(6);
        assert_eq!(&genome1.nodes[&4], &node4);
        assert_eq!(&genome1.genes[&2], &Gene::new(2, 0, 3, 1.0));
        assert_eq!(&genome1.genes[&4], &Gene::new(4, 0, 4, 2.0));
        assert_eq!(&genome1.genes[&5], &Gene::new(5, 3, 4, 3.0));
        assert_eq!(&genome1.genes[&6], &Gene::new(6, 4, 2, 4.0));
    }

    #[test]
    fn combines_genes_average() {
        let mut config = GeneticConfig::default();
        config.input_count = NonZeroUsize::new(2).unwrap();
        config.output_count = NonZeroUsize::new(1).unwrap();
        config.initial_expression_chance = 0.0;

        let mut genome1 = Genome::new(&config);
        genome1.add_node(3, ActivationType::Sigmoid);

        let mut genome2 = genome1.clone();

        genome1.add_gene(0, 0, 2, 1.0);
        genome2.add_gene(0, 0, 2, 3.0);
        genome1.add_gene(1, 0, 3, 2.0);
        genome2.add_gene(1, 0, 3, -2.0);
        genome1.add_gene(2, 3, 2, 3.0);
        genome2.add_gene(2, 3, 2, 3.5);
        genome1.add_gene(3, 1, 2, 4.0);
        genome2.add_gene(3, 1, 2, 4.0);

        genome1.average_common_genes(&genome2);

        assert_eq!(genome1.genes[&0].weight, 2.0);
        assert_eq!(genome1.genes[&1].weight, 0.0);
        assert_eq!(genome1.genes[&2].weight, 3.25);
        assert_eq!(genome1.genes[&3].weight, 4.0);
    }

    #[test]
    fn combines_genes_random_choice() {
        let mut config = GeneticConfig::default();
        config.input_count = NonZeroUsize::new(2).unwrap();
        config.output_count = NonZeroUsize::new(1).unwrap();
        config.initial_expression_chance = 0.0;

        let mut genome1 = Genome::new(&config);
        genome1.add_node(3, ActivationType::Sigmoid);

        let mut genome2 = genome1.clone();

        genome1.add_gene(0, 0, 2, 1.0);
        genome2.add_gene(0, 0, 2, 3.0);
        genome1.add_gene(1, 0, 3, 2.0);
        genome2.add_gene(1, 0, 3, -2.0);
        genome1.add_gene(2, 3, 2, 3.0);
        genome2.add_gene(2, 3, 2, 3.5);
        genome1.add_gene(3, 1, 2, 4.0);
        genome2.add_gene(3, 1, 2, 4.0);

        genome1.randomly_choose_common_genes(&genome2);

        assert!([1.0, 3.0].contains(&genome1.genes[&0].weight));
        assert!([2.0, -2.0].contains(&genome1.genes[&1].weight));
        assert!([3.0, 3.5].contains(&genome1.genes[&2].weight));
        assert!([4.0].contains(&genome1.genes[&3].weight));
    }

    #[test]
    fn reset_suppressions() {
        let mut config = GeneticConfig::default();
        config.input_count = NonZeroUsize::new(2).unwrap();
        config.output_count = NonZeroUsize::new(1).unwrap();
        config.initial_expression_chance = 0.0;
        config.suppression_reset_chance = 1.0;

        let mut genome = Genome::new(&config);
        genome.add_gene(0, 0, 2, 3.0).suppressed = true;
        genome.add_gene(1, 1, 2, 4.0).suppressed = true;
        genome.add_gene(2, 2, 2, 5.0).suppressed = true;

        genome.reset_suppressions(&config);

        assert!(genome.genes().values().all(|g| !g.suppressed));
    }

    #[test]
    fn genetic_distance_to() {
        const WEIGHT_FACTOR: f32 = 0.8;
        const DISJOINT_FACTOR: f32 = 0.6;
        const EXCESS_FACTOR: f32 = 0.4;
        let mut config = GeneticConfig::default();
        config.input_count = NonZeroUsize::new(2).unwrap();
        config.output_count = NonZeroUsize::new(1).unwrap();
        config.initial_expression_chance = 0.0;
        config.common_weight_factor = WEIGHT_FACTOR;
        config.disjoint_gene_factor = DISJOINT_FACTOR;
        config.excess_gene_factor = EXCESS_FACTOR;

        let mut genome1 = Genome::new(&config);
        let mut genome2 = Genome::new(&config);

        genome1.add_node(3, ActivationType::Sigmoid);
        genome2.add_node(4, ActivationType::Sigmoid);

        genome1.add_gene(1, 0, 2, -2.0);
        genome2.add_gene(1, 0, 2, 2.0);

        genome1.add_gene(2, 1, 3, 5.0);
        genome2.add_gene(3, 2, 4, 5.0);

        genome1.add_gene(4, 1, 2, 3.0);
        genome2.add_gene(4, 1, 2, 6.0);

        genome1.add_gene(5, 2, 2, 5.0);
        genome2.add_gene(6, 4, 4, 1.0);

        assert_eq!(
            genome1.genetic_distance_to(&genome2, &config),
            DISJOINT_FACTOR * 2.0 + EXCESS_FACTOR * 2.0 + WEIGHT_FACTOR * ((4.0 + 3.0) / 2.0)
        );
    }

    #[test]
    fn genetic_distance_to_equal() {
        let mut config = GeneticConfig::default();
        config.input_count = NonZeroUsize::new(2).unwrap();
        config.output_count = NonZeroUsize::new(1).unwrap();
        config.initial_expression_chance = 1.0;
        config.common_weight_factor = 1.0;
        config.disjoint_gene_factor = 0.5;
        config.excess_gene_factor = 0.5;

        let mut genome = Genome::new(&config);

        genome.add_node(3, ActivationType::Sigmoid);
        genome.add_node(4, ActivationType::Sigmoid);
        genome.add_gene(3, 2, 4, 5.0);
        genome.add_gene(5, 2, 2, 5.0);
        genome.add_gene(6, 4, 4, 1.0);

        assert_eq!(genome.genetic_distance_to(&genome.clone(), &config), 0.0);
    }
}
