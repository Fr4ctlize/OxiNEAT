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
use std::collections::hash_map::{Entry, HashMap};
use std::collections::HashSet;

/// Genomes are the focus of evolution in NEAT.
/// They are a collection of genes and nodes that can be instantiated
/// as a phenotype (a neural network) and evaluated
/// for performance in a task, which results numerically in
/// their fitness score. Genomes can be progressively mutated,
/// thus adding complexity and functionality.
#[derive(Clone, Debug, PartialEq)]
pub struct Genome {
    genes: HashMap<Innovation, Gene>,
    nodes: HashMap<Innovation, Node>,
    node_pairings: HashSet<(Innovation, Innovation)>,
    pub(in crate) fitness: f32,
    max_innovation: Innovation,
}

impl Genome {
    /// Create a new genome with the specified configuration.
    pub fn new(config: &GeneticConfig) -> Genome {
        let input_count = config.input_count.get();
        let output_count = config.output_count.get();

        let mut nodes = HashMap::with_capacity(input_count + output_count);
        let mut genes = HashMap::new();
        let mut node_pairings = HashSet::new();

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

        let mut max_innovation = 0;
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
                        nodes.get_mut(&i).unwrap().add_output_gene(id).unwrap();
                        nodes
                            .get_mut(&(o + input_count))
                            .unwrap()
                            .add_input_gene(id)
                            .unwrap();
                        max_innovation = id;
                    }
                }
            }
        }

        Genome {
            genes,
            nodes,
            fitness: 0.0,
            max_innovation,
            node_pairings,
        }
    }

    /// Add a new gene to the genome.
    /// Returns a reference to the new gene.
    ///
    /// # Errors
    ///
    /// This function will return an error if a gene with the same
    /// `gene_id` already existed in the genome, or if either `input_id`
    /// or `output_id` do not correspond to nodes present in the genome.
    pub fn add_gene(
        &mut self,
        gene_id: Innovation,
        input_id: Innovation,
        output_id: Innovation,
        weight: f32,
    ) -> Result<&mut Gene, String> {
        match self.check_gene_viability(gene_id, input_id, output_id) {
            Ok(_) => unsafe { self.add_gene_unchecked(gene_id, input_id, output_id, weight) },
            Err(e) => Err(e),
        }
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
    ) -> Result<&mut Gene, String> {
        self.nodes
            .get_mut(&input_id)
            .unwrap()
            .add_output_gene(gene_id)?;
        self.nodes
            .get_mut(&output_id)
            .unwrap()
            .add_input_gene(gene_id)?;
        self.max_innovation = self.max_innovation.max(gene_id);
        self.node_pairings.insert((input_id, output_id));
        Ok(self
            .genes
            .entry(gene_id)
            .or_insert(Gene::new(gene_id, input_id, output_id, weight)))
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
    /// # Errors
    ///
    /// This function returns an error if a node of the
    /// same ID already existed in the genome.
    pub fn add_node(
        &mut self,
        node_id: Innovation,
        activation_type: ActivationType,
    ) -> Result<&mut Node, String> {
        match self.check_node_viability(node_id) {
            Ok(_) => unsafe { Ok(self.add_node_unchecked(node_id, activation_type)) },
            Err(e) => Err(e),
        }
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
            .or_insert(Node::new(node_id, NodeType::Neuron, activation_type))
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
        // All nodes that are not inputs can be potential
        // outputs for the new gene.
        let potential_outputs: HashSet<Innovation> = self
            .nodes
            .iter()
            .filter_map(|(id, n)| {
                if n.node_type() != NodeType::Sensor {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect();

        // Only nodes that aren't connected to all potential
        // outputs should be considered as potential inputs.
        let mut potential_inputs: Vec<Innovation> = self
            .nodes
            .iter()
            .filter_map(|(id, n)| {
                if n.output_genes().len() < potential_outputs.len() {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect();
        if potential_inputs.is_empty() {
            return Err("no viable input found".to_string());
        }

        let mut rng = rand::thread_rng();

        // Give all possible input nodes an equal chance at being chosen.
        potential_inputs.shuffle(&mut rng);

        // The network must have at least two nodes, and
        // if max_gene_mutation_attempts is 0 dest_node will
        // remain None, so it is safe to set source_node to 0
        // temporarily without risking unwanted gene creation.
        let mut source_node = 0;
        let mut dest_node = None;

        for i in potential_inputs
            .iter()
            .take(config.max_gene_mutation_attempts)
        {
            let candidate_input = self.nodes.get(i).unwrap();
            // Try to make a recursive gene.
            if candidate_input.node_type() != NodeType::Sensor
                && rng.gen::<f32>() < config.recursion_chance
            {
                source_node = candidate_input.innovation();
                dest_node = Some(source_node);
                break;
            // Try to find another possible output node.
            } else {
                let mut candidate_outputs = &potential_outputs
                    - &candidate_input
                        .output_genes()
                        .iter()
                        .map(|id| self.genes.get(id).unwrap().output())
                        .collect();
                candidate_outputs.remove(&candidate_input.innovation());
                if let Some(output) = candidate_outputs.iter().choose(&mut rng) {
                    source_node = candidate_input.innovation();
                    dest_node = Some(*output);
                    break;
                }
            }
        }

        // If a viable node pair was found, add a new gene
        // in between. Otherwise, return a nonfatal error.
        match dest_node {
            Some(dest_node) => {
                let gene_id = history.next_gene_innovation(source_node, dest_node);
                let gene =
                    self.add_gene(gene_id, source_node, dest_node, Gene::random_weight(config))?;
                history.add_gene_innovation(source_node, dest_node);
                Ok(gene)
            }
            None => Err("no valid input-output pair found".to_string()),
        }
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
        // All unsuppressed genes are candidates to
        // be "split" in a node mutation.
        let candidate_genes: Vec<&Gene> = self.genes.values().filter(|g| !g.suppressed).collect();

        // Choose a random candidate to split.
        let mut rng = rand::thread_rng();
        let gene_to_split = match candidate_genes.choose(&mut rng) {
            Some(gene_to_split) => gene_to_split.innovation(),
            None => {
                return Err(
                    "attempted node mutation on empty or completely suppressed genome".to_string(),
                )
            }
        };

        // Get the innovation numbers for the mutation.
        let (mut input_gene, mut new_node, mut output_gene) =
            history.next_node_innovation(gene_to_split);

        // Check if this node innovation has previously
        // occurred in this same genome. If so, get new
        // innovation numbers. In either case, add the
        // innovation to the history.
        if self.nodes.contains_key(&new_node) {
            let (input, node, output) = history.add_node_innovation(gene_to_split, true);
            input_gene = input;
            new_node = node;
            output_gene = output;
        } else {
            history.add_node_innovation(gene_to_split, true);
        }

        let (input_node, output_node) = {
            let gene_to_split = self.genes.get(&gene_to_split).unwrap();
            (gene_to_split.input(), gene_to_split.output())
        };
        // Do all safety checks before modifying the genome,
        // to avoid leaving the genome in an incorrect state.
        if let Err(e) = self.check_node_viability(new_node) {
            return Err(e);
        }
        if let Err(e) = self.check_gene_viability(input_gene, input_node, new_node) {
            return Err(e);
        }
        if let Err(e) = self.check_gene_viability(output_gene, new_node, output_node) {
            return Err(e);
        }
        // Borrow the split gene to suppress it and get
        // its input and output.
        let gene_to_split = self.genes.get_mut(&gene_to_split).unwrap();
        gene_to_split.suppressed = true;

        // Add the new node and genes to the genome, and return.
        unsafe {
            self.add_node_unchecked(new_node, *config.activation_types.choose(&mut rng).unwrap());
            self.add_gene_unchecked(
                input_gene,
                input_node,
                new_node,
                Gene::random_weight(config),
            )?;
            self.add_gene_unchecked(
                output_gene,
                new_node,
                output_node,
                Gene::random_weight(config),
            )?;
        }

        Ok((
            self.genes.get(&input_gene).unwrap(),
            self.nodes.get(&new_node).unwrap(),
            self.genes.get(&output_gene).unwrap(),
        ))
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
    ) -> Result<Genome, String> {
        let (parent1, parent2) = if self.fitness >= other.fitness {
            (self, other)
        } else {
            (other, self)
        };

        let mut child = parent1.clone();

        let mut rng = rand::thread_rng();
        if rng.gen::<f32>() < config.mate_only_chance {
            child.combine(parent2, config)?;
        } else if rng.gen::<f32>() < config.mutate_only_chance {
            child.mutate_all(history, config)?;
        } else {
            child.combine(parent2, config)?;
            child.mutate_all(history, config)?;
        }

        child.reset_suppressions(config);

        Ok(child)
    }

    /// Performs all mutations on self.
    fn mutate_all(&mut self, history: &mut History, config: &GeneticConfig) -> Result<(), String> {
        self.mutate_weights(config);
        self.mutate_nodes(history, config)?;
        self.mutate_genes(history, config)?;
        Ok(())
    }

    /// Adds all uncommon structure and combines genes to `self`.
    fn combine(&mut self, other: &Genome, config: &GeneticConfig) -> Result<(), String> {
        self.add_noncommon_structure(other)?;
        if rand::thread_rng().gen::<f32>() < config.mate_by_averaging_chance {
            self.combines_genes_average(other);
        } else {
            self.combines_genes_random_choice(other);
        }
        Ok(())
    }

    /// Adds all genes and nodes not common to both genomes to `self`.
    fn add_noncommon_structure(&mut self, other: &Genome) -> Result<(), String> {
        for (id, node) in &other.nodes {
            if !self.nodes.contains_key(&id) {
                self.add_node(*id, node.activation_type())?;
            }
        }

        for (id, gene) in &other.genes {
            if !self.node_pairings.contains(&(gene.input(), gene.output())) {
                self.add_gene(*id, gene.input(), gene.output(), gene.weight)?;
            }
        }

        Ok(())
    }

    /// Combines all common genes by averaging weights, and suppresses
    /// genes that are suppresssed in either genome.
    fn combines_genes_average(&mut self, other: &Genome) {
        for (id, gene) in &other.genes {
            if let Some(own) = self.genes.get_mut(&id) {
                own.weight = (own.weight + gene.weight) / 2.0;
            }
        }
    }

    /// Combines all common genes by chosing weights randomly between genomes, and suppresses
    /// genes that are suppresssed in either genome.
    fn combines_genes_random_choice(&mut self, other: &Genome) {
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
    pub fn genetic_distance_to(&mut self, other: &Genome, config: &GeneticConfig) -> f32 {
        let max_g1 = self.genes.keys().max().unwrap_or(&0);
        let max_g2 = self.genes.keys().max().unwrap_or(&0);

        let common_ids: HashSet<Innovation> = self
            .genes
            .keys()
            .cloned()
            .collect::<HashSet<Innovation>>()
            .intersection(&other.genes.keys().cloned().collect::<HashSet<Innovation>>())
            .cloned()
            .collect();

        let common_weight_diff: f32 = common_ids
            .iter()
            .map(|id| {
                (self.genes.get(id).unwrap().weight - other.genes.get(id).unwrap().weight).abs()
            })
            .sum();

        let disjoint_gene_count1: usize = self
            .genes()
            .keys()
            .cloned()
            .filter(|id| !common_ids.contains(id) && id < max_g2)
            .count();
        let disjoint_gene_count2: usize = other
            .genes()
            .keys()
            .cloned()
            .filter(|id| !common_ids.contains(id) && id < max_g1)
            .count();

        let excess_gene_count = (self.genes.len() - common_ids.len() - disjoint_gene_count1)
            + (other.genes.len() - common_ids.len() - disjoint_gene_count2);

        config.disjoint_gene_factor * (disjoint_gene_count1 + disjoint_gene_count2) as f32
            + config.excess_gene_factor * excess_gene_count as f32
            + config.common_weight_factor * common_weight_diff
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
    fn add_gene() -> Result<(), String> {
        const INNOVATION: Innovation = 631;
        const INPUT: Innovation = 0;
        const OUTPUT: Innovation = 1;
        const WEIGHT: f32 = 3.0;

        let mut config = GeneticConfig::default();
        config.initial_expression_chance = 0.0;

        let mut genome = Genome::new(&config);
        let gene = genome.add_gene(INNOVATION, INPUT, OUTPUT, WEIGHT)?;

        assert_eq!(gene.innovation(), INNOVATION);
        assert_eq!(gene.input(), INPUT);
        assert_eq!(gene.output(), OUTPUT);
        assert_eq!(gene.weight, WEIGHT);

        let gene = gene.clone();

        assert_eq!(genome.genes().len(), 1);
        assert_eq!(genome.genes.get(&INNOVATION).unwrap(), &gene);

        Ok(())
    }

    #[test]
    fn add_gene_duplicate_gene_innovation() {
        const INNOVATION: Innovation = 0;
        const INPUT: Innovation = 0;
        const OUTPUT: Innovation = 1;
        const WEIGHT: f32 = 3.0;

        let mut config = GeneticConfig::default();
        config.initial_expression_chance = 1.0;

        let mut genome = Genome::new(&config);
        let gene = genome.add_gene(INNOVATION, INPUT, OUTPUT, WEIGHT);
        if let Ok(_) = gene {
            panic!("duplicate gene addition should return error");
        }
    }

    #[test]
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

        let gene = genome.add_gene(INNOVATION, input, output, WEIGHT);
        if let Ok(_) = gene {
            panic!("duplicate gene addition should return error");
        }
    }

    #[test]
    fn add_gene_invalid_input() {
        const INNOVATION: Innovation = 631;
        const INPUT: Innovation = 500;
        const OUTPUT: Innovation = 1;
        const WEIGHT: f32 = 3.0;

        let mut config = GeneticConfig::default();
        config.initial_expression_chance = 0.0;

        let mut genome = Genome::new(&config);
        let gene = genome.add_gene(INNOVATION, INPUT, OUTPUT, WEIGHT);
        if let Ok(_) = gene {
            panic!("invalid input node should return error");
        }
    }

    #[test]
    fn add_gene_invalid_output() {
        const INNOVATION: Innovation = 631;
        const INPUT: Innovation = 0;
        const OUTPUT: Innovation = 500;
        const WEIGHT: f32 = 3.0;

        let mut config = GeneticConfig::default();
        config.initial_expression_chance = 0.0;

        let mut genome = Genome::new(&config);
        let gene = genome.add_gene(INNOVATION, INPUT, OUTPUT, WEIGHT);
        if let Ok(_) = gene {
            panic!("invalid output node should return error");
        }
    }

    #[test]
    fn add_node() -> Result<(), String> {
        const INPUTS: usize = 1;
        const OUTPUTS: usize = 1;
        const INNOVATION: Innovation = 42;
        const ACTIVATION_TYPE: ActivationType = ActivationType::Gaussian;

        let mut config = GeneticConfig::default();
        config.initial_expression_chance = 0.0;
        config.input_count = NonZeroUsize::new(INPUTS).unwrap();
        config.output_count = NonZeroUsize::new(OUTPUTS).unwrap();

        let mut genome = Genome::new(&config);
        let node = genome.add_node(INNOVATION, ACTIVATION_TYPE)?;

        assert_eq!(node.innovation(), INNOVATION);
        assert_eq!(node.node_type(), NodeType::Neuron);
        assert_eq!(node.activation_type(), ACTIVATION_TYPE);
        assert_eq!(node.input_genes().len(), 0);
        assert_eq!(node.output_genes().len(), 0);

        let node = node.clone();

        assert_eq!(genome.nodes.len(), INPUTS + OUTPUTS + 1);
        assert_eq!(genome.nodes.get(&INNOVATION).unwrap(), &node);

        Ok(())
    }

    #[test]
    fn add_node_duplicate() {
        const INNOVATION: Innovation = 0;
        const ACTIVATION_TYPE: ActivationType = ActivationType::Gaussian;

        let mut config = GeneticConfig::default();
        config.initial_expression_chance = 0.0;

        let mut genome = Genome::new(&config);
        let node = genome.add_node(INNOVATION, ACTIVATION_TYPE);
        if let Ok(_) = node {
            panic!("duplicate node addition should return error");
        }
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
        let initial_weight = genome.genes.get(&0).unwrap().weight;
        genome.mutate_weights(&config);
        assert_ne!(initial_weight, genome.genes.get(&0).unwrap().weight);
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
        let initial_weight = genome.genes.get(&0).unwrap().weight;
        genome.mutate_weights(&config);
        assert_ne!(initial_weight, genome.genes.get(&0).unwrap().weight);
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
        let initial_weight = genome.genes.get(&0).unwrap().weight;
        genome.mutate_weights(&config);
        assert_eq!(initial_weight, genome.genes.get(&0).unwrap().weight);
    }

    #[test]
    fn mutate_genes() -> Result<(), String> {
        let mut config = GeneticConfig::default();
        config.initial_expression_chance = 1.0;
        config.gene_mutation_chance = 1.0;
        config.max_gene_mutation_attempts = 20;
        config.recursion_chance = 0.0;
        config.weight_mutation_power = 3.0;

        let mut history = History::new(&config);

        let mut genome = Genome::new(&config);
        genome.add_node(2, ActivationType::Sigmoid)?;
        let gene = genome.mutate_genes(&mut history, &config)?;

        assert_eq!(
            gene.innovation(),
            history.next_gene_innovation(gene.input(), gene.output())
        );
        assert!((0..=2).contains(&gene.input()));
        assert!((1..=2).contains(&gene.output()));

        Ok(())
    }

    #[test]
    fn mutate_genes_recursive() -> Result<(), String> {
        let mut config = GeneticConfig::default();
        config.initial_expression_chance = 1.0;
        config.gene_mutation_chance = 1.0;
        config.max_gene_mutation_attempts = 20;
        config.recursion_chance = 1.0;
        config.weight_mutation_power = 3.0;

        let mut history = History::new(&config);

        let mut genome = Genome::new(&config);
        genome.add_node(50, ActivationType::Sigmoid)?;
        genome.add_gene(42, 0, 50, 2.0)?;
        genome.add_gene(43, 50, 1, 2.0)?;
        genome.add_gene(44, 1, 50, 2.0)?;
        let gene = genome.mutate_genes(&mut history, &config)?;

        assert_eq!(gene.input(), gene.output());

        Ok(())
    }

    #[test]
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
        genome.add_gene(42, 1, 1, 5.0).unwrap();

        let gene = genome.mutate_genes(&mut history, &config);

        if let Ok(_) = gene {
            panic!("gene mutation with no viable node pairs should return an error");
        }
    }

    #[test]
    fn mutate_nodes() -> Result<(), String> {
        let mut config = GeneticConfig::default();
        config.activation_types = vec![ActivationType::Sigmoid, ActivationType::ReLU];
        config.initial_expression_chance = 1.0;
        config.node_mutation_chance = 1.0;

        let mut history = History::new(&config);

        let mut genome = Genome::new(&config);
        let (input, node, output) = genome.mutate_nodes(&mut history, &config)?;

        assert_eq!(
            input.innovation(),
            *node.input_genes().iter().nth(0).unwrap()
        );
        assert_eq!(
            output.innovation(),
            *node.output_genes().iter().nth(0).unwrap()
        );
        assert!(config.activation_types.contains(&node.activation_type()));
        assert_eq!(
            node.innovation(),
            genome.nodes.get(&2).unwrap().innovation()
        );
        assert_eq!(genome.genes.len(), 3);
        assert_eq!(genome.nodes.len(), 3);
        assert_eq!(genome.genes.get(&0).unwrap().suppressed, true);

        Ok(())
    }

    #[test]
    fn mutate_nodes_no_gene_found() {
        let mut config = GeneticConfig::default();
        config.initial_expression_chance = 0.0;
        config.node_mutation_chance = 1.0;

        let mut history = History::new(&config);

        let mut genome = Genome::new(&config);
        let node = genome.mutate_nodes(&mut history, &config);

        if let Ok(_) = node {
            panic!("node mutation on empty genome should return an error");
        }
    }

    #[test]
    fn add_noncommon_structure() -> Result<(), String> {
        let mut config = GeneticConfig::default();
        config.input_count = NonZeroUsize::new(2).unwrap();
        config.output_count = NonZeroUsize::new(1).unwrap();
        config.initial_expression_chance = 1.0;

        let mut genome1 = Genome::new(&config);
        let mut genome2 = Genome::new(&config);
        genome1.add_node(3, ActivationType::Sigmoid)?;
        genome2.add_node(3, ActivationType::Sigmoid)?;
        genome2.add_node(4, ActivationType::ReLU)?;
        genome1.add_gene(2, 0, 3, 1.0)?;
        genome2.add_gene(3, 0, 3, 1.0)?;
        genome2.add_gene(4, 0, 4, 2.0)?;
        genome2.add_gene(5, 3, 4, 3.0)?;
        genome2.add_gene(6, 4, 2, 4.0)?;
        genome1.add_noncommon_structure(&genome2)?;

        let mut node4 = Node::new(4, NodeType::Neuron, ActivationType::ReLU);
        node4.add_input_gene(4)?;
        node4.add_input_gene(5)?;
        node4.add_output_gene(6)?;
        assert_eq!(genome1.nodes.get(&4).unwrap(), &node4);
        assert_eq!(genome1.genes.get(&2).unwrap(), &Gene::new(2, 0, 3, 1.0));
        assert_eq!(genome1.genes.get(&4).unwrap(), &Gene::new(4, 0, 4, 2.0));
        assert_eq!(genome1.genes.get(&5).unwrap(), &Gene::new(5, 3, 4, 3.0));
        assert_eq!(genome1.genes.get(&6).unwrap(), &Gene::new(6, 4, 2, 4.0));

        Ok(())
    }

    #[test]
    fn combines_genes_average() -> Result<(), String> {
        let mut config = GeneticConfig::default();
        config.input_count = NonZeroUsize::new(2).unwrap();
        config.output_count = NonZeroUsize::new(1).unwrap();
        config.initial_expression_chance = 0.0;

        let mut genome1 = Genome::new(&config);
        genome1.add_node(3, ActivationType::Sigmoid)?;

        let mut genome2 = genome1.clone();

        genome1.add_gene(0, 0, 2, 1.0)?;
        genome2.add_gene(0, 0, 2, 3.0)?;
        genome1.add_gene(1, 0, 3, 2.0)?;
        genome2.add_gene(1, 0, 3, -2.0)?;
        genome1.add_gene(2, 3, 2, 3.0)?;
        genome2.add_gene(2, 3, 2, 3.5)?;
        genome1.add_gene(3, 1, 2, 4.0)?;
        genome2.add_gene(3, 1, 2, 4.0)?;

        genome1.combines_genes_average(&genome2);

        assert_eq!(genome1.genes.get(&0).unwrap().weight, 2.0);
        assert_eq!(genome1.genes.get(&1).unwrap().weight, 0.0);
        assert_eq!(genome1.genes.get(&2).unwrap().weight, 3.25);
        assert_eq!(genome1.genes.get(&3).unwrap().weight, 4.0);

        Ok(())
    }

    #[test]
    fn combines_genes_random_choice() -> Result<(), String> {
        let mut config = GeneticConfig::default();
        config.input_count = NonZeroUsize::new(2).unwrap();
        config.output_count = NonZeroUsize::new(1).unwrap();
        config.initial_expression_chance = 0.0;

        let mut genome1 = Genome::new(&config);
        genome1.add_node(3, ActivationType::Sigmoid)?;

        let mut genome2 = genome1.clone();

        genome1.add_gene(0, 0, 2, 1.0)?;
        genome2.add_gene(0, 0, 2, 3.0)?;
        genome1.add_gene(1, 0, 3, 2.0)?;
        genome2.add_gene(1, 0, 3, -2.0)?;
        genome1.add_gene(2, 3, 2, 3.0)?;
        genome2.add_gene(2, 3, 2, 3.5)?;
        genome1.add_gene(3, 1, 2, 4.0)?;
        genome2.add_gene(3, 1, 2, 4.0)?;

        genome1.combines_genes_random_choice(&genome2);

        assert!([1.0, 3.0].contains(&genome1.genes.get(&0).unwrap().weight));
        assert!([2.0, -2.0].contains(&genome1.genes.get(&1).unwrap().weight));
        assert!([3.0, 3.5].contains(&genome1.genes.get(&2).unwrap().weight));
        assert!([4.0].contains(&genome1.genes.get(&3).unwrap().weight));

        Ok(())
    }

    #[test]
    fn reset_suppressions() -> Result<(), String> {
        let mut config = GeneticConfig::default();
        config.input_count = NonZeroUsize::new(2).unwrap();
        config.output_count = NonZeroUsize::new(1).unwrap();
        config.initial_expression_chance = 0.0;
        config.suppression_reset_chance = 1.0;

        let mut genome = Genome::new(&config);
        genome.add_gene(0, 0, 2, 3.0)?.suppressed = true;
        genome.add_gene(1, 1, 2, 4.0)?.suppressed = true;
        genome.add_gene(2, 2, 2, 5.0)?.suppressed = true;

        genome.reset_suppressions(&config);

        assert!(genome.genes().values().all(|g| !g.suppressed));

        Ok(())
    }

    #[test]
    fn genetic_distance_to() -> Result<(), String> {
        let mut config = GeneticConfig::default();
        config.input_count = NonZeroUsize::new(2).unwrap();
        config.output_count = NonZeroUsize::new(1).unwrap();
        config.initial_expression_chance = 0.0;
        config.common_weight_factor = 1.0;
        config.disjoint_gene_factor = 0.5;
        config.excess_gene_factor = 0.5;

        let mut genome1 = Genome::new(&config);
        let mut genome2 = Genome::new(&config);

        genome1.add_node(3, ActivationType::Sigmoid)?;
        genome2.add_node(4, ActivationType::Sigmoid)?;

        genome1.add_gene(1, 0, 2, -2.0)?;
        genome2.add_gene(1, 0, 2, 2.0)?;

        genome1.add_gene(2, 1, 3, 5.0)?;
        genome2.add_gene(3, 2, 4, 5.0)?;

        genome1.add_gene(4, 1, 2, 3.0)?;
        genome2.add_gene(4, 1, 2, 6.0)?;

        genome1.add_gene(5, 2, 2, 5.0)?;
        genome2.add_gene(6, 4, 4, 1.0)?;

        assert_eq!(
            genome1.genetic_distance_to(&genome2, &config),
            0.5 * 2.0 + 0.5 * 2.0 + 1.0 * (4.0 + 3.0)
        );
        Ok(())
    }

    #[test]
    fn genetic_distance_to_equal() -> Result<(), String> {
        let mut config = GeneticConfig::default();
        config.input_count = NonZeroUsize::new(2).unwrap();
        config.output_count = NonZeroUsize::new(1).unwrap();
        config.initial_expression_chance = 1.0;
        config.common_weight_factor = 1.0;
        config.disjoint_gene_factor = 0.5;
        config.excess_gene_factor = 0.5;

        let mut genome = Genome::new(&config);

        genome.add_node(3, ActivationType::Sigmoid)?;
        genome.add_node(4, ActivationType::Sigmoid)?;
        genome.add_gene(3, 2, 4, 5.0)?;
        genome.add_gene(5, 2, 2, 5.0)?;
        genome.add_gene(6, 4, 4, 1.0)?;

        assert_eq!(genome.genetic_distance_to(&genome.clone(), &config), 0.0);

        Ok(())
    }
}
