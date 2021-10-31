use crate::genomics::GeneticConfig;
use crate::Innovation;

use ahash::RandomState;
use oxineat::InnovationHistory;
use serde::{Deserialize, Serialize};

use std::collections::hash_map::{Entry, HashMap};

/// A `History` keeps track of gene and node innovations in a
/// population, in order to make sure identical mutations
/// are assigned the same innovation numbers.
///
/// For gene innovations the input and output nodes are used to
/// identify identical mutations, and the corresponding innovation
/// number is recorded.
///
/// For node innovations the split gene is used to identify
/// identical mutations, and the innovation numbers for the
/// corresponding input gene, new node, and output gene are
/// recorded, in that order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct History {
    next_gene_innovation: Innovation,
    next_node_innovation: Innovation,
    gene_innovations: HashMap<(Innovation, Innovation), Innovation, RandomState>,
    gene_endpoints: Vec<(Innovation, Innovation)>,
    node_innovations: HashMap<Innovation, (Innovation, Innovation, Innovation), RandomState>,
}

impl InnovationHistory for History {
    type Config = GeneticConfig;

    fn new(config: &GeneticConfig) -> History {
        Self::new(config)
    }
}

impl History {
    /// Creates a new History using the specified configuration.
    ///
    /// Initially generated genes are given the innovation number
    /// `o + i ⨯ output_count`, where `i` is the innovation number
    /// of their input node and `o` is that of their output node.
    /// Thus, the next availabe gene innovation number returnable by
    /// the `History` starts start at input_count ⨯ output_count`.
    ///
    /// # Examples
    /// ```
    /// use oxineat_nn::genomics::{GeneticConfig, History};
    ///
    /// let history = History::new(&GeneticConfig::zero());
    /// ```
    pub fn new(config: &GeneticConfig) -> History {
        let (gene_innovations, gene_endpoints) = (0..config.input_count.get())
            // Cartesian product of input and outputs...
            .flat_map(|i| (0..config.output_count.get()).map(move |o| (i, o)))
            // Get the output innovation number, as we only have indices...
            .map(|(i, o)| (i, o, o + config.input_count.get()))
            // Get both gene innovations and gene endpoints...
            .map(|(i, o_idx, o)| (((i, o), o_idx + i * config.output_count.get()), (i, o)))
            .unzip();
        History {
            // Pre-allocate innovation numbers for all possible initial
            // genes, and the input and output nodes.
            next_gene_innovation: config.input_count.get() * config.output_count.get(),
            next_node_innovation: config.input_count.get() + config.output_count.get(),
            gene_innovations,
            gene_endpoints,
            node_innovations: HashMap::default(),
        }
    }

    /// Returns the next gene innovation number, or the
    /// previously assigned number to the same gene mutation.
    pub(crate) fn next_gene_innovation(
        &self,
        input_id: Innovation,
        output_id: Innovation,
    ) -> Innovation {
        *self
            .gene_innovations
            .get(&(input_id, output_id))
            .unwrap_or(&self.next_gene_innovation)
    }

    /// Returns the next node and gene innovation numbers,
    /// or the previously assigned numbers to the same node mutation,
    /// in the format `(input gene, new node, output gene)`.
    ///
    /// If `duplicate` is `true` and the node mutation is
    /// already registered, the returned innovation numbers
    /// will be computed as if it were a new mutation. This
    /// is used in situations in which the mutating genome
    /// already split the same gene in a previous mutation,
    /// which would result in duplicate genes and nodes within
    /// the same genome. This can be detected if the numbers
    /// returned by this function without setting `duplicate`
    /// to `true` refer to genes/nodes already present in the
    /// genome.
    pub(crate) fn next_node_innovation(
        &self,
        split_gene: Innovation,
        duplicate: bool,
    ) -> (Innovation, Innovation, Innovation) {
        if !self.node_innovations.contains_key(&split_gene) || duplicate {
            (
                self.next_gene_innovation,
                self.next_node_innovation,
                self.next_gene_innovation + 1,
            )
        } else {
            self.node_innovations[&split_gene]
        }
    }

    /// Adds a gene mutation to the history and returns
    /// the corresponding new gene innovation number, or the
    /// previously assigned number for the same gene mutation.
    pub(crate) fn add_gene_innovation(&mut self, input_id: Innovation, output_id: Innovation) {
        if let Entry::Vacant(entry) = self.gene_innovations.entry((input_id, output_id)) {
            entry.insert(self.next_gene_innovation);
            self.gene_endpoints.push((input_id, output_id));
            self.next_gene_innovation += 1;
        }
    }

    /// Adds a node mutation to the history, if the mutation
    /// is new.
    ///
    /// If `duplicate` is `true` and the node mutation is
    /// already registered, the returned innovation numbers
    /// will be computed as if it were a new mutation, and
    /// the new numbers will substitute the previously assigned
    /// ones. This is used in situations in which the mutating genome
    /// already split the same gene in a previous mutation,
    /// which would result in duplicate genes and nodes within
    /// the same genome. This can be detected if the numbers
    /// returned by this function without setting `duplicate`
    /// to `true` refer to genes/nodes already present in the
    /// genome.
    pub(crate) fn add_node_innovation(&mut self, split_gene: Innovation, duplicate: bool) {
        if !self.node_innovations.contains_key(&split_gene) || duplicate {
            let (split_gene_input_node, split_gene_output_node) = self.gene_endpoints[split_gene];
            let new_node = self.next_node_innovation;

            let new_input_gene = self.next_gene_innovation;
            self.add_gene_innovation(split_gene_input_node, new_node);
            let new_output_gene = self.next_gene_innovation;
            self.add_gene_innovation(new_node, split_gene_output_node);
            let innovation_record = (new_input_gene, new_node, new_output_gene);

            self.node_innovations.insert(split_gene, innovation_record);
            self.next_node_innovation += 1;
        }
    }

    // This has been removed to simplify keep track of
    // innovations and to maintain a more "complete" picture
    // of the innovation history.
    //
    // /// Clears the history's lists of mutations, but keeps
    // /// its innovation number counts.
    // pub fn clear(&mut self) {
    //     self.gene_innovations.clear();
    //     self.node_innovations.clear();
    // }

    /// Returns the highest gene innovation number generated.
    ///
    /// # Examples
    /// ```
    /// use oxineat_nn::genomics::{GeneticConfig, History};
    ///
    /// let history = History::new(&GeneticConfig::zero());
    ///
    /// assert_eq!(history.max_gene_innovation(), 0);
    /// ```
    pub fn max_gene_innovation(&self) -> Innovation {
        self.next_gene_innovation - 1
    }

    /// Returns the hightest node innovation number generated.
    ///
    /// # Examples
    /// ```
    /// use oxineat_nn::genomics::{GeneticConfig, History};
    ///
    /// let history = History::new(&GeneticConfig::zero());
    ///
    /// assert_eq!(history.max_node_innovation(), 1);
    /// ```
    pub fn max_node_innovation(&self) -> Innovation {
        self.next_node_innovation - 1
    }

    /// Returns an iterator over the complete record of
    /// gene innovations, in the format
    /// `((input node, output node), gene innovation)`.
    /// No ordering is guaranteed.
    ///
    /// # Examples
    /// ```
    /// use oxineat_nn::genomics::{GeneticConfig, History, NNGenome};
    ///
    /// let config = GeneticConfig {
    ///     initial_expression_chance: 1.0,
    ///     recursion_chance: 1.0,
    ///     ..GeneticConfig::zero()
    /// };
    /// let mut history = History::new(&config);
    /// 
    /// // Add mutations to the history through genome mutation.
    /// NNGenome::new(&config).mutate_add_gene(&mut history, &config);
    ///
    /// for ((input_node, output_node), gene) in history.gene_innovation_history() {
    ///     println!("gene innovation with id {} from node {} to node {}",
    ///         gene, input_node, output_node);
    /// }
    /// ```
    pub fn gene_innovation_history(
        &self,
    ) -> impl Iterator<Item = (&(Innovation, Innovation), &Innovation)> {
        self.gene_innovations.iter()
    }

    /// Returns an iterator over the complete record of
    /// node innovations, in the format
    /// `(split gene, (input gene, new node, output gene))`.
    /// No ordering is guaranteed.
    ///
    /// # Examples
    /// ```
    /// use oxineat_nn::genomics::{GeneticConfig, History, NNGenome};
    ///
    /// let config = GeneticConfig {
    ///     initial_expression_chance: 1.0,
    ///     ..GeneticConfig::zero()
    /// };
    /// let mut history = History::new(&config);
    /// 
    /// // Add mutations to the history through genome mutation.
    /// NNGenome::new(&config).mutate_add_node(&mut history, &config);
    ///
    /// for (split_gene, (input_gene, new_node, output_gene)) in history.node_innovation_history() {
    ///     println!("gene {} split into genes {} and {} with node {} in between",
    ///         split_gene, input_gene, output_gene, new_node);
    /// }
    /// ```
    pub fn node_innovation_history(
        &self,
    ) -> impl Iterator<Item = (&Innovation, &(Innovation, Innovation, Innovation))> {
        self.node_innovations.iter()
    }
}

#[cfg(test)]
mod tests {}
