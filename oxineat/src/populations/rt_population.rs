use crate::{
    populations::{Population, PopulationConfig, Species},
    Genome, InnovationHistory,
};

/// Opaque token representing a genome
/// to get around borrow rules.
/// 
/// TODO: this representation will require `O(n)` 
/// replacement time, and force all GenomeTokens
/// to update, probably. Maybe reconsider?
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GenomeToken {
    species: usize,
    index: usize,
}

/// A `RealTimePopulation` supports rt-NEAT
/// through N-genome replacement without
/// generational evolution. 
/// 
/// If it makes more sense, this may become
/// the base Population class, and the generational
/// population a special case of this.
pub struct RealTimePopulation<C, H, G> {
    pop: Population<C, H, G>,
}

impl<C, H, G> RealTimePopulation<C, H, G>
where
    G: Genome<InnovationHistory = H, Config = C> + Clone,
{
    pub fn new(
        population_config: PopulationConfig,
        genetic_config: C,
    ) -> RealTimePopulation<C, H, G>
    where
        H: InnovationHistory<Config = C>,
    {
        unimplemented!()
    }

    pub fn remove_genomes<const N: usize>(&mut self, tokens: &[GenomeToken; N]) -> Option<G> {
        unimplemented!()
    }

    pub fn replace_genomes<F, P, T, const N: usize>(
        &mut self,
        tokens: &[GenomeToken; N],
        f: F,
        p: P,
    ) -> Option<(GenomeToken, T)>
    where
        F: Fn(GenomeToken) -> f32,
        P: Fn(&G) -> (GenomeToken, T),
    {
        unimplemented!()
    }

    pub fn add_child<P, T>(
        &mut self,
        parent1: GenomeToken,
        parent2: GenomeToken,
        p: P,
    ) -> Option<(GenomeToken, T)>
    where
        P: Fn(&G) -> (GenomeToken, T),
    {
        unimplemented!()
    }

    pub fn reset(&mut self)
    where
        C: Clone,
        H: InnovationHistory<Config = C>,
    {
        self.pop.reset();
    }

    pub fn get(&self, token: GenomeToken) -> Option<&G> {
        self.pop
            .species
            .get(token.species)
            .map(|s| s.genomes.get(token.index))
            .flatten()
    }

    pub fn genomes(&self) -> impl Iterator<Item = &G> {
        self.pop.genomes()
    }

    pub fn species(&self) -> impl Iterator<Item = &Species<G>> {
        self.pop.species.iter()
    }

    pub fn history(&self) -> &H
    where
        H: InnovationHistory<Config = C>,
    {
        &self.pop.history
    }
}
