use crate::{
    populations::{Population, PopulationConfig, Species, SpeciesID},
    Genome, InnovationHistory,
};

use std::marker::PhantomData;
use std::num::NonZeroUsize;

use rand::prelude::*;

/// An ID for a genome inside a species. Is guaranteed
/// to be unique inside the species for the lifetime
/// of each instance.
/// NonZeroUsize used for Option niche optimization.
type GenomeID = NonZeroUsize;
/// A genome that is tagged for the token system of RealTimePopulation.
/// A value of None indicates an uninitialized tag.
#[derive(Debug)]
struct TaggedGenome<G>(G, Option<GenomeID>);

impl<G: Genome> Genome for TaggedGenome<G> {
    type Config = G::Config;
    type InnovationHistory = G::InnovationHistory;

    fn new(config: &Self::Config) -> Self {
        TaggedGenome(G::new(config), None)
    }

    fn genetic_distance(first: &Self, second: &Self, config: &Self::Config) -> f32 {
        G::genetic_distance(&first.0, &second.0, config)
    }

    fn mate(
        parent1: &Self,
        parent2: &Self,
        history: &mut Self::InnovationHistory,
        config: &Self::Config,
    ) -> Self {
        TaggedGenome(G::mate(&parent1.0, &parent2.0, history, config), None)
    }

    fn set_fitness(&mut self, fitness: f32) {
        self.0.set_fitness(fitness)
    }

    fn fitness(&self) -> f32 {
        self.0.fitness()
    }

    fn conforms_to(&self, config: &Self::Config) -> bool {
        self.0.conforms_to(config)
    }
}

impl<G: Clone> Clone for TaggedGenome<G> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), self.1.clone())
    }
}

/// Opaque token representing a genome.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct GenomeToken<'id> {
    inner: GenomeTokenInner,
    _marker: PhantomData<&'id ()>,
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
struct GenomeTokenInner {
    species: SpeciesID,
    genome_id: GenomeID,
}

impl From<&GenomeTokenInner> for GenomeToken<'_> {
    fn from(t: &GenomeTokenInner) -> Self {
        GenomeToken {
            inner: GenomeTokenInner {
                species: t.species,
                genome_id: t.genome_id,
            },
            _marker: PhantomData,
        }
    }
}

/// A `RealTimePopulation` supports rt-NEAT
/// through N-genome replacement without
/// generational evolution.
pub struct RealTimePopulation<'id, C, H, G> {
    pop: Population<C, H, TaggedGenome<G>>,
    _marker: PhantomData<&'id ()>,
}

/// Implementation invariant:
///     GenomeIDs are added in increasing order to token map
///     vectors, and maintain a total ordering of insertion order,
///     mirroring ordering in the species' Genome vector.
/// DO NOT MODIFY INSERTION ORDER INTO SPECIES WITHOUT UPDATING
/// THIS INVARIANT ACCORDINGLY.
impl<'id, C, H, G> RealTimePopulation<'id, C, H, G>
where
    G: Genome<InnovationHistory = H, Config = C> + Clone,
{
    pub fn new<F, R>(population_config: PopulationConfig, genetic_config: C, f: F) -> R
    where
        H: InnovationHistory<Config = C>,
        F: for<'a> FnOnce(RealTimePopulation<C, H, TaggedGenome<G>>) -> R,
    {
        let realtime_population = RealTimePopulation {
            pop: Population::new(population_config, genetic_config),
            _marker: PhantomData,
        };
        f(realtime_population)
    }

    pub fn remove_genomes(&mut self, tokens: impl Iterator<Item = GenomeToken<'id>>) -> Vec<G> {
        // Maybe use Vec::drain_filter when it stabilizes to reduce unnecessary
        // copies: https://github.com/rust-lang/rust/issues/43244.
        tokens
            .map(|token| {
                let species = &mut self.get_species_mut(&token.inner.species);
                let index = species
                    .genomes
                    .binary_search_by_key(&token.inner.genome_id, |g| {
                        g.1.expect("retrieve genome id that should be initialized")
                    })
                    .expect("retrieve genome index from species that contains it");
                species.genomes.remove(index).0
            })
            .collect()
    }

    pub fn replace_genomes<F, P, T>(
        &mut self,
        tokens: impl Iterator<Item = GenomeToken<'id>>,
        f: F,
        p: P,
    ) -> (Vec<G>, Vec<(GenomeToken<'id>, T)>)
    where
        F: Fn(&GenomeToken<'id>) -> Result<f32, f32>,
        P: Fn(&G) -> T,
    {
        let replaced = self.remove_genomes(tokens);

        let history = &mut self.pop.history;
        // let children = Vec::with_capacity(replaced.len());

        self.pop.evaluate_fitness(|g| {
            let token = GenomeTokenInner {
                species: todo!(),
                genome_id: g.1.unwrap(),
            };
            match f(&(&token).into()) {
                Ok(fitness) => fitness,
                Err(fitness) => todo!(),
            }
        });
        self.pop.get_species_adjusted_fitness();

        // for _ in 0..replaced.len() {
        //     let parent1 =
        // }
        // let parent1 = try_get_parent(self, parent1);
        // let child = parent1
        //     .map(|parent1| {
        //         if thread_rng().gen::<f32>() < self.pop.population_config.sexual_reproduction_chance
        //         {
        //             let parent2 = try_get_parent(self, parent2);
        //             parent2.map(|parent2| {
        //                 Genome::mate(parent1, parent2, history, &self.pop.genetic_config)
        //             })
        //         } else {
        //             Some(Genome::mate(
        //                 parent1,
        //                 parent1,
        //                 &mut self.pop.history,
        //                 &self.pop.genetic_config,
        //             ))
        //         }
        //     })
        //     .flatten();

        (replaced, unimplemented!())
    }

    pub fn insert<P, T>(&mut self, genome: G, p: P) -> (GenomeToken<'id>, T)
    where
        P: Fn(&G) -> T,
    {
        // Generate the phenotype before the genome is moved
        // during speciation.
        let phenotype = p(&genome);

        let assigned_species = {
            let next_species_id = SpeciesID(self.pop.historical_species_count, 0);
            // Assign a tag of None here, as we don't yet know where it will end up.
            let assigned_species = self
                .pop
                .respeciate(TaggedGenome(genome, None), next_species_id);
            if assigned_species == next_species_id {
                self.pop.historical_species_count += 1;
            }
            let idx = self
                .pop
                .species
                .binary_search_by_key(&assigned_species, |s| s.id())
                .expect("search for species known to exist");
            &mut self.pop.species[idx]
        };

        let genome_id = if assigned_species.genomes.len() == 1 {
            let intialized_id = GenomeID::new(1);
            assigned_species.genomes[0].1 = intialized_id.clone();
            intialized_id.unwrap()
        } else {
            let initialized_id = assigned_species.genomes[assigned_species.genomes.len() - 2]
                .1
                .map(|id| NonZeroUsize::new(id.get() + 1).unwrap());
            assigned_species.genomes.iter_mut().last().unwrap().1 = initialized_id.clone();
            initialized_id.unwrap()
        };

        let token = {
            let map_token = GenomeTokenInner {
                species: assigned_species.id(),
                genome_id,
            };
            (&map_token).into()
        };

        (token, phenotype)
    }

    pub fn get(&self, token: &GenomeToken<'id>) -> &G {
        let token_species = self.get_species(&token.inner.species);
        let index = token_species
            .genomes
            .binary_search_by_key(&token.inner.genome_id, |g| g.1.unwrap())
            .unwrap();
        &token_species.genomes[index].0
    }

    pub fn genomes(&self) -> impl Iterator<Item = &G> {
        self.pop.genomes().map(|g| &g.0)
    }

    pub fn species(&self) -> impl Iterator<Item = &Species<G>> {
        todo!()
    }

    pub fn history(&self) -> &H
    where
        H: InnovationHistory<Config = C>,
    {
        &self.pop.history
    }

    fn get_species(&self, id: &SpeciesID) -> &Species<TaggedGenome<G>> {
        self.pop
            .species
            .binary_search_by_key(id, Species::id)
            .ok()
            .map(|i| &self.pop.species[i])
            .expect("retrieve species of genome token")
    }

    fn get_species_mut(&mut self, id: &SpeciesID) -> &mut Species<TaggedGenome<G>> {
        self.pop
            .species
            .binary_search_by_key(id, Species::id)
            .ok()
            .map(move |i| &mut self.pop.species[i])
            .expect("retrieve species of genome token")
    }
}

impl<'id, C, H, G> From<RealTimePopulation<'id, C, H, G>> for Population<C, H, G> {
    fn from(rt_pop: RealTimePopulation<'id, C, H, G>) -> Self {
        todo!()
    }
}

impl<'a, 'id, C, H, G> From<&'a RealTimePopulation<'id, C, H, G>> for &'a Population<C, H, G> {
    fn from(rt_pop: &'a RealTimePopulation<'id, C, H, G>) -> Self {
        todo!()
    }
}
