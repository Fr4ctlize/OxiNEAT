use crate::{Genome, InnovationHistory, PopulationConfig, Species, SpeciesID};

use std::collections::HashMap;

use ahash::RandomState;
use rand::{
    seq::{IteratorRandom, SliceRandom},
    Rng,
};

/// Auxiliary type for offspring generation.
/// Handles all the tasks of generating a population's
/// offspring according to the specified configs
/// and allotted offspring.
pub(super) struct OffspringFactory<'a, C, H, G> {
    species: &'a [Species<G>],
    history: &'a mut H,
    genetic_config: &'a C,
    population_config: &'a PopulationConfig,
}

impl<'a, C, H, G> OffspringFactory<'a, C, H, G>
where
    H: InnovationHistory,
    G: Genome<InnovationHistory = H, Config = C> + Clone,
{
    /// Create a new offspring factory.
    pub(super) fn new(
        species: &'a [Species<G>],
        history: &'a mut H,
        genetic_config: &'a C,
        population_config: &'a PopulationConfig,
    ) -> OffspringFactory<'a, C, H, G> {
        OffspringFactory {
            species,
            history,
            genetic_config,
            population_config,
        }
    }

    /// Generate the alloted offspring.
    pub(super) fn generate_offspring(
        &mut self,
        allotted_offspring: &[usize],
    ) -> HashMap<SpeciesID, Vec<G>, RandomState> {
        let mut offspring_of_species = self
            .species
            .iter()
            .enumerate()
            .map(|(i, s)| (s.id(), Vec::with_capacity(allotted_offspring[i])))
            .collect();

        for (species_index, allotted_offspring) in allotted_offspring.iter().enumerate() {
            let current_species = &self.species[species_index];
            let elite = current_species
                .count_elite(self.population_config)
                .min(*allotted_offspring);
            let offspring = (*allotted_offspring - elite).max(0);

            self.add_species_elite(&mut offspring_of_species, species_index, elite);
            self.add_mated_offspring(offspring, &mut offspring_of_species, species_index);
        }

        offspring_of_species
    }

    /// Add the top "elite" members of the species
    /// to the offspring.
    fn add_species_elite(
        &mut self,
        offpring_map: &mut HashMap<SpeciesID, Vec<G>, RandomState>,
        species_index: usize,
        elite: usize,
    ) {
        let species = &self.species[species_index];
        offpring_map
            .get_mut(&species.id())
            .unwrap()
            .extend_from_slice(&species.genomes[0..elite])
    }

    /// Choose parents from the species or from
    /// other species and mate them, adding the child
    /// to the species' offspring.
    fn add_mated_offspring(
        &mut self,
        offspring: usize,
        species_offspring: &mut HashMap<SpeciesID, Vec<G>, RandomState>,
        species_index: usize,
    ) {
        let species = &self.species[species_index];
        let survivors = species.count_survivors(self.population_config);
        let eligible_parents: Vec<&G> = species.genomes[..survivors].iter().collect();
        let mut rng = rand::thread_rng();
        // Mate parents
        for _ in 0..offspring {
            let parent1 = *eligible_parents
                .choose(&mut rng)
                .unwrap_or_else(|| panic!("no eligible parents in species {:?}", species.id()));
            let (parent2_species, parent2) =
                if rng.gen::<f32>() < self.population_config.sexual_reproduction_chance {
                    Self::choose_second_parent(species, self.species, self.population_config)
                } else {
                    (species.id(), parent1)
                };
            let child_species = if rng.gen::<bool>() {
                species.id()
            } else {
                parent2_species
            };
            species_offspring
                .get_mut(&child_species)
                .unwrap()
                .push(Genome::mate(
                    parent1,
                    parent2,
                    self.history,
                    self.genetic_config,
                ));
        }
    }

    /// Choose a parent from the currents species,
    /// or from another randomly selected.
    fn choose_second_parent(
        current_species: &'a Species<G>,
        all_species: &'a [Species<G>],
        population_config: &PopulationConfig,
    ) -> (SpeciesID, &'a G) {
        let mut rng = rand::thread_rng();

        if all_species.len() > 1 && rng.gen::<f32>() < population_config.interspecies_mating_chance
        {
            let other_species = all_species
                .iter()
                .filter(|s| s.id() != current_species.id())
                .choose(&mut rng)
                .unwrap();
            (
                other_species.id(),
                other_species.genomes.choose(&mut rng).unwrap_or_else(|| {
                    panic!("no eligible parents in species {:?}", other_species.id())
                }),
            )
        } else {
            (
                current_species.id(),
                current_species.genomes.choose(&mut rng).unwrap_or_else(|| {
                    panic!("no eligible parents in species {:?}", current_species.id())
                }),
            )
        }
    }
}
