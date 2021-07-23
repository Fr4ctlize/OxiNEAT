use super::*;

pub(super) struct OffspringFactory<'a> {
    species: &'a [Species],
    history: &'a mut History,
    genetic_config: &'a GeneticConfig,
    population_config: &'a PopulationConfig,
}

impl<'a> OffspringFactory<'a> {
    pub(super) fn new(
        species: &'a [Species],
        history: &'a mut History,
        genetic_config: &'a GeneticConfig,
        population_config: &'a PopulationConfig,
    ) -> OffspringFactory<'a> {
        OffspringFactory {
            species,
            history,
            genetic_config,
            population_config,
        }
    }

    pub(super) fn generate_offspring(
        &mut self,
        allotted_offspring: &[usize],
    ) -> HashMap<SpeciesID, Vec<Genome>> {
        let mut species_offspring = self
            .species
            .iter()
            .enumerate()
            .map(|(i, s)| (s.id(), Vec::with_capacity(allotted_offspring[i])))
            .collect();

        for (species_index, allotted_offspring) in allotted_offspring.iter().enumerate() {
            let current_species = &self.species[species_index];
            let elite = current_species
                .count_elite(&self.population_config)
                .min(*allotted_offspring);
            let offspring = (*allotted_offspring - elite).max(0);

            self.add_species_elite(&mut species_offspring, species_index, elite);
            self.add_mated_offspring(offspring, &mut species_offspring, species_index);
        }

        species_offspring
    }

    fn add_species_elite(
        &mut self,
        offpring_map: &mut HashMap<SpeciesID, Vec<Genome>>,
        species_index: usize,
        elite: usize,
    ) {
        let species = &self.species[species_index];
        offpring_map
            .get_mut(&species.id())
            .unwrap()
            .extend_from_slice(&species.genomes[0..elite])
    }

    fn add_mated_offspring(
        &mut self,
        offspring: usize,
        species_offspring: &mut HashMap<SpeciesID, Vec<Genome>>,
        species_index: usize,
    ) {
        // We need to get the species this way to avoid
        // problems with the borrow-checker, as it can't
        // see that we aren't touching it when we modify
        // self.history.
        let species = &self.species[species_index];
        let survivors = species.count_survivors(&self.population_config);
        let eligible_parents: Vec<&Genome> = species.genomes[..survivors].iter().collect();
        let mut rng = rand::thread_rng();
        // Mate parents
        for _ in 0..offspring {
            let parent1 = eligible_parents.choose(&mut rng).unwrap();
            let (parent2_species, parent2) =
                Self::choose_second_parent(species, &self.species, &self.population_config);
            let child_species = if rng.gen::<usize>() % 2 == 0 {
                species.id()
            } else {
                parent2_species
            };
            species_offspring.get_mut(&child_species).unwrap().push(
                parent1
                    .mate_with(parent2, &mut self.history, &self.genetic_config)
                    .unwrap(),
            );
        }
    }

    fn choose_second_parent(
        current_species: &'a Species,
        all_species: &'a [Species],
        population_config: &PopulationConfig,
    ) -> (SpeciesID, &'a Genome) {
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
                other_species.genomes.choose(&mut rng).unwrap(),
            )
        } else {
            (
                current_species.id(),
                current_species.genomes.choose(&mut rng).unwrap(),
            )
        }
    }
}
