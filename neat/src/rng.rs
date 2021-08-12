use rand::Rng;

/// Simple wrapper for a T: Rng,
/// needed for dependancy inversion
/// (monomorphization implies Rng's
/// can't be passed around directly,
/// probably can't build a vtbl).
#[derive(Clone)]
pub struct SimpleBernoulli<T: Rng + Clone>(T);

impl<T> SimpleBernoulli<T>
where
    T: Rng + Clone,
{
    pub fn new(rng: T) -> SimpleBernoulli<T> {
        SimpleBernoulli(rng)
    }

    pub fn gen_bool(&mut self, chance: f32) -> bool {
        self.0.gen::<f32>() < chance
    }
}
