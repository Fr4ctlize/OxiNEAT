pub mod genomes;
pub mod networks;
pub mod populations;

pub use genomes::GeneticConfig;
pub use populations::PopConfig;

/// Identifier type used to designate historically
/// identical mutations for the purposes of
/// genome comparison and genetic tracking.
pub type Innovation = usize;

#[cfg(test)]
mod tests {}
