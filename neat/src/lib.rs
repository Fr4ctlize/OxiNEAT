pub mod genomes;
pub mod networks;
pub mod populations;
mod result;

pub use genomes::GeneticConfig;
pub use populations::PopConfig;
pub use result::Result;

/// Identifier type used to designate historically
/// identical mutations for the purposes of
/// genome comparison and genetic tracking.
pub type Innovation = usize;

#[cfg(test)]
mod tests {}
