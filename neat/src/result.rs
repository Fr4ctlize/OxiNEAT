/// Specialized result type for NEAT.
/// Differentiaties fatal and non-fatal
/// errors for internal purposes.
pub type Result<T> = std::result::Result<T, Error>;

pub struct Error {
    error: ErrorType,
}

/// Fatal errors should return control
/// to the user, while non-fatal errors
/// can be safely ignored, but are returned
/// to the user for informative purposes.
enum ErrorType {
    Fatal(Box<dyn std::error::Error + Send>),
    Nonfatal(Box<dyn std::error::Error + Send>),
}