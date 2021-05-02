/// Specialized result type for NEAT.
/// Differentiaties fatal and non-fatal
/// errors for internal purposes.
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub struct Error {
    pub(crate) error: ErrorType,
}

pub(crate) fn fatal(e: &str) -> Error {
    Error {
        error: ErrorType::Fatal(e.to_string()),
    }
}

pub(crate) fn nonfatal(e: &str) -> Error {
    Error {
        error: ErrorType::Nonfatal(e.to_string()),
    }
}

/// Fatal errors should return control
/// to the user, while non-fatal errors
/// can be safely ignored, but are returned
/// to the user for informative purposes.
#[derive(Debug)]
pub(crate) enum ErrorType {
    Fatal(String),
    Nonfatal(String),
}
