use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("unsupported model format")]
    UnsupportedFormat,

    #[error("{format} data is invalid: {message}")]
    InvalidData {
        format: &'static str,
        message: String,
    },

    #[error("model invariant failed: {0}")]
    Invariant(String),
}
