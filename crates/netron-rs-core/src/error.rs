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

    #[error("access denied: {path}")]
    AccessDenied { path: String },

    #[error("model invariant failed: {0}")]
    Invariant(String),
}
