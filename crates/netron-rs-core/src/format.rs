use std::path::Path;

use crate::{Model, ModelError};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Confidence {
    None,
    Low,
    Medium,
    High,
}

#[derive(Clone, Copy, Debug)]
pub struct ModelInput<'a> {
    pub data: &'a [u8],
    pub path: Option<&'a Path>,
}

#[derive(Clone, Copy, Debug)]
pub struct FormatMetadata {
    pub name: &'static str,
    pub extensions: &'static [&'static str],
}

pub trait ModelFormat {
    fn metadata(&self) -> FormatMetadata;

    fn detect(&self, input: ModelInput<'_>) -> Confidence;

    fn parse(&self, input: ModelInput<'_>) -> Result<Model, ModelError>;
}
