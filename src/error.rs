use thiserror::Error;

#[derive(Error, Debug)]
pub enum KNNError {
    #[error("internal library error")]
    Internal,
    #[error("not inserted elements (expected to return {expected:?}, found {found:?})")]
    InsufficientInsertions { expected: String, found: String },
    #[error("unknown knn error")]
    Unknown,
}
