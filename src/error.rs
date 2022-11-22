use thiserror::Error;

#[derive(Error, Debug)]
pub enum KNNError {
    #[error("internal library error")]
    Internal,
    #[error("not inserted elements (expected to return {expected:?}, found {found:?})")]
    InsufficientInsertions { expected: usize, found: usize },
    #[error("maximum connections reached, unable to push element")]
    FullCapacity,
    #[error("unknown knn error")]
    Unknown,
}
