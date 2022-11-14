use num::{cast, Float};
use std::iter::Sum;

#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Distance {
    Euclidean,
}

impl Distance {
    /// * `N` - N dimensional space
    pub fn calculate<const N: usize, T>(&self, q: [T; N], p: [T; N]) -> T
    where
        T: Float + Sum + Clone + Copy,
    {
        match self {
            // TODO SIMD
            Self::Euclidean => q
                .iter()
                .cloned()
                .zip(p.iter().cloned())
                .map(|(q_i, p_i)| (q_i - p_i).powf(cast::<usize, T>(2).unwrap()))
                .sum::<T>()
                .sqrt(),
        }
    }
}
