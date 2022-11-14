use crate::distance::Distance;
use crate::enter_point::EnterPoint;
use num::Float;
use std::iter::Sum;

pub trait Node<const N: usize, const M: usize, T>
where
    T: Float + Sum,
{
    fn value(&self) -> [T; N];

    fn nearest(
        &self,
        closest_found_elements: &[EnterPoint<N, M, T>],
        distance: &Distance,
    ) -> Option<EnterPoint<N, M, T>>;

    fn furthest(
        &self,
        neighbors: &[EnterPoint<N, M, T>],
        distance: &Distance,
    ) -> EnterPoint<N, M, T>;

    fn distance(&self, enter_point: &EnterPoint<N, M, T>, distance: &Distance) -> T;
}
