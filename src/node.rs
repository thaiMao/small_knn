use crate::distance::Distance;
use crate::enter_point::EnterPoint;
use num::Float;
use std::iter::Sum;

pub trait Node<const N: usize, T>
where
    T: Float + Sum,
{
    fn value(&self) -> [T; N];

    fn nearest(
        &self,
        closest_found_elements: &[EnterPoint<N, T>],
        distance: &Distance,
    ) -> Option<EnterPoint<N, T>>;

    fn furthest(
        &self,
        neighbors: &[EnterPoint<N, T>],
        distance: &Distance,
    ) -> Option<EnterPoint<N, T>>;

    fn distance(&self, enter_point: &EnterPoint<N, T>, distance: &Distance) -> T;
}
