use crate::distance::Distance;
use crate::enter_point::EnterPoint;
use crate::node::Node;
use num::Float;
use std::iter::Sum;

#[derive(Clone, Copy, Debug)]
pub struct QueryElement<const N: usize, T>
where
    T: Float,
{
    value: [T; N],
}

impl<const N: usize, const M: usize, T> Node<N, M, T> for QueryElement<N, T>
where
    T: Float + Sum,
{
    fn value(&self) -> [T; N] {
        self.value
    }

    fn nearest(
        &self,
        closest_found_elements: &[EnterPoint<N, M, T>],
        distance: &Distance,
    ) -> EnterPoint<N, M, T> {
        let mut lowest = None;
        let mut lowest_index = 0;
        for (index, element) in closest_found_elements.iter().enumerate() {
            let temp = distance.calculate(element.get_value(), self.value);

            if lowest.is_none() {
                lowest = Some(temp);
            }

            if temp < lowest.unwrap() {
                lowest = Some(temp);
                lowest_index = index;
            }
        }

        closest_found_elements.get(lowest_index).unwrap().clone()
    }

    fn furthest(
        &self,
        neighbors: &[EnterPoint<N, M, T>],
        distance: &Distance,
    ) -> EnterPoint<N, M, T> {
        let mut highest = None;
        let mut highest_index = 0;
        for (index, element) in neighbors.iter().enumerate() {
            let temp = distance.calculate(element.get_value(), self.value);

            if highest.is_none() {
                highest = Some(temp);
            }

            if temp < highest.unwrap() {
                highest = Some(temp);
                highest_index = index;
            }
        }

        neighbors.get(highest_index).unwrap().clone()
    }

    fn distance(&self, enter_point: &EnterPoint<N, M, T>, distance: &Distance) -> T {
        distance.calculate(self.value, enter_point.get_value())
    }
}

impl<const N: usize, T> QueryElement<N, T>
where
    T: Float + Sum + Clone + Copy + PartialOrd,
{
    fn new(value: [T; N]) -> Self {
        Self { value }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Element {
    index: usize,
    layer: usize,
}

impl Element {
    /// * `index` - Index location of enter point in self.enter_points.
    /// * `layer` - Layer of the enter point.
    pub fn new(index: usize, layer: usize) -> Self {
        Self { index, layer }
    }

    pub fn get_index(&self) -> usize {
        self.index
    }

    pub fn get_layer(&self) -> usize {
        self.layer
    }
}
