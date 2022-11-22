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
    pub value: [T; N],
}

impl<const N: usize, T> Node<N, T> for QueryElement<N, T>
where
    T: Float + Sum,
{
    fn value(&self) -> [T; N] {
        self.value
    }

    fn nearest(
        &self,
        closest_found_elements: &[EnterPoint<N, T>],
        distance: &Distance,
    ) -> Option<EnterPoint<N, T>> {
        // Closest found elements should contain at least one element.
        debug_assert!(closest_found_elements.len() > 0);

        let mut lowest = T::zero();
        let mut lowest_index = 0;
        for (index, element) in closest_found_elements.iter().enumerate() {
            let temp = distance.calculate(element.get_value(), self.value);

            // Reinitialize lowest value at index 0.
            if index == 0 {
                lowest = temp;
            }

            if temp < lowest {
                lowest = temp;
                lowest_index = index;
            }
        }

        closest_found_elements.get(lowest_index).copied()
    }

    fn furthest(
        &self,
        neighbors: &[EnterPoint<N, T>],
        distance: &Distance,
    ) -> Option<EnterPoint<N, T>> {
        // Closest found elements should contain at least one element.
        debug_assert!(neighbors.len() > 0);

        let mut highest = T::zero();
        let mut highest_index = 0;
        for (index, element) in neighbors.iter().enumerate() {
            let temp = distance.calculate(element.get_value(), self.value);

            if index == 0 {
                highest = temp;
            }

            if temp < highest {
                highest = temp;
                highest_index = index;
            }
        }

        neighbors.get(highest_index).copied()
    }

    fn distance(&self, enter_point: &EnterPoint<N, T>, distance: &Distance) -> T {
        distance.calculate(self.value, enter_point.get_value())
    }
}

impl<const N: usize, T> QueryElement<N, T>
where
    T: Float + Sum + Clone + Copy + PartialOrd,
{
    pub fn new(value: [T; N]) -> Self {
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
