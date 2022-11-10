use crate::array_vec::ArrayVec;
use crate::distance::Distance;
use crate::node::Node;
use crate::query::Element;
use num::Float;
use std::iter::Sum;

/// * `N` - Fixed array size.
/// * `M` - Number of established connections.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EnterPoint<const N: usize, const M: usize, T> {
    index: usize,
    layer: usize,
    value: [T; N],
    connections: ArrayVec<Element, M>,
    enter_point_index: usize,
}

impl<const N: usize, const M: usize, T> EnterPoint<N, M, T> {
    fn new(value: [T; N], index: usize, layer: usize, enter_point_index: usize) -> Self {
        Self {
            index,
            layer,
            value,
            connections: ArrayVec::new(),
            enter_point_index,
        }
    }

    pub fn get_value(&self) -> [T; N] {
        self.value
    }

    /// Returns connections for a given layer.
    pub fn neighbourhood(&self, layer: usize) -> impl Iterator<Item = Element> {
        self.connections
            .into_iter()
            .flatten()
            .filter(move |connection| connection.get_layer() == layer)
    }

    /// Return the number of connections for a given layer.
    fn number_of_connections(&self, layer: usize) -> usize {
        self.connections
            .into_iter()
            .flatten()
            .filter(move |connection| connection.get_layer() == layer)
            .fold(0, |total, _| total + 1)
    }
}

impl<const N: usize, const M: usize, T> Node<N, M, T> for EnterPoint<N, M, T>
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
            let temp = distance.calculate(element.value, self.value);

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
            let temp = distance.calculate(element.value, self.value);

            if highest.is_none() {
                highest = Some(temp);
            }

            if temp > highest.unwrap() {
                highest = Some(temp);
                highest_index = index;
            }
        }

        neighbors.get(highest_index).unwrap().clone()
    }

    fn distance(&self, enter_point: &EnterPoint<N, M, T>, distance: &Distance) -> T {
        distance.calculate(self.value, enter_point.value)
    }
}

impl<const N: usize, const M_MAX: usize, T> PartialEq<Element> for EnterPoint<N, M_MAX, T>
where
    T: Float,
{
    fn eq(&self, connection: &Element) -> bool {
        self.index == connection.get_index()
    }
}
