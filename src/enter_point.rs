use crate::array_vec::ArrayVec;
use crate::distance::Distance;
use crate::node::Node;
use crate::query::Element;
use num::Float;
use std::iter::Sum;

/// Size of array that stores all connections to neighbors at all levels.
const SIZE: usize = 128;

/// * `N` - Fixed array size.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EnterPoint<const N: usize, T> {
    pub index: usize,
    pub layer: usize,
    pub value: [T; N],
    pub connections: ArrayVec<Element, SIZE>,
}

impl<const N: usize, T> EnterPoint<N, T>
where
    T: Float,
{
    pub fn new(value: [T; N], index: usize, layer: usize) -> Self {
        Self {
            index,
            layer,
            value,
            connections: ArrayVec::new(),
        }
    }

    pub fn get_value(&self) -> [T; N] {
        self.value
    }

    pub fn get_index(&self) -> usize {
        self.index
    }

    pub fn get_layer(&self) -> usize {
        self.layer
    }

    /// Returns connections for a given layer.
    pub fn neighbourhood(&self, layer: usize) -> impl Iterator<Item = Element> {
        self.connections
            .into_iter()
            .flatten()
            .filter(move |connection| connection.get_layer() == layer)
    }

    /// Remove connected neighbors for a given layer.
    pub fn clear_connections(&mut self, layer: usize) {
        let mut filtered_connections = [None; 128];

        for (connection, conn) in self
            .connections
            .into_iter()
            .flatten()
            .filter(|c| c.get_layer() != layer)
            .zip(filtered_connections.iter_mut())
        {
            *conn = Some(connection);
        }

        self.connections.inner = filtered_connections;
    }

    /// Return the number of connections for a given layer.
    pub fn number_of_connections(&self, layer: usize) -> usize {
        self.connections
            .into_iter()
            .flatten()
            .filter(move |connection| connection.get_layer() == layer)
            .fold(0, |total, _| total + 1)
    }
}

impl<const N: usize, T> Node<N, T> for EnterPoint<N, T>
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
            let temp = distance.calculate(element.value, self.value);

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
            let temp = distance.calculate(element.value, self.value);

            if index == 0 {
                highest = temp;
            }

            if temp > highest {
                highest = temp;
                highest_index = index;
            }
        }

        neighbors.get(highest_index).copied()
    }

    fn distance(&self, enter_point: &EnterPoint<N, T>, distance: &Distance) -> T {
        distance.calculate(self.value, enter_point.value)
    }
}

impl<const N: usize, T> PartialEq<Element> for EnterPoint<N, T>
where
    T: Float,
{
    fn eq(&self, connection: &Element) -> bool {
        self.index == connection.get_index()
    }
}

#[cfg(test)]
mod enter_point_tests {
    use super::{Element, EnterPoint};
    use crate::array_vec::ArrayVec;

    #[test]
    fn neighbourhood_test() {
        let mut connections = ArrayVec::<Element, 128>::new();

        connections.try_push(Element::new(4, 3)).unwrap();
        connections.try_push(Element::new(5, 2)).unwrap();
        connections.try_push(Element::new(2, 0)).unwrap();
        connections.try_push(Element::new(6, 1)).unwrap();
        connections.try_push(Element::new(8, 3)).unwrap();
        connections.try_push(Element::new(1, 3)).unwrap();
        connections.try_push(Element::new(3, 5)).unwrap();
        connections.try_push(Element::new(0, 6)).unwrap();
        connections.index = 8;
        let enter_point = EnterPoint {
            index: 0,
            layer: 2,
            value: [42_f32],
            connections,
        };

        let layer = 3;
        let neighbors: Vec<usize> = enter_point
            .neighbourhood(layer)
            .map(|element| element.get_index())
            .collect();

        assert_eq!(neighbors, vec![4, 8, 1]);
    }
}
