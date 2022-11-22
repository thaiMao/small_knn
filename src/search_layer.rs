use crate::distance::Distance;
use crate::enter_point::EnterPoint;
use crate::node::Node;
use num::Float;
use std::{cmp::Ordering, collections::HashMap, fmt::Debug, iter::Sum};

#[derive(Clone, Debug)]
pub struct SearchLayer<const N: usize, const M: usize, T>
where
    T: Float,
{
    visited_elements: Vec<EnterPoint<N, M, T>>,
    candidates: Vec<EnterPoint<N, M, T>>,
    found_nearest_neighbors: Vec<EnterPoint<N, M, T>>,
    distance: Distance,
}

impl<'a, const N: usize, const M: usize, T> SearchLayer<N, M, T>
where
    T: Float + Sum + Debug,
{
    pub fn new(distance: Distance, capacity: usize) -> Self {
        Self {
            visited_elements: Vec::with_capacity(capacity),
            candidates: Vec::with_capacity(capacity),
            found_nearest_neighbors: Vec::with_capacity(capacity),
            distance,
        }
    }

    pub fn clear(&mut self) {
        self.visited_elements.clear();
        self.candidates.clear();
        self.found_nearest_neighbors.clear();
    }

    pub fn search<const NUMBER_OF_NEAREST_TO_Q_ELEMENTS_TO_RETURN: usize>(
        &mut self,
        query_element: impl Node<N, M, T>,
        enter_points: &[EnterPoint<N, M, T>],
        layer: usize,
        hnsw: &HashMap<usize, EnterPoint<N, M, T>>,
    ) -> &[EnterPoint<N, M, T>] {
        // v ← ep // set of visited elements
        self.visited_elements.clear();
        self.visited_elements.extend_from_slice(enter_points);
        // C ← ep // set of candidates
        self.candidates.clear();
        self.candidates.extend_from_slice(enter_points);
        // W ← ep (Dynamic list of found nearest neighbors).
        self.found_nearest_neighbors.clear();
        self.found_nearest_neighbors.extend_from_slice(enter_points);

        while self.candidates.len() > 0 {
            // Extract closest element.
            // Sort candidate elements.
            // Extract closest neighbor to element from the candidates.
            // Sort the working queue from furthest to nearest.
            self.candidates.sort_by(|a, b| {
                let distance_a_q = self
                    .distance
                    .calculate(query_element.value(), a.get_value());
                let distance_b_q = self
                    .distance
                    .calculate(query_element.value(), b.get_value());
                let x = distance_a_q - distance_b_q;

                if x < T::zero() {
                    Ordering::Greater
                } else if x == T::zero() {
                    Ordering::Equal
                } else {
                    Ordering::Less
                }
            });

            if let Some(nearest) = self.candidates.pop() {
                let furthest = match query_element
                    .furthest(self.found_nearest_neighbors.as_slice(), &self.distance)
                {
                    Some(furthest) => furthest,
                    None => {
                        panic!("Cannot find furthest neighbor. Check neighbors list is not empty.");
                    }
                };

                if query_element.distance(&nearest, &self.distance)
                    > query_element.distance(&furthest, &self.distance)
                {
                    // All elements in W are evaluated
                    break;
                }

                for e in nearest
                    .neighbourhood(layer)
                    .map(|element| hnsw.get(&element.get_index()))
                    .flatten()
                    .cloned()
                {
                    // Update C and W
                    if self.visited_elements.iter().find(|&v| v == &e).is_none() {
                        self.visited_elements.push(e.clone());
                        let furthest = match query_element
                            .furthest(&self.found_nearest_neighbors, &self.distance)
                        {
                            Some(furthest) => furthest,
                            None => {
                                panic!("Cannot find furthest neighbor. Check neighbors list is not empty.");
                            }
                        };

                        if query_element.distance(&e, &self.distance)
                            < query_element.distance(&furthest, &self.distance)
                            || self.found_nearest_neighbors.len()
                                < NUMBER_OF_NEAREST_TO_Q_ELEMENTS_TO_RETURN
                        {
                            self.candidates.push(e.clone());
                            self.found_nearest_neighbors.push(e);
                            if self.found_nearest_neighbors.len()
                                > NUMBER_OF_NEAREST_TO_Q_ELEMENTS_TO_RETURN
                            {
                                // Remove furthest element from W to q
                                // Sort from nearest to furthest.
                                self.found_nearest_neighbors.sort_by(|a, b| {
                                    let distance_a_q = self
                                        .distance
                                        .calculate(query_element.value(), a.get_value());
                                    let distance_b_q = self
                                        .distance
                                        .calculate(query_element.value(), b.get_value());
                                    let x = distance_a_q - distance_b_q;

                                    if x < T::zero() {
                                        Ordering::Less
                                    } else if x == T::zero() {
                                        Ordering::Equal
                                    } else {
                                        Ordering::Greater
                                    }
                                });
                                // Remove furthest element.
                                self.found_nearest_neighbors.pop();
                            }
                        }
                    }
                }
            }
        }

        if self.found_nearest_neighbors.len() < NUMBER_OF_NEAREST_TO_Q_ELEMENTS_TO_RETURN {
            &self.found_nearest_neighbors
        } else {
            &self.found_nearest_neighbors[..NUMBER_OF_NEAREST_TO_Q_ELEMENTS_TO_RETURN]
        }
    }
}

#[cfg(test)]
mod search_layer_test {
    use std::collections::HashMap;

    use super::SearchLayer;
    use crate::query::Element;
    use crate::{enter_point::EnterPoint, query::QueryElement};

    #[test]
    fn test_search_layer() {
        let capacity = 1024;
        let distance = crate::distance::Distance::Euclidean;
        let mut search_layer = SearchLayer::<2, 16, f32>::new(distance, capacity);
        let query_element = QueryElement { value: [2.1, 2.1] };
        let mut enter_point = EnterPoint::new([11.0, 15.0], 3, 3);
        let _ = enter_point.connections.try_push(Element::new(1, 1));
        let _ = enter_point.connections.try_push(Element::new(2, 0));
        let enter_points = vec![enter_point];
        let layer = 1;
        let mut hnsw = HashMap::new();

        let mut ep = EnterPoint::new([1.0, 1.0], 0, 1);
        let _ = ep.connections.try_push(Element::new(1, 1));
        let _ = ep.connections.try_push(Element::new(1, 0));
        hnsw.insert(0, ep);

        let mut ep = EnterPoint::new([10.0, 5.0], 2, 0);
        let _ = ep.connections.try_push(Element::new(1, 0));
        let _ = ep.connections.try_push(Element::new(3, 0));
        hnsw.insert(2, ep);

        let mut ep = EnterPoint::new([11.0, 15.0], 3, 3);
        let _ = ep.connections.try_push(Element::new(1, 1));
        let _ = ep.connections.try_push(Element::new(2, 0));
        hnsw.insert(3, ep);

        let mut ep = EnterPoint::new([2.0, 2.0], 1, 1);
        let _ = ep.connections.try_push(Element::new(0, 1));
        let _ = ep.connections.try_push(Element::new(0, 0));
        let _ = ep.connections.try_push(Element::new(2, 0));
        let _ = ep.connections.try_push(Element::new(3, 1));
        hnsw.insert(1, ep);

        let output = search_layer.search::<1>(query_element, enter_points.as_slice(), layer, &hnsw);

        assert_eq!(output, &[ep]);
    }
}

#[test]
fn search_layer_zero_neighbors_test() {
    use crate::QueryElement;

    const K: usize = 5;
    const DIMENSIONS: usize = 8;
    const M: usize = 32;
    let mut search_layer = SearchLayer::<DIMENSIONS, f32>::new(Distance::Euclidean, 1024);

    let mut connections = ArrayVec::<Element, 128>::new();

    connections.try_push(Element::new(4, 1));
    connections.try_push(Element::new(33, 1));
    connections.try_push(Element::new(9, 1));
    connections.try_push(Element::new(8, 1));
    connections.try_push(Element::new(10, 1));
    connections.try_push(Element::new(3, 1));
    connections.try_push(Element::new(5, 1));
    connections.try_push(Element::new(15, 1));
    connections.try_push(Element::new(34, 0));
    connections.index = 9;
    let enter_points = vec![EnterPoint {
        index: 4,
        layer: 1,
        value: [
            0.0230469704,
            -0.62307477,
            -0.20195961,
            0.0387507677,
            -0.658557177,
            -0.197433949,
            0.0382528305,
            -0.658577204,
        ],
        connections,
    }];

    let layer = 0;
    let mut hnsw = HashMap::new();
    hnsw.insert(
        4,
        EnterPoint {
            index: 4,
            layer: 1,
            value: [
                0.0230469704,
                -0.62307477,
                -0.20195961,
                0.0387507677,
                -0.658557177,
                -0.197433949,
                0.0382528305,
                -0.658577204,
            ],
            connections,
        },
    );

    let mut connections = ArrayVec::new();
    connections.try_push(Element::new(22, 8));
    connections.try_push(Element::new(22, 7));
    connections.try_push(Element::new(22, 6));
    connections.try_push(Element::new(22, 5));
    connections.try_push(Element::new(22, 4));
    connections.try_push(Element::new(22, 3));
    connections.try_push(Element::new(22, 2));
    connections.try_push(Element::new(26, 1));
    connections.try_push(Element::new(15, 1));
    connections.try_push(Element::new(32, 1));
    connections.try_push(Element::new(21, 1));
    connections.try_push(Element::new(22, 1));
    connections.try_push(Element::new(33, 1));
    connections.try_push(Element::new(23, 1));
    connections.try_push(Element::new(29, 1));
    connections.try_push(Element::new(25, 1));
    connections.try_push(Element::new(14, 1));
    connections.try_push(Element::new(4, 1));
    connections.try_push(Element::new(24, 1));
    connections.try_push(Element::new(5, 1));
    connections.try_push(Element::new(27, 1));
    connections.try_push(Element::new(9, 1));
    connections.try_push(Element::new(28, 1));
    connections.try_push(Element::new(3, 1));
    connections.try_push(Element::new(8, 1));
    connections.try_push(Element::new(10, 1));
    connections.try_push(Element::new(30, 1));
    connections.try_push(Element::new(26, 1));
    connections.try_push(Element::new(15, 0));
    connections.try_push(Element::new(32, 0));
    connections.try_push(Element::new(31, 0));
    connections.try_push(Element::new(21, 0));
    connections.index = 32;
    hnsw.insert(
        34,
        EnterPoint {
            index: 34,
            layer: 8,
            value: [
                0.0210614204,
                -0.624067008,
                -0.187818527,
                0.0429267883,
                -0.658624887,
                -0.18597126,
                0.0424735546,
                -0.658663272,
            ],
            connections,
        },
    );

    hnsw.insert(
        15,
        EnterPoint {
            index: 15,
            layer: 0,
            value: [0_f32; 8],
            connections: ArrayVec::new(),
        },
    );

    hnsw.insert(
        32,
        EnterPoint {
            index: 32,
            layer: 0,
            value: [0_f32; 8],
            connections: ArrayVec::new(),
        },
    );

    hnsw.insert(
        31,
        EnterPoint {
            index: 31,
            layer: 0,
            value: [0_f32; 8],
            connections: ArrayVec::new(),
        },
    );

    hnsw.insert(
        21,
        EnterPoint {
            index: 21,
            layer: 0,
            value: [0_f32; 8],
            connections: ArrayVec::new(),
        },
    );

    let query_element = QueryElement {
        value: [
            -0.00517272949,
            -0.663250923,
            -0.220837593,
            0.010566473,
            -0.698032141,
            -0.212332249,
            0.010065794,
            -0.698020935,
        ],
    };
    let eps = search_layer.search::<K>(query_element, &enter_points, layer, &mut hnsw);

    assert_eq!(eps.len(), K);
}
