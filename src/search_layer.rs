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
                let furthest =
                    query_element.furthest(self.found_nearest_neighbors.as_slice(), &self.distance);

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
                        let furthest =
                            query_element.furthest(&self.found_nearest_neighbors, &self.distance);

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
