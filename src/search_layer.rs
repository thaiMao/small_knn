use crate::distance::Distance;
use crate::enter_point::EnterPoint;
use crate::query::QueryElement;
use num::Float;
use std::{cmp::Ordering, fmt::Debug, iter::Sum};

#[derive(Clone, Debug)]
pub struct SearchLayer<const N: usize, const M_MAX: usize, T>
where
    T: Float,
{
    visited_elements: Vec<EnterPoint<N, M_MAX, T>>,
    candidates: Vec<EnterPoint<N, M_MAX, T>>,
    found_nearest_neighbors: Vec<EnterPoint<N, M_MAX, T>>,
    distance: Distance,
}

impl<'a, const N: usize, const M_MAX: usize, T> SearchLayer<N, M_MAX, T>
where
    T: Float + Sum + Debug,
{
    fn new(distance: Distance, capacity: usize) -> Self {
        Self {
            visited_elements: Vec::with_capacity(capacity),
            candidates: Vec::with_capacity(capacity),
            found_nearest_neighbors: Vec::with_capacity(capacity),
            distance,
        }
    }
    fn search<const NUMBER_OF_NEAREST_TO_Q_ELEMENTS_TO_RETURN: usize>(
        &mut self,
        query_element: &QueryElement<N, M_MAX, T>,
        enter_points: &[EnterPoint<N, M_MAX, T>],
        layer: usize,
    ) -> &[EnterPoint<N, M_MAX, T>] {
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
                let distance_a_q = self.distance.calculate(query_element.value(), a.value);
                let distance_b_q = self.distance.calculate(query_element.value(), b.value);
                let x = distance_a_q - distance_b_q;

                if x < T::zero() {
                    Ordering::Greater
                } else if x == T::zero() {
                    Ordering::Equal
                } else {
                    Ordering::Less
                }
            });
            let nearest = self.candidates.pop().unwrap();

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
                .map(|element| enter_points.get(element.index).unwrap())
                .cloned()
            {
                // Update C and W
                if self.visited_elements.iter().find(|&v| v == &e).is_none() {
                    self.visited_elements.push(e.clone());
                    let furthest =
                        query_element.furthest(&self.found_nearest_neighbors, &self.distance);

                    if query_element.distance(&e, &self.distance)
                        > query_element.distance(&furthest, &self.distance)
                    {
                        self.candidates.push(e.clone());
                        self.found_nearest_neighbors.push(e);
                        if self.found_nearest_neighbors.len()
                            > NUMBER_OF_NEAREST_TO_Q_ELEMENTS_TO_RETURN
                        {
                            // Remove furthest element from W to q
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
