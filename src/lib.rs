use crate::distance::Distance;
use crate::enter_point::EnterPoint;
use crate::node::Node;
use crate::query::{Element, QueryElement};
use num::{cast, Float, Num};
use rand::prelude::*;
use search_layer::SearchLayer;
use std::collections::HashMap;
use std::{cmp::Ordering, fmt::Debug, iter::Sum, marker::PhantomData, ops::Deref};

mod array_vec;
pub mod distance;
mod enter_point;
mod node;
mod query;
mod search_layer;
pub struct Setup;
pub struct Ready;

const DEFAULT_CAPACITY: usize = 128;
const DEFAULT_NORMALIZATION_FACTOR: f32 = 2.0;
const DEFAULT_NEIGHBOR_SELECTION_ALGORTHIM: NeighborSelectionAlgorithm =
    NeighborSelectionAlgorithm::Heuristic;
const DEFAULT_EXTEND_CANDIDATES: bool = true;
const DEFAULT_KEEP_PRUNED_CONNECTIONS: bool = false;
const DEFAULT_DISTANCE: Distance = Distance::Euclidean;

/// A reasonable range of M is from 5 to 48.
const M: usize = 16;
/// Simulations suggest that 2 * M is a good choice for M_MAX_ZERO
/// The maximum connections that an element can have for the ground layer.
const M_MAX_ZERO: usize = M * 2;
const DEFAULT_MAX_CONNECTIONS: usize = 16;
const EF_CONSTRUCTION: usize = 32;
/// * `N` - Number of dimensions.
#[derive(Clone, Debug)]
pub struct HNSW<const N: usize, T, Stage = Setup>
where
    T: Float + Sum + Debug,
{
    stage: PhantomData<Stage>,
    enter_points: Vec<EnterPoint<N, M, T>>,
    rng: rand::rngs::ThreadRng,
    search_layer: SearchLayer<N, M, T>,
    found_nearest_neighbors: Vec<EnterPoint<N, M, T>>,
    working_queue: Vec<EnterPoint<N, M, T>>,
    neighbors: Vec<EnterPoint<N, M, T>>,
    discarded_candidates: Vec<EnterPoint<N, M, T>>,
    normalization_factor: T,
    neighbor_selection_algorithm: NeighborSelectionAlgorithm,
    extend_candidates: bool,
    keep_pruned_connections: bool,
    distance: Distance,
    capacity: usize,
    nearest_elements: Vec<EnterPoint<N, M, T>>,
    enter_point: Option<EnterPoint<N, M, T>>,
    econn: Vec<EnterPoint<N, M, T>>,
    max_connections: usize,
    hnsw: HashMap<usize, EnterPoint<N, M, T>>,
}

impl<const N: usize, T> Default for HNSW<N, T, Setup>
where
    T: Float + Sum + Debug,
{
    fn default() -> Self {
        let rng = rand::thread_rng();

        Self {
            stage: PhantomData::<Setup>,
            enter_points: Vec::with_capacity(DEFAULT_CAPACITY),
            rng,
            search_layer: SearchLayer::new(DEFAULT_DISTANCE, DEFAULT_CAPACITY),
            found_nearest_neighbors: Vec::with_capacity(DEFAULT_CAPACITY),
            working_queue: Vec::with_capacity(DEFAULT_CAPACITY),
            neighbors: Vec::with_capacity(DEFAULT_CAPACITY),
            discarded_candidates: Vec::with_capacity(DEFAULT_CAPACITY),
            normalization_factor: cast::<f32, T>(DEFAULT_NORMALIZATION_FACTOR).unwrap(),
            neighbor_selection_algorithm: DEFAULT_NEIGHBOR_SELECTION_ALGORTHIM,
            extend_candidates: DEFAULT_EXTEND_CANDIDATES,
            keep_pruned_connections: DEFAULT_KEEP_PRUNED_CONNECTIONS,
            distance: DEFAULT_DISTANCE,
            capacity: DEFAULT_CAPACITY,
            nearest_elements: Vec::with_capacity(DEFAULT_CAPACITY),
            enter_point: None,
            econn: Vec::with_capacity(DEFAULT_CAPACITY),
            max_connections: DEFAULT_MAX_CONNECTIONS,
            hnsw: HashMap::with_capacity(DEFAULT_CAPACITY),
        }
    }
}

impl<const N: usize, T> HNSW<N, T, Setup>
where
    T: Float + Sum + Debug,
{
    pub fn new(distance: Distance) -> Self {
        let rng = rand::thread_rng();

        Self {
            stage: PhantomData::<Setup>,
            enter_points: Vec::with_capacity(DEFAULT_CAPACITY),
            rng,
            search_layer: SearchLayer::new(DEFAULT_DISTANCE, DEFAULT_CAPACITY),
            found_nearest_neighbors: Vec::with_capacity(DEFAULT_CAPACITY),
            working_queue: Vec::with_capacity(DEFAULT_CAPACITY),
            neighbors: Vec::with_capacity(DEFAULT_CAPACITY),
            discarded_candidates: Vec::with_capacity(DEFAULT_CAPACITY),
            normalization_factor: cast::<f32, T>(DEFAULT_NORMALIZATION_FACTOR).unwrap(),
            neighbor_selection_algorithm: DEFAULT_NEIGHBOR_SELECTION_ALGORTHIM,
            extend_candidates: DEFAULT_EXTEND_CANDIDATES,
            keep_pruned_connections: DEFAULT_KEEP_PRUNED_CONNECTIONS,
            distance,
            capacity: DEFAULT_CAPACITY,
            nearest_elements: Vec::with_capacity(DEFAULT_CAPACITY),
            enter_point: None,
            econn: Vec::with_capacity(DEFAULT_CAPACITY),
            max_connections: DEFAULT_MAX_CONNECTIONS,
            hnsw: HashMap::with_capacity(DEFAULT_CAPACITY),
        }
    }

    pub fn set_capacity(&mut self, capacity: usize) -> &mut Self {
        self.capacity = capacity;
        self
    }

    pub fn set_normalization_factor(&mut self, normalization_factor: T) -> &mut Self {
        self.normalization_factor = normalization_factor;
        self
    }

    pub fn set_neighbor_selection_algorithm(
        &mut self,
        neighbor_selection_algorithm: NeighborSelectionAlgorithm,
    ) -> &mut Self {
        self.neighbor_selection_algorithm = neighbor_selection_algorithm;
        self
    }

    pub fn set_extend_candidates(&mut self, extend_candidates: bool) -> &mut Self {
        self.extend_candidates = extend_candidates;
        self
    }

    pub fn set_keep_pruned_connections(&mut self, keep_pruned_connections: bool) -> &mut Self {
        self.keep_pruned_connections = keep_pruned_connections;
        self
    }

    pub fn set_distance(&mut self, distance: Distance) -> &mut Self {
        self.distance = distance;
        self
    }

    /// Set the maximum number of connections for each element per layer.
    pub fn set_max_connections(&mut self, max_connections: usize) -> &mut Self {
        self.max_connections = max_connections;
        self
    }

    pub fn build(&mut self) -> HNSW<N, T, Ready> {
        let enter_points = Vec::with_capacity(self.capacity);
        let found_nearest_neighbors = Vec::with_capacity(self.capacity);
        let working_queue = Vec::with_capacity(self.capacity);
        let neighbors = Vec::with_capacity(self.capacity);
        let discarded_candidates = Vec::with_capacity(self.capacity);
        let nearest_elements = Vec::with_capacity(self.capacity);
        let econn = Vec::with_capacity(self.capacity);
        let hnsw = HashMap::with_capacity(self.capacity);

        let rng = rand::thread_rng();

        HNSW::<N, T, Ready> {
            stage: PhantomData::<Ready>,
            enter_points,
            rng,
            search_layer: SearchLayer::new(DEFAULT_DISTANCE, DEFAULT_CAPACITY),
            found_nearest_neighbors,
            working_queue,
            neighbors,
            discarded_candidates,
            normalization_factor: cast::<f32, T>(DEFAULT_NORMALIZATION_FACTOR).unwrap(),
            neighbor_selection_algorithm: NeighborSelectionAlgorithm::Heuristic,
            extend_candidates: true,
            keep_pruned_connections: false,
            distance: self.distance,
            capacity: self.capacity,
            nearest_elements,
            enter_point: None,
            econn,
            max_connections: self.max_connections,
            hnsw,
        }
    }
}

impl<const N: usize, T> HNSW<N, T, Ready>
where
    T: Float + Sum + PartialEq + Debug + PartialOrd,
{
    /// Insert elements into a graph structure.
    /// * `M` - Number of established connections.
    /// * `M_MAX` - Maximum number of connections for each element per layer.
    pub fn insert<Q>(&mut self, index: usize, value: Q)
    where
        Q: Deref + Deref<Target = [T; N]>,
    {
        self.found_nearest_neighbors.clear();
        let query_element = QueryElement::new(*value);

        // Get enter point for HNSW.
        self.enter_points.clear();
        if let Some(enter_point) = self.enter_point {
            self.enter_points.push(enter_point);
        }
        // Top layer for HNSW.
        let top_layer_level = match self.enter_point {
            Some(ep) => ep.get_layer(),
            None => 0,
        };

        // New element's level.
        let random_number: T = cast::<f32, T>(self.rng.gen()).unwrap();
        let new_element_level = (-random_number.ln() * self.normalization_factor).floor();
        let new_element_level = cast::<T, usize>(new_element_level).unwrap();

        // Add bidirectional connections from neighbors to q at layer lc
        let mut ep = EnterPoint::new(query_element.value, index, new_element_level);

        if self.enter_points.len() > 0 {
            for layer in top_layer_level..new_element_level + 1 {
                let nearest_element =
                    self.search_layer
                        .search::<1>(query_element, &self.enter_points[0..1], layer);

                self.found_nearest_neighbors.clear();
                self.found_nearest_neighbors
                    .extend_from_slice(&nearest_element);

                let nearest_element_to_query =
                    query_element.nearest(self.found_nearest_neighbors.as_slice(), &self.distance);
                self.enter_points.clear();
                self.enter_points.push(nearest_element_to_query);
            }

            for layer in top_layer_level.min(new_element_level)..0 {
                let found_nearest_neighbors = self.search_layer.search::<EF_CONSTRUCTION>(
                    query_element,
                    self.enter_points.as_slice(),
                    layer,
                );

                self.found_nearest_neighbors.clear();
                self.found_nearest_neighbors
                    .extend_from_slice(&found_nearest_neighbors);

                let mut neighbors = match self.neighbor_selection_algorithm {
                    NeighborSelectionAlgorithm::Simple => {
                        self.select_neighbors_simple::<M>(query_element, Candidate::Neighbors)
                    }
                    NeighborSelectionAlgorithm::Heuristic => self.select_neighbors_heuristic::<M>(
                        query_element,
                        Candidate::Neighbors,
                        layer,
                        self.extend_candidates,
                        self.keep_pruned_connections,
                    ),
                };

                for neighbor in neighbors.iter_mut() {
                    let overflow = neighbor
                        .connections
                        .try_push(Element::new(index, new_element_level));

                    let overflow = ep
                        .connections
                        .try_push(Element::new(neighbor.get_index(), neighbor.get_layer()));
                }

                for mut e in neighbors.iter().cloned() {
                    self.econn.clear();
                    e.neighbourhood(layer)
                        .map(|element| self.hnsw.get(&element.get_index()).unwrap())
                        .cloned()
                        .for_each(|enter_point| {
                            self.econn.push(enter_point);
                        });

                    if layer == 0 && e.number_of_connections(layer) > M_MAX_ZERO
                        || e.number_of_connections(layer) > self.max_connections
                    {
                        let new_econn = match self.neighbor_selection_algorithm {
                            NeighborSelectionAlgorithm::Simple => self
                                .select_neighbors_simple::<M>(
                                    e.clone(),
                                    Candidate::ElementConnections,
                                ),
                            NeighborSelectionAlgorithm::Heuristic => self
                                .select_neighbors_heuristic::<M>(
                                    e.clone(),
                                    Candidate::ElementConnections,
                                    layer,
                                    self.extend_candidates,
                                    self.keep_pruned_connections,
                                ),
                        };

                        e.connections.clear();
                        // Set neighbourhood(e) at layer lc to eNewConn
                        for element in new_econn.iter().map(|enter_point| {
                            Element::new(enter_point.get_index(), enter_point.get_layer())
                        }) {
                            e.connections.try_push(element);
                        }
                    }
                }
                // ep ← W
                self.enter_points
                    .extend_from_slice(&self.found_nearest_neighbors);
            }
        }

        if self.enter_point.is_none() || new_element_level > top_layer_level {
            // Set enter point for hnsw to query element.
            self.enter_point = Some(ep);
        }

        // Update HNSW inserting element q.
        self.hnsw.insert(index, ep);

        self.econn.clear();
        self.neighbors.clear();
        self.found_nearest_neighbors.clear();
    }

    pub fn clear(&mut self) {
        self.enter_points.clear();
        self.found_nearest_neighbors.clear();
        self.working_queue.clear();
        self.neighbors.clear();
        self.discarded_candidates.clear();
        self.nearest_elements.clear();
    }

    // Algorithm 5 - Search KNN
    /// * `K` -Number of nearest neighbors to return.
    pub fn search_neighbors<const K: usize, Q>(&mut self, value: Q) -> Result<[usize; K], ()>
    where
        Q: Deref + Deref<Target = [T; N]>,
        T: Num + PartialOrd,
    {
        let mut enter_point = match self.enter_point {
            Some(enter_point) => enter_point,
            None => return Err(()),
        };
        let query_element = QueryElement::new(*value);
        self.nearest_elements.clear();
        let level = enter_point.get_layer();

        for layer in level..=1 {
            let closest_neighbor =
                self.search_layer
                    .search::<1>(query_element, &[enter_point.clone()], layer);
            self.nearest_elements.extend_from_slice(&closest_neighbor);
            enter_point = query_element.nearest(&self.nearest_elements, &self.distance);
        }

        let closest_neighbors = self.search_layer.search::<EF_CONSTRUCTION>(
            query_element,
            self.enter_points.as_slice(),
            0,
        );
        self.nearest_elements.extend_from_slice(&closest_neighbors);

        // Return nearest elements
        // Sort elements by nearest to furthest order from query element.
        self.nearest_elements.sort_by(|a, b| {
            let distance_a_q = self.distance.calculate(*value, a.get_value());
            let distance_b_q = self.distance.calculate(*value, b.get_value());
            let x = distance_a_q - distance_b_q;

            if x < T::zero() {
                Ordering::Less
            } else if x == T::zero() {
                Ordering::Equal
            } else {
                Ordering::Greater
            }
        });

        if self.nearest_elements.len() < K {
            panic!("Not enough elements inserted");
        }

        let mut output = [0; K];

        for i in 0..K {
            output[i] = self.nearest_elements[i].get_index();
        }

        Ok(output)
    }

    // Algorithm 3 - Select neighbours simple
    fn select_neighbors_simple<const NUMBER_OF_NEIGHBOURS_TO_RETURN: usize>(
        &mut self,
        base_element: impl Node<N, M, T>,
        candidate: Candidate,
    ) -> [EnterPoint<N, M, T>; NUMBER_OF_NEIGHBOURS_TO_RETURN] {
        self.neighbors.clear();

        match candidate {
            Candidate::Neighbors => {
                // Return nearest elements
                // Sort elements by nearest to furthest order from query element.
                self.found_nearest_neighbors.sort_by(|a, b| {
                    let distance_a_q = self.distance.calculate(base_element.value(), a.value());
                    let distance_b_q = self.distance.calculate(base_element.value(), b.value());
                    let x = distance_a_q - distance_b_q;

                    if x < T::zero() {
                        Ordering::Less
                    } else if x == T::zero() {
                        Ordering::Equal
                    } else {
                        Ordering::Greater
                    }
                });

                if self.found_nearest_neighbors.len() < NUMBER_OF_NEIGHBOURS_TO_RETURN {
                    panic!("Not enough elements inserted");
                }

                for i in 0..NUMBER_OF_NEIGHBOURS_TO_RETURN {
                    self.neighbors.push(self.found_nearest_neighbors[i]);
                }

                self.neighbors.clone().try_into().unwrap()
            }
            Candidate::ElementConnections => {
                // Return nearest elements
                // Sort elements by nearest to furthest order from query element.
                self.econn.sort_by(|a, b| {
                    let distance_a_q = self.distance.calculate(base_element.value(), a.value());
                    let distance_b_q = self.distance.calculate(base_element.value(), b.value());
                    let x = distance_a_q - distance_b_q;

                    if x < T::zero() {
                        Ordering::Less
                    } else if x == T::zero() {
                        Ordering::Equal
                    } else {
                        Ordering::Greater
                    }
                });

                if self.econn.len() < NUMBER_OF_NEIGHBOURS_TO_RETURN {
                    panic!("Not enough elements inserted");
                }

                for i in 0..NUMBER_OF_NEIGHBOURS_TO_RETURN {
                    self.neighbors.push(self.econn[i]);
                }

                self.neighbors.clone().try_into().unwrap()
            }
        }
    }
    // Algorithm 4 - Select neighbours heuristic
    fn select_neighbors_heuristic<const NUMBER_OF_NEIGHBOURS_TO_RETURN: usize>(
        &mut self,
        base_element: impl Node<N, M, T>,
        candidate: Candidate,
        layer: usize,
        extend_candidates: bool,
        keep_pruned_connections: bool,
    ) -> [EnterPoint<N, M, T>; NUMBER_OF_NEIGHBOURS_TO_RETURN] {
        let candidate_elements = match candidate {
            Candidate::ElementConnections => self.econn.as_slice(),
            Candidate::Neighbors => self.found_nearest_neighbors.as_slice(),
        };
        self.neighbors.clear();
        self.working_queue.clear();
        self.working_queue.extend_from_slice(candidate_elements);

        if extend_candidates {
            for e in self.found_nearest_neighbors.iter() {
                for e_adjacent in e
                    .neighbourhood(layer)
                    .map(|element| self.hnsw.get(&element.get_index()).unwrap())
                {
                    if self
                        .working_queue
                        .iter()
                        .find(|w| w.get_index() == e_adjacent.get_index())
                        .is_none()
                    {
                        self.working_queue.push(e_adjacent.clone());
                    }
                }
            }
        }

        self.discarded_candidates.clear();

        while self.working_queue.len() > 0 && self.neighbors.len() < NUMBER_OF_NEIGHBOURS_TO_RETURN
        {
            // Extract closest neighbor to element from the queue.
            // Sort the working queue from furthest to nearest.
            self.working_queue.sort_by(|a, b| {
                let distance_a_q = self.distance.calculate(base_element.value(), a.value());
                let distance_b_q = self.distance.calculate(base_element.value(), b.value());
                let x = distance_a_q - distance_b_q;

                if x < T::zero() {
                    Ordering::Greater
                } else if x == T::zero() {
                    Ordering::Equal
                } else {
                    Ordering::Less
                }
            });
            let nearest = self.working_queue.pop().unwrap();

            if base_element.distance(&nearest, &self.distance)
                < self
                    .neighbors
                    .iter()
                    .map(|element| self.hnsw.get(&element.get_index()).unwrap())
                    .map(|n| base_element.distance(n, &self.distance))
                    .reduce(T::min)
                    .unwrap_or(T::max_value())
            {
                self.neighbors.push(nearest);
            } else {
                self.discarded_candidates.push(nearest);
            }
        }

        if keep_pruned_connections {
            while self.discarded_candidates.len() > 0
                && self.neighbors.len() < NUMBER_OF_NEIGHBOURS_TO_RETURN
            {
                // Extract closest neighbor to element from the queue.
                // Sort the working queue from furthest to nearest.
                self.working_queue.sort_by(|a, b| {
                    let distance_a_q = self.distance.calculate(base_element.value(), a.value());
                    let distance_b_q = self.distance.calculate(base_element.value(), b.value());
                    let x = distance_a_q - distance_b_q;

                    if x < T::zero() {
                        Ordering::Greater
                    } else if x == T::zero() {
                        Ordering::Equal
                    } else {
                        Ordering::Less
                    }
                });
                let nearest = self.working_queue.pop().unwrap();

                self.neighbors.push(nearest);
            }
        }
        self.neighbors.clone().try_into().unwrap()
    }
    // Algorithm 2 - Search layer
}

#[derive(Clone, Debug)]
pub enum NeighborSelectionAlgorithm {
    Simple,
    Heuristic,
}

#[cfg(test)]
mod knn_tests {
    use super::HNSW;
    use crate::distance::Distance;
    use std::ops::Deref;

    const DIMENSIONS: usize = 2;
    const K: usize = 2;

    #[test]
    fn test_search() {
        struct MyNode<const N: usize, T>
        where
            T: Clone + Copy,
        {
            value: [T; N],
        }

        impl<const N: usize, T> Deref for MyNode<N, T>
        where
            T: Clone + Copy,
        {
            type Target = [T; N];
            fn deref(&self) -> &Self::Target {
                &self.value
            }
        }

        let mut knn = HNSW::<DIMENSIONS, f32>::default()
            .set_distance(Distance::Euclidean)
            .build();

        knn.insert(0, MyNode { value: [1.0, 1.0] });
        knn.insert(1, MyNode { value: [2.0, 2.0] });
        knn.insert(2, MyNode { value: [10.0, 5.0] });
        knn.insert(
            4,
            MyNode {
                value: [11.0, 15.0],
            },
        );
        let neighbors = knn.search_neighbors::<K, _>(MyNode { value: [2.1, 2.1] });
        assert_eq!(neighbors.unwrap(), [1, 0]);

        knn.clear();
    }
}

#[derive(Clone, Copy)]
enum Candidate {
    ElementConnections,
    Neighbors,
}
