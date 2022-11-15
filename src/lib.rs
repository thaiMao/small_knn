use crate::distance::Distance;
use crate::enter_point::EnterPoint;
use crate::node::Node;
use crate::query::{Element, QueryElement};
use error::KNNError;
use num::{cast, Float, Num};
use rand::{distributions::Standard, prelude::*};
use search_layer::SearchLayer;
use std::collections::HashMap;
use std::{cmp::Ordering, fmt::Debug, iter::Sum, marker::PhantomData, ops::Deref};

mod array_vec;
pub mod distance;
mod enter_point;
mod error;
mod node;
mod query;
mod search_layer;

pub struct Setup;
pub struct Ready;

const DEFAULT_CAPACITY: usize = 128;
// A simple choice for the optimal normalization factor is 1 / ln(M).
const DEFAULT_NORMALIZATION_FACTOR: f32 = 3.0;
const DEFAULT_NEIGHBOR_SELECTION_ALGORTHIM: NeighborSelectionAlgorithm =
    NeighborSelectionAlgorithm::Heuristic;
const DEFAULT_EXTEND_CANDIDATES: bool = true;
const DEFAULT_KEEP_PRUNED_CONNECTIONS: bool = true;
const DEFAULT_DISTANCE: Distance = Distance::Euclidean;

/// A reasonable range of M is from 5 to 48.
const M: usize = 32;
/// Simulations suggest that 2 * M is a good choice for M_MAX_ZERO
/// The maximum connections that an element can have for the ground layer.
const M_MAX_ZERO: usize = M * 2;
// TODO - Set proper M_MAX
const M_MAX: usize = M_MAX_ZERO;
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
    neighbors: Vec<Option<usize>>,
    discarded_candidates: Vec<EnterPoint<N, M, T>>,
    normalization_factor: T,
    neighbor_selection_algorithm: NeighborSelectionAlgorithm,
    extend_candidates: bool,
    keep_pruned_connections: bool,
    distance: Distance,
    capacity: usize,
    nearest_elements: Vec<EnterPoint<N, M, T>>,
    enter_point_index: Option<usize>,
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
            enter_point_index: None,
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
            enter_point_index: None,
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
            keep_pruned_connections: self.keep_pruned_connections,
            distance: self.distance,
            capacity: self.capacity,
            nearest_elements,
            enter_point_index: None,
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
    pub fn insert<Q>(&mut self, index: usize, value: Q) -> Result<(), KNNError>
    where
        Q: Deref + Deref<Target = [T; N]>,
        Standard: Distribution<T>,
    {
        self.found_nearest_neighbors.clear();
        let query_element = QueryElement::new(*value);

        // Get enter point for HNSW.
        self.enter_points.clear();
        let enter_point = match self.enter_point_index {
            Some(index) => self.hnsw.get(&index),
            None => None,
        };
        // Top layer for HNSW.
        let top_layer_level = match enter_point {
            Some(ep) => ep.get_layer(),
            None => 0,
        };

        let random_number: T = self.rng.gen::<T>();
        // New element's level.
        let new_element_level = (-random_number.ln() * self.normalization_factor).floor();
        let new_element_level = cast::<T, usize>(new_element_level).unwrap();
        let mut ep = EnterPoint::new(query_element.value, index, new_element_level);

        if let Some(enter_point) = enter_point {
            let mut enter_point = enter_point.clone();
            let r = if top_layer_level >= new_element_level + 1 {
                new_element_level + 1..=top_layer_level
            } else {
                top_layer_level..=new_element_level + 1
            };
            for layer in r.rev() {
                //debug_assert!(self.enter_points.len() == 1);
                let nearest_element = self.search_layer.search::<1>(
                    query_element,
                    &[enter_point.clone()],
                    layer,
                    &self.hnsw,
                );

                self.found_nearest_neighbors.clear();
                self.found_nearest_neighbors
                    .extend_from_slice(&nearest_element);

                let nearest_element_to_query =
                    query_element.nearest(self.found_nearest_neighbors.as_slice(), &self.distance);

                if let Some(nearest_element_to_query) = nearest_element_to_query {
                    enter_point = nearest_element_to_query;
                    self.enter_points.clear();
                    self.enter_points.push(nearest_element_to_query);
                }
            }

            for layer in (0..=top_layer_level.min(new_element_level)).rev() {
                let found_nearest_neighbors = self.search_layer.search::<EF_CONSTRUCTION>(
                    query_element,
                    self.enter_points.as_slice(),
                    layer,
                    &self.hnsw,
                );

                self.found_nearest_neighbors.clear();
                self.found_nearest_neighbors
                    .extend_from_slice(&found_nearest_neighbors);

                if index == 2 {
                    println!("W: {:#?}", self.found_nearest_neighbors);
                    println!("layer: {:#?}", layer);
                    println!("query: {:#?}", query_element);
                    println!("enter_points: {:#?}", self.enter_points);
                }

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

                if index == 2 {
                    println!("neighbors: {:#?}", neighbors);
                    println!("extend_candidates: {:#?}", self.extend_candidates);
                    println!(
                        "keep_pruned_connections: {:#?}",
                        self.keep_pruned_connections
                    );
                }

                // Add bidirectional connections from neighbors to q at layer lc
                for k in neighbors.iter().flatten() {
                    let neighbor = match self.hnsw.get_mut(k) {
                        Some(n) => n,
                        None => {
                            return Err(KNNError::Internal);
                        }
                    };

                    let overflow = neighbor.connections.try_push(Element::new(index, layer));

                    let overflow = ep
                        .connections
                        .try_push(Element::new(neighbor.get_index(), layer));
                }

                // Update HNSW inserting element q.
                self.hnsw.insert(index, ep);

                for k in neighbors.iter().flatten() {
                    self.econn.clear();

                    let e = match self.hnsw.get(k) {
                        Some(e) => e.clone(),
                        None => return Err(KNNError::Internal),
                    };

                    // Popuate self.econn which is used in select_neighbors methods
                    e.neighbourhood(layer)
                        .map(|element| {
                            let key = &element.get_index();

                            self.hnsw.get(&key)
                        })
                        .flatten()
                        .for_each(|enter_point| {
                            self.econn.push(enter_point.clone());
                        });

                    // TODO Split M_MAX_ZERO and M_MAX
                    if layer == 0 && e.number_of_connections(layer) > M_MAX_ZERO {
                        let new_econn = match self.neighbor_selection_algorithm {
                            NeighborSelectionAlgorithm::Simple => self
                                .select_neighbors_simple::<M_MAX_ZERO>(
                                    e.clone(),
                                    Candidate::ElementConnections,
                                ),
                            NeighborSelectionAlgorithm::Heuristic => self
                                .select_neighbors_heuristic::<M_MAX_ZERO>(
                                    e.clone(),
                                    Candidate::ElementConnections,
                                    layer,
                                    self.extend_candidates,
                                    self.keep_pruned_connections,
                                ),
                        };

                        // Convert to array of elements.
                        let mut new_econn_elements = [None; M_MAX_ZERO];

                        for (i, e) in new_econn
                            .iter()
                            .flatten()
                            .zip(new_econn_elements.iter_mut())
                        {
                            let ep = match self.hnsw.get(i) {
                                Some(ep) => ep.clone(),
                                None => return Err(KNNError::Internal),
                            };

                            *e = Some(Element::new(ep.get_index(), layer));
                        }

                        // Get enter point from hashmap, clear and replace connections with new_econn.
                        let e = match self.hnsw.get_mut(k) {
                            Some(e) => e,
                            None => {
                                return Err(KNNError::Internal);
                            }
                        };
                        e.connections.clear();
                        // Set neighbourhood(e) at layer lc to eNewConn
                        for element in new_econn_elements.iter().flatten().cloned() {
                            let overflow = e.connections.try_push(element);
                        }
                    }

                    if layer > 0 && e.number_of_connections(layer) > M_MAX {
                        let new_econn = match self.neighbor_selection_algorithm {
                            NeighborSelectionAlgorithm::Simple => self
                                .select_neighbors_simple::<M_MAX>(
                                    e.clone(),
                                    Candidate::ElementConnections,
                                ),
                            NeighborSelectionAlgorithm::Heuristic => self
                                .select_neighbors_heuristic::<M_MAX>(
                                    e.clone(),
                                    Candidate::ElementConnections,
                                    layer,
                                    self.extend_candidates,
                                    self.keep_pruned_connections,
                                ),
                        };

                        // Convert to array of elements.
                        let mut new_econn_elements = [None; M_MAX];

                        for (index, e) in new_econn
                            .iter()
                            .flatten()
                            .zip(new_econn_elements.iter_mut())
                        {
                            let ep = match self.hnsw.get(index) {
                                Some(ep) => ep.clone(),
                                None => return Err(KNNError::Internal),
                            };

                            *e = Some(Element::new(ep.get_index(), layer));
                        }

                        // Get enter point from hashmap, clear and replace connections with new_econn.
                        let e = match self.hnsw.get_mut(k) {
                            Some(e) => e,
                            None => return Err(KNNError::Internal),
                        };
                        e.connections.clear();
                        // Set neighbourhood(e) at layer lc to eNewConn
                        for element in new_econn_elements.iter().flatten().cloned() {
                            let overflow = e.connections.try_push(element);
                        }
                    }
                }

                // ep ← W
                self.enter_points.clear();
                self.enter_points
                    .extend_from_slice(&self.found_nearest_neighbors);
            }
        } else {
            // Update HNSW inserting element q.
            self.hnsw.insert(index, ep);
        }

        if self.enter_point_index.is_none() || new_element_level > top_layer_level {
            // Set enter point for hnsw to query element.
            self.enter_point_index = Some(index);
        }

        self.found_nearest_neighbors.clear();
        self.working_queue.clear();
        self.neighbors.clear();
        self.discarded_candidates.clear();
        self.nearest_elements.clear();

        self.enter_points.clear();
        self.search_layer.clear();
        self.econn.clear();

        Ok(())
    }

    pub fn clear(&mut self) {
        self.enter_points.clear();
        self.found_nearest_neighbors.clear();
        self.working_queue.clear();
        self.neighbors.clear();
        self.discarded_candidates.clear();
        self.nearest_elements.clear();
        self.search_layer.clear();
        self.econn.clear();

        self.hnsw.clear();
        self.enter_point_index = None;
    }

    // Algorithm 5 - Search KNN
    /// * `K` -Number of nearest neighbors to return.
    pub fn search_neighbors<const K: usize, Q>(&mut self, value: Q) -> Result<[usize; K], KNNError>
    where
        Q: Deref + Deref<Target = [T; N]>,
        T: Num + PartialOrd,
    {
        // Get enter point for hnsw.
        let mut enter_point = match self.enter_point_index {
            Some(index) => match self.hnsw.get(&index) {
                Some(enter_point) => enter_point.clone(),
                None => return Err(KNNError::Internal),
            },
            None => return Err(KNNError::Internal),
        };
        self.enter_points.clear();
        // Initialize enter points to handle case where top level is zero (and does not enter
        // for loop).
        self.enter_points.extend_from_slice(&[enter_point]);

        let query_element = QueryElement::new(*value);
        // Top layer for hnsw.
        let level = enter_point.get_layer();

        for layer in (1..=level).rev() {
            let closest_neighbor =
                self.search_layer
                    .search::<1>(query_element, &[enter_point], layer, &self.hnsw);
            self.nearest_elements.clear();
            self.nearest_elements.extend_from_slice(&closest_neighbor);

            // Get nearest element from W to q.
            let nearest = query_element.nearest(&self.nearest_elements, &self.distance);

            if let Some(nearest) = nearest {
                enter_point = nearest;
                self.enter_points.clear();
                self.enter_points.extend_from_slice(&[nearest]);
            }
        }

        let closest_neighbors = self.search_layer.search::<EF_CONSTRUCTION>(
            query_element,
            self.enter_points.as_slice(),
            0,
            &self.hnsw,
        );
        self.nearest_elements.clear();
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
            return Err(KNNError::InsufficientInsertions {
                expected: K,
                found: self.nearest_elements.len(),
            });
        }

        let mut output = [0; K];

        for i in 0..K {
            output[i] = self.nearest_elements[i].get_index();
        }
        // Need to clear nearest elements so elements can be inserted after
        // search_neighbors is called and search_neighbors return the
        // correct nearest neighbors.
        self.nearest_elements.clear();
        Ok(output)
    }

    // Algorithm 3 - Select neighbours simple
    fn select_neighbors_simple<const NUMBER_OF_NEIGHBORS_TO_RETURN: usize>(
        &mut self,
        base_element: impl Node<N, M, T>,
        candidate: Candidate,
    ) -> [Option<usize>; NUMBER_OF_NEIGHBORS_TO_RETURN] {
        self.neighbors.clear();

        match candidate {
            Candidate::Neighbors => {
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

                for (n, e) in self
                    .neighbors
                    .iter_mut()
                    .take(NUMBER_OF_NEIGHBORS_TO_RETURN)
                    .zip(
                        self.found_nearest_neighbors
                            .iter()
                            .take(NUMBER_OF_NEIGHBORS_TO_RETURN),
                    )
                {
                    *n = Some(e.get_index());
                }
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

                for (n, e) in self
                    .neighbors
                    .iter_mut()
                    .take(NUMBER_OF_NEIGHBORS_TO_RETURN)
                    .zip(self.econn.iter().take(NUMBER_OF_NEIGHBORS_TO_RETURN))
                {
                    *n = Some(e.get_index());
                }
            }
        }

        let mut output = [None; NUMBER_OF_NEIGHBORS_TO_RETURN];

        if self.neighbors.len() < NUMBER_OF_NEIGHBORS_TO_RETURN {
            output.copy_from_slice(self.neighbors.as_slice());
        } else {
            output.copy_from_slice(&self.neighbors[..NUMBER_OF_NEIGHBORS_TO_RETURN]);
        };

        output
    }
    // Algorithm 4 - Select neighbours heuristic
    fn select_neighbors_heuristic<const NUMBER_OF_NEIGHBORS_TO_RETURN: usize>(
        &mut self,
        base_element: impl Node<N, M, T>,
        candidate: Candidate,
        layer: usize,
        extend_candidates: bool,
        keep_pruned_connections: bool,
    ) -> [Option<usize>; NUMBER_OF_NEIGHBORS_TO_RETURN] {
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
                    .map(|element| self.hnsw.get(&element.get_index()))
                    .flatten()
                {
                    if self
                        .working_queue
                        .iter()
                        .find(|w| w.get_index() == e_adjacent.get_index())
                        .is_none()
                    {
                        self.working_queue.push(*e_adjacent);
                    }
                }
            }
        }

        self.discarded_candidates.clear();

        while self.working_queue.len() > 0 && self.neighbors.len() < NUMBER_OF_NEIGHBORS_TO_RETURN {
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
            // Extract nearest element from working queue.
            // Unwrap: Working queue must be greater than zero to enter body of while loop.
            let nearest = self.working_queue.pop().unwrap();

            // If nearest element is closer to base element compared to any element from self.neighbors
            if base_element.distance(&nearest, &self.distance)
                < self
                    .neighbors
                    .iter()
                    .flatten()
                    .map(|k| self.hnsw.get(k))
                    .flatten()
                    .map(|n| base_element.distance(n, &self.distance))
                    .reduce(T::min)
                    .unwrap_or(T::max_value())
            {
                self.neighbors.push(Some(nearest.get_index()));
            } else {
                self.discarded_candidates.push(nearest);
            }
        }

        if keep_pruned_connections {
            while self.discarded_candidates.len() > 0
                && self.neighbors.len() < NUMBER_OF_NEIGHBORS_TO_RETURN
            {
                // Extract closest neighbor to element from the queue.
                // Sort the working queue from furthest to nearest.
                self.discarded_candidates.sort_by(|a, b| {
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

                match self.discarded_candidates.pop() {
                    Some(nearest) => {
                        let exists = self
                            .neighbors
                            .iter()
                            .flatten()
                            .any(|n| *n == nearest.get_index());

                        if !exists {
                            self.neighbors.push(Some(nearest.get_index()));
                        }
                    }
                    None => {}
                };
            }
        }

        let mut output = [None; NUMBER_OF_NEIGHBORS_TO_RETURN];

        if self.neighbors.len() < NUMBER_OF_NEIGHBORS_TO_RETURN {
            let (a, _) = output.split_at_mut(self.neighbors.len());
            a.copy_from_slice(&self.neighbors[..self.neighbors.len()]);
        } else {
            output.copy_from_slice(&self.neighbors[..NUMBER_OF_NEIGHBORS_TO_RETURN])
        }

        output
    }
    // Algorithm 2 - Search layer
}

#[derive(Clone, Debug)]
pub enum NeighborSelectionAlgorithm {
    Simple,
    Heuristic,
}

#[derive(Clone, Copy)]
enum Candidate {
    ElementConnections,
    Neighbors,
}

#[test]
fn flakey_test() {
    struct MyStruct<const N: usize, T>
    where
        T: Clone + Copy,
    {
        value: [T; N],
    }

    impl<const N: usize, T> Deref for MyStruct<N, T>
    where
        T: Clone + Copy,
    {
        type Target = [T; N];
        fn deref(&self) -> &Self::Target {
            &self.value
        }
    }

    const DIMENSIONS: usize = 2;
    const K: usize = 2;
    let mut knn = HNSW::<DIMENSIONS, f32>::default()
        .set_distance(Distance::Euclidean)
        .build();

    // Clear KNN and assert search call returns an error when there are not enough examples inserted.

    knn.clear();
    _ = knn.insert(
        0,
        MyStruct {
            value: [42.0, 42.0],
        },
    );

    _ = knn.insert(
        1,
        MyStruct {
            value: [24.0, 24.0],
        },
    );

    _ = knn.insert(
        2,
        MyStruct {
            value: [25.0, 25.0],
        },
    );

    _ = knn.insert(
        3,
        MyStruct {
            value: [34.0, 34.0],
        },
    );

    let neighbors = knn.search_neighbors::<K, _>(MyStruct {
        value: [40.0, 40.0],
    });
    assert_eq!(neighbors.unwrap(), [0, 3]);
}

#[test]
fn select_neighbors_heuristic_test() {
    use crate::array_vec::ArrayVec;

    struct MyStruct<const N: usize, T>
    where
        T: Clone + Copy,
    {
        value: [T; N],
    }

    impl<const N: usize, T> Deref for MyStruct<N, T>
    where
        T: Clone + Copy,
    {
        type Target = [T; N];
        fn deref(&self) -> &Self::Target {
            &self.value
        }
    }

    const DIMENSIONS: usize = 2;
    let mut knn = HNSW::<DIMENSIONS, f32>::default()
        .set_distance(Distance::Euclidean)
        .build();
    knn.found_nearest_neighbors = vec![
        EnterPoint {
            index: 1,
            layer: 7,
            value: [24.0, 24.0],
            connections: ArrayVec {
                inner: [
                    Some(Element::new(0, 2)),
                    Some(Element::new(0, 1)),
                    Some(Element::new(0, 0)),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
                index: 3,
            },
        },
        EnterPoint {
            index: 0,
            layer: 2,
            value: [42.0, 42.0],
            connections: ArrayVec {
                inner: [
                    Some(Element::new(1, 2)),
                    Some(Element::new(1, 1)),
                    Some(Element::new(1, 0)),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
                index: 3,
            },
        },
    ];
    let layer = 0;
    let extend_candidates = true;
    let keep_pruned_connections = true;
    const NUMBER_OF_NEIGHBORS_TO_RETURN: usize = 2;

    let base_element = QueryElement::new([25.0, 25.0]);
    let candidate = Candidate::Neighbors;
    let neighbors = knn.select_neighbors_heuristic::<NUMBER_OF_NEIGHBORS_TO_RETURN>(
        base_element,
        candidate,
        layer,
        extend_candidates,
        keep_pruned_connections,
    );

    assert_eq!(neighbors, [Some(1), Some(0)]);
}
