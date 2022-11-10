use crate::array_vec::ArrayVec;
use num::{cast, Float, Num};
use rand::prelude::*;
use std::{cmp::Ordering, fmt::Debug, iter::Sum, marker::PhantomData, ops::Deref};

mod array_vec;

pub struct Setup;
pub struct Ready;

const DEFAULT_CAPACITY: usize = 128;
const DEFAULT_NORMALIZATION_FACTOR: f32 = 2.0;
const DEFAULT_NEIGHBOR_SELECTION_ALGORTHIM: NeighborSelectionAlgorithm =
    NeighborSelectionAlgorithm::Heuristic;
const DEFAULT_EXTEND_CANDIDATES: bool = true;
const DEFAULT_KEEP_PRUNED_CONNECTIONS: bool = false;
const DEFAULT_DISTANCE: Distance = Distance::Euclidean;

/// * `N` - Number of dimensions.
#[derive(Clone, Debug)]
pub struct HNSW<const EF_CONSTRUCTION: usize, const N: usize, const M_MAX: usize, T, Stage = Setup>
where
    T: Float + Sum + Debug,
{
    stage: PhantomData<Stage>,
    enter_points: Vec<EnterPoint<N, M_MAX, T>>,
    rng: rand::rngs::ThreadRng,
    search_layer: SearchLayer<N, M_MAX, T>,
    found_nearest_neighbors: Vec<EnterPoint<N, M_MAX, T>>,
    working_queue: Vec<EnterPoint<N, M_MAX, T>>,
    neighbors: Vec<EnterPoint<N, M_MAX, T>>,
    discarded_candidates: Vec<EnterPoint<N, M_MAX, T>>,
    normalization_factor: T,
    neighbor_selection_algorithm: NeighborSelectionAlgorithm,
    extend_candidates: bool,
    keep_pruned_connections: bool,
    distance: Distance,
    capacity: usize,
    nearest_elements: Vec<EnterPoint<N, M_MAX, T>>,
    enter_point_key: usize,
    econn: Vec<EnterPoint<N, M_MAX, T>>,
}

impl<const EF_CONSTRUCTION: usize, const N: usize, const M_MAX: usize, T> Default
    for HNSW<EF_CONSTRUCTION, N, M_MAX, T, Setup>
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
            enter_point_key: 0,
            econn: Vec::with_capacity(DEFAULT_CAPACITY),
        }
    }
}

impl<const EF_CONSTRUCTION: usize, const N: usize, const M_MAX: usize, T>
    HNSW<EF_CONSTRUCTION, N, M_MAX, T, Setup>
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
            enter_point_key: 0,
            econn: Vec::with_capacity(DEFAULT_CAPACITY),
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

    pub fn build(&mut self) -> HNSW<EF_CONSTRUCTION, N, M_MAX, T, Ready> {
        let enter_points = Vec::with_capacity(self.capacity);
        let found_nearest_neighbors = Vec::with_capacity(self.capacity);
        let working_queue = Vec::with_capacity(self.capacity);
        let neighbors = Vec::with_capacity(self.capacity);
        let discarded_candidates = Vec::with_capacity(self.capacity);
        let nearest_elements = Vec::with_capacity(self.capacity);
        let econn = Vec::with_capacity(self.capacity);

        let rng = rand::thread_rng();

        HNSW::<EF_CONSTRUCTION, N, M_MAX, T, Ready> {
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
            enter_point_key: 0,
            econn,
        }
    }
}

impl<'a, const EF_CONSTRUCTION: usize, const N: usize, const M_MAX: usize, T>
    HNSW<EF_CONSTRUCTION, N, M_MAX, T, Ready>
where
    T: Float + Sum + PartialEq + Debug + PartialOrd,
{
    /// Insert elements into a graph structure.
    /// * `M` - Number of established connections.
    /// * `M_MAX` - Maximum number of connections for each element per layer.
    pub fn insert<const M: usize, Q>(&mut self, index: usize, value: Q)
    where
        Q: Deref + Deref<Target = [T; N]>,
    {
        debug_assert!(M <= M_MAX);

        self.found_nearest_neighbors.clear();
        let query_element = QueryElement::new(*value);

        // Top layer for HNSW.
        let top_layer_level = match self.enter_points.get(self.enter_point_key) {
            Some(ep) => ep.layer,
            None => 0,
        };

        // New element's level.
        let random_number: T = cast::<f32, T>(self.rng.gen()).unwrap();
        let new_element_level = (-random_number.ln() * self.normalization_factor).floor();
        let new_element_level = cast::<T, usize>(new_element_level).unwrap();

        let enter_point_index = self.enter_points.len();

        if self.enter_points.len() > 0 {
            for layer in top_layer_level..new_element_level + 1 {
                let nearest_elements = self.search_layer.search::<1>(
                    &query_element,
                    self.enter_points.as_slice(),
                    layer,
                );
                self.found_nearest_neighbors
                    .extend_from_slice(&nearest_elements);

                let nearest_element_to_query =
                    query_element.nearest(self.found_nearest_neighbors.as_slice(), &self.distance);
                self.enter_points.push(nearest_element_to_query);
            }

            for layer in top_layer_level.min(new_element_level)..0 {
                let found_nearest_neighbors = self.search_layer.search::<EF_CONSTRUCTION>(
                    &query_element,
                    self.enter_points.as_slice(),
                    layer,
                );

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

                // Add bidirectional connections from neighbors to q at layer lc
                //let mut ep = EnterPoint::new(
                //    query_element.value,
                //    index,
                //    new_element_level,
                //    enter_point_index,
                //);
                for neighbor in neighbors.iter_mut() {
                    neighbor
                        .connections
                        .try_push(Element::new(enter_point_index, new_element_level));

                    //let overflow = ep
                    //    .connections
                    //    .try_push(Element::new(neighbor.enter_point_index, neighbor.layer));
                }

                for mut e in neighbors.iter().cloned() {
                    self.econn.clear();
                    e.neighbourhood(layer)
                        .map(|element| self.enter_points.get(element.index).unwrap())
                        .cloned()
                        .for_each(|enter_point| {
                            self.econn.push(enter_point);
                        });

                    if e.number_of_connections(layer) > M_MAX {
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
                            Element::new(enter_point.enter_point_index, enter_point.layer)
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

        if new_element_level > top_layer_level || top_layer_level == 0 {
            // Set enter point for hnsw to query element.
            self.enter_points.push(EnterPoint::new(
                query_element.value,
                index,
                new_element_level,
                enter_point_index,
            ));
            self.enter_point_key = index;
        }

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
        let enter_point = match self.enter_points.get(self.enter_point_key) {
            Some(enter_point) => enter_point.clone(),
            None => return Err(()),
        };
        let query_element = QueryElement::new(*value);
        self.nearest_elements.clear();
        let level = enter_point.layer;

        for layer in level..=1 {
            let closest_neighbor =
                self.search_layer
                    .search::<1>(&query_element, &[enter_point.clone()], layer);
            self.nearest_elements.extend_from_slice(&closest_neighbor);
            self.enter_points
                .push(query_element.nearest(&self.nearest_elements, &self.distance));
        }

        let closest_neighbors = self.search_layer.search::<EF_CONSTRUCTION>(
            &query_element,
            self.enter_points.as_slice(),
            0,
        );
        self.nearest_elements.extend_from_slice(&closest_neighbors);

        // Return nearest elements
        // Sort elements by nearest to furthest order from query element.
        self.nearest_elements.sort_by(|a, b| {
            let distance_a_q = self.distance.calculate(*value, a.value);
            let distance_b_q = self.distance.calculate(*value, b.value);
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
            output[i] = self.nearest_elements[i].index;
        }

        Ok(output)
    }

    // Algorithm 3 - Select neighbours simple
    fn select_neighbors_simple<const NUMBER_OF_NEIGHBOURS_TO_RETURN: usize>(
        &mut self,
        base_element: impl Node<N, M_MAX, T>,
        candidate: Candidate,
    ) -> [EnterPoint<N, M_MAX, T>; NUMBER_OF_NEIGHBOURS_TO_RETURN] {
        self.neighbors.clear();

        match candidate {
            Candidate::Neighbors => {
                // Return nearest elements
                // Sort elements by nearest to furthest order from query element.
                self.found_nearest_neighbors.sort_by(|a, b| {
                    let distance_a_q = self.distance.calculate(base_element.value(), a.value);
                    let distance_b_q = self.distance.calculate(base_element.value(), b.value);
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
                    let distance_a_q = self.distance.calculate(base_element.value(), a.value);
                    let distance_b_q = self.distance.calculate(base_element.value(), b.value);
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
        base_element: impl Node<N, M_MAX, T>,
        candidate: Candidate,
        layer: usize,
        extend_candidates: bool,
        keep_pruned_connections: bool,
    ) -> [EnterPoint<N, M_MAX, T>; NUMBER_OF_NEIGHBOURS_TO_RETURN] {
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
                    .map(|element| self.enter_points.get(element.index).unwrap())
                {
                    if self
                        .working_queue
                        .iter()
                        .find(|w| w.index == e_adjacent.index)
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
                let distance_a_q = self.distance.calculate(base_element.value(), a.value);
                let distance_b_q = self.distance.calculate(base_element.value(), b.value);
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
                    .map(|element| self.enter_points.get(element.index).unwrap())
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
                    let distance_a_q = self.distance.calculate(base_element.value(), a.value);
                    let distance_b_q = self.distance.calculate(base_element.value(), b.value);
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct EnterPoint<const N: usize, const M_MAX: usize, T> {
    index: usize,
    layer: usize,
    value: [T; N],
    // TODO Explain why 64.
    connections: ArrayVec<Element, 64>,
    enter_point_index: usize,
}

impl<const N: usize, const M_MAX: usize, T> EnterPoint<N, M_MAX, T> {
    fn new(value: [T; N], index: usize, layer: usize, enter_point_index: usize) -> Self {
        Self {
            index,
            layer,
            value,
            connections: ArrayVec::new(),
            enter_point_index,
        }
    }

    /// Returns connections for a given layer.
    fn neighbourhood(&self, layer: usize) -> impl Iterator<Item = Element> {
        self.connections
            .into_iter()
            .flatten()
            .filter(move |connection| connection.layer == layer)
    }

    /// Return the number of connections for a given layer.
    fn number_of_connections(&self, layer: usize) -> usize {
        self.connections
            .into_iter()
            .flatten()
            .filter(move |connection| connection.layer == layer)
            .fold(0, |total, _| total + 1)
    }
}

impl<const N: usize, const M_MAX: usize, T> Node<N, M_MAX, T> for EnterPoint<N, M_MAX, T>
where
    T: Float + Sum,
{
    fn value(&self) -> [T; N] {
        self.value
    }
    fn nearest(
        &self,
        closest_found_elements: &[EnterPoint<N, M_MAX, T>],
        distance: &Distance,
    ) -> EnterPoint<N, M_MAX, T> {
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
        neighbors: &[EnterPoint<N, M_MAX, T>],
        distance: &Distance,
    ) -> EnterPoint<N, M_MAX, T> {
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

    fn distance(&self, enter_point: &EnterPoint<N, M_MAX, T>, distance: &Distance) -> T {
        distance.calculate(self.value, enter_point.value)
    }
}

impl<const N: usize, const M_MAX: usize, T> PartialEq<Element> for EnterPoint<N, M_MAX, T>
where
    T: Float,
{
    fn eq(&self, connection: &Element) -> bool {
        self.index == connection.index
    }
}

/// * `M_MAX` - Maximum number of connections for each element per layer.
#[derive(Clone, Copy, Debug)]
struct QueryElement<const N: usize, const M_MAX: usize, T>
where
    T: Num,
{
    value: [T; N],
}

impl<const N: usize, const M_MAX: usize, T> Node<N, M_MAX, T> for QueryElement<N, M_MAX, T>
where
    T: Float + Sum,
{
    fn value(&self) -> [T; N] {
        self.value
    }

    fn nearest(
        &self,
        closest_found_elements: &[EnterPoint<N, M_MAX, T>],
        distance: &Distance,
    ) -> EnterPoint<N, M_MAX, T> {
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
        neighbors: &[EnterPoint<N, M_MAX, T>],
        distance: &Distance,
    ) -> EnterPoint<N, M_MAX, T> {
        let mut highest = None;
        let mut highest_index = 0;
        for (index, element) in neighbors.iter().enumerate() {
            let temp = distance.calculate(element.value, self.value);

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

    fn distance(&self, enter_point: &EnterPoint<N, M_MAX, T>, distance: &Distance) -> T {
        distance.calculate(self.value, enter_point.value)
    }
}

impl<const N: usize, const M_MAX: usize, T> QueryElement<N, M_MAX, T>
where
    T: Float + Sum + Clone + Copy + PartialOrd,
{
    fn new(value: [T; N]) -> Self {
        Self { value }
    }
}

trait Node<const N: usize, const M_MAX: usize, T>
where
    T: Float + Sum,
{
    fn value(&self) -> [T; N];

    fn nearest(
        &self,
        closest_found_elements: &[EnterPoint<N, M_MAX, T>],
        distance: &Distance,
    ) -> EnterPoint<N, M_MAX, T>;

    fn furthest(
        &self,
        neighbors: &[EnterPoint<N, M_MAX, T>],
        distance: &Distance,
    ) -> EnterPoint<N, M_MAX, T>;

    fn distance(&self, enter_point: &EnterPoint<N, M_MAX, T>, distance: &Distance) -> T;
}

#[derive(Clone, Debug)]
pub enum NeighborSelectionAlgorithm {
    Simple,
    Heuristic,
}

#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Distance {
    Euclidean,
}

impl Distance {
    /// * `N` - N-space
    pub fn calculate<const N: usize, T>(&self, q: [T; N], p: [T; N]) -> T
    where
        T: Float + Sum + Clone + Copy,
    {
        match self {
            // TODO SIMD
            Self::Euclidean => q
                .iter()
                .cloned()
                .zip(p.iter().cloned())
                .map(|(q_i, p_i)| (q_i - p_i).powf(cast::<usize, T>(2).unwrap()))
                .sum::<T>()
                .sqrt(),
        }
    }
}

#[cfg(test)]
mod knn_tests {
    use std::ops::Deref;

    use super::{Distance, HNSW};

    const EF_CONSTRUCTION: usize = 4;
    const DIMENSIONS: usize = 2;
    const MAX_CONNECTIONS: usize = 16;
    const M: usize = 8;
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

        let mut knn = HNSW::<EF_CONSTRUCTION, DIMENSIONS, MAX_CONNECTIONS, f32>::default()
            .set_distance(Distance::Euclidean)
            .build();

        knn.insert::<M, _>(0, MyNode { value: [1.0, 1.0] });
        knn.insert::<M, _>(1, MyNode { value: [2.0, 2.0] });
        knn.insert::<M, _>(2, MyNode { value: [10.0, 5.0] });
        knn.insert::<M, _>(
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Element {
    index: usize,
    layer: usize,
}

impl Element {
    /// * `index` - Index location of enter point in self.enter_points.
    /// * `layer` - Layer of the enter point.
    fn new(index: usize, layer: usize) -> Self {
        Self { index, layer }
    }
}

#[derive(Clone, Debug)]
struct SearchLayer<const N: usize, const M_MAX: usize, T>
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
    ) -> [EnterPoint<N, M_MAX, T>; NUMBER_OF_NEAREST_TO_Q_ELEMENTS_TO_RETURN] {
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
            let nearest = query_element.nearest(self.candidates.as_slice(), &self.distance);
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

        self.found_nearest_neighbors.clone().try_into().unwrap()
    }
}

#[derive(Clone, Copy)]
enum Candidate {
    ElementConnections,
    Neighbors,
}
