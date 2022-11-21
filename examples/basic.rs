use small_knn::{distance::Distance, HNSW};

fn main() {
    const DIMENSIONS: usize = 2;
    const K: usize = 2;
    let mut knn = HNSW::<DIMENSIONS, f32>::default()
        .set_distance(Distance::Euclidean)
        .build();

    _ = knn.insert(0, [1.0, 1.0]);
    _ = knn.insert(1, [2.0, 2.0]);
    _ = knn.insert(2, [10.0, 5.0]);
    _ = knn.insert(4, [11.0, 15.0]);
    let neighbors = knn.search_neighbors::<K>([2.1, 2.1]);
    assert_eq!(neighbors.unwrap(), [1, 0]);
}
