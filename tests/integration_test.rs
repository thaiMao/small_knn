use small_knn::{distance::Distance, HNSW};

#[test]
fn test_clear() {
    const DIMENSIONS: usize = 2;
    const K: usize = 2;
    let mut knn = HNSW::<DIMENSIONS, f32>::default()
        .set_distance(Distance::Euclidean)
        .build();

    _ = knn.insert(0, [1.0, 1.0]);
    _ = knn.insert(1, [2.0, 2.0]);
    _ = knn.insert(2, [10.0, 5.0]);
    _ = knn.insert(4, [11.0, 15.0]);
    let mut neighbors = knn.search_neighbors::<K>([2.1, 2.1]);
    assert_eq!(neighbors.unwrap(), [1, 0]);

    // Insert another value and search again.
    _ = knn.insert(5, [2.5, 2.5]);
    neighbors = knn.search_neighbors::<K>([2.1, 2.1]);
    assert_eq!(neighbors.unwrap(), [1, 5]);

    // 2.4 is closer to index 5 than index 1
    neighbors = knn.search_neighbors::<K>([2.4, 2.4]);
    assert_eq!(neighbors.unwrap(), [5, 1]);

    // Clear KNN and assert search call returns an error when there are not enough examples inserted.
    knn.clear();
    neighbors = knn.search_neighbors::<K>([2.4, 2.4]);
    assert!(neighbors.is_err());

    knn.clear();
    _ = knn.insert(0, [42.0, 42.0]);

    _ = knn.insert(1, [24.0, 24.0]);

    _ = knn.insert(2, [25.0, 25.0]);

    _ = knn.insert(3, [34.0, 34.0]);

    neighbors = knn.search_neighbors::<K>([40.0, 40.0]);
    assert_eq!(neighbors.unwrap(), [0, 3]);
}

#[test]
#[should_panic]
fn test_insufficient_insertions() {
    const DIMENSIONS: usize = 2;
    const K: usize = 2;
    let mut knn = HNSW::<DIMENSIONS, f32>::default()
        .set_distance(Distance::Euclidean)
        .build();

    _ = knn.insert(0, [1.0, 1.0]);

    let neighbors = knn.search_neighbors::<K>([2.1, 2.1]);
    assert_eq!(neighbors.unwrap(), [1, 0]);
}

#[test]
fn test_minimum_insertions() {
    const DIMENSIONS: usize = 2;
    const K: usize = 2;
    let mut knn = HNSW::<DIMENSIONS, f32>::default()
        .set_distance(Distance::Euclidean)
        .build();

    _ = knn.insert(0, [1.0, 1.0]);
    _ = knn.insert(1, [2.0, 2.0]);
    let neighbors = knn.search_neighbors::<K>([2.1, 2.1]);
    assert_eq!(neighbors.unwrap(), [1, 0]);

    let neighbors = knn.search_neighbors::<K>([1.1, 1.1]);
    assert_eq!(neighbors.unwrap(), [0, 1]);
}
