# small_knn

[![CircleCI](https://circleci.com/gh/thaiMao/small_knn.svg?style=shield)](https://app.circleci.com/pipelines/github/thaiMao/small_knn)

This library is an approximate K-nearest neighbor search based on Hierarchical
Navigable Small World (https://arxiv.org/pdf/1603.09320.pdf).

## Usage

- Set customizable parameters.

```rust
let mut knn = HNSW::<DIMENSIONS, f32>::default()
    .set_distance(Distance::Euclidean)
    .build();
```

- Insert data.

```rust
knn.insert(0, [1.0, 1.0]);
knn.insert(1, [2.0, 2.0]);
knn.insert(2, [10.0, 5.0]);
knn.insert(4, [11.0, 15.0]);
```

- Search for neighbors.

```rust
let neighbors = knn.search_neighbors::<K>([2.1, 2.1]);

// Returns the index of the nearest neighbours
assert_eq!(neighbors.unwrap(), [1, 0]);
```

## Design goals

The goal is to carry out a similarity search for a fixed number (K) of nearest
neighbors for a given query without incurring additional allocations during
search.

## Documentation

TODO

## License

This project is licensed under the MIT License.
