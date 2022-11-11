# CI:

[![CircleCI](https://circleci.com/gh/thaiMao/love_thy.svg?style=svg)](https://app.circleci.com/pipelines/github/thaiMao/love_thy)

Under active development - DO NOT USE.

An approximate K-nearest neighbor search based on navigable small world
graphs written in Rust.

Original paper: https://arxiv.org/pdf/1603.09320.pdf

Design goals

Preallocate all heap memory required upfront during the "build" phase.

Avoid memory allocations and deallocations when searching for nearest neighbors.

Usage

```
struct MyNode<const N: usize, T> where T: Clone + Copy, {
    value: [T; N],
}

// Implement Deref
impl<const N: usize, T> Deref for MyNode<N, T>
where
    T: Clone + Copy,
{
    type Target = [T; N];
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}
```

```
// Set parameters
let mut knn = HNSW::<DIMENSIONS, f32>::default()
    .set_distance(Distance::Euclidean)
    .build();
```

```
// Add data
knn.insert(0, MyNode { value: [1.0, 1.0] });
knn.insert(1, MyNode { value: [2.0, 2.0] });
knn.insert(2, MyNode { value: [10.0, 5.0] });
knn.insert(4, MyNode { value: [11.0, 15.0] },);
```

```
// Search for neighbors
let neighbors = knn.search_neighbors::<K,>(MyNode { value: [2.1, 2.1] });

// Returns the index of the nearest neighbours
assert_eq!(neighbors.unwrap(), [1, 0]);
```
