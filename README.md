# Small KNN

[![CircleCI](https://circleci.com/gh/thaiMao/small_knn.svg?style=shield)](https://app.circleci.com/pipelines/github/thaiMao/small_knn)

This library is an approximate K-nearest neighbor search based on navigable small world
graphs (https://arxiv.org/pdf/1603.09320.pdf) written in Rust.

Design goals

Preallocate all heap memory required upfront during the "build" phase.

Avoid memory allocations and deallocations when searching for nearest neighbors.

## Usage

- Declare a type.

```rust
struct MyStruct<const N: usize, T> where T: Clone + Copy, {
    value: [T; N],
}
```

- Implement Deref.

```rust
impl<const N: usize, T> Deref for MyStruct<N, T>
where
    T: Clone + Copy,
{
    type Target = [T; N];
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}
```

- Set customizable parameters.

```rust
let mut knn = HNSW::<DIMENSIONS, f32>::default()
    .set_distance(Distance::Euclidean)
    .build();
```

- Insert data.

```rust
knn.insert(0, MyNode { value: [1.0, 1.0] });
knn.insert(1, MyNode { value: [2.0, 2.0] });
knn.insert(2, MyNode { value: [10.0, 5.0] });
knn.insert(4, MyNode { value: [11.0, 15.0] },);
```

- Search for neighbors.

```rust
let neighbors = knn.search_neighbors::<K,>(MyNode { value: [2.1, 2.1] });

// Returns the index of the nearest neighbours
assert_eq!(neighbors.unwrap(), [1, 0]);
```

## Documentation

## License

This project is licensed under the MIT License.
