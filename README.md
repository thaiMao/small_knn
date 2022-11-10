Under active development - DO NOT USE.

An approximate K-nearest neighbor search based on navigable small world
graphs written in Rust.

Design goals

Preallocate all heap memory required upfront during the "build" phase.

Avoid memory allocations and deallocations when searching for nearest neighbors.

Usage

```
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
        knn.insert(4, MyNode { value: [11.0, 15.0] },);

        let neighbors = knn.search_neighbors::<K,>(MyNode { value: [2.1, 2.1] });
        assert_eq!(neighbors.unwrap(), [1, 0]);
```
