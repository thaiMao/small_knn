use love_thy::{Distance, HNSW};
use std::ops::Deref;

fn main() {
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

    const EF_CONSTRUCTION: usize = 4;
    const DIMENSIONS: usize = 2;
    const K: usize = 2;
    let mut knn = HNSW::<EF_CONSTRUCTION, DIMENSIONS, f32>::default()
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
}
