#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct ArrayVec<T, const CAP: usize>
where
    T: Clone + Copy,
{
    inner: [Option<T>; CAP],
    index: usize,
}

impl<T, const CAP: usize> ArrayVec<T, CAP>
where
    T: Copy + Clone + PartialEq,
{
    pub fn new() -> Self {
        Self {
            inner: [None; CAP],
            index: 0,
        }
    }

    // TODO Introduce proper error handling.
    pub fn try_push(&mut self, element: T) -> Result<(), ()> {
        // Check if element already exists.
        //let duplicate = self.inner.iter().flatten().any(|c| *c == element);

        //if duplicate {
        //    return Err(());
        //}
        match self.inner.get_mut(self.index) {
            Some(e) => {
                *e = Some(element);
                self.index += 1;
                Ok(())
            }
            None => Err(()),
        }
    }

    pub fn clear(&mut self) {
        self.inner = [None; CAP];
        self.index = 0;
    }
}

impl<T, const CAP: usize> IntoIterator for ArrayVec<T, CAP>
where
    T: Clone + Copy,
{
    type Item = Option<T>;
    type IntoIter = std::array::IntoIter<Self::Item, CAP>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}
