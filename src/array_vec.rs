use crate::error::KNNError;

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct ArrayVec<T, const CAP: usize>
where
    T: Clone + Copy,
{
    pub inner: [Option<T>; CAP],
    pub index: usize,
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

    pub fn try_push(&mut self, element: T) -> Result<(), KNNError> {
        match self.inner.get_mut(self.index) {
            Some(e) => {
                *e = Some(element);
                self.index += 1;
                Ok(())
            }
            None => {
                // Returns an error when attempting to push an element when
                // the array vec has reached full capacity.
                // Ways to avoid this error include:
                // 1. Trying to limit the potential height of the top layer by
                // setting a low normalization factor.
                // 2. Reducing the max number of allowed connections for an
                // element at layer zero `M_MAX_ZERO`.
                // 3. Or similarly, reduce `M_MAX`, the max number of
                // connections for an element at each layer above layer zero.
                Err(KNNError::FullCapacity)
            }
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
