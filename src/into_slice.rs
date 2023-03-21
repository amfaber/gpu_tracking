use crate::my_dtype;
use ndarray::{Array2, ArrayView2};

pub trait IntoSlice{
    fn into_slice(&self) -> &[my_dtype];
}

impl IntoSlice for Array2<my_dtype>{
    fn into_slice(&self) -> &[my_dtype]{
        self.as_slice().unwrap()
    }
}

impl<'a> IntoSlice for ArrayView2<'a, my_dtype>{
    fn into_slice(&self) -> &[my_dtype]{
        self.as_slice().unwrap()
    }
}

impl IntoSlice for [my_dtype]{
    fn into_slice(&self) -> &[my_dtype]{
        self
    }
}

impl IntoSlice for Vec<my_dtype>{
    fn into_slice(&self) -> &[my_dtype]{
        self.as_slice()
    }
}

impl IntoSlice for &[my_dtype]{
    fn into_slice(&self) -> &[my_dtype]{
        *self
    }
}
