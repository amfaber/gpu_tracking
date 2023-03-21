use crate::my_dtype;
use ndarray::{Array2, ArrayView2};

pub struct slice_wrap<'a>(&'a[my_dtype]);

impl<'a> From<&'a[my_dtype]> for slice_wrap<'a>{
    fn from(slice: &'a[my_dtype]) -> Self{
        Self(slice)
    }
}

impl<'a> From<&'a Array2<my_dtype>> for slice_wrap<'a>{
    fn from(array: &'a Array2<my_dtype>) -> Self{
        Self(array.as_slice().unwrap())
    }
}

impl<'a> From<&'a ArrayView2<'a, my_dtype>> for slice_wrap<'a>{
    fn from(array: &'a ArrayView2<'a, my_dtype>) -> Self{
        Self(array.as_slice().unwrap())
    }
}

impl<'a> From<&'a Vec<my_dtype>> for slice_wrap<'a>{
    fn from(vec: &'a Vec<my_dtype>) -> Self{
        Self(vec.as_slice())
    }
}

impl<'a> Into<&'a[my_dtype]> for slice_wrap<'a>{
    fn into(self) -> &'a[my_dtype]{
        self.0
    }
}
