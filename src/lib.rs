// #![allow(warnings)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

pub mod decoderiter;
pub mod execute_gpu;
pub mod into_slice;
pub mod kernels;
pub mod slice_wrapper;
pub type my_dtype = f32;
pub mod error;
pub mod gpu_setup;
pub mod linking;
pub mod progressfuture;
pub mod utils;

// pub mod python_bindings;
