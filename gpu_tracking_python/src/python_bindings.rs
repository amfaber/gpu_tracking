#![allow(warnings)]
use std::{fs::File, path::PathBuf, sync::{atomic::AtomicUsize, Arc}, collections::HashMap};

use ::gpu_tracking::{
    decoderiter::MinimalETSParser,
    error::{Error, Result},
    execute_gpu::{execute_file, execute_ndarray, mean_from_iter, path_to_iter, CommandBuilder, FileOrArray},
    gpu_setup::{new_adapter, new_device_queue, ParamStyle, TrackingParams},
    linking,
    linking::SubsetterType,
    my_dtype,
    progressfuture::{PollResult, ProgressFuture, ScopedProgressFuture},
};
use gpu_tracking_app;
use gpu_tracking_macros::gen_python_functions;
use ndarray::{Array, Array2, ArrayView2, Axis, ArrayView3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3, PyArrayDyn, Element, PyReadonlyArrayDyn};
use pollster::FutureExt;
use pyo3::{
    prelude::*,
    pyclass::IterNextOutput,
    types::{IntoPyDict, PyDict, PyType},
};
use tiff::encoder::*;
use lazy_static::lazy_static;
use std::any::Any;

lazy_static! {
    static ref PANDAS: PyObject = {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let pandas = PyModule::import(py, "pandas").unwrap();
        pandas.to_object(py)
    };
}

// lazy_static! {
//     static ref NP_NDARRAY: PyObject = {
//         let gil = Python::acquire_gil();
//         let py = gil.python();
//         let numpy = PyModule::import(py, "numpy").unwrap();
//         let ndarray = numpy.getattr("ndarray").unwrap();
//         ndarray.to_object(py)
//     };
// }

lazy_static! {
    static ref NUMPY: PyObject = {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let numpy = PyModule::import(py, "numpy").unwrap();
        // let ndarray = numpy.getattr("ndarray").unwrap();
        numpy.to_object(py)
    };
}

trait ToPyErr {
    fn pyerr(self) -> PyErr;
}


// fn helper(file_or_array: &PyAny){
//     if let Ok(path) = file_or_array.extract::<String>(){
//         let closure = Box::new(move |job: &dyn Any, progress, interrupt|{
//             let (path, channel, params, characterize_points): (&String, Option<usize>, TrackingParams, Option<(ArrayView2<my_dtype>, bool, bool)>) = job.downcast().unwrap();
//             execute_file(&path, channel, params, 0, characterize_points, Some(interrupt), Some(progress), &device_queue)
//         });
//         let mut worker = ScopedProgressFuture::new(scope, closure);
//         worker.submit_same((&path, channel, params, characterize_points));
//         worker
//     } else if let Ok(array) = make_arr_3df32(py, file_or_array){
//         let rust_array = array.as_array();
//         let closure = Box::new(move |job, progress, interrupt|{
//             let job = job.downcast::<(&ArrayView3<f32>, Option<usize>, TrackingParams, Option<(ArrayView2<my_dtype>, bool, bool)>)>();
//             let (array, channel, params, characterize_points): (&ArrayView3<f32>, Option<usize>, TrackingParams, Option<(ArrayView2<my_dtype>, bool, bool)>) = job.downcast().unwrap();
//             execute_ndarray(array, params, 0, characterize_points, Some(interrupt), Some(progress), &device_queue)
//         }) as Box<dyn Fn(&dyn Any, &Arc<Mutex>)>;
        
//         let mut worker = ScopedProgressFuture::new(scope, closure as &dyn Fn(&dyn Any, ));
//         worker.submit_same((&rust_array, channel, params, characterize_points));
//         worker
//     } else {
//         return Err(pyo3::exceptions::PyValueError::new_err("First argument must be a path or a numpy array"))
//     }
// }

// enum ArrayOrFile<'a>{
//     File(&'a String),
//     Array(ArrayView3<'a, f32>),
// }

// fn extract_array_inner<'p>(py: Python<'p>, any: &'p PyAny) -> PyResult<PyObject>{

enum OwnedFileOrArray<'p>{
    File(String),
    Array(ArrayView3<'p, f32>)
}

impl<'p> OwnedFileOrArray<'p>{
    fn new(py: Python<'p>, any: &'p PyAny) -> PyResult<Self>{
        if let Ok(path) = any.extract::<String>(){
            return Ok(Self::File(path))
        }
        
        match make_arr_3df32(py, any){
            Ok(array) => return {
                let array = array.as_array();
                let array = unsafe{ ArrayView3::from_shape_ptr(array.raw_dim(), array.as_ptr()) };
                Ok(Self::Array(array))
            },
            Err(e) => return Err(e),
        }
        
        return Err(pyo3::exceptions::PyValueError::new_err("First argument must be a path or a numpy array"))
    }

    fn borrow(&'p self) -> FileOrArray<'p, String>{

        match self{
            Self::File(path) => FileOrArray::File(path.clone()),
            Self::Array(array) => {
                // let arr3d = unsafe{ ArrayView3::from_shape_ptr(arr3d.raw_dim(), arr3d.as_ptr()) };
                // let arr3d = unsafe{ &*(&arr3d as *const ArrayView3<f32>) };
                FileOrArray::Array(&array)
            }
        }
    }
}

// fn test<'p>(py: Python<'p>, any: &'p PyAny){
//     let py_foa = PyFileOrArray::new(py, any).unwrap();
//     let rust_foa = py_foa.to_rust();
//     let idk = CommandBuilder::new()
//         .set_file_or_array(rust_foa);
// }

fn make_arr_3df32<'p>(py: Python<'p>, any: &'p PyAny) -> PyResult<PyReadonlyArray3<'p, f32>>{
// fn make_arr_3df32<'p>(py: Python<'p>, any: &'p PyAny) -> PyResult<&'p ArrayView3<'p, f32>>{
    
    let locals = [
        ("np", NUMPY.as_ref(py)),
        ("arr", any),
    ].into_py_dict(py);
    let arr3d = py.eval(r#"np.atleast_3d(arr).astype("float32", copy = False)"#, None, Some(&locals))?;

    let arr3d = arr3d.extract::<&PyArray3<f32>>()?;
    let arr3d = arr3d.readonly();
    // let arr3d = arr3d.as_array();
    
    // let arr3d = unsafe{ ArrayView3::from_shape_ptr(arr3d.raw_dim(), arr3d.as_ptr()) };
    // let arr3d = unsafe{ &*(&arr3d as *const ArrayView3<f32>) };
    // Ok(arr3d.as_array())
    // todo!()
    // Ok(arr3d)
    Ok(arr3d)
    
}

impl ToPyErr for Error {
    fn pyerr(self) -> PyErr {
        match self {
            Error::Interrupted => pyo3::exceptions::PyKeyboardInterrupt::new_err(self.to_string()),

            Error::GpuAdapterError | Error::GpuDeviceError(_) => {
                pyo3::exceptions::PyConnectionError::new_err(self.to_string())
            }

            Error::ThreadError | Error::PolledAfterTermination | Error::TiffWrite => {
                pyo3::exceptions::PyBaseException::new_err(self.to_string())
            }

            Error::InvalidFileName { .. } | Error::FileNotFound { .. } => {
                pyo3::exceptions::PyFileNotFoundError::new_err(self.to_string())
            }

            Error::DimensionMismatch { .. }
            | Error::EmptyIterator
            | Error::NonSortedCharacterization
            | Error::FrameOutOfBounds { .. }
            | Error::NonStandardArrayLayout
            | Error::UnsupportedFileformat { .. }
            | Error::ArrayDimensionsError { .. }
            | Error::NoExtensionError { .. }
            | Error::ReadError
            | Error::FrameOOB
            | Error::ChannelNotFound
            | Error::CastError
            | Error::TooDenseToLink
            | Error::WrongBuilder => pyo3::exceptions::PyValueError::new_err(self.to_string()),
        }
    }
}

impl ToPyErr for ::gpu_tracking::progressfuture::Error {
    fn pyerr(self) -> PyErr {
        Error::ThreadError.pyerr()
    }
}



gen_python_functions!();



#[pymodule]
fn gpu_tracking(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(batch_rust, m)?)?;

    // m.add_function(wrap_pyfunction!(batch_file_rust, m)?)?;

    m.add_function(wrap_pyfunction!(LoG_rust, m)?)?;

    // m.add_function(wrap_pyfunction!(log_file_rust, m)?)?;

    m.add_function(wrap_pyfunction!(characterize_points_rust, m)?)?;

    // m.add_function(wrap_pyfunction!(characterize_file_rust, m)?)?;

    // m.add_function(wrap_pyfunction!(test, m)?)?;

    #[pyfn(m)]
    #[pyo3(name = "load_rust")]
    fn load<'py>(
        py: Python<'py>,
        path: &str,
        keys: Option<Vec<usize>>,
        channel: Option<usize>,
    ) -> PyResult<&'py PyArray3<my_dtype>> {
        let path = PathBuf::from(path);
        let (provider, dims) = path_to_iter(&path, channel).map_err(|err| err.pyerr())?;
        let mut output = Vec::new();
        let mut n_frames = 0;
        match keys {
            Some(keys) => {
                if keys.iter().enumerate().all(|(idx, &key)| idx == key) {
                    let image_iter = provider.into_iter().take(keys.len());
                    for image in image_iter {
                        let image = image.map_err(|err| err.pyerr())?;
                        output.extend(image.into_iter());
                        n_frames += 1;
                    }
                } else {
                    for key in keys {
                        let image = provider
                            .get_frame(key)
                            .map_err(|err| match err {
                                Error::FrameOOB => Error::FrameOutOfBounds {
                                    vid_len: provider.len(Some(key)),
                                    problem_idx: key,
                                },
                                _ => err,
                            })
                            .map_err(|err| err.pyerr())?;
                        output.extend(image.into_iter());
                        n_frames += 1;
                    }
                }
            }
            None => {
                for image in provider.into_iter() {
                    let image = image.map_err(|err| err.pyerr())?;
                    output.extend(image.into_iter());
                    n_frames += 1;
                }
            }
        }
        let arr =
            Array::from_shape_vec([n_frames, dims[0] as usize, dims[1] as usize], output).unwrap();
        let pyarr = arr.into_pyarray(py);
        Ok(pyarr)
    }

    #[pyfn(m)]
    #[pyo3(name = "link_rust")]
    fn link_py<'py>(
        py: Python<'py>,
        pyarr: PyReadonlyArray2<my_dtype>,
        search_range: my_dtype,
        memory: Option<usize>,
    ) -> PyResult<&'py PyArray1<usize>> {
        let memory = memory.unwrap_or(0);
        let array = pyarr.as_array();
        let frame_iter =
            linking::FrameSubsetter::new(array, Some(0), (1, 2), None, SubsetterType::Linking)
                .into_linking_iter();
        let res =
            linking::linker_all(frame_iter, search_range, memory).map_err(|err| err.pyerr())?;
        Ok(res.into_pyarray(py))
    }

    #[pyfn(m)]
    #[pyo3(name = "connect_rust")]
    fn connect<'py>(
        py: Python<'py>,
        pyarr1: PyReadonlyArray2<my_dtype>,
        pyarr2: PyReadonlyArray2<my_dtype>,
        search_range: my_dtype,
    ) -> PyResult<(&'py PyArray1<usize>, &'py PyArray1<usize>)> {
        let array1 = pyarr1.as_array();
        let array2 = pyarr2.as_array();
        let frame_iter1 =
            linking::FrameSubsetter::new(array1, Some(0), (1, 2), None, SubsetterType::Linking)
                .into_linking_iter();
        let frame_iter2 =
            linking::FrameSubsetter::new(array2, Some(0), (1, 2), None, SubsetterType::Linking)
                .into_linking_iter();
        let (res1, res2) = linking::connect_all(frame_iter1, frame_iter2, search_range)
            .map_err(|err| err.pyerr())?;
        Ok((res1.into_pyarray(py), res2.into_pyarray(py)))
    }

    #[pyfn(m)]
    #[pyo3(name = "parse_ets")]
    fn parse_ets<'py>(py: Python<'py>, path: &str) -> PyResult<&'py PyDict> {
        let mut file = File::open(path).unwrap();
        let parser = MinimalETSParser::new(&mut file).unwrap();
        let output = PyDict::new(py);
        for channel in parser.offsets.keys() {
            let iter = parser
                .iterate_channel(file.try_clone().unwrap(), *channel)
                .map_err(|err| err.pyerr())?;
            let n_frames = iter.len();
            let mut vec = Vec::with_capacity(n_frames * parser.dims.iter().product::<usize>());
            vec.extend(iter.flatten().flatten());
            let array =
                Array::from_shape_vec((n_frames, parser.dims[1], parser.dims[0]), vec).unwrap();
            let array = array.into_pyarray(py);
            output.set_item(*channel, array).unwrap();
        }
        Ok(output)
    }

    #[pyfn(m)]
    #[pyo3(name = "parse_ets_with_keys")]
    fn parse_ets_with_keys<'py>(
        py: Python<'py>,
        path: &str,
        keys: Vec<usize>,
        channel: Option<usize>,
    ) -> PyResult<&'py PyArray3<u16>> {
        let mut file = File::open(path).unwrap();
        let parser = MinimalETSParser::new(&mut file).unwrap();
        let channel = channel.unwrap_or(0);
        let mut iter = parser
            .iterate_channel(file.try_clone().unwrap(), channel)
            .map_err(|err| err.pyerr())?;
        let n_frames = keys.len();
        let mut vec = Vec::with_capacity(n_frames * parser.dims.iter().product::<usize>());
        for key in keys {
            iter.seek(key).map_err(|err| err.pyerr())?;
            vec.extend(iter.next().unwrap().into_iter().flatten());
        }
        let array = Array::from_shape_vec((n_frames, parser.dims[1], parser.dims[0]), vec).unwrap();
        let array = array.into_pyarray(py);
        Ok(array)
    }

    #[pyfn(m)]
    #[pyo3(name = "mean_from_file_rust")]
    fn mean_from_file<'py>(
        py: Python<'py>,
        path: &str,
        channel: Option<usize>,
    ) -> PyResult<&'py PyArray2<f32>> {
        let path = PathBuf::from(path);
        let (provider, dims) = path_to_iter(&path, channel).map_err(|err| err.pyerr())?;
        let iter = provider.into_iter();
        let mean_arr = mean_from_iter(iter, &dims).map_err(|err| err.pyerr())?;
        Ok(mean_arr.into_pyarray(py))
    }

    #[pyfn(m)]
    #[pyo3(name = "tracking_app")]
    fn app<'py>(_py: Python<'py>, doc_dir: PathBuf) {
        gpu_tracking_app::run::run_python(doc_dir);
    }

    #[pyfn(m)]
    #[pyo3(name = "to_tiff")]
    fn to_tiff<'py>(
        py: Python<'py>,
        path: &str,
        pyarr: PyReadonlyArray3<my_dtype>,
    ) -> PyResult<()> {
        let file = std::fs::File::create(path)?;
        let writer = std::io::BufWriter::new(file);
        let mut encoder = TiffEncoder::new(writer).map_err(|_| Error::TiffWrite.pyerr())?;
        let arr = pyarr.as_array();
        let arr = arr.mapv(|ele| ele as u16);
        let iter = arr.axis_iter(Axis(0));
        for image in iter {
            let slice = image.as_slice().unwrap();
            encoder
                .write_image::<colortype::Gray16>(
                    image.shape()[0] as u32,
                    image.shape()[1] as u32,
                    slice,
                )
                .map_err(|_| Error::TiffWrite.pyerr())?;
        }
        Ok(())
    }

    Ok(())
}
