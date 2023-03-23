#![allow(warnings)]
use std::{fs::File, path::PathBuf, sync::{atomic::AtomicUsize, Arc}};

use ::gpu_tracking::{
    decoderiter::MinimalETSParser,
    error::{Error, Result},
    execute_gpu::{execute_file, execute_ndarray, mean_from_iter, path_to_iter},
    gpu_setup::{new_adapter, new_device_queue, ParamStyle, TrackingParams},
    linking,
    linking::SubsetterType,
    my_dtype,
    progressfuture::{PollResult, ProgressFuture, ScopedProgressFuture},
};
use gpu_tracking_app;
use gpu_tracking_macros::gen_python_functions;
use ndarray::{Array, Array2, ArrayView2, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
use pollster::FutureExt;
use pyo3::{
    prelude::*,
    pyclass::IterNextOutput,
    types::{IntoPyDict, PyDict},
};
use tiff::encoder::*;

trait ToPyErr {
    fn pyerr(self) -> PyErr;
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
            | Error::TooDenseToLink => pyo3::exceptions::PyValueError::new_err(self.to_string()),
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

    m.add_function(wrap_pyfunction!(batch_file_rust, m)?)?;

    m.add_function(wrap_pyfunction!(log_rust, m)?)?;

    m.add_function(wrap_pyfunction!(log_file_rust, m)?)?;

    m.add_function(wrap_pyfunction!(characterize_rust, m)?)?;

    m.add_function(wrap_pyfunction!(characterize_file_rust, m)?)?;

    // m.add_function(wrap_pyfunction!(test, m)?)?;

    #[pyfn(m)]
    #[pyo3(name = "load")]
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
    #[pyo3(name = "mean_from_disk")]
    fn mean_from_disk<'py>(
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
    fn app<'py>(_py: Python<'py>) {
        gpu_tracking_app::run::run_ignore();
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
