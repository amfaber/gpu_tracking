use std::path::PathBuf;

use thiserror;
use wgpu;

#[derive(Debug, thiserror::Error)]
pub enum Error{
	#[error("could not get gpu adapter")]
	GpuAdapterError,
	
	#[error("could not get gpu device")]
	GpuDeviceError(#[from] wgpu::RequestDeviceError),
	
	#[error("could not find file '{filename}'")]
	FileNotFound{
		source: std::io::Error,
		filename: std::path::PathBuf,
	},

	#[error("dimension mismatch. provided dimensions are ({}, {}), but frame {} contains {} pixels", .dimensions[0], .dimensions[1], idx, frame_len)]
	DimensionMismatch{
		idx: usize,
		frame_len: usize,
		dimensions: [u32; 2],
	},

	#[error("the provided video doesn't contain any frames.")]
	EmptyIterator,

	#[error("the points provided for characterization aren't sorted in ascending frame number.")]
	NonSortedCharacterization,

	#[error("Requested a frame outside the length
 of the video. Video length is {}, requested frame {}", vid_len, problem_idx)]
	// #[error("Requested characterization for a frame outside the length
 // of the video. Video length is {}, requested characterization in frame {}", vid_len, problem_idx)]
	FrameOutOfBounds{
		vid_len: usize,
		problem_idx: usize,
	},

	#[error("Internal frame out of bounds error")]
	FrameOOB,

	
	#[error("unexpected threading error. this will require some more debugging.")]
	ThreadError,
	
	#[error("the passed array is not in standard memory layout. perhaps it is a view of a larger array?")]
	NonStandardArrayLayout,

	
	#[error("could not determine extension of file '{filename}'")]
	NoExtensionError{
		filename: std::path::PathBuf,
	},

	#[error("unsupported file format: {extension}")]
	UnsupportedFileformat{
		extension: String,
	},

	#[error("unsupported characterize array dimensions.
accepted dimensions: (Nx2) or (Nx3). received: {:?}", dims)]
	ArrayDimensionsError{
		dims: Vec<usize>,
	},

	#[error("error in casting image datatype")]
	CastError,

	#[error("error in seeking to the requested image in file")]
	ReadError,

	#[error("filename is not valid utf-8")]
	InvalidFileName{
		filename: PathBuf,
	},

	#[error("the requested channel wasn't found in the file")]
	ChannelNotFound,

	#[error("computational load during linking is abnormally high. try setting a smaller
 search range or track with settings that give fewer detections.")]
	TooDenseToLink,

	#[error("Keyboard Interrupt, my custom version")]
	Interrupted,

	#[error("Future polled after termination")]
	PolledAfterTermination,

	#[error("Tiff write error")]
	TiffWrite,
}

pub type Result<T> = std::result::Result<T, Error>;

