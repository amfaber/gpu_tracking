use tiff::decoder::{Decoder, DecodingResult::*};
use ndarray::{ArrayView3, ArrayView2, Axis};
use crate::{my_dtype, into_slice::IntoSlice, error::Error};
use byteorder::{ReadBytesExt, LittleEndian};
use std::{io::{self, Read, Seek, SeekFrom}, cell::RefCell};
use std::collections::HashMap;

// fn cast_vec_to_f32<T: num_traits::NumCast>(to_cast: Vec<T>) -> Result<Vec<my_dtype>, Error>{
//     let converted = to_cast.into_iter().map(|datum| {num_traits::cast(datum).ok_or(Error::CastError)}).collect();
//     converted
// }

fn cast_vec_to_f32<T: Into<my_dtype> + Copy>(res: Vec<T>) -> Vec<my_dtype>{
    let data = res.iter().map(|&x| x.into()).collect::<Vec<my_dtype>>();
    data
}

pub trait FrameProvider{
    type Frame: IntoSlice;
    type FrameIter: Iterator<Item = Result<Self::Frame, Error>>;
    // type IntoIter: Iterator<Item = Self::Frame>;
    fn get_frame(&self, frame_idx: usize) -> Result<Self::Frame, Error>;
    fn len(&self, too_high: Option<usize>) -> usize;
    fn light_len(&self) -> Option<usize>;
    fn into_iter(self: Box<Self>) -> Self::FrameIter;
    // fn into_iter(self) -> Box<dyn Iterator<Item = Result<Self::Frame, Error>>>;
}

impl<F: IntoSlice, I: Iterator<Item = Result<F, Error>>> FrameProvider for Box<dyn FrameProvider<Frame = F, FrameIter = I>>{
    type Frame = F;
    type FrameIter = I;
    fn get_frame(&self, frame_idx: usize) -> Result<Self::Frame, Error> {
        // T::get_frame(self, frame_idx)
        (**self).get_frame(frame_idx)
    }
    fn len(&self, too_high: Option<usize>) -> usize{
        (**self).len(too_high)
    }
    fn light_len(&self) -> Option<usize>{
        (**self).light_len()
    }
    fn into_iter(self: Box<Self>) -> I{
    // fn into_iter(self) -> Box<dyn Iterator<Item = Result<Self::Frame, Error>>>{
        (*self).into_iter()
        // Box::new(T::into_iter(*self))
    }
    
}

// #[derive(Debug)]
// pub enum GetFrameError{
    // OutOfBounds{
    //     vidlen,
        
    // },
//     ReadError,
//     CastError
// }

impl<R: Read + Seek + 'static> FrameProvider for RefCell<Decoder<R>>{
    type Frame = Vec<my_dtype>;
    type FrameIter = Box<dyn Iterator<Item = Result<Self::Frame, Error>>>;
    // type FrameIter = IterDecoder<R>;
    // type frame = Vec<my_dtype>;
    fn get_frame(&self, frame_idx: usize) -> Result<Vec<my_dtype>, Error>{
        // dbg!("can i get a frame?");
        let seek_result = self.borrow_mut().seek_to_image(frame_idx);
        seek_result.map_err(|_| Error::FrameOOB)?;
        let result = self.borrow_mut().read_image().map_err(|_| Error::ReadError)?;
        let out = match result{
            U8(vec) => Ok(cast_vec_to_f32(vec)),
            U16(vec) => Ok(cast_vec_to_f32(vec)),
            I8(vec) => Ok(cast_vec_to_f32(vec)),
            I16(vec) => Ok(cast_vec_to_f32(vec)),
            F32(vec) => Ok(cast_vec_to_f32(vec)),
            
            U32(_) => Err(Error::CastError)?,
            U64(_) => Err(Error::CastError)?,
            F64(_) => Err(Error::CastError)?,
            I32(_) => Err(Error::CastError)?,
            I64(_) => Err(Error::CastError)?,
        };
        out
    }

    fn len(&self, too_high: Option<usize>) -> usize{
        let mut lo = 0;
        let mut hi = match too_high{
            Some(hi) => hi,
            None => {
                let mut hi = 1024;
                loop{
                    match self.borrow_mut().seek_to_image(hi){
                        Ok(_) => hi = hi * 2,
                        Err(_) => break,
                    }
                }
                hi
            }
        };
        let mut mid = (hi - lo) / 2;
        while mid != lo{
            match self.borrow_mut().seek_to_image(mid){
                Ok(_) => lo = mid,
                Err(_) => hi = mid,
            }
            mid = lo + (hi - lo) / 2;
        }
        hi
    }

    fn light_len(&self) -> Option<usize>{
        None
    }

    fn into_iter(self: Box<Self>) -> Self::FrameIter{
        let iter = IterDecoder::from(self.into_inner());
        Box::new(iter)
    }
}

impl<R: Read + Seek + 'static> FrameProvider for RefCell<ETSIterator<R>>{
    type Frame = Vec<my_dtype>;
    type FrameIter = Box<dyn Iterator<Item = Result<Self::Frame, Error>>>;
    // type FrameIter = ETSIterator<R>;
    // type frame = Vec<my_dtype>;
    fn get_frame(&self, frame_idx: usize) -> Result<Self::Frame, Error>{
        self.borrow_mut().seek(frame_idx)?;
        let result = self.borrow_mut().next().unwrap()?;
        let out = cast_vec_to_f32(result);
        Ok(out)
    }

    fn len(&self, _too_high: Option<usize>) -> usize{
        self.borrow().offsets.len()
    }

    fn light_len(&self) -> Option<usize>{
        Some(self.len(None))
    }

    fn into_iter(self: Box<Self>) -> Box<dyn Iterator<Item = Result<Self::Frame, Error>>>{
        let mut inner = self.into_inner();
        inner.seek(0).unwrap();
        let out = inner.map(|res_image| {
            res_image.map(|image| {
                let idk1 = image.iter().map(|&pixel| pixel as f32);
                let idk: Vec<_> = idk1.collect();
                idk
                })
            });
        Box::new(out)
    }
}


// struct LifeTimeWrap<'a, 'b>(&'b ArrayView3<'a, my_dtype>);

// impl <'a, 'b> FrameProvider<'a> for LifeTimeWrap<'a, 'b>{
//     type frame = ArrayView2<'a, my_dtype>;
//     fn get_frame(&self, frame_index: usize) -> Result<Self::frame, Error>{
//         // let self = &*self;
//         let n_frames = self.0.shape()[0];
//         if frame_index >= n_frames{
//             return Err(Error::FrameOutOfBounds { vid_len: n_frames, problem_idx: frame_index })
//         }
//         Ok(self.0.index_axis(Axis(0), frame_index))
//     }
// }

impl<'a, 'b: 'a> FrameProvider for &'b ArrayView3<'a, my_dtype>{
    type Frame = ArrayView2<'a, my_dtype>;
    type FrameIter = Box<dyn Iterator<Item = Result<Self::Frame, Error>> + 'a>;
    // type IntoIter = ndarray::iter::AxisIter<'b, f32, ndarray::Dim<[usize; 2]>>;
    // type frame = ArrayView2<'a, my_dtype>;
    // type frame = &'a[my_dtype];
    fn get_frame(&self, frame_index: usize) -> Result<ArrayView2<'a, my_dtype>, Error>{
        // let self = &*self;
        let n_frames = self.shape()[0];
        if frame_index >= n_frames{
            return Err(Error::FrameOOB)
        }
        Ok(self.index_axis(Axis(0), frame_index))
    }

    fn len(&self, _too_high: Option<usize>) -> usize{
        self.shape()[0]
    }
    
    fn light_len(&self) -> Option<usize>{
        Some(self.len(None))
    }
    
    fn into_iter(self: Box<Self>) -> Box<dyn Iterator<Item = Result<Self::Frame, Error>> + 'a>{
        let iter = self.axis_iter(Axis(0)).map(|frame| Ok(frame));
        Box::new(iter)
    }
}


pub struct IterDecoder<R: std::io::Read + std::io::Seek>{
    decoder: Decoder<R>,
    // pub dims: [u32; 2],
    pub dims: (usize, usize),
    first: bool
}


impl <R: std::io::Read + std::io::Seek> From::<Decoder<R>> for IterDecoder<R>{
    fn from(mut decoder: Decoder<R>) -> IterDecoder<R>{
        let (width, height) = decoder.dimensions().unwrap();
        // let dims = [height, width];
        let dims = (height as usize, width as usize);
        IterDecoder {
            decoder,
            dims,
            first: true
        }
    }
}

impl<R: std::io::Read + std::io::Seek> IterDecoder<R>{
    fn _treat_data<T: Into<my_dtype> + Copy>(&self, res: Vec<T>) -> Vec<my_dtype>{
        let data = res.iter().map(|&x| x.into()).collect::<Vec<my_dtype>>();
        data
    }
    // fn _treat_data<T: num_traits::NumCast + Copy>(&self, to_cast: Vec<T>) -> Vec<my_dtype>{
    //     let converted = to_cast.iter().map(|&datum| {num_traits::cast(datum).ok_or(Error::CastError)}).collect::<Result<Vec<_>, _>>().unwrap();
    //     converted
    // }
}

impl <R: std::io::Read + std::io::Seek> Iterator for IterDecoder<R>{
    type Item = Result<Vec<my_dtype>, Error>;
    fn next(&mut self) -> Option<Self::Item>{
        self.next_image()?;
        let res = self.decoder.read_image();
        match res{
            Ok(U8(vec)) => Some(Ok(cast_vec_to_f32(vec))),
            Ok(U16(vec)) => Some(Ok(cast_vec_to_f32(vec))),
            Ok(I8(vec)) => Some(Ok(cast_vec_to_f32(vec))),
            Ok(I16(vec)) => Some(Ok(cast_vec_to_f32(vec))),
            Ok(F32(vec)) => Some(Ok(cast_vec_to_f32(vec))),
            
            Ok(U32(_)) => Some(Err(Error::CastError)),
            Ok(U64(_)) => Some(Err(Error::CastError)),
            Ok(F64(_)) => Some(Err(Error::CastError)),
            Ok(I32(_)) => Some(Err(Error::CastError)),
            Ok(I64(_)) => Some(Err(Error::CastError)),
            Err(_) => return Some(Err(Error::ReadError))
        }
    }
}
impl<R: Read + Seek> IterDecoder<R>{
    fn next_image(&mut self) -> Option<()>{
        if self.first{
            self.first = false;
        } else {
            self.decoder.next_image().ok()?;
        }
        Some(())
    }
}


// use bencher::black_box;

#[derive(Debug, Clone)]
pub struct MinimalETSParser{//<R: Read + Seek>{
    // pub reader: R,
    pub offsets: HashMap<usize, Vec<Option<u64>>>,
    pub dims: [usize; 2],
    data_type: ETSDataType,
}

pub enum ETSDataBuffer<'a>{
    I8(&'a mut [i8]),
    U8(&'a mut [u8]),
    I16(&'a mut [i16]),
    U16(&'a mut [u16]),
    I32(&'a mut [i32]),
    U32(&'a mut [u32]),
    F32(&'a mut [f32]),
}

#[derive(Debug, Clone, Copy)]

enum ETSDataType {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
}

impl ETSDataType {
    pub fn size(&self) -> usize {
        match self {
            ETSDataType::I8 => 1,
            ETSDataType::U8 => 1,
            ETSDataType::I16 => 2,
            ETSDataType::U16 => 2,
            ETSDataType::I32 => 4,
            ETSDataType::U32 => 4,
            ETSDataType::F32 => 4,
        }
    }
    pub fn from_code(code: u32) -> ETSDataType {
        match code {
            1 => ETSDataType::I8,
            2 => ETSDataType::U8,
            3 => ETSDataType::I16,
            4 => ETSDataType::U16,
            5 => ETSDataType::I32,
            6 => ETSDataType::U32,
            9 => ETSDataType::F32,
            _ => panic!("Unknown ETS data type code: {}", code),
        }
    }
}


impl MinimalETSParser{
    pub fn new<R: Read + Seek>(reader: &mut R) -> io::Result<Self>{
        reader.seek(SeekFrom::Start(0))?;
        let mut buf4 = vec![0; 4];
        // let mut buf8 = vec![0; 8];
        reader.read_exact(&mut buf4)?;
        assert_eq!("SIS\x00".as_bytes(), &buf4[..]);
        reader.seek(SeekFrom::Current(8))?;
        let ndimensions = reader.read_u32::<LittleEndian>()?;
        let additional_header_offset = reader.read_u64::<LittleEndian>()?;
        reader.seek(SeekFrom::Current(8))?;
        let used_chunk_offset = reader.read_u64::<LittleEndian>()?;
        let n_used_chunks = reader.read_u32::<LittleEndian>()?;
        
        reader.seek(SeekFrom::Start(additional_header_offset.into()))?;
        reader.read_exact(&mut buf4)?;
        assert_eq!("ETS\x00".as_bytes(), &buf4[..]);
        reader.seek(SeekFrom::Current(4))?;

        let data_type = ETSDataType::from_code(reader.read_u32::<LittleEndian>()?);

        reader.seek(SeekFrom::Current(16))?;
        let dims = [reader.read_u32::<LittleEndian>()? as usize, reader.read_u32::<LittleEndian>()? as usize];

        reader.seek(SeekFrom::Start(used_chunk_offset))?;
        let mut offsets = HashMap::new();
        for _ in 0..n_used_chunks{
            reader.seek(SeekFrom::Current(4))?;
            let coord = (0..ndimensions).map(|_| reader.read_u32::<LittleEndian>().unwrap() as usize).collect::<Vec<_>>();
            let entry = offsets.entry(coord[4]).or_insert(Vec::new());
            if coord[2] + 1 == entry.len(){
                entry.push(Some(reader.read_u64::<LittleEndian>().unwrap()));
            }
            else if coord[2] + 1 > entry.len(){
                entry.resize(coord[2] + 1, None);
                entry[coord[2]] = Some(reader.read_u64::<LittleEndian>().unwrap());
            }
            else {
                entry[coord[2]] = Some(reader.read_u64::<LittleEndian>().unwrap());
            }
            reader.seek(SeekFrom::Current(8))?;
        }

        let parser = MinimalETSParser{
            // reader,
            offsets,
            dims,
            data_type,
        };
        Ok(parser)
    }

    // pub fn u32_with_buf(reader: &mut R, buf: &mut [u8]) -> io::Result<u32> {
    //     reader.read_exact(buf)?;
    //     Ok(u32::from_le_bytes(buf.try_into().unwrap()))
    // }

    pub fn iterate_channel<R: Read + Seek>(&self, reader: R, channel: usize) -> crate::error::Result<ETSIterator<R>>{
        let read_size = self.dims[0] * self.dims[1] * self.data_type.size();
        let offsets = self.offsets.get(&channel).ok_or(crate::error::Error::ChannelNotFound)?.clone();
        Ok(ETSIterator{
            reader,
            // parser: self.reader,
            current: 0,
            offsets,
            read_size,
        })
    }
}



pub struct ETSIterator<R: Read + Seek>{
    reader: R,
    pub current: usize,
    pub offsets: Vec<Option<u64>>,
    read_size: usize,
}

impl<R: Read + Seek> ETSIterator<R>{
    pub fn new(reader: R, offsets: Vec<Option<u64>>, read_size: usize) -> Self{
        ETSIterator{
            reader,
            current: 0,
            offsets,
            read_size,
        }
    }

    pub fn len(&self) -> usize{
        self.offsets.iter().flatten().count()
    }

    pub fn seek(&mut self, to: usize) -> crate::error::Result<()>{
        match self.offsets.get(to){
            Some(Some(_val)) => { self.current = to; Ok(()) },
            Some(None) => Err(Error::ReadError),
            None => Err(Error::FrameOutOfBounds { vid_len: self.offsets.len(), problem_idx: to }),
        }
    }
}

impl<R: Read + Seek> Iterator for ETSIterator<R>{
    type Item = crate::error::Result<Vec<u16>>;
    fn next(&mut self) -> Option<Self::Item> {
        let offset = self.offsets.get(self.current)?;
        match offset{
            Some(offset) => {
                self.reader.seek(SeekFrom::Start(*offset)).unwrap();
                
                #[cfg(target_endian = "big")]
                let mut buf = vec![0u8; self.read_size];
                #[cfg(target_endian = "little")]
                let mut buf = vec![0u16; self.read_size / 2];
                // println!("u16 buf.len {}", buf.len());
                #[cfg(target_endian = "little")]
                let byte_buf = unsafe{std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, buf.len() * 2)};
                // println!("u8 buf.len {}", buf.len());
                
                self.reader.read_exact(byte_buf).unwrap();
                
                #[cfg(target_endian = "big")]
                let buf = buf.chunks_exact(2).map(|chunk| {
                    u16::from_le_bytes(chunk.try_into().unwrap())
                }).collect::<Vec<_>>();
                self.current += 1;
                Some(Ok(buf))
            },
            None => {
                self.current += 1;
                Some(Err(Error::ReadError))
            }
        }
    }
}