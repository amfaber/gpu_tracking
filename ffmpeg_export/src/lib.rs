use std::{ffi::{CString, c_char}, ptr::{null_mut, null}, path::{Path, PathBuf}};

// use ffmpeg_next::ffi::*;
use ffmpeg_sys_next::*;

#[derive(Debug)]
#[repr(C)]
struct OutputStream {
    st: *mut AVStream,
    enc: *mut AVCodecContext,

    /* pts of the next frame that will be generated */
    next_pts: i64,
    samples_count: i32,

    frame: *mut AVFrame,
    tmp_frame: *mut AVFrame,

    tmp_pkt: *mut AVPacket,

    t: f32,
    tincr: f32,
    tincr2: f32,

    sws_ctx: *mut SwsContext,
}

impl OutputStream {
    fn new() -> Self {
        Self {
            st: std::ptr::null_mut(),
            enc: std::ptr::null_mut(),

            /* pts of the next frame that will be generated */
            next_pts: 0,
            samples_count: 0,

            frame: std::ptr::null_mut(),
            tmp_frame: std::ptr::null_mut(),

            tmp_pkt: std::ptr::null_mut(),

            t: 0.,
            tincr: 0.,
            tincr2: 0.,

            sws_ctx: std::ptr::null_mut(),
        }
    }
}

unsafe fn add_stream(
    ost: &mut OutputStream,
    oc: *mut AVFormatContext,
    codec: &mut *const AVCodec,
    codec_id: AVCodecID,
    width: i32,
    height: i32,
    fps: AVRational,
) -> Result<()> {
    ost.tmp_pkt = av_packet_alloc();
    if ost.tmp_pkt.is_null(){
        return Err(PacketAlloc)
    }

    *codec = avcodec_find_encoder(codec_id);
    if (*codec).is_null(){
        return Err(CodecNotFound)
    }

    ost.st = avformat_new_stream(oc, *codec);
    if ost.st.is_null(){
        return Err(StreamAlloc)
    }

    (*ost.st).id = ((*oc).nb_streams - 1) as _;
    (*(*ost.st).codecpar).codec_id = codec_id;
    // (*(*ost.st).codecpar).codec_type = AVMediaType::AVMEDIA_TYPE_VIDEO;
    
    let c = avcodec_alloc_context3(*codec);
    if c.is_null(){
        return Err(CodecContextAlloc)
    }
    ost.enc = c;

    (*c).codec_id = codec_id;

    (*c).bit_rate = 400000;
    /* Resolution must be a multiple of two. */
    (*c).width = width;
    (*c).height = height;
    /* timebase: This is the fundamental unit of time (in seconds) in terms
     * of which frame timestamps are represented. For fixed-fps content,
     * timebase should be 1/framerate and timestamp increments should be
     * identical to 1. */
    (*(*ost).st).time_base = AVRational { num: fps.den, den: fps.num };
    (*c).time_base = (*(*ost).st).time_base;
    (*c).framerate = fps;


    (*c).gop_size = 12; /* emit one intra frame every twelve frames at most */
    (*c).pix_fmt = AVPixelFormat::AV_PIX_FMT_YUV420P;

    if ((*(*oc).oformat).flags & AVFMT_GLOBALHEADER) != 0 {
        (*c).flags = AV_CODEC_FLAG_GLOBAL_HEADER as _;
    }

    Ok(())
}

unsafe fn alloc_picture(pix_fmt: AVPixelFormat, width: i32, height: i32) -> Result<*mut AVFrame> {
    let picture = av_frame_alloc();
    if picture.is_null(){
        return Err(FrameAlloc)
    }

    (*picture).format = pix_fmt as _;
    (*picture).width = width;
    (*picture).height = height;

    if av_frame_get_buffer(picture, 0) < 0{
        return Err(FrameBufferAlloc)
    }

    Ok(picture)
}

unsafe fn _fill_yuv_image(pict: *mut AVFrame, frame_index: i64, width: i32, height: i32) {
    let i = frame_index as i32;
    for y in 0..height {
        for x in 0..width {
            *(*pict).data[0].add((y * (*pict).linesize[0] + x) as usize) = (x + y + i * 3) as u8;
        }
    }

    for y in 0..height / 2 {
        for x in 0..width / 2 {
            *(*pict).data[1].add((y * (*pict).linesize[1] + x) as usize) = (128 + y + i * 2) as u8;
            *(*pict).data[2].add((y * (*pict).linesize[2] + x) as usize) = (64 + x + i * 5) as u8;
        }
    }
}

// unsafe fn fill_rgba_image(pict: *mut AVFrame, frame_index: i64, width: i32, height: i32) {
//     let i = frame_index as i32;
//     for y in 0..height {
//         for x in 0..width {
//             for c in 0..3 {
//                 *(*pict).data[0].add((y * (*pict).linesize[0] + x * 4 + c) as usize) =
//                     ((x + y + i * 3 + c * 200) & 255) as u8;
//             }
//             *(*pict).data[0].add((y * (*pict).linesize[0] + x * 4 + 3) as usize) = 100;
//         }
//     }
// }

unsafe fn copy_single_plane_image(pict: *mut AVFrame,
    width: i32,
    height: i32,
    data: &[u8],
    size: i32,
 ){
    for y in 0..height {
        for x in 0..width {
            for c in 0..4 {
                let ffmpeg_idx = (y * (*pict).linesize[0] + x * size + c) as usize;
                let plain_idx = (y * width * size + x * size + c) as usize;
                *(*pict).data[0].add(ffmpeg_idx) =
                     data[plain_idx];
            }
        }
    }
}

unsafe fn open_video(
    codec: *const AVCodec,
    ost: &mut OutputStream,
    opt_arg: *const AVDictionary,
    input_format: PixelFormat,
    true_width: i32,
    true_height: i32,
    crf: i32,
) -> Result<()>{
    let mut opt = null_mut();

    if av_dict_copy(&mut opt, opt_arg, 0) < 0{
        return Err(DictCopy)
    }
    
    let key = b"preset\0".as_ptr() as *const c_char;
    let val = b"slow\0".as_ptr() as *const c_char;
    if av_dict_set(&mut opt, key, val, 0) < 0{
        return Err(DictSet)
    }

    
    let key = b"crf\0".as_ptr() as *const c_char;
    if !(0..=51).contains(&crf){
        return Err(InvalidCRF)
    }
    let mut crf = crf.to_string();
    crf.push('\0');
    let val = crf.as_ptr() as *const c_char;
    if av_dict_set(&mut opt, key, val, 0) < 0{
        return Err(DictSet)
    }

    let c = ost.enc;
    if avcodec_open2(c, codec, &mut opt) < 0{
        return Err(CodecOpen)
    }

    av_dict_free(&mut opt);

    ost.frame = alloc_picture((*c).pix_fmt, (*c).width, (*c).height)?;
    ost.tmp_frame = alloc_picture(input_format.to_ffmpeg(), true_width, true_height)?;

    if avcodec_parameters_from_context((*(*ost).st).codecpar, c) < 0{
        return Err(ParametersFromContext)
    }
    Ok(())
}

unsafe fn get_video_frame(ost: &mut OutputStream, data: Option<&[u8]>, format: PixelFormat, true_width: i32, true_height: i32) -> Result<*mut AVFrame>{
    let c = ost.enc;

    let Some(data) = data else {
        return Ok(null_mut())
    };
    // if av_compare_ts(
    //     ost.next_pts,
    //     (*c).time_base,
    //     STREAM_DURATION,
    //     AVRational { num: 1, den: 1 },
    // ) > 0
    // {
    //     return null_mut();
    // }

    if av_frame_make_writable(ost.frame) < 0{
        return Err(MakeWritable)
    }

    let ffmpeg_format = format.to_ffmpeg();
    // if (*c).pix_fmt != ffmpeg_format {
    if ost.sws_ctx.is_null() {
        ost.sws_ctx = sws_getContext(
            true_width,
            true_height,
            ffmpeg_format,
            (*c).width,
            (*c).height,
            (*c).pix_fmt,
            SWS_BICUBIC,
            null_mut(),
            null_mut(),
            null_mut(),
        );
        if ost.sws_ctx.is_null(){
            return Err(ConversionContextAlloc)
        }
    }

    match format {
        // AVPixelFormat::AV_PIX_FMT_YUV420P => {
        //     fill_yuv_image(ost.tmp_frame, ost.next_pts, (*c).width, (*c).height)
        // }
        PixelFormat::RGBA => {
            copy_single_plane_image(ost.tmp_frame, true_width, true_height, data, format.size())
        }
        // _ => todo!(),
    }
    sws_scale(
        ost.sws_ctx,
        (*ost.tmp_frame).data.as_ptr() as *const *const u8,
        (*ost.tmp_frame).linesize.as_ptr(),
        0,
        true_height,
        (*ost.frame).data.as_ptr(),
        (*ost.frame).linesize.as_ptr(),
    );
    // } else {
    //     fill_yuv_image(ost.frame, ost.next_pts, (*c).width, (*c).height);
    // }

    (*ost.frame).pts = ost.next_pts;
    ost.next_pts += 1;

    Ok(ost.frame)
}

// unsafe fn log_packet(fmt_ctx: *mut AVFormatContext, pkt: *mut AVPacket) {
//     let time_base = (**(*fmt_ctx).streams.add((*pkt).stream_index as _)).time_base;
//     let float_time = time_base.num as f64 / time_base.den as f64;
//     println!(
//         "pts:{} pts_time:{} dts:{} dts_time:{} duration:{} duration_time:{} stream_index:{}\n",
//         (*pkt).pts,
//         (*pkt).pts as f64 * float_time,
//         (*pkt).dts,
//         (*pkt).dts as f64 * float_time,
//         (*pkt).duration,
//         (*pkt).duration as f64 * float_time,
//         (*pkt).stream_index
//     );
// }


unsafe fn write_frame(
    oc: *mut AVFormatContext,
    ost: &mut OutputStream,
    data: Option<&[u8]>,
    format: PixelFormat,
    true_width: i32,
    true_height: i32,
) -> Result<()> {
    let fmt_ctx = oc;
    let c = ost.enc;
    let st = ost.st;
    let frame = get_video_frame(ost, data, format, true_width, true_height)?;
    let pkt = ost.tmp_pkt;

    if avcodec_send_frame(c, frame) != 0{
        return Err(SendFrame)
    }

    loop {
        let ret = avcodec_receive_packet(c, pkt);
        if ret == AVERROR(EAGAIN) || ret == AVERROR_EOF {
            break;
        } else if ret < 0 {
            return Err(ReceivePacket)
        }

        av_packet_rescale_ts(pkt, (*c).time_base, (*st).time_base);
        (*pkt).stream_index = (*st).index;

        // log_packet(fmt_ctx, pkt);
        if av_interleaved_write_frame(fmt_ctx, pkt) < 0{
            return Err(WritePacket)
        }
    };

    Ok(())
}

// pub unsafe fn mux_main() -> std::result::Result<(), ()> {

//     let mut oc = std::ptr::null_mut();
//     let path =
//         CString::new(r"C:\Users\andre\Documents\gpu_tracking_testing\ffmpeg-adventures\test.mp4")
//             .unwrap();

//     let format = CString::new("H264").unwrap();
//     // let format = null();
//     avformat_alloc_output_context2(&mut oc, std::ptr::null(), format.as_ptr(), path.as_ptr());

//     let mut video_codec = std::ptr::null();

//     let fmt = (*oc).oformat;

//     let mut ost = OutputStream::new();
//     add_stream(&mut ost, oc, &mut video_codec, (*fmt).video_codec, 400, 200, AVRational{num: 25, den: 1}).unwrap();

//     let mut opt = std::ptr::null_mut();
//     open_video(video_codec, &mut ost, opt, ).unwrap();

//     av_dump_format(oc, 0, path.as_ptr(), 1);

//     avio_open(&mut (*oc).pb, path.as_ptr(), AVIO_FLAG_WRITE);

//     avformat_write_header(oc, &mut opt);

//     // loop {
//     //     if write_video_frame(oc, &mut ost) {
//     //         break;
//     //     }
//     // }

//     av_write_trailer(oc);
//     if ((*fmt).flags & AVFMT_NOFILE) != 0 {
//         avio_closep(&mut (*oc).pb);
//     }

//     avformat_free_context(oc);

//     Ok(())
// }

type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error,)]
pub enum Error{
    #[error("Couldn't allocate an output context")]
    OutputContextAlloc,
    #[error("Couldn't allocate a codec context")]
    CodecContextAlloc,
    #[error("Couldn't convert to CString")]
    CStringConvert,
    #[error("Couldn't allocate packet")]
    PacketAlloc,

    #[error("Couldn't find codec")]
    CodecNotFound,

    #[error("Couldn't allocate stream")]
    StreamAlloc,

    #[error("Couldn't copy dictionary in FFMPEG")]
    DictCopy,

    #[error("Couldn't set dictionary value in FFMPEG")]
    DictSet,

    #[error("CRF has to in [0; 51]")]
    InvalidCRF,

    #[error("Couldn't open codec")]
    CodecOpen,

    #[error("Couldn't allocate frame")]
    FrameAlloc,

    #[error("Couldn't allocate frame buffer")]
    FrameBufferAlloc,

    #[error("Error in avcodec_parameters_from_context")]
    ParametersFromContext,

    #[error("Error opening and writing to {0}")]
    AVOpen(PathBuf),

    #[error("Error writing header to {0}")]
    HeaderWrite(PathBuf),

    #[error("Error sending frame")]
    SendFrame,

    #[error("Error sending frame")]
    ReceivePacket,

    #[error("Error writing packet to output file")]
    WritePacket,

    #[error("Error making frame writable")]
    MakeWritable,

    #[error("Couldn't allocate conversion context")]
    ConversionContextAlloc,

    #[error("Couldn't find the specified codec")]
    FindCodec,

    #[error("The passed data must have length=width*height")]
    InvalidDataLen
    
}
use Error::*;

#[derive(Clone, Copy, Debug)]
pub enum PixelFormat{
    RGBA,
}

impl PixelFormat{
    fn to_ffmpeg(&self) -> AVPixelFormat{
        match self{
            Self::RGBA => AVPixelFormat::AV_PIX_FMT_RGBA,
        }
    }

    // Size of the data format in bytes. RGBA is 4 for example
    fn size(&self) -> i32{
        match self{
            Self::RGBA => 4,
        }
    }
}

pub struct Rational{
    num: i32,
    den: i32,
}

impl Rational{
    fn to_ffmpeg(&self) -> AVRational{
        AVRational{
            num: self.num,
            den: self.den,
        }
    }
}

impl From<i32> for Rational{
    fn from(value: i32) -> Self {
        Self{
            num: value,
            den: 1,
        }
    }
}

impl From<f32> for Rational{
    fn from(value: f32) -> Self {
        if (-1. < value) && (value < 1.){
            Self{
                num: 1,
                den: (1./value) as _,
            }
        } else {
            Self{
                num: value as _,
                den: 1,
            }
        }
    }
}

#[derive(Debug)]
pub struct VideoEncoder{
    oc: *mut AVFormatContext,
    // opt: *mut AVDictionary,
    ost: OutputStream,
    input_format: PixelFormat,
    pub height: i32,
    pub width: i32,
}

fn to_cstring<P: AsRef<Path>>(inp: P) -> Result<CString>{
    inp
        .as_ref()
        .to_str()
        .and_then(|s| CString::new(s).ok())
        .ok_or(CStringConvert)
}

impl VideoEncoder{
    pub fn new<P: AsRef<Path>, R: Into<Rational>>(
        width: i32,
        height: i32,
        input_pixel_format: PixelFormat, 
        output_path: P,
        fps: R,
        crf: Option<i32>,
    ) -> Result<Self>{
        unsafe{
            let fps: Rational = fps.into();
            let fps = fps.to_ffmpeg();
            let path = output_path.as_ref().to_owned();
            let cpath = to_cstring(&path)?;
            let mut oc = std::ptr::null_mut();
            if avformat_alloc_output_context2(&mut oc, null(), null(), cpath.as_ptr()) < 0{
                return Err(OutputContextAlloc)
            }
            let mut video_codec = std::ptr::null();
            let codec_id = AVCodecID::AV_CODEC_ID_H264;

            let mut ost = OutputStream::new();
            add_stream(
                &mut ost,
                oc,
                &mut video_codec,
                codec_id,
                width + width % 2,
                height + height % 2,
                fps
            )?;

            let mut opt = std::ptr::null_mut();

            open_video(video_codec, &mut ost, opt, input_pixel_format, width, height, crf.unwrap_or(10))?;

            if avio_open(&mut (*oc).pb, cpath.as_ptr(), AVIO_FLAG_WRITE) < 0{
                return Err(AVOpen(path))
            }

            if avformat_write_header(oc, &mut opt) < 0{
                return Err(HeaderWrite(path))
            }

            Ok(Self{
                oc,
                ost,
                input_format: input_pixel_format,
                height,
                width,
            })
        }
    }
    pub fn encode_frame(&mut self, data: &[u8]) -> Result<()>{
        if data.len() != (self.width * self.height * 4) as usize{
            return Err(InvalidDataLen)
        }
        unsafe{
            write_frame(self.oc, &mut self.ost, Some(data), self.input_format, self.width, self.height)?;
        }
        Ok(())
    }

    pub fn finish(self){}
}

impl Drop for VideoEncoder{
    fn drop(&mut self) {
        unsafe{
            let _ = write_frame(self.oc, &mut self.ost, None, self.input_format, self.width, self.height);
            av_write_trailer(self.oc);
            if ((*(*self.oc).oformat).flags & AVFMT_NOFILE) != 0 {
                avio_closep(&mut (*self.oc).pb);
            }

            avformat_free_context(self.oc);
        }
    }
}


#[cfg(test)]
mod tests{
    use super::*;

    #[test]
    pub fn encoder_main(){
        let mut enc = VideoEncoder::new(
            500, 600, PixelFormat::RGBA, "h264.mp4", 10, None
        ).unwrap();

        let mut data = vec![0; (enc.height * enc.width * 4) as usize];
        let n_frames = 100;
        for i in 0..n_frames{
            for y in 0..enc.height {
                for x in 0..enc.width {
                    // data[(y * enc.width * 4 + x * 4 + 1) as usize] = 255
                    for c in 0..3 {
                        data[(y * enc.width * 4 + x * 4 + c) as usize] =
                            ((x + y + i * 3 + c * 200) & 255) as u8;
                    }
                    data[(y * enc.width + x * 4 + 3) as usize] = 100;
                }
            }
            enc.encode_frame(&data).unwrap();
        }
        enc.finish();
    }

    // #[test]
    // pub fn h264(){
    //     // println!("what");
    //     std::io::stdout().flush().unwrap();
    //     let mut enc = VideoEncoder::new_h264(
    //         500, 600, PixelFormat::RGBA, "h264.mp4", 10
    //     ).unwrap();

    //     let mut data = vec![0; (enc.height * enc.width * 4) as usize];
    //     let n_frames = 100;
    //     for i in 0..n_frames{
    //         for y in 0..enc.height {
    //             for x in 0..enc.width {
    //                 // data[(y * enc.width * 4 + x * 4 + 1) as usize] = 255
    //                 for c in 0..3 {
    //                     data[(y * enc.width * 4 + x * 4 + c) as usize] =
    //                         ((x + y + i * 3 + c * 200) & 255) as u8;
    //                 }
    //                 data[(y * enc.width + x * 4 + 3) as usize] = 100;
    //             }
    //         }
    //         enc.encode_frame(&data).unwrap();
    //     }
    //     enc.finish();
    // }


    #[test]
    fn libx264_attempts(){
        unsafe{
            let mut codec_ptr;
            let mut iter_state = null_mut();

            println!("Available codecs:");

            let mut file = std::fs::File::create("output.txt").unwrap();

            use std::io::Write;
            use std::ffi::CStr;

            while {
                codec_ptr = av_codec_iterate(&mut iter_state);
                !codec_ptr.is_null()
            } {
                let codec = &*codec_ptr;
                // if !avcodec_find_encoder_by_name(codec.name).is_null() {
                    let codec_name = CStr::from_ptr(codec.name).to_str().unwrap_or("Unknown");
                    writeln!(&mut file, "{}", codec_name).unwrap();
                // }
            }
        }
    }
}
