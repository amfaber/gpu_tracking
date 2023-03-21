// #![allow(warnings)]
#![allow(unused)]
#![allow(non_camel_case_types)]
use futures;
use futures_intrusive;
// use gpu_tracking::execute_gpu::{path_to_iter, ProgressFuture, PollResult};
use gpu_tracking::gpu_setup::{ParamStyle, setup_state};
use gpu_tracking::linking::FrameSubsetter;
use gpu_tracking::progressfuture::{ProgressFuture, PollResult};
use gpu_tracking::{
    decoderiter::{IterDecoder, FrameProvider},
    execute_gpu::{
        self, execute_gpu, execute_ndarray, path_to_iter,
    },
    gpu_setup::TrackingParams,
};
use ndarray::{s, Array2, Axis, Array, Array3};
use pollster::FutureExt;
use std::cell::RefCell;
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::{fs, time::Instant};
use tiff::decoder::{Decoder, DecodingResult};
pub type my_dtype = f32;
use clap::Parser;
use std::path;
use ctrlc;
use indicatif;

#[derive(Parser, Debug)]
struct Args {
    // #[arg(short, long)]
    input: Option<String>,
    #[arg(short, long)]
    debug: Option<bool>,
    #[arg(short, long)]
    filter: Option<bool>,
    #[arg(short, long)]
    characterize: Option<bool>,
    // #[arg(short, long)]
    // processed_cpu: Option<bool>,
}


fn test_trackpy_easy() -> gpu_tracking::error::Result<()>{
    let atomic = Arc::new(AtomicBool::new(false));
    let interrupted = atomic.clone();
    ctrlc::set_handler(move || interrupted.store(true, Ordering::Relaxed))
        .expect("Couldn't set ctrl-c handler");
    let args: Args = Args::parse();
    let now_top = Instant::now();
    // let path = args.input.unwrap_or("testing/frank_freeze/frame_t_0.ets".to_string());
    // let path = args.input.unwrap_or(r"C:\Users\andre\Documents\tracking_optimizations\gpu-tracking\testing\marcus_blobs\big_blobs.tif".to_string());
    // let path = args.input.unwrap_or(r"C:\Users\andre\Documents\tracking_optimizations\gpu-tracking\testing\kdtree_panic\emily_she_kdtree_panic.tif".to_string());
    // let path = args.input.unwrap_or(r"C:\Users\andre\Documents\tracking_optimizations\gpu-tracking\tiff_vsi\vsi dummy\_Process_9747_\stack1\frame_t_0.ets".to_string());
    // let path = args.input.unwrap_or(r"C:\Users\andre\Documents\tracking_optimizations\gpu-tracking\tiff_vsi\vsi dummy\Process_9747.vsi".to_string());
    
    let path = args.input.unwrap_or("testing/easy_test_data.tif".to_string());
    let debug = args.debug.unwrap_or(false);
    let filter = args.filter.unwrap_or(true);
    let characterize = args.characterize.unwrap_or(false);
    // let file = fs::File::open(&path).expect("didn't find the file");
    // let mut decoder = Decoder::new(file).expect("Can't create decoder");
    // let (width, height) = decoder.dimensions().unwrap();
    // let dims = [height, width];
    // let mut decoderiter = IterDecoder::from(decoder).take(10);
    let params = TrackingParams {
        style: ParamStyle::Trackpy {
            separation: 10,
            diameter: 9,
            maxsize: 0.0,
            threshold: 0.0,
            invert: false,
            percentile: 0.,
            topn: 0,
            preprocess: true,
            filter_close: true,
        },
        noise_size: 1.,
        // style: ParamStyle::Log{
        //     min_radius: 3.0,
        //     max_radius: 25.0,
        //     log_spacing: true,
        //     overlap_threshold: 0.5,
        //     n_radii: 10,
        // },
        snr: Some(1.5),
        minmass_snr: Some(0.3),
        // adaptive_background: Some(4),
        characterize: true,
        // illumination_sigma: Some(30.),
        search_range: Some(10.),
        doughnut_correction: true,
        
        // include_r_in_output: true,
        // truncate_preprocessed: true,
        // keys: Some(vec![0, 1, 2, 3, 4, 5]),
        ..Default::default()
    };
    let now = Instant::now();
    let points = vec![
        0f32, 300f32, 300f32, 5f32,
        0f32, 200f32, 200f32, 5f32,
        // 2000f32, 200f32, 200f32,
        // 10000f32, 200f32, 200f32,
        // 200f32, 300f32, 300f32,
        // 1999f32, 200f32, 200f32,
    ];
    let width = 4;
    let points = Array2::from_shape_vec((points.len()/4, 4), points).unwrap();
    // let point_view = points.view();
    // let point_iter = FrameSubsetter::new(&point_view, Some(0), (1, 2));
    // let tmp = points.view();
    // let inp = Some(FrameSubsetter::new(tmp, Some(0), (1, 2), Some(3), gpu_tracking::linking::SubsetterType::Characterization::Characterization));
    let (provider, dims) = path_to_iter(&path, None)?;
    let array = provider.into_iter().flat_map(|frame| frame.unwrap().into_iter()).collect::<Vec<_>>();
    let array = Array::from_shape_vec([array.len() / (dims[0] * dims[1]) as usize, dims[0] as usize, dims[1] as usize], array).unwrap();

    let now = Instant::now();
    let (results, column_names) = if false{
        execute_gpu::execute_file(
            &path,
            None,
            params,
            1,
            // Some((points.view(), true, true)),
            // Some(&points.view()),
            None::<_>,
            None,
            None,
        )?
    } else {
        let mut future = ProgressFuture::from_interrupt_signal(atomic, |job, progress, interruption|{
            let (array, params, channel): (Array3<f32>, _, Option<usize>)= job;
            execute_gpu::execute_ndarray(
                &array.view(),
                params,
                0,
                None::<_>,
                Some(interruption),
                Some(progress),
            )
        });

        future.submit_same((array, params, None));

    let bar = indicatif::ProgressBar::with_draw_target(None, indicatif::ProgressDrawTarget::stderr());
        let idk = loop{
            match future.poll().unwrap(){
                PollResult::Done(res) => break res,
                PollResult::Pending(prog) => {
                    bar.set_position(prog.0 as u64);
                    // bar.set_message(prog.0.to_string());
                    // if prog >= 1000{
                    //     break match future.interrupt(){
                    //         PollResult::Done(res) => res,
                    //         _ => panic!("oof2")
                    //     }
                    // }
                },
                _ => panic!("Oof")
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }?;
        // bar.set_message("");
        bar.finish_and_clear();
        let prog = future.progress.lock().unwrap();
        println!("progress = {}, {}", prog.0, prog.1.unwrap_or(0));
        println!("interrupt = {}", future.interrupter.load(Ordering::Relaxed));
        idk
    };



    
    let function_time = now.elapsed().as_millis() as f64 / 1000.;
    dbg!(function_time);
    dbg!(&results.shape());
    dbg!(&results.lanes(Axis(1)).into_iter().next().unwrap());
    dbg!(&results.slice(s![0..12, ..]));

    Ok(())
}

fn main(){
    dbg!(test_trackpy_easy().unwrap());
}


fn test_unedited() -> gpu_tracking::error::Result<()>{
    let args: Args = Args::parse();
    let now_top = Instant::now();
    let path = args.input.unwrap_or("testing/easy_test_data.tif".to_string());
    // let path = args
    //     .input
    //     .unwrap_or("testing/scuffed_blobs_1C_even_dims.tif".to_string());
    let debug = args.debug.unwrap_or(false);
    let filter = args.filter.unwrap_or(true);
    let characterize = args.characterize.unwrap_or(false);
    // let processed_cpu = args.processed_cpu.unwrap_or(false);
    let file = fs::File::open(&path).expect("didn't find the file");
    let mut decoder = Decoder::new(file).expect("Can't create decoder");
    let (width, height) = decoder.dimensions().unwrap();
    let dims = [height, width];
    // // // dbg!(dims);
    let mut decoderiter = IterDecoder::from(decoder);
    // let all_frames = decoderiter.collect::<Vec<_>>();
    // let all_views = all_frames.iter().map(|x| x.view()).collect::<Vec<_>>();
    // let arr = ndarray::stack(ndarray::Axis(0), &all_views).unwrap();

    // let results = execute_ndarray(&arr.view(), TrackingParams::default(), true);
    let params = TrackingParams {
        // diameter: 9,
        minmass: 800.,
        // separation: 10,
        // filter_close: filter,
        // search_range: Some(9.),
        // characterize,
        // cpu_processed: processed_cpu,
        // sig_radius: Some(2.),
        // bg_radius: Some((60 as f32).sqrt()),
        // gap_radius: Some(0.5),
        // varcheck: Some(1.),

        // style: ParamStyle::Log {
        //     min_radius: 2.,
        //     max_radius: 5.,
        //     n_radii: 10,
        //     log_spacing: false,
        //     overlap_threshold: 1.,
        // },
        include_r_in_output: true,
        ..Default::default()
    };
    // let mut decoderiter = match debug {
    //         true => decoderiter.take(1),
    //         false => decoderiter.take(usize::MAX)
    //     };
    // let now = Instant::now();
    // let points = vec![(0usize, vec![[300f32, 300f32], [200f32, 200f32]]), (200usize, vec![[300f32, 300f32], [200f32, 200f32]])];
    // let (results, column_names) = execute_gpu::execute_file(&path, Some(1), params, debug, 1, Some(points.into_iter()));
    // // let (results, shape) = execute_gpu(&mut decoderiter, &dims, params, debug, 1);
    // dbg!(&results);
    // let function_time = now.elapsed().as_millis() as f64 / 1000.;
    // dbg!(function_time);
    let now = Instant::now();
    let points = vec![
        0f32, 300f32, 300f32,
        0f32, 200f32, 200f32,
        200f32, 300f32, 300f32,
        1999f32, 200f32, 200f32,
    ];
    let points = Array2::from_shape_vec((4, 3), points).unwrap();
    let point_view = points.view();
    let state = setup_state(&params, &dims, false)?;
    // let point_iter = FrameSubsetter::new(&point_view, Some(0), (1, 2));
    // let (results, column_names) = execute_gpu::execute_gpu(
    //     decoderiter,
    //     &dims,
    //     &params,
    //     debug,
    //     1,
    //     None::<_>,
    //     &state,
    //     // Some(point_iter),
    // )?;
    // let (results, column_names) = execute_gpu::execute_ndarray(&arr.view(), params, debug, 1, None);
    // let (results, shape) = execute_gpu(&mut decoderiter, &dims, params, debug, 1);
    // dbg!(&results.slice(s![..600, ..]));
    let function_time = now.elapsed().as_millis() as f64 / 1000.;
    dbg!(function_time);
    // dbg!(&results);
    // dbg!(&results.len());

    // std::fs::write("testing/dump.bin", bytemuck::cast_slice(&results));
    // if debug{
    // let mut file = fs::OpenOptions::new().write(true).create(true).truncate(true).open("test").unwrap();
    // let raw_bytes = unsafe{std::slice::from_raw_parts(results.as_ptr() as *const u8, results.len() * std::mem::size_of::<my_dtype>())};
    // file.write_all(raw_bytes).unwrap();
    // }

    // let total = now_top.elapsed().as_millis() as f64 / 1000.;
    // dbg!(total);
    Ok(())
}

