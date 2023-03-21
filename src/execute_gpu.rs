use bytemuck::{Pod, Zeroable};
use futures_intrusive;
type my_dtype = f32;
type my_dtype_u = u32;
use ndarray::{Array, Array2, ArrayView3, ArrayView, ArrayView2};
use num_traits::Pow;
use pollster::FutureExt;
use tiff::decoder::Decoder;
use std::{collections::{VecDeque, HashMap}, time::Instant, sync::{mpsc::Sender, Arc, atomic::{Ordering, AtomicBool}, Mutex}, path::{Path, PathBuf}, fs::File, f32::consts::PI, cell::RefCell, thread::{Scope, ScopedJoinHandle}, rc::Rc};
use wgpu::{Buffer, SubmissionIndex};
use crate::{kernels, into_slice::IntoSlice,
    gpu_setup::{
    self, GpuState, ParamStyle, TrackingParams, GpuStateFlavor
}, linking::{
    Linker, FrameSubsetter, ReturnDistance2, SubsetterOutput, SubsetterType, DurationBookkeep,
}, decoderiter::{
    self, FrameProvider,
}, utils::{
    any_as_u8_slice, 
    // inspect_buffers, inspect_through_state,
    Dispatcher,
}, error::{Error, Result}};
use kd_tree::{self, KdPoint};
use tracing::*;

// type channel_type = Option<(Vec<ResultRow>, usize, Option<Vec<my_dtype>>, Option<f32>)>;
type channel_type = Option<(Vec<ResultRow>, usize, Option<Vec<my_dtype>>)>;

type ThreadHandle<'a> = ScopedJoinHandle<'a, Result<(Vec<f32>, Option<Vec<DurationBookkeep>>)>>;


#[derive(Clone, Copy, Debug, Zeroable, Pod, Default)]
#[repr(C)]
pub struct ResultRow{
    pub x: my_dtype,
    pub y: my_dtype,
    pub mass: my_dtype,
    pub r: my_dtype,
    pub max_intensity: my_dtype,
    pub Rg: my_dtype,
    pub raw_mass: my_dtype,
    pub signal: my_dtype,
    pub ecc: my_dtype,
    pub count: my_dtype,
}

const MY_DTYPE_SIZE: usize = std::mem::size_of::<my_dtype>();
const N_RESULT_COLUMNS: usize = std::mem::size_of::<ResultRow>() / MY_DTYPE_SIZE;



impl ResultRow{

    pub fn insert_in_output(&self, output: &mut Vec<my_dtype>, params: &TrackingParams){
        output.push(self.x);
        output.push(self.y);
        output.push(self.mass);
        if params.include_r_in_output{
            output.push(self.r);
        }
        if params.characterize{
            output.push(self.Rg);
            output.push(self.raw_mass);
            output.push(self.signal);
            output.push(self.ecc);
        }
    }
}

impl KdPoint for ResultRow{    
    type Scalar = f32;
    type Dim = typenum::U2;
    fn at(&self, k: usize) -> Self::Scalar{
        match k{
            0 => self.x,
            1 => self.y,
            _ => panic!("Invalid index"),
        }
    }
}


fn submit_work<'a, F: IntoSlice>(
    frame: F,
    staging_buffer: &wgpu::Buffer,
    state: &GpuState,
    tracking_params: &TrackingParams,
    // debug: bool,
    positions: Option<SubsetterOutput>,
    // frame_idx: usize,
    frame_sender: &Option<Sender<F>>,
    mut thread_handle: ScopedJoinHandle<'a, Result<(Vec<f32>, Option<Vec<DurationBookkeep>>)>>
    ) -> Result<(SubmissionIndex, ThreadHandle<'a>)> {
    let frame_slice = frame.into_slice();
    let common_buffers = &state.common_buffers;

    let mut encoder =
    state.device.create_command_encoder(&Default::default());
    state.queue.write_buffer(staging_buffer, 0, bytemuck::cast_slice(frame_slice));
    state.queue.write_buffer(staging_buffer, state.pic_byte_size, bytemuck::cast_slice(&[state.pic_size as u32]));

    // thread_handle = thread_handle_error(sender.send(frame), thread_handle)?;
    if let Some(sender) = frame_sender{
        thread_handle = handle_thread_error(sender.send(frame), thread_handle)?;
    }
    
    encoder.copy_buffer_to_buffer(staging_buffer, 0,
        &common_buffers.frame_buffer, 0, state.pic_byte_size);
    encoder.copy_buffer_to_buffer(staging_buffer, state.pic_byte_size,
        &common_buffers.counter_buffer, 0, common_buffers.counter_buffer.size());

    if tracking_params.illumination_sigma.is_some(){
        if tracking_params.illumination_correction_per_frame{
            encoder.copy_buffer_to_buffer(&common_buffers.frame_buffer, 0,
                &common_buffers.illumination_correcter.as_ref().unwrap().buffer, 0, state.pic_byte_size);
            let sigma = tracking_params.illumination_sigma.unwrap();
            common_buffers.illumination_correcter.as_ref().unwrap().pass.execute(&mut encoder, bytemuck::cast_slice(&[sigma, 1.0]));
            // state.common_buffers.illumination_correcter.as_mut().unwrap().initialized = true;
            // common_buffers.illumination_correcter.as_ref().unwrap().pass.execute(&mut encoder, );
        }
        // if !state.common_buffers.illumination_correcter.as_ref().unwrap().initialized{
        //     panic!("Tried to correct illumination profile without initializing the gpu-side buffer")
        // }
        state.passes["correct_illumination"][0].execute(&mut encoder, bytemuck::cast_slice(&[state.pic_size as u32]));
    }

    // let to_print = vec![
    //     &state.common_buffers.illumination_correcter.as_ref().unwrap().buffer,
    //     &state.common_buffers.frame_buffer,
    // ];
    // fs::write("testing/mean_pic/shape0.dump", bytemuck::cast_slice(&[state.dims[0], state.dims[1], 0]));
    // fs::write("testing/mean_pic/shape1.dump", bytemuck::cast_slice(&[state.dims[0], state.dims[1], 0]));
    // inspect_buffers(
    //     &to_print[..],
    //     staging_buffer,
    //     &state.queue,
    //     &mut encoder,
    //     &state.device,
    //     "testing/mean_pic",
    // );
    
    if let Some(ref pass) = state.std_pass{
        pass.execute(&mut encoder)
        // pass.execute(&mut encoder, state, staging_buffer)
    }
    

    match state.flavor{
        GpuStateFlavor::Trackpy{ ref order, .. } => {

            if let Some(SubsetterOutput::Characterization(ref positions)) = positions {
                // let positions = positions.iter().map(|pos| ResultRow{
                //     x: pos[0],
                //     y: pos[1],
                //     mass: 0.,
                //     max_intensity: 0.0,
                //     r: 0.,
                //     Rg: 0.,
                //     raw_mass: 0.,
                //     signal: 0.,
                //     ecc: 0.,
                //     count: 0.,
                // }).collect::<Vec<_>>();
                state.queue.write_buffer(staging_buffer, state.pic_byte_size, bytemuck::cast_slice(&positions));
                state.queue.write_buffer(&common_buffers.atomic_filtered_buffer, 0, bytemuck::cast_slice(&[positions.len() as u32]));
                encoder.copy_buffer_to_buffer(staging_buffer, state.pic_byte_size,
                    &common_buffers.result_buffer, 0, state.pic_byte_size);
                let (mut charpass, workgroup1d) = state.characterize.take().expect("The characterize pass should always be set at this point");
                charpass.dispatcher = Some(Rc::new(Dispatcher::new_direct(&[positions.len() as u32, 1, 1], &workgroup1d)));
                state.characterize.set(Some((charpass, workgroup1d)));
            };
            
            for cpassname in order.iter(){
                let fullpass = &state.passes[cpassname][0];
                fullpass.execute(&mut encoder, &[]);
                fullpass.reset_indirect(&mut encoder)
            }
            
        },
        GpuStateFlavor::Log{ ref buffers, ref radii, .. } => {
            
            state.passes["preprocess_rows"][0].execute(&mut encoder, &[]);
            state.passes["preprocess_cols"][0].execute(&mut encoder, &[]);
            
            let n_radii = radii.len();
            let mut iterator = radii.iter().enumerate();
            
            let (i, radius) = iterator.next().unwrap();
            let modder = buffers.logspace_buffers.len();
            let convolution = &buffers.logspace_buffers[i % modder].1;
            let sigma = radius / (2 as my_dtype).sqrt();
            convolution.execute(&mut encoder, bytemuck::cast_slice(&[sigma]));
            
            let mut edge = -1;
            
            for (i, radius) in iterator{
                let convolution = &buffers.logspace_buffers[i % modder].1;
                let sigma = radius / (2 as my_dtype).sqrt();
                convolution.execute(&mut encoder, bytemuck::cast_slice(&[sigma]));
                
                let find_max = &state.passes["logspace_max"][(i-1) % 3];
                let prev_radius = radii[i-1];
                let push_constants_tuple = (edge, prev_radius);
                let push_constants = unsafe{ any_as_u8_slice(&push_constants_tuple) };
                find_max.execute(&mut encoder, push_constants);
                
                let walk = &state.passes["walk"][0];
                walk.execute(&mut encoder, &[]);
                encoder.clear_buffer(&common_buffers.particles_buffer, 0, None);
                encoder.clear_buffer(&common_buffers.atomic_buffer, 0, None);
                edge = 0;
            }
            
            edge = 1;
            
            let find_max = &state.passes["logspace_max"][(n_radii - 1) % modder];
            let radius = radii[n_radii - 1];
            let push_constants_tuple = (edge, radius);
            let push_constants = unsafe{ any_as_u8_slice(&push_constants_tuple) };
            find_max.execute(&mut encoder, push_constants);
            
            let walk = &state.passes["walk"][0];
            walk.execute(&mut encoder, &[]);
            encoder.clear_buffer(&common_buffers.particles_buffer, 0, None);
            encoder.clear_buffer(&common_buffers.atomic_buffer, 0, None);
        }
    }

    if let Some(n_iter) = tracking_params.adaptive_background{
        let the_atomics = [&state.common_buffers.atomic_buffer, &state.common_buffers.atomic_buffer2];
        let mut iter = the_atomics.iter().zip(state.passes["adaptive_filter"].iter()).cycle().take(n_iter);
        state.common_buffers.raw_frame.as_ref().map(|buf| encoder.copy_buffer_to_buffer(&state.common_buffers.frame_buffer, 0, buf, 0, buf.size()));

        {
            let (to_reset, pass) = iter.next().unwrap();
            encoder.clear_buffer(to_reset, 0, None);
            pass.execute(&mut encoder, &[]);
            pass.reset_indirect(&mut encoder);
        }
        for (to_reset, pass) in iter{
            state.std_pass.as_ref().unwrap().execute(&mut encoder);
            // state.std_pass.as_ref().unwrap().execute(&mut encoder, state, staging_buffer);
            encoder.clear_buffer(to_reset, 0, None);
            pass.execute(&mut encoder, &[]);
            pass.reset_indirect(&mut encoder);
            // state.std_pass.execute(&mut encoder, state, staging_buffer);
            // if i == 3{
            //     let to_print = vec![
            //         &state.common_buffers.frame_buffer,
            //         &state.common_buffers.result_buffer,
            //         &state.common_buffers.particles_buffer,
            //         &state.common_buffers.particles_buffer2,
            //         &state.common_buffers.std_buffer,
            //         &state.common_buffers.atomic_buffer,
            //         &state.common_buffers.atomic_buffer2,
            //         &state.common_buffers.atomic_filtered_buffer,
            //         state.passes["adaptive_filter"][0].dispatcher.as_ref().unwrap().as_ref().get_buffer(),
            //         state.passes["adaptive_filter"][1].dispatcher.as_ref().unwrap().as_ref().get_buffer(),
            //         &state.common_buffers.counter_buffer,
            //         // state.passes["characterize"][0].dispatcher.as_ref().unwrap().as_ref().get_buffer(),
            //     ];
            //     let width = 10;
            //     fs::write("testing/mean_pic/shape0.dump",
            //         bytemuck::cast_slice(&[state.dims[0], state.dims[1], 0]));
            //     fs::write("testing/mean_pic/shape1.dump",
            //         bytemuck::cast_slice(&[(state.pic_size * 4) as u32 / width, width, 0]));
            //     fs::write("testing/mean_pic/shape2.dump",
            //         bytemuck::cast_slice(&[(state.pic_size) as u32 / width, width, 0]));
            //     fs::write("testing/mean_pic/shape3.dump",
            //         bytemuck::cast_slice(&[(state.pic_size) as u32 / width, width, 0]));
            //     fs::write("testing/mean_pic/shape4.dump",
            //         bytemuck::cast_slice(&[1u32, 0]));
            //     fs::write("testing/mean_pic/shape5.dump",
            //         bytemuck::cast_slice(&[1u32, 1]));
            //     fs::write("testing/mean_pic/shape6.dump",
            //         bytemuck::cast_slice(&[1u32, 1]));
            //     fs::write("testing/mean_pic/shape7.dump",
            //         bytemuck::cast_slice(&[1u32, 1]));
            //     fs::write("testing/mean_pic/shape8.dump",
            //         bytemuck::cast_slice(&[1u32, 1]));
            //     fs::write("testing/mean_pic/shape9.dump",
            //         bytemuck::cast_slice(&[1u32, 1]));
            //     fs::write("testing/mean_pic/shape10.dump",
            //         bytemuck::cast_slice(&[1u32, 1]));
            //     inspect_buffers(
            //         &to_print[..],
            //         staging_buffer,
            //         &state.queue,
            //         &mut encoder,
            //         &state.device,
            //         "testing/mean_pic",
            //     );
            // }
        // state.std_pass.execute(&mut encoder, state, staging_buffer);
            // state.std_pass.execute(&mut encoder);
        }
        
    }

    let characterize = state.characterize.take();
    if let Some(ref inner) = characterize{
        let (pass, _workgroup1d) = inner;
        
        pass.execute(&mut encoder, &[]);
        pass.reset_indirect(&mut encoder);
    }
    state.characterize.set(characterize);
    
    // if let Some(ref pass) = state.characterize{
        
    //     // pass.as_ptr.execute(&mut encoder, &[]);
    //     pass.reset_indirect(&mut encoder);
        
    // }

    // let to_print = vec![
    //     &state.common_buffers.result_buffer,
    //     &state.common_buffers.atomic_filtered_buffer,
    // ];
    // fs::write("testing/mean_pic/shape0.dump", bytemuck::cast_slice(&[1000u32, 10, 0]));
    // fs::write("testing/mean_pic/shape1.dump", bytemuck::cast_slice(&[1, 1]));
    // inspect_buffers(
    //     &to_print[..],
    //     staging_buffer,
    //     &state.queue,
    //     &mut encoder,
    //     &state.device,
    //     "testing/mean_pic",
    // );
    
    let output_buffer = &common_buffers.result_buffer;
    
    encoder.copy_buffer_to_buffer(&common_buffers.atomic_filtered_buffer, 0,
        staging_buffer, 0, 4);
    encoder.copy_buffer_to_buffer(output_buffer, 0,
        staging_buffer, 4, state.pic_byte_size);
    
    let index = state.queue.submit(Some(encoder.finish()));
    let mut encoder =
        state.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.clear_buffer(&common_buffers.result_buffer, 0, None);
    encoder.clear_buffer(&common_buffers.atomic_buffer, 0, None);
    encoder.clear_buffer(&common_buffers.atomic_buffer2, 0, None);
    encoder.clear_buffer(&common_buffers.atomic_filtered_buffer, 0, None);
    encoder.clear_buffer(&common_buffers.particles_buffer, 0, None);
    encoder.clear_buffer(&common_buffers.particles_buffer2, 0, None);
    state.queue.submit(Some(encoder.finish()));
    Ok((index, thread_handle))
}




fn get_work<'a>(
    finished_staging_buffer: &Buffer,
    state: &GpuState,
    // tracking_params: &TrackingParams,
    old_submission: wgpu::SubmissionIndex,
    wait_gpu_time: &mut Option<f64>,
    frame_index: usize,
    // debug: bool,
    job_sender: &Sender<channel_type>,
    thread_handle: ScopedJoinHandle<'a, Result<(Vec<f32>, Option<Vec<DurationBookkeep>>)>>,
    // circle_inds: &Vec<[i32; 2]>,
    // dims: &[u32; 2],
    ) -> Result<ScopedJoinHandle<'a, Result<(Vec<f32>, Option<Vec<DurationBookkeep>>)>>> {
    // println!("getting frame: {}", frame_index);
    let buffer_slice = finished_staging_buffer.slice(..);
    let (sender, receiver) = 
            futures_intrusive::channel::shared::oneshot_channel();
    
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    let now = wait_gpu_time.as_ref().map(|_| Instant::now());
    state.device.poll(wgpu::Maintain::WaitForSubmissionIndex(old_submission));
    wait_gpu_time.as_mut().map(|t| *t += now.unwrap().elapsed().as_secs_f64());
    receiver.receive().block_on().unwrap().unwrap();
    let data = buffer_slice.get_mapped_range();
    let neighborhoods = None;
    let n_parts = *bytemuck::from_bytes::<u32>(&data[..MY_DTYPE_SIZE]);
    
    let particle_data_endpoint = (n_parts as usize)*MY_DTYPE_SIZE*N_RESULT_COLUMNS + MY_DTYPE_SIZE;
    let results = bytemuck::cast_slice::<u8, ResultRow>(&data[MY_DTYPE_SIZE..particle_data_endpoint]).to_vec();

    drop(data);
    finished_staging_buffer.unmap();

    handle_thread_error(job_sender.send(Some((results, frame_index, neighborhoods))), thread_handle)
}

fn handle_thread_error<'a, E, T>(send_res: std::result::Result<(), E>, thread_handle: ScopedJoinHandle<'a, Result<T>>) -> Result<ScopedJoinHandle<'a, Result<T>>>{
    match send_res{
        Ok(()) => return Ok(thread_handle),
        Err(_) => {
            match thread_handle.join(){
                Ok(Ok(_)) | Err(_) => return Err(Error::ThreadError),
                Ok(Err(e)) => return Err(e),
            }
        }
    }
}

pub fn column_names(params: &TrackingParams) -> (Vec<(&'static str, &'static str)>, Option<usize>){
// pub fn column_names(params: &TrackingParams) -> Vec<(&'static str, &'static str)>{
    let mut names = vec![("frame", "int"), ("y", "float"), ("x", "float"), ("mass", "float")];

    if params.include_r_in_output{
        names.push(("r", "float"))
    }
    
    if params.characterize{
        names.push(("Rg", "float"));
        names.push(("raw", "float"));
        names.push(("signal", "float"));
        names.push(("ecc", "float"));
    }
    
    if params.doughnut_correction{
        names.push(("raw_mass", "float"));
        names.push(("raw_bg_median", "float"));
        names.push(("raw_mass_corrected", "float"));
    }

    // duration column HAS TO BE THE LAST, as it is added
    // in a postprocessing step
    let particle_column = if params.search_range.is_some(){
        names.push(("particle", "int"));
        names.push(("duration", "int"));
        Some(names.len() - 2)
    } else {
        None
    };
    
    (names, particle_column)
}

pub struct PostProcessKernels<F1, F2>
where F1: Fn(my_dtype) -> (Vec<[i32; 2]>, usize), F2: Fn(my_dtype) -> Vec<[i32; 2]>{
    raw_sig_inds: FloatMemoizer<(Vec<[i32; 2]>, usize), F1>,
    raw_bg_inds: FloatMemoizer<Vec<[i32; 2]>, F2>,
}

struct FloatMemoizer<O, F>
where F: Fn(my_dtype) -> O{
    map: HashMap<my_dtype_u, O>,
    function: Box<F>,
}

impl<O, F> FloatMemoizer<O, F>
where F: Fn(my_dtype) -> O{
    pub fn new(function: F) -> Self{
        Self{
            map: HashMap::new(),
            function: Box::new(function),
        }
    }

    pub fn call(&mut self, args: my_dtype) -> &O{
        self.map.entry(args.to_bits()).or_insert((self.function)(args))
    }
}


fn filter_close_trackpy(results: &Vec<ResultRow>, separation: my_dtype) -> Vec<ResultRow>{
    
    let tree = kd_tree::KdIndexTree::build_by_ordered_float(&results);
    let output = results.iter().enumerate()
    .map(|(idx, query)| {
        let mass = query.mass;
        let neighbors = tree.within_radius_rd2(query, separation);
        // let neighbors = tree.within_radius(query, tracking_params.separation as my_dtype);
        let mut keep_point = true;
        for (&neighbor_idx, _distance) in neighbors{
            let neighbor_mass = results[neighbor_idx].mass;
            if neighbor_idx != idx{
                if mass < neighbor_mass{
                    keep_point = false;
                    break;
                } else if mass == neighbor_mass{
                    if idx > neighbor_idx{
                        keep_point = false;
                        break;
                    }
                }
            }
        }
        if keep_point{
            Some(*query)
        } else {
            None
        }
    }).flatten().collect::<Vec<_>>();

    return output
}

fn prune_blobs_log(results: &Vec<ResultRow>, overlap_threshold: my_dtype,
    interruption: Option<&Arc<AtomicBool>>,
    ) -> Result<Vec<ResultRow>> {
    // RESULTS NEED TO BE PRESORTED ACCORDING TO BLOB PRIORITY.
    // priority is here chosen to be blob radius, with mass as tiebreaker.
    fn disk_overlap(d: my_dtype, r1: my_dtype, r2: my_dtype) -> my_dtype{
        
        fn make_acos(d: my_dtype, r1: my_dtype, r1s: my_dtype, r2s: my_dtype) -> my_dtype{
            let mut ratio: my_dtype = (d.pow(2) + r1s - r2s) / (2f32 * d * r1);
            ratio = ratio.clamp(-1., 1.);
            return ratio.acos()
        }
        let r1s = r1.pow(2);
        let r2s = r2.pow(2);
        let acos1 = make_acos(d, r1, r1s, r2s);
        let acos2 = make_acos(d, r2, r2s, r1s);

        let a = -d + r2 + r1;
        let b =  d - r2 + r1;
        let c =  d + r2 - r1;
        let d =  d + r2 + r1;
        let area = r1s * acos1 + r2s * acos2 - 0.5 * (a * b * c * d).abs().sqrt();

        let overlap = area / (PI * r1s.min(r2s));
        return overlap
    }

    fn blob_overlap(blob1: &ResultRow, blob2: &ResultRow, distance: my_dtype) -> my_dtype{
        if distance > blob1.r + blob2.r{
            return 0.
        } else if distance <= (blob1.r - blob2.r).abs() {
            return 1.
        } else {
            return disk_overlap(distance, blob1.r, blob2.r)
        }
    }
    let tree = kd_tree::KdIndexTree::build_by_ordered_float(&results);
    // let mut output = Vec::new();

    let mut to_keep: Vec<_> = results.iter().map(|ele| Some(ele)).collect();

    // This is pretty much a copy-paste from skimage's feature.blob's _prune_blobs.
    // It is not invariant to the order of the blobs, which in turn can make it non-deterministic
    // as we have no guarantees on the ordering of the blobs received from the gpu.
    // It can be made deterministic by first sorting the results.

    for idx in 0..results.len(){
        if let Some(atomic) = interruption{
            if atomic.load(Ordering::Relaxed){
                return Err(Error::Interrupted)
            }
        }
        let query = match to_keep[idx]{
            None => continue,
            Some(query) => query,
        };
        let query_radius = query.r * 2.;
        let neighbors = tree.within_radius_rd2(query, query_radius);
        for (&neighbor_idx, distance2) in neighbors{
            if neighbor_idx == idx{
                continue
            }
            let neighbor = match to_keep[neighbor_idx]{
                Some(neighbor) => neighbor,
                None => continue,
            };
            let overlap = blob_overlap(query, neighbor, distance2.sqrt());
            if overlap > overlap_threshold{
                to_keep[neighbor_idx] = None //Since the array has been sorted before calling the function we can be sure that we are setting the
                // blob with the smallest r (with mass as tiebreaker) to None
            }
        }
    }

    let output: Vec<_> = to_keep.into_iter().flatten().cloned().collect();
    Ok(output)
}

fn post_process<A: IntoSlice>(
    mut results: Vec<ResultRow>,
    output: &mut Vec<my_dtype>,
    frame_index: usize,
    mut linker: Option<&mut Linker>,
    kernels: Option<&mut PostProcessKernels<impl Fn(my_dtype) -> (Vec<[i32; 2]>, usize), impl Fn(my_dtype) -> Vec<[i32; 2]>>>,
    frame: Option<A>,
    dims: [u32; 2],
    tracking_params: &TrackingParams,
    interruption: Option<&Arc<AtomicBool>>,
    ) -> Result<()> {
    
    let relevant_points = match tracking_params.style{
        ParamStyle::Trackpy{filter_close, separation, ..} => {
            if filter_close{
                filter_close_trackpy(&results, separation as my_dtype)
            } else {
                results
            }
        },
        ParamStyle::Log{ overlap_threshold, ..} => {
            if overlap_threshold < 1.{
                results.sort_by(|a, b|{
                    let r_cmp = b.r.partial_cmp(&a.r).unwrap();
                    match r_cmp{
                        std::cmp::Ordering::Equal => b.mass.partial_cmp(&a.mass).unwrap(),
                        _ => r_cmp,
                    }
                });
                prune_blobs_log(&results, overlap_threshold, interruption)?
            } else {
                results
            }
        },
    };
    
    let raw_properties = frame.map(|inner| {
        
        let frame = inner.into_slice();
        let dims = [dims[0] as usize, dims[1] as usize];
        let frame = ArrayView::from_shape(dims, frame).unwrap();
        let kernels = kernels.unwrap();

        let (mut raw_sig_inds, mut raw_bg_inds) = match tracking_params.style{
            ParamStyle::Trackpy{ diameter, .. } => {
                // let radius = tracking_params.doughnut_correction.unwrap();
                let radius = (diameter as my_dtype) / 2.0;
                let (raw_sig_inds, _) = kernels.raw_sig_inds.call(radius);
                let raw_bg_inds = kernels.raw_bg_inds.call(radius);
                (Some(raw_sig_inds), Some(raw_bg_inds))
            },
            ParamStyle::Log { .. } => {
                (None, None)
            }
        };
        let mut sig = Vec::new();
        let mut bg = Vec::new();

        let mut sig_sums = Vec::new();
        let mut bg_medians = Vec::new();
        let mut correcteds = Vec::new();


        for row in relevant_points.iter() {
            match tracking_params.style{
                ParamStyle::Trackpy{ .. } => {},
                ParamStyle::Log{ .. } => {
                    let (inner_raw_sig_inds, _) = kernels.raw_sig_inds.call(row.r);
                    let inner_raw_bg_inds = kernels.raw_bg_inds.call(row.r);
                    raw_sig_inds = Some(inner_raw_sig_inds);
                    raw_bg_inds = Some(inner_raw_bg_inds);
                },
            }
            let raw_sig_inds = raw_sig_inds.unwrap();
            let raw_bg_inds = raw_bg_inds.unwrap();
            let middle_index = [row.x.round() as i32, row.y.round() as i32];
            for ind in raw_sig_inds{
                let curidx = [(middle_index[0] + ind[0]) as usize, (middle_index[1] + ind[1]) as usize];
                match frame.get(curidx){
                    Some(val) => {
                        sig.push(*val);
                    },
                    None => {}
                };
            }
            for ind in raw_bg_inds{
                let curidx = [(middle_index[0] + ind[0]) as usize, (middle_index[1] + ind[1]) as usize];
                match frame.get(curidx){
                    Some(val) => {
                        bg.push(*val);
                    },
                    None => {}
                };
            }
            let sig_sum: my_dtype = sig.iter().sum();
            bg.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let bg_median = if bg.len() == 0{
                0.
            } else {
                bg[bg.len()/2]
            };
            let corrected = sig_sum - bg_median * sig.len() as f32;

            sig_sums.push(sig_sum);
            bg_medians.push(bg_median);
            correcteds.push(corrected);
            sig.clear();
            bg.clear();
        }
        
        (sig_sums, bg_medians, correcteds)
    });

    if let (Some(linker), Some(reset_points)) = (&mut linker, tracking_params.linker_reset_points.as_ref()){
        if reset_points.contains(&frame_index){
            linker.reset()
        }
    }
    let part_ids = match linker{
        Some(linker) => Some(linker.advance(&relevant_points, interruption)?),
        None => None,
    };
    
    for (idx, row) in relevant_points.iter().enumerate(){
        output.push(frame_index as my_dtype);
        row.insert_in_output(output, tracking_params);
        raw_properties.as_ref().map(|raw_properties|{
            output.push(raw_properties.0[idx]);
            output.push(raw_properties.1[idx]);
            output.push(raw_properties.2[idx]);
        });
        part_ids.as_ref().map(|part_ids| output.push(part_ids[idx] as my_dtype));
    }
    Ok(())
}

pub fn mean_from_iter<A: IntoSlice, T: Iterator<Item = Result<A>>>(mut iter: T, dims: &[u32; 2]) -> crate::error::Result<Array2<my_dtype>>{
    let mut n_frames = 0;
    let image_size = (dims[0] * dims[1]) as usize;
    let mut mean_vec = iter.try_fold(vec![0f32; image_size], |mut acc, ele|{
        let ele = ele?;
        let slc = ele.into_slice();
        if slc.len() != image_size{
            return Err(crate::error::Error::DimensionMismatch { idx: n_frames, frame_len: slc.len(), dimensions: dims.clone() })
        }
        for (a, e) in acc.iter_mut().zip(slc.iter()){
            *a += e
        }
        n_frames += 1;
        Ok(acc)
    })?;
    let n_frames = n_frames as f32;
    for e in mean_vec.iter_mut(){
        *e /= n_frames;
    }
    let mean_arr = Array::from_shape_vec((dims[0] as usize, dims[1] as usize), mean_vec).unwrap();
    Ok(mean_arr)
}


pub fn setup_illumination_profile(
    mean_frame: Array2<my_dtype>,
    tracking_params: &TrackingParams,
    state: &mut GpuState,
    ){
    
    let mut encoder = state.device.create_command_encoder(&Default::default());
    let staging_buffer = &state.common_buffers.staging_buffers[0];
    let target_buffer = &state.common_buffers.illumination_correcter.as_ref().unwrap().buffer;
    state.queue.write_buffer(staging_buffer, 0, bytemuck::cast_slice(mean_frame.into_slice()));
    encoder.copy_buffer_to_buffer(staging_buffer, 0, target_buffer, 0, state.pic_byte_size);
    let sigma = tracking_params.illumination_sigma.unwrap();
    state.common_buffers.illumination_correcter.as_ref().unwrap().pass.execute(&mut encoder, bytemuck::cast_slice(&[sigma, 1.0]));
    state.queue.submit(Some(encoder.finish()));
    
    state.common_buffers.illumination_correcter.as_mut().unwrap().initialized = true;
}

pub fn execute_gpu<F: IntoSlice + Send, P: FrameProvider<Frame = F> + ?Sized>(
    frames: Box<P>,
    dims: [u32; 2],
    tracking_params: &TrackingParams,
    verbosity: u32,
    pos_iter: Option<FrameSubsetter>,
    state: &GpuState,
    interruption: Option<&Arc<AtomicBool>>,
    progress: Option<&Arc<Mutex<(usize, Option<usize>)>>>,
    ) -> Result<(Vec<my_dtype>, Vec<(&'static str, &'static str)>)>{
    
    let mut wait_gpu_time = if verbosity > 0 { Some(0.) } else {None};

    let (inp_sender,
        inp_receiver) = std::sync::mpsc::channel();
    
    let prog_max = match (progress, &tracking_params.keys){
        (None, _) => None,
        (Some(_), None) => frames.light_len(),
        (Some(_), Some(keys)) => Some(keys.len()),
    };

    let send_frame = tracking_params.doughnut_correction;
    let (frame_sender, frame_receiver) = if send_frame{
        let (s, r) = std::sync::mpsc::channel::<P::Frame>();
        (Some(s), Some(r))
    } else {
        (None, None)
    };
    error!("entering execute_gpu");
    
    let output = std::thread::scope(|scope: &Scope| -> Result<(Vec<f32>, Option<Vec<crate::linking::DurationBookkeep>>)>{
        let handle = {
            let thread_tracking_params = tracking_params.clone();
            let mut linker = tracking_params.search_range
                .map(|range| Linker::new(range, tracking_params.memory.unwrap_or(0)));
            let progress = progress;
            scope.spawn(move || -> Result<(Vec<f32>, Option<Vec<crate::linking::DurationBookkeep>>)>{
                let mut kernels = if send_frame {
                    let raw_sig_inds = FloatMemoizer::new(kernels::circle_inds);
                    let raw_bg_inds = FloatMemoizer::new(|radius| {
                            kernels::annulus_inds(tracking_params.bg_radius.unwrap_or(2.0*radius), 
                            tracking_params.gap_radius.unwrap_or(0.) + radius)});
                    Some(
                        PostProcessKernels{
                            raw_sig_inds,
                            raw_bg_inds,
                        }
                    )
                } else {
                    None
                };
                let mut output: Vec<my_dtype> = Vec::with_capacity(100000);
                let mut thread_sleep = if verbosity > 0 {Some(0.)} else {None};
                let mut cur_prog = 1;
                loop{
                    let now = thread_sleep.map(|_| std::time::Instant::now());
                    match inp_receiver.recv().map_err(|_| crate::error::Error::ThreadError)?{
                        None => break,
                        Some(inp) => {
                            thread_sleep.as_mut().map(|thread_sleep| *thread_sleep += now.unwrap().elapsed().as_nanos() as f64 / 1e9);
                            let (results, frame_index,
                                _neighborhoods) = inp;
                            let frame = match frame_receiver{
                                Some(ref receiver) => Some(receiver.recv()
                                    .map_err(|_| crate::error::Error::ThreadError)?),
                                None => None,
                            };
                            post_process(results, &mut output, frame_index, linker.as_mut(),
                                kernels.as_mut(), frame, dims.clone(), &thread_tracking_params,
                                interruption)?;
                            if let Some(prog) = progress{
                                *prog.lock().unwrap() = (cur_prog, prog_max)
                            }
                            cur_prog += 1;
                        }
                    }
                }
                thread_sleep.map(|thread_sleep| println!("Thread sleep: {} s", thread_sleep));
                let part_durations = linker.map(|linker| linker.finish());
                Ok((output, part_durations))
            })
        };
        
        let mut free_staging_buffers = state.common_buffers.staging_buffers.iter().collect::<Vec<&wgpu::Buffer>>();
        let mut in_use_staging_buffers = VecDeque::new();

        let mut the_iter: Box<dyn Iterator<Item = crate::error::Result<_>>> = match pos_iter{
            Some(subsetter) => {
                match subsetter.frame_col{
                    Some(_) => {
                        Box::new(subsetter.map(|result|{
                            let (frame_to_take, positions) = result?;
                            let frame_idx = frame_to_take.unwrap();
                            let frame = frames.get_frame(frame_idx)
                                .map_err(|err|{
                                    match err{
                                        Error::FrameOOB => Error::FrameOutOfBounds { vid_len: frames.len(Some(frame_idx)), problem_idx: frame_idx },
                                        _ => err,
                                    }
                                })?;
                            Ok((frame, Some(positions), frame_idx))
                        }))
                    },
                    None => {
                        let frames_iter = frames.into_iter().enumerate();
                        Box::new(frames_iter.zip(subsetter.cycle()).map(|(res1, res2)|{
                            let (frame_idx, frame) = res1;
                            let frame = frame?;
                            let (_frame_to_take, positions) = res2?;
                            Ok((frame, Some(positions), frame_idx))
                        }))
                    }
                }
            },
            None => {
                let keys = &tracking_params.keys;
                match keys{
                    Some(keys) => {
                        if keys.iter().enumerate().all(|(idx, &key)| idx == key){
                            let frames_iter = frames.into_iter().enumerate().take(keys.len());
                            Box::new(frames_iter.map(|res|{
                                let (frame_idx, frame) = res;
                                let frame = frame?;
            
                                Ok((frame, None, frame_idx))
                            }))
                        } else {
                            Box::new(keys.into_iter().map(|&frame_idx|{
                                let frame = frames.get_frame(frame_idx)
                                    .map_err(|err|{
                                        match err{
                                            Error::FrameOOB => Error::FrameOutOfBounds { vid_len: frames.len(Some(frame_idx)), problem_idx: frame_idx },
                                            _ => err,
                                        }
                                    })?;
                                Ok((frame, None, frame_idx))
                            }))
                        }
                        
                    },
                    None => {
                        let frames_iter = frames.into_iter().enumerate();
                        Box::new(frames_iter.map(|res|{
                            let (frame_idx, frame) = res;
                            let frame = frame?;
            
                            Ok((frame, None, frame_idx))
                        }))
                    }
                }
            }
        };

        
        let (frame, positions, frame_index) = match the_iter.next(){
            Some(Ok(inner)) => inner,
            Some(Err(err)) => {
                drop(inp_sender);
                return Err(err)
            },
            None => {
                drop(inp_sender);
                return Err(Error::EmptyIterator)
            },
        };
        let staging_buffer = free_staging_buffers.pop().unwrap();
        in_use_staging_buffers.push_back(staging_buffer);
        
        let slc = frame.into_slice();
        if slc.len() != state.pic_size{
            return Err(crate::error::Error::DimensionMismatch { idx: frame_index, frame_len: slc.len(), dimensions: state.dims.clone() })
        }
        
        let (mut old_submission, mut handle) =
            submit_work(frame, staging_buffer, state, &tracking_params, positions, &frame_sender, handle)?;
        

        let mut get_work_frame_idx = frame_index;
        for res in the_iter{
            if let Some(atomic) = interruption{
                if atomic.load(Ordering::Relaxed){
                    drop(inp_sender);
                    return Err(Error::Interrupted)
                }
            }
            let (frame, positions, frame_index) = match res{
                Ok(inner) => inner,
                Err(err) => {
                    drop(inp_sender);
                    return Err(err)
                }
            };
            
            let staging_buffer = free_staging_buffers.pop().unwrap();
            in_use_staging_buffers.push_back(staging_buffer);
            let slc = frame.into_slice();
            if slc.len() != state.pic_size{
                return Err(crate::error::Error::DimensionMismatch { idx: frame_index, frame_len: slc.len(), dimensions: state.dims.clone() })
            }
            let (new_submission, new_handle) = submit_work(frame, staging_buffer, &state, &tracking_params, positions, &frame_sender, handle)?;
            handle = new_handle;
            
            let finished_staging_buffer = in_use_staging_buffers.pop_front().unwrap();

            handle = get_work(finished_staging_buffer,
                &state, 
                old_submission, 
                &mut wait_gpu_time,
                get_work_frame_idx,
                &inp_sender,
                handle,
            )?;


            free_staging_buffers.push(finished_staging_buffer);
            old_submission = new_submission;
            get_work_frame_idx = frame_index;
        }
        let finished_staging_buffer = in_use_staging_buffers.pop_front().unwrap();
        
        handle = get_work(finished_staging_buffer,
            &state, 
            old_submission, 
            &mut wait_gpu_time,
            get_work_frame_idx,
            &inp_sender,
            handle,
        )?;

        wait_gpu_time.map(|wait_gpu_time| println!("Wait GPU time: {} s", wait_gpu_time));
        drop(inp_sender.send(None));
        handle.join()
            .map_err(|_err| crate::error::Error::ThreadError)?
    });
    
    
    let (output, part_durations) = output?;
    let (col_names, particle_column) = column_names(tracking_params);
    let output = match (part_durations, particle_column){
        (Some(durations), Some(particle_column)) => {
            let n_parts = output.len()/(col_names.len() - 1);
            let mut new_output = Vec::with_capacity(n_parts * col_names.len());
            for chunk in output.chunks_exact(col_names.len() - 1){
                new_output.extend_from_slice(chunk);
                new_output.push(
                    durations[chunk[particle_column] as usize].duration as f32
                )
            }
            new_output
        },
        (None, None) => {
            output
        }
        _ => {
            panic!("this should never happen")
        }
    };

    Ok((output, col_names))
}



pub fn path_to_iter<P: AsRef<std::path::Path>>(path: P, channel: Option<usize>)
    -> crate::error::Result<(
        Box<
            dyn FrameProvider<Frame = Vec<f32>, FrameIter = Box<dyn Iterator<Item = Result<Vec<f32>>>>> + 'static>,
            [u32; 2]
    )> {
    let path: &Path = path.as_ref();
    let ext = path.extension()
        .ok_or_else(|| Error::NoExtensionError{ filename: path.to_path_buf() })?
        .to_ascii_lowercase();
    let (iter, dims): (Box<dyn FrameProvider<Frame = Vec<my_dtype>, FrameIter = Box<dyn Iterator<Item = Result<Vec<f32>>>>>>, _) = match ext.to_str().unwrap() {
        "tif" | "tiff" => {
            let file = File::open(path).map_err(|ioerr| crate::error::Error::FileNotFound { source: ioerr, filename: path.to_path_buf() })?;
            let decoder = RefCell::new(Decoder::new(file).unwrap());
            let (width, height) = decoder.borrow_mut().dimensions().unwrap();
            let dims = [height, width];
            (Box::new(decoder), dims)
        },
        "ets" => {
            let mut file = File::open(path).map_err(|ioerr| crate::error::Error::FileNotFound { source: ioerr, filename: path.to_path_buf() })?;
            let parser = decoderiter::MinimalETSParser::new(&mut file).unwrap();
            let dims = [parser.dims[1] as u32, parser.dims[0] as u32];
            let provider = RefCell::new(parser.iterate_channel(file, channel.unwrap_or(0))?);
            (Box::new(provider), dims)
        },
        "vsi" => {
            let gen_error = || Error::FileNotFound {
                source: std::io::Error::new(std::io::ErrorKind::NotFound, "vsi doesn't have a filename"),
                filename: path.to_path_buf()
            };
            
            let file_stem = path.file_stem().ok_or_else(gen_error)?.to_str().ok_or_else(|| Error::InvalidFileName { filename: path.to_path_buf() })?;
            let ets_path = path.parent()
                .ok_or_else(gen_error)?
                .join(format!("_{}_", file_stem))
                .join("stack1")
                .join("frame_t_0.ets");
            
            let mut file = File::open(ets_path).map_err(|ioerr| crate::error::Error::FileNotFound { source: ioerr, filename: path.to_path_buf() })?;
            let parser = decoderiter::MinimalETSParser::new(&mut file).unwrap();
            let dims = [parser.dims[1] as u32, parser.dims[0] as u32];
            let provider = RefCell::new(parser.iterate_channel(file, channel.unwrap_or(0))?);
            (Box::new(provider), dims)
        }
        _ => Err(crate::error::Error::UnsupportedFileformat { extension: ext.to_str().unwrap().to_string() })?,
    };
    Ok((iter, dims))
}

pub fn execute_provider<'a, F: IntoSlice + Send, P: FrameProvider<Frame = F> + ?Sized, G: Fn() -> Result<(Box<P>, [u32; 2])>>(
    provider_generator: G,
    params: TrackingParams,
    verbosity: u32,
    pos_array: Option<(ArrayView2<'a, my_dtype>, bool, bool)>,
    interruption: Option<&Arc<AtomicBool>>,
    progress: Option<&Arc<Mutex<(usize, Option<usize>)>>>,
    ) -> crate::error::Result<(Array2<my_dtype>, Vec<(&'static str, &'static str)>)> {
    error!("entering provider");
    let pos_iter = match pos_array{
        Some((pos_array, true, true)) => 
            Some(FrameSubsetter::new(pos_array, Some(0), (1, 2), Some(3), SubsetterType::Characterization)),
        
        Some((pos_array, true, false)) => 
            Some(FrameSubsetter::new(pos_array, Some(0), (1, 2), None, SubsetterType::Characterization)),
        
        Some((pos_array, false, true)) => 
            Some(FrameSubsetter::new(pos_array, None, (0, 1), Some(2), SubsetterType::Characterization)),
        
        Some((pos_array, false, false)) => 
            Some(FrameSubsetter::new(pos_array, None, (0, 1), None, SubsetterType::Characterization)),
        None => { None }
    };

    let (provider, dims) = provider_generator()?;
    error!("before state setup");
    let mut state = gpu_setup::setup_state(&params, &dims, pos_iter.is_some())?;
    error!("state was set up");
    
    if params.illumination_sigma.is_some() && !params.illumination_correction_per_frame{
        // let (provider, dims) = path_to_iter(&path, None)?;
        let (provider, dims) = provider_generator()?;
        let iter = provider.into_iter();
        let mean_frame = mean_from_iter(iter, &dims)?;
        setup_illumination_profile(mean_frame, &params, &mut state);
    }

    
    let (res, column_names) = 
            execute_gpu(provider, dims, &params, verbosity, pos_iter, &mut state, interruption, progress)?;
    let res_len = res.len();
    let shape = (res_len / column_names.len(), column_names.len());
    let res = Array2::from_shape_vec(shape, res)
        .expect(format!("Could not convert to ndarray. Shape is ({}, {}) but length is {}", shape.0, shape.1, &res_len).as_str());
    Ok((res, column_names))
}

pub fn execute_file<'a>(
    path: impl Into<PathBuf>,
    channel: Option<usize>,
    params: TrackingParams,
    verbosity: u32,
    pos_array: Option<(ArrayView2<'a, my_dtype>, bool, bool)>,
    interruption: Option<&Arc<AtomicBool>>,
    progress: Option<&Arc<Mutex<(usize, Option<usize>)>>>,
    ) -> crate::error::Result<(Array2<my_dtype>, Vec<(&'static str, &'static str)>)> {
    error!("entering execute_file");
    let path = Into::<PathBuf>::into(path);
    let generator = || path_to_iter(&path, channel);
    
    execute_provider(generator, params, verbosity, pos_array, interruption, progress)
    // let pos_iter = match pos_array{
    //     Some((pos_array, true, true)) => 
    //         Some(FrameSubsetter::new(pos_array, Some(0), (1, 2), Some(3), SubsetterType::Characterization)),
        
    //     Some((pos_array, true, false)) => 
    //         Some(FrameSubsetter::new(pos_array, Some(0), (1, 2), None, SubsetterType::Characterization)),
        
    //     Some((pos_array, false, true)) => 
    //         Some(FrameSubsetter::new(pos_array, None, (0, 1), Some(2), SubsetterType::Characterization)),
        
    //     Some((pos_array, false, false)) => 
    //         Some(FrameSubsetter::new(pos_array, None, (0, 1), None, SubsetterType::Characterization)),
    //     None => { None }
    // };
    // let mut state = gpu_setup::setup_state(&params, &dims, pos_iter.is_some())?;

    // if params.illumination_sigma.is_some() && !params.illumination_correction_per_frame{
    //     let (provider, dims) = path_to_iter(&path, None)?;
    //     // let iter = (0..)
    //     //     .map(|i| provider.get_frame(i))
    //     //     .take_while(|res| !matches!(res, Err(crate::error::Error::FrameOOB)));
    //     let iter = provider.into_iter();
    //     // let iter = (0..).map(|i| provider.get_frame(i).map(|inner| (i, inner))).take_while(|res| !matches!(res, Err(crate::error::Error::FrameOutOfBounds { .. })));
    //     let mean_frame = mean_from_iter(iter, &dims)?;
    //     setup_illumination_profile(mean_frame, &params, &mut state);
    // }
    // let (res, column_names) = 
    //         execute_gpu(Box::new(provider), dims, &params, verbosity, pos_iter, &mut state)?;
    // let res_len = res.len();
    // let shape = (res_len / column_names.len(), column_names.len());
    // let res = Array2::from_shape_vec(shape, res)
    //     .expect(format!("Could not convert to ndarray. Shape is ({}, {}) but length is {}", shape.0, shape.1, &res_len).as_str());
    // Ok((res, column_names))
}

pub fn execute_ndarray<'a>(
    array: &'a ArrayView3<'a, my_dtype>,
    params: TrackingParams,
    verbosity: u32,
    pos_array: Option<(ArrayView2<'a, my_dtype>, bool, bool)>,
    interruption: Option<&Arc<AtomicBool>>,
    progress: Option<&Arc<Mutex<(usize, Option<usize>)>>>,
    ) -> crate::error::Result<(Array2<my_dtype>, Vec<(&'static str, &'static str)>)> {
    if !array.is_standard_layout(){
        return Err(crate::error::Error::NonStandardArrayLayout)
    }
    let dims_usize = array.shape();
    let dims = [dims_usize[1] as u32, dims_usize[2] as u32];
    let generator = move || Ok((Box::new(array), dims));
    execute_provider(generator, params, verbosity, pos_array, interruption, progress)
    // let pos_iter = match pos_array{
    //     Some((pos_array, true, true)) => 
    //         Some(FrameSubsetter::new(pos_array, Some(0), (1, 2), Some(3), SubsetterType::Characterization)),
        
    //     Some((pos_array, true, false)) => 
    //         Some(FrameSubsetter::new(pos_array, Some(0), (1, 2), None, SubsetterType::Characterization)),
        
    //     Some((pos_array, false, true)) => 
    //         Some(FrameSubsetter::new(pos_array, None, (0, 1), Some(2), SubsetterType::Characterization)),
        
    //     Some((pos_array, false, false)) => 
    //         Some(FrameSubsetter::new(pos_array, None, (0, 1), None, SubsetterType::Characterization)),
    //     None => { None }
    // };

    // let mut state = gpu_setup::setup_state(&params, &dims, pos_iter.is_some())?;
    
    // if params.illumination_sigma.is_some() && !params.illumination_correction_per_frame{
    //     let axisiter = array.axis_iter(ndarray::Axis(0)).map(|x| Ok(x));
    //     let mean_array = mean_from_iter(axisiter, &dims)?;
    //     setup_illumination_profile(mean_array, &params, &mut state);
    // }
    // // let idk = &array.view();
    // let (res, column_names) = 
    //     // if debug {
    //     //     sequential_execute(axisiter, &[dims[1] as u32, dims[2] as u32], params, debug, verbosity)
    //     // } else {
    //         execute_gpu(Box::new(array), dims.clone(), &params, verbosity, pos_iter, &mut state)?;
    //     // };
    // let res_len = res.len();
    // let shape = (res_len / column_names.len(), column_names.len());
    // let res = Array2::from_shape_vec(shape, res)
    //     .expect(format!("Could not convert to ndarray. Shape is ({}, {}) but length is {}", shape.0, shape.1, &res_len).as_str());
    // Ok((res, column_names))
}
