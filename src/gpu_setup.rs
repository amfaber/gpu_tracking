use std::{collections::HashMap, rc::Rc, cell::Cell};

use crate::{my_dtype, utils::*};

use pollster::FutureExt;
use wgpu::{Buffer, Device, self, util::DeviceExt, ComputePipeline, BindGroupLayout};
use tracing::*;


#[derive(Clone)]
pub enum ParamStyle{
    Trackpy{
        separation: u32,
        diameter: u32,
        // minmass: f32,
        maxsize: f32,
        // noise_size: f32,
        // smoothing_size: u32,
        threshold: f32,
        invert: bool,
        percentile: f32,
        topn: u32,
        preprocess: bool,
        filter_close: bool,
    },
    Log{
        min_radius: my_dtype,
        max_radius: my_dtype,
        n_radii: usize,
        log_spacing: bool,
        overlap_threshold: my_dtype,
    },
}




#[derive(Clone)]
pub struct TrackingParams{
    pub style: ParamStyle,
    pub minmass: my_dtype,
    pub max_iterations: u32,
    pub characterize: bool,
    pub search_range: Option<my_dtype>,
    pub memory: Option<usize>,
    // pub cpu_processed: bool,
    pub doughnut_correction: bool,
    pub bg_radius: Option<my_dtype>,
    pub gap_radius: Option<my_dtype>,
    pub snr: Option<my_dtype>,
    pub minmass_snr: Option<my_dtype>,
    pub truncate_preprocessed: bool,
    pub illumination_sigma: Option<my_dtype>,
    pub adaptive_background: Option<usize>,
    pub include_r_in_output: bool,
    pub shift_threshold: f32,
    pub linker_reset_points: Option<Vec<usize>>,
    pub keys: Option<Vec<usize>>,
    pub noise_size: f32,
    pub smoothing_size: Option<u32>,
    pub illumination_correction_per_frame: bool,
    // pub keys: Cell<Option<Vec<usize>>>,
}

impl Default for TrackingParams{
    fn default() -> Self {
        TrackingParams{
            characterize: false,
            search_range: None,
            memory: None,
            doughnut_correction: false,
            bg_radius: None,
            gap_radius: None,
            snr: None,
            minmass_snr: None,
            max_iterations: 10,
            minmass: 0.,
            truncate_preprocessed: true,
            illumination_sigma: None,
            adaptive_background: None,
            include_r_in_output: false,
            shift_threshold: 0.6,
            linker_reset_points: None,
            keys: None,
            noise_size: 1.,
            smoothing_size: None,
            illumination_correction_per_frame: false,
            style: ParamStyle::Trackpy{
                diameter: 9,
                threshold: 0.0,
                invert: false,
                percentile: 0.,
                topn: 0,
                maxsize: 0.0,
                preprocess: true,
                separation: 10,
                filter_close: true,
            }
        }
    }
}

#[derive(Debug)]
pub struct GpuParams{
    pub pic_dims: [u32; 2],
    pub composite_dims: [u32; 2],
    pub sigma: f32,
    pub circle_dims: [u32; 2],
    pub dilation_dims: [u32; 2],
    pub max_iterations: u32,
    pub shift_threshold: f32,
    pub minmass: f32,
    pub margin: i32,
    pub snr: f32,
    pub minmass_snr: f32,
    pub rough_snr_factor: f32,
}

pub struct IlluminationCorrecter{
    pub buffer: wgpu::Buffer,
    pub pass: SeparableConvolution<2>,
    pub initialized: bool,
}

pub struct CommonBuffers{
    pub staging_buffers: Vec<Buffer>,
    pub frame_buffer: Buffer,
    pub result_buffer: Buffer,
    pub atomic_buffer: Buffer,
    pub atomic_buffer2: Buffer,
    pub particles_buffer: Buffer,
    pub particles_buffer2: Buffer,
    pub atomic_filtered_buffer: Buffer,
    pub param_buffer: Buffer,
    pub processed_buffer: Buffer,
    // pub sum_buffer: Buffer,
    pub counter_buffer: Buffer,
    pub std_buffer: Buffer,
    pub illumination_correcter: Option<IlluminationCorrecter>,
    pub temp_buffer: Buffer,
    pub raw_frame: Option<Buffer>,
}

pub struct TrackpyGpuBuffers{
    // pub processed_buffer: Buffer,
    // pub temp_buffer: Buffer,
    pub masses_buffer: Buffer,
    pub max_rows: Buffer,
}

pub struct LogGpuBuffers{
    // pub raw_padded: (Buffer, (FullComputePass, [u32; 4]), FftPass<2>),
    // pub logspace_buffers: Vec<(Buffer, Vec<FullComputePass>, FftPass<2>, Laplace<2>)>,
    pub logspace_buffers: Vec<(Buffer, Laplace<2>)>,
    // pub filter_buffers: Vec<(Buffer, FftPass<2>, FullComputePass)>,
    // pub laplacian_buffer: (Buffer, FftPass<2>),
    // pub unpadded_laplace: (Buffer, [u32; 2]),
    pub global_max: Buffer,
    // pub temp_buffer: Buffer,
    pub temp_buffer2: Buffer,
}


impl CommonBuffers{
    fn create(
        tracking_params: &TrackingParams,
        device: &wgpu::Device,
        size: u64,
        dims: &[u32; 2],
        pipelines: &mut HashMap<String, (Rc<ComputePipeline>, BindGroupLayout, Option<Rc<Dispatcher>>)>,
        ) -> Self{
        // let result_buffer_depth = if tracking_params.characterize { 7 } else { 3 };
        let mut staging_buffers = Vec::new();
        for i in 0..2{
            let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(format!("Staging {}", i).as_str()),
                size: size * 8,
                usage: wgpu::BufferUsages::COPY_SRC 
                | wgpu::BufferUsages::COPY_DST 
                | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            staging_buffers.push(staging_buffer);
        }
        
        let frame_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let raw_frame = if tracking_params.adaptive_background.is_some() & tracking_params.characterize{
            Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }))
        } else { None };
        
        let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (size * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let atomic_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC |
                wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        let atomic_buffer2 = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC |
                wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        let atomic_filtered_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let particles_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let particles_buffer2 = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let params = gpuparams_from_tracking_params(&tracking_params, *dims);
        let param_buffer = unsafe{
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Width Buffer"),
                contents: any_as_u8_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM
            })
        };

        // let sum_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        //     label: None,
        //     size: 4 * 50,
        //     usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        //     mapped_at_creation: false,
        // });
        
        let counter_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let std_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC |
                 wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        
        let processed_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let temp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (2 * size) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let illumination_correcter = if tracking_params.illumination_sigma.is_some() {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let pipeline = pipelines.remove("smooth_mean_frame").unwrap();
            let conv = SeparableConvolution::new(
                device,
                &pipeline,
                &buffer,
                &buffer,
                &temp_buffer,
                [&param_buffer],
            );
            let correcter = IlluminationCorrecter{
                buffer,
                pass: conv,
                initialized: false,
            };
            Some(correcter)
            }
        else {
            None
        };
        
        Self {
            staging_buffers,
            frame_buffer,
            result_buffer,
            atomic_buffer,
            atomic_buffer2,
            particles_buffer,
            particles_buffer2,
            atomic_filtered_buffer,
            param_buffer,
            processed_buffer,
            // sum_buffer,
            counter_buffer,
            std_buffer,
            illumination_correcter,
            temp_buffer,
            raw_frame,
        }
    }
}



impl TrackpyGpuBuffers{
    fn create(
        device: &wgpu::Device,
        size: u64,
        ) -> Self{
        
        let masses_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
    
        let max_rows = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Self{
            // temp_buffer,
            masses_buffer,
            max_rows,
            // processed_buffer,
        }
    }
}


impl LogGpuBuffers{
    fn create(
        device: &wgpu::Device,
        size: u64,
        common_buffers: &CommonBuffers,
        pipelines: &mut HashMap<String, (Rc<ComputePipeline>, BindGroupLayout, Option<Rc<Dispatcher>>)>
    ) -> Self {
        let frame_buffer = &common_buffers.frame_buffer;
        let param_buffer = &common_buffers.param_buffer;

        // let laplacian: [f32; 9] = [
        //     -1., -1., -1.,
        //     -1.,  8., -1.,
        //     -1., -1., -1.,
        // ];

        // let laplacian_buffer = device.create_buffer_init(&BufferInitDescriptor{
        //     label: None,
        //     contents: bytemuck::cast_slice(&laplacian),
        //     usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        // });

        // let padded_laplacian_buffer = fftplan.create_buffer(device, false);

        
        // let laplacian_fft_pass = fftplan.fft_pass(&padded_laplacian_buffer, device);

        // let filter_buffers: Vec<_> = (0..n_radii).map(|_|{
        //     let buffer = fftplan.create_buffer(device, false);
        //     let pass = fftplan.fft_pass(&buffer, device);
        //     let laplace_convolve = fftplan.inplace_spectral_convolution_pass(device, &buffer, &padded_laplacian_buffer);
        //     (buffer, pass, laplace_convolve)
        // }).collect();

        // let raw_padded = fftplan.create_buffer(device, false);
        // let raw_pad_pass = fftplan.pad_pass(device, &frame_buffer, &raw_padded, dims);
        // let raw_fft_pass = fftplan.fft_pass(&raw_padded, device);
        // let raw_padded = (raw_padded, raw_pad_pass, raw_fft_pass);
        
        let separable_pipeline = pipelines.remove("separable_log").unwrap();
        // let temp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        //     label: None,
        //     size: size * 2,
        //     usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        //     mapped_at_creation: false,
        // });

        let temp_buffer2 = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let logspace_buffers: Vec<_> = (0..3)
        .map(|_|{
            // let log_space_buffer = fftplan.create_buffer(device, false);
            let log_space_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            // let fftpass = fftplan.fft_pass(&log_space_buffer, device);
            // let convolutionpasses = filter_buffers.iter().map(|(filterbuffer, _fftpass, _lap_conv)|{
            //     fftplan.spectral_convolution_pass(device, &raw_padded.0, filterbuffer, &log_space_buffer)
            // }).collect();
            let separable_pass = Laplace::<2>::new(
                device,
                &separable_pipeline,
                frame_buffer,
                &log_space_buffer,
                &common_buffers.temp_buffer,
                &temp_buffer2,
                [param_buffer]
            );
            // (log_space_buffer, convolutionpasses, fftpass, separable_pass)
            (log_space_buffer, separable_pass)
        }).collect();

        let global_max = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self{
            // filter_buffers,
            global_max,
            // laplacian_buffer: (padded_laplacian_buffer, laplacian_fft_pass),
            logspace_buffers,
            // raw_padded,
            // temp_buffer,
            // unpadded_laplace: (laplacian_buffer, [3, 3]),
            temp_buffer2,
        }
    }
}

pub struct GpuState{
    pub device: Device,
    pub queue: wgpu::Queue,
    pub common_buffers: CommonBuffers,
    pub passes: HashMap<String, Vec<FullComputePass>>,
    pub pic_size: usize,
    // pub result_read_depth: u64,
    pub pic_byte_size: u64,
    pub flavor: GpuStateFlavor,
    pub dims: [u32; 2],
    pub std_pass: Option<StdArray>,
    pub characterize: Cell<Option<(FullComputePass, [u32; 3])>>,
}

pub enum GpuStateFlavor{
    Trackpy{
        order: Vec<String>,
        buffers: TrackpyGpuBuffers,
    },
    Log{
        // fftplan: FftPlan,
        buffers: LogGpuBuffers,
        radii: Vec<my_dtype>,
    },
}


fn gpuparams_from_tracking_params(params: &TrackingParams, pic_dims: [u32; 2]) -> GpuParams {
    let (kernel_size, circle_size, dilation_size, margin, sigma) = match params.style{
        ParamStyle::Trackpy{separation, diameter, ..} => {
            let smoothing_size = params.smoothing_size.unwrap_or(diameter);
            let circle_size = diameter;
            let dilation_size = (2. * separation as f32 / (2 as f32).sqrt()) as u32;
            let margin = vec![diameter / 2, separation / 2 - 1, smoothing_size / 2].iter().max().unwrap().clone() as i32;
            (smoothing_size, circle_size, dilation_size, margin, params.noise_size)
        },
        ParamStyle::Log { max_radius, .. } => {
            let smoothing_size = params.smoothing_size.unwrap_or({
                let radius = (max_radius + 0.5) as u32;
                let diameter = radius * 2 + 1;
                diameter
            });
            (smoothing_size, 0, 0, 0, params.noise_size)
        }
    };

    GpuParams{
        pic_dims,
        composite_dims: [kernel_size, kernel_size],
        sigma,
        // constant_dims: [kernel_size, kernel_size],
        circle_dims: [circle_size, circle_size],
        dilation_dims: [dilation_size, dilation_size],
        max_iterations: params.max_iterations,
        shift_threshold: params.shift_threshold,
        minmass: params.minmass,
        margin,
        snr: params.snr.unwrap_or(0.0),
        minmass_snr: params.minmass_snr.unwrap_or(0.0),
        rough_snr_factor: if params.adaptive_background.is_some() { 0.5 } else {1.0},
    }
}

// struct AdapterError{
//     msg: &'static str,
// }

// // impl 

// impl std::error::Error for AdapterError{}

pub fn setup_state(
    tracking_params: &TrackingParams,
    dims: &[u32; 2],
    characterize_new_points: bool,
    ) -> crate::error::Result<GpuState> {
    error!("entering state setup");
    
    let instance = wgpu::Instance::new(
        {
            let backends = wgpu::Backends::all();
            error!("backends");
            let dx12_shader_compiler = wgpu::Dx12Compiler::default();
            error!("compiler");
            wgpu::InstanceDescriptor{
                backends,
                dx12_shader_compiler,
            }
            // wgpu::InstanceDescriptor{
            //     backends: wgpu::Backends::all(),
            //     dx12_shader_compiler: Default::default(),
            // }
        }
    );
    error!("instance created");

    let adapter = instance
    .request_adapter(&wgpu::RequestAdapterOptionsBase{
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    })
    .block_on().ok_or(crate::error::Error::GpuAdapterError)?;
    error!("adapter created");

    let mut desc = wgpu::DeviceDescriptor::default();
    desc.features = wgpu::Features::MAPPABLE_PRIMARY_BUFFERS | wgpu::Features::PUSH_CONSTANTS;
    desc.limits.max_push_constant_size = 16;
    desc.limits.max_storage_buffers_per_shader_stage = 12;
    // desc.limits.max_compute_invocations_per_workgroup = 1024;
    // desc.limits.max_compute_workgroup_size_x = 1024;
    let (device, queue) = adapter
    .request_device(&desc, None)
    .block_on()?;
    error!("device created");
    
    
    // let workgroup_size2d = [32u32, 32, 1];
    let workgroup_size2d = [16u32, 16, 1];
    // let workgroup_size1d = [1024u32, 1, 1];
    let workgroup_size1d = [256u32, 1, 1];

    // let n_workgroups = |dims: &[u32; 3], wgsize: &[u32; 3]| { 
    //     let mut n_workgroups = [0, 0, 0];
    //     for i in 0..3 {
    //         n_workgroups[i] = (dims[i] + wgsize[i] - 1) / wgsize[i];
    //     }
    //     n_workgroups
    // };

    // let result_read_depth: u64 = match debug{
    //     false => match tracking_params.characterize{
    //         false => 3,
    //         true => 7,
    //     },
    //     true => 2,
    // };
    
    let pic_size = dims.iter().product::<u32>() as usize;
    let slice_size = pic_size * std::mem::size_of::<my_dtype>();
    let size = slice_size as wgpu::BufferAddress;
    
    let preprocess_source = |source: &str, wg_size: &[u32; 3], common_header: &str| -> String {
        let mut result = String::new();
        result.push_str(common_header);
        result.push_str(source);
        if characterize_new_points{
            result = result.replace("//_feat_characterize_points ", "");
        }
        // if tracking_params.varcheck.is_some(){
            // result = result.replace("//_feat_varcheck ", "");
        // }
        if tracking_params.truncate_preprocessed{
            result = result.replace("//_feat_truncate_preprocessed ", "")
        }
        if let ParamStyle::Log{..} = tracking_params.style{
            result = result.replace("//_feat_LoG ", "");
        }
        result = result.replace("@workgroup_size(_)",
        format!("@workgroup_size({}, {}, {})", wg_size[0], wg_size[1], wg_size[2]).as_str());
        result = result.replace("_workgroup1d_", format!("{}u", wg_size[0]).as_str());
        result
    };

    let wg_dims = [dims[0], dims[1], 1];
    let wg_dims_1d = [dims[0] * dims[1], 1, 1];
    let common_header = include_str!("shaders/params.wgsl");

    let direct_dispatcher1d = Rc::new(Dispatcher::new_direct(&wg_dims_1d, &workgroup_size1d));
    let direct_dispatcher2d = Rc::new(Dispatcher::new_direct(&wg_dims, &workgroup_size2d));
    let indirect_default = wgpu::util::DispatchIndirect{ x: 0, y: 1, z: 1 };
    let indirect_dispatcher = Rc::new(Dispatcher::new_indirect(&device, indirect_default));
    let indirect_dispatcher2 = Rc::new(Dispatcher::new_indirect(&device, indirect_default));
    let indirect_dispatcher_result = Rc::new(Dispatcher::new_indirect(&device, indirect_default));

    let mut shaders = HashMap::new();
    shaders.extend([
        ("preprocess_rows", (include_str!("shaders/preprocess_rows.wgsl"), Some(Rc::clone(&direct_dispatcher2d)), workgroup_size2d)),
        ("preprocess_cols", (include_str!("shaders/preprocess_cols.wgsl"), Some(Rc::clone(&direct_dispatcher2d)), workgroup_size2d)),
        ("walk", (include_str!("shaders/walk.wgsl"), Some(Rc::clone(&indirect_dispatcher)), workgroup_size1d)),
    ]);

    // let shader = std::fs::read_to_string("src/shaders/characterize.wgsl").unwrap();
    if tracking_params.characterize | characterize_new_points {
        shaders.extend([
            ("characterize", (include_str!("shaders/characterize.wgsl"), Some(Rc::clone(&indirect_dispatcher_result)), workgroup_size1d)),
            // ("characterize", (shader.as_str(), Some(Rc::clone(&indirect_dispatcher_result)), workgroup_size1d))
        ]);
    }
    
    if tracking_params.snr.is_some() | tracking_params.minmass_snr.is_some() {
        shaders.extend([
            ("sum_image_outplace", (include_str!("shaders/sum_image_outplace.wgsl"), None, workgroup_size1d)),
            ("std_image", (include_str!("shaders/std_image.wgsl"), None, workgroup_size1d)),
            ("last_reduction", (include_str!("shaders/last_reduction.wgsl"), None, [1, 1, 1])),
            ("sum_image_inplace", (include_str!("shaders/sum_image_inplace.wgsl"), None, workgroup_size1d)),
        ]);
    }
    
    if tracking_params.illumination_sigma.is_some(){
        shaders.extend([
            ("sum_frames", (include_str!("shaders/correct_illumination/sum_frames.wgsl"), Some(Rc::clone(&direct_dispatcher1d)), workgroup_size1d)),
            // ("divide_frames", (include_str!("shaders/correct_illumination/divide_frames.wgsl"), Some(Rc::clone(&direct_dispatcher1d)), workgroup_size1d)),
            ("smooth_mean_frame", (include_str!("shaders/correct_illumination/smooth_mean_frame.wgsl"), Some(Rc::clone(&direct_dispatcher2d)), workgroup_size2d)),
            ("correct_illumination", (include_str!("shaders/correct_illumination/correct_illumination.wgsl"), Some(Rc::clone(&direct_dispatcher1d)), workgroup_size1d)),
        ]);
    }

    if tracking_params.adaptive_background.is_some(){
        shaders.extend([
            ("adaptive_filter", (include_str!("shaders/adaptive_filter.wgsl"), Some(Rc::clone(&indirect_dispatcher2)), workgroup_size1d)),
        ]);
    }

    let (flavor, mut pipelines, common_buffers) = match &tracking_params.style{
        
        ParamStyle::Trackpy{..} => {
            // let mut shaders = HashMap::from([
            shaders.extend([
                // ("sum_image", (include_str!("shaders/sum_image.wgsl"), Some(Rc::clone(&direct_dispatcher)), workgroup_size2d)),
                // ("std_image", (include_str!("shaders/std_image.wgsl"), Some(Rc::clone(&direct_dispatcher)), workgroup_size2d)),
                ("max_rows", (include_str!("shaders/max_rows.wgsl"), Some(Rc::clone(&direct_dispatcher2d)), workgroup_size2d)),
                ("extract_max", (include_str!("shaders/extract_max.wgsl"), Some(Rc::clone(&direct_dispatcher2d)), workgroup_size2d)),
                // ("preprocess_rows", (include_str!("shaders/preprocess_rows.wgsl"), Some(Rc::clone(&direct_dispatcher1d)), workgroup_size2d)),
                // ("preprocess_cols", (include_str!("shaders/preprocess_cols.wgsl"), Some(Rc::clone(&direct_dispatcher1d)), workgroup_size2d)),
                // ("walk", (include_str!("shaders/walk.wgsl"), Some(Rc::clone(&indirect_dispatcher1d)), workgroup_size1d)),
            ]);

            let mut pipelines = 
            make_pipelines_from_source(shaders, preprocess_source, common_header, &device);
            
            let common_buffers = CommonBuffers::create(
                &tracking_params, &device, size, dims, &mut pipelines,
            );
            
            let order = vec![
                // Some("sum_image".to_string()),
                // Some("std_image".to_string()),
                Some("preprocess_rows".to_string()),
                Some("preprocess_cols".to_string()),
                if !characterize_new_points { Some("max_rows".to_string()) } else {None},
                if !characterize_new_points { Some("extract_max".to_string()) } else {None},
                if !characterize_new_points { Some("walk".to_string()) } else {None},
                // if (tracking_params.characterize | characterize_new_points) { Some("characterize".to_string()) } else {None},
            ];

            let order = order.into_iter().flatten().collect();
            
            // let GpuBuffers::Trackpy(buffers) = create_buffers(&tracking_params, &device, size * 2, dims, None, &mut pipelines) else {unreachable!()};
            let buffers = TrackpyGpuBuffers::create(
                &device, size,
            );

            let flavor = GpuStateFlavor::Trackpy{
                order,
                buffers,
            };


            (flavor, pipelines, common_buffers)
        },
        ParamStyle::Log{ max_radius, min_radius, log_spacing, n_radii, .. } => {
            // let direct_dispatcher = Rc::new(Dispatcher::new_direct(&wg_dims, &workgroup_size2d));
            // let indirect_default = wgpu::util::DispatchIndirect{ x: 0, y: 1, z: 1 };
            // let indirect_dispatcher = Rc::new(Dispatcher::new_indirect(&device, indirect_default));

            // let fftshaders = wgpu_fft::fft::compile_shaders(&device, 
            //     Some(&workgroup_size2d),
            //     Some(&workgroup_size1d));
            // let max_sigma = max_radius / (2 as my_dtype).sqrt();
            // let gauss_size = (max_sigma * 4. + 0.5) as u32 * 2 + 1;
            // let fft_shape = wgpu_fft::fft::get_shape(dims, &[gauss_size, gauss_size]);
            // let fftplan = wgpu_fft::fft::FftPlan::create(&fft_shape, fftshaders, &device, &queue);
            // let init_wg_dims = [gauss_size, gauss_size, 1];
            
            // let shaders = HashMap::from([
            shaders.extend([
                // ("init_log", (include_str!("shaders/log_style/init_log.wgsl"), init_wg_dims, workgroup_size1d)),
                
                
                ("logspace_max", (include_str!("shaders/log_style/logspace_max.wgsl"), Some(Rc::clone(&direct_dispatcher2d)), workgroup_size2d)),
                // ("walk", (include_str!("shaders/walk.wgsl"), Some(Rc::clone(&indirect_dispatcher)), workgroup_size1d)),
                ("separable_log", (include_str!("shaders/log_style/separable_log.wgsl"), Some(Rc::clone(&direct_dispatcher2d)), workgroup_size2d)),
                // ("preprocess_rows", (include_str!("shaders/preprocess_rows.wgsl"), Some(Rc::clone(&direct_dispatcher)), workgroup_size2d)),
                // ("preprocess_cols", (include_str!("shaders/preprocess_cols.wgsl"), Some(Rc::clone(&direct_dispatcher)), workgroup_size2d)),

                // ("characterize", (include_str!("shaders/characterize.wgsl"), [10000, 1, 1], workgroup_size1d)),
            ]);
            
            let mut pipelines = 
            make_pipelines_from_source(shaders, preprocess_source, common_header, &device);
            
            let common_buffers = CommonBuffers::create(
                &tracking_params, &device, size, dims, &mut pipelines,
            );
            
            let buffers = LogGpuBuffers::create(
                &device, size, &common_buffers, &mut pipelines,
            );

            let radii = if *log_spacing{
                let mut start = min_radius.log(10.);
                let end = max_radius.log(10.);
                let diff = (end - start) / (*n_radii - 1) as my_dtype;
                let mut out = (0..(n_radii - 1)).map(|_| {let temp = start; start += diff; (10f32).powf(temp)}).collect::<Vec<_>>();
                out.push((10f32).powf(start));
                out
            } else {
                let mut start = *min_radius;
                let end = *max_radius;
                let diff = (end - start) / (*n_radii - 1) as my_dtype;
                let mut out = (0..(n_radii - 1)).map(|_| {let temp = start; start += diff; temp}).collect::<Vec<_>>();
                out.push(start);
                out
            };
            let flavor = GpuStateFlavor::Log{
                buffers,
                radii,
            };
            (flavor, pipelines, common_buffers,)
        },
    };
    
    
    let std_pass = if tracking_params.snr.is_some() || tracking_params.minmass_snr.is_some(){
        Some(
            StdArray::new(
                &device,
                &pipelines.remove("sum_image_inplace").unwrap(),
                &pipelines.remove("sum_image_outplace").unwrap(),
                &pipelines.remove("last_reduction").unwrap(),
                &pipelines.remove("std_image").unwrap(),
                &common_buffers.frame_buffer,
                &common_buffers.temp_buffer,
                &common_buffers.std_buffer,
                &common_buffers.counter_buffer,
                pic_size as u32,
                &workgroup_size1d,
            )
        )
    } else { None };

    /* Dynamically load shaders to avoid recompilation when debugging. This requires the shaders to be
    in the directory layout as in the repo.
    let shaders = HashMap::from([
        ("max_rows", ("src/shaders/max_rows.wgsl", wg_dims, workgroup_size2d)),
        ("extract_max", ("src/shaders/extract_max.wgsl", wg_dims, workgroup_size2d)),
        ("preprocess_rows", ("src/shaders/preprocess_rows.wgsl", wg_dims, workgroup_size2d)),
        ("preprocess_cols", ("src/shaders/preprocess_cols.wgsl", wg_dims, workgroup_size2d)),
        ("walk", ("src/shaders/walk.wgsl", [10000, 1, 1], [256, 1, 1])),
        ("characterize", ("src/shaders/characterize.wgsl", [10000, 1, 1], [256, 1, 1])),
    ]);

    let shaders = shaders.into_iter().map(|(name, (shader, dims, group_size))| {
        let mut shader_file = File::open(shader).expect(format!("{} not found", shader).as_str());
        let mut shader_string = String::new();
        shader_file.read_to_string(&mut shader_string).unwrap();
        (name, (shader_string, dims, group_size))
    }).collect::<HashMap<_, _>>();
    */

    // let shaders = shaders.into_iter().map(|(name, (source, dims, group_size))|{
    //     let shader_source = preprocess_source(source, &group_size, common_header);
    //     let bindgrouplayout = infer_compute_bindgroup_layout(&device, &shader_source);
    //     let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
    //         label: None,
    //         source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    //     });
    //     (name, (shader, n_workgroups(&dims, &group_size), bindgrouplayout))
    // }).collect::<HashMap<_, _>>();
    
    let mut bind_group_entries = HashMap::new();

    bind_group_entries.extend([
        ("preprocess_rows", vec![vec![
            Some((0, &common_buffers.param_buffer)),
            Some((1, &common_buffers.frame_buffer)),
            Some((3, &common_buffers.temp_buffer)),
        ]]),
        ("preprocess_cols", vec![vec![
            Some((0, &common_buffers.param_buffer)),
            Some((2, &common_buffers.temp_buffer)),
            Some((3, &common_buffers.processed_buffer)), 
        ]]),
    ]);

    if tracking_params.illumination_sigma.is_some(){
        bind_group_entries.extend([
            ("sum_frames", vec![vec![
                Some((0, &common_buffers.frame_buffer)),
                Some((1, &common_buffers.illumination_correcter.as_ref().unwrap().buffer)), 
            ]]),
            ("correct_illumination", vec![vec![
                Some((0, &common_buffers.illumination_correcter.as_ref().unwrap().buffer)), 
                Some((1, &common_buffers.frame_buffer)),
            ]]),
        ]);
    }
    
    let (walk_result, walk_atomic, walk_dispatch) = if tracking_params.adaptive_background.is_some(){
        bind_group_entries.extend([
            ("adaptive_filter", vec![vec![
                Some((0, &common_buffers.param_buffer)),
                Some((1, &common_buffers.particles_buffer2)), 
                Some((2, &common_buffers.particles_buffer)), 
                Some((3, &common_buffers.result_buffer)), 
                Some((4, &common_buffers.std_buffer)), 
                Some((5, &common_buffers.atomic_buffer2)), 
                Some((6, &common_buffers.atomic_buffer)), 
                Some((7, &common_buffers.atomic_filtered_buffer)), 
                Some((8, indirect_dispatcher.as_ref().get_buffer())),
                Some((9, indirect_dispatcher_result.as_ref().get_buffer())),
                Some((10, &common_buffers.frame_buffer)),
                Some((11, &common_buffers.counter_buffer)),
            ], vec![
                Some((0, &common_buffers.param_buffer)),
                Some((1, &common_buffers.particles_buffer)), 
                Some((2, &common_buffers.particles_buffer2)), 
                Some((3, &common_buffers.result_buffer)), 
                Some((4, &common_buffers.std_buffer)), 
                Some((5, &common_buffers.atomic_buffer)), 
                Some((6, &common_buffers.atomic_buffer2)), 
                Some((7, &common_buffers.atomic_filtered_buffer)), 
                Some((8, indirect_dispatcher2.as_ref().get_buffer())),
                Some((9, indirect_dispatcher_result.as_ref().get_buffer())),
                Some((10, &common_buffers.frame_buffer)),
                Some((11, &common_buffers.counter_buffer)),
            ]]),
        ]);
        (&common_buffers.particles_buffer2, &common_buffers.atomic_buffer2, &indirect_dispatcher2)
    } else {
        (&common_buffers.result_buffer, &common_buffers.atomic_filtered_buffer, &indirect_dispatcher_result)
    };

    bind_group_entries.extend([
        ("walk", vec![vec![
            Some((0, &common_buffers.param_buffer)),
            Some((1, &common_buffers.processed_buffer)), 
            Some((2, &common_buffers.particles_buffer)),
            Some((3, &common_buffers.atomic_buffer)),
            Some((4, walk_atomic)),
            Some((5, walk_result)),
            Some((6, &common_buffers.frame_buffer)),
            Some((7, &common_buffers.std_buffer)),
            Some((8, walk_dispatch.as_ref().get_buffer())),
        ]]),
    ]);

    if tracking_params.characterize{
        bind_group_entries.extend([
            ("characterize", vec![vec![
                Some((0, &common_buffers.param_buffer)),
                Some((1, &common_buffers.processed_buffer)), 
                Some((2, common_buffers.raw_frame.as_ref().unwrap_or(&common_buffers.frame_buffer))),
                Some((3, &common_buffers.atomic_filtered_buffer)),
                Some((4, &common_buffers.result_buffer)),
            ]]),
        ]);
    }
    
    match flavor{
    GpuStateFlavor::Trackpy{ref buffers, ..} => {
            bind_group_entries.extend([
            ("max_rows", vec![vec![
                Some((0, &common_buffers.param_buffer)),
                Some((1, &common_buffers.processed_buffer)), 
                Some((2, &buffers.max_rows)),
            ]]),
            ("extract_max", vec![vec![
                Some((0, &common_buffers.param_buffer)),
                Some((1, &common_buffers.processed_buffer)), 
                Some((2, &buffers.max_rows)),
                Some((3, &common_buffers.atomic_buffer)),
                Some((4, &common_buffers.particles_buffer)),
                Some((5, indirect_dispatcher.as_ref().get_buffer())),
                Some((6, &common_buffers.std_buffer)),
            ]]),
        ])
    },

    GpuStateFlavor::Log{ref buffers, ..} => {
        bind_group_entries.extend([
            ("logspace_max", (0..3isize).map(|i| {
                vec![
                    Some((0, &common_buffers.param_buffer)),
                    Some((1, &buffers.logspace_buffers[((i-1).rem_euclid(3)) as usize].0)),
                    Some((2, &buffers.logspace_buffers[i as usize].0)),
                    Some((3, &buffers.logspace_buffers[((i+1).rem_euclid(3)) as usize].0)),
                    Some((4, &common_buffers.atomic_buffer)),
                    Some((5, &common_buffers.particles_buffer)),
                    Some((7, indirect_dispatcher.as_ref().get_buffer())),
                    Some((8, &common_buffers.std_buffer)),
                    Some((9, &common_buffers.processed_buffer)),
                ]}).collect::<Vec<_>>()),
        ])
    }
    };

    let bind_group_entries = bind_group_entries
        .iter().map(|(&name, bind_entries_vector)| {
            let bind_entries_vector = bind_entries_vector.iter().map(|bind_entries|{
                bind_entries.iter().flatten().map(|(i, buffer)|
                    wgpu::BindGroupEntry {
                    binding: *i as u32,
                    resource: buffer.as_entire_binding()}).collect::<Vec<_>>()
                }).collect::<Vec<_>>();
                (name, bind_entries_vector)
            })
        .collect::<HashMap<_, _>>();


    let mut passes = pipelines.into_iter()
        .map(|(name, (pipeline, layout, dispatcher))|{
        let entries = &bind_group_entries[name.as_str()];
        let passes_for_name = entries.iter().map(|entry|{
            let bindgroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &entry[..],
            });
            let pass = FullComputePass{
                pipeline: Rc::clone(&pipeline),
                bindgroup,
                dispatcher: dispatcher.as_ref().map(|d| Rc::clone(d)),
            };
            pass
        }).collect::<Vec<_>>();
        
        (name, passes_for_name)
    }).collect::<HashMap<_, _>>();

    passes.get_mut("adaptive_filter").as_mut().map(|v| v[1].dispatcher = Some(indirect_dispatcher));
    
    
    let characterize = Cell::new(passes.remove("characterize").map(|mut v| (v.remove(0), workgroup_size1d)));

    let state = GpuState{
        device,
        queue,
        passes,
        common_buffers,
        pic_byte_size: size,
        pic_size,
        flavor,
        dims: dims.clone(),
        std_pass,
        characterize
    };
    Ok(state)

}



fn make_pipelines_from_source<F: Fn(&str, &[u32; 3], &str) -> String>(
    shaders: HashMap<&str, (&str, Option<Rc<Dispatcher>>, [u32; 3])>,
    preprocessor: F,
    common_header: &str,
    device: &wgpu::Device)
    -> HashMap<String, (Rc<wgpu::ComputePipeline>, wgpu::BindGroupLayout, Option<Rc<Dispatcher>>)>
    {
    
    let output = shaders.into_iter().map(|(name, (source, dispatcher, group_size))|{
        let shader_source = preprocessor(source, &group_size, common_header);
        let bindgrouplayout = infer_compute_bindgroup_layout(&device, &shader_source);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipelinelayout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label: None,
            bind_group_layouts: &[&bindgrouplayout],
            push_constant_ranges: &[wgpu::PushConstantRange{
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..16,
            }],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
            label: None,
            layout: Some(&pipelinelayout),
            module: &shader,
            entry_point: "main",
        });
        (name.to_string(), (Rc::new(pipeline), bindgrouplayout, dispatcher))
    }).collect::<HashMap<_, _>>();

    output
}
