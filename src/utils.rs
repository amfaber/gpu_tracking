
use std::rc::Rc;
use wgpu::{self, util::DeviceExt};
use crate::gpu_setup::GpuState;
use regex;


pub unsafe fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    ::std::slice::from_raw_parts(
        (p as *const T) as *const u8,
        ::std::mem::size_of::<T>(),
    )
}

pub struct SeparableConvolution<const N: usize>{
    input_pass: FullComputePass,
    passes: [FullComputePass; 2],
}

pub enum Dispatcher{
    Direct([u32; 3]),
    Indirect{
        dispatcher: wgpu::Buffer,
        resetter: wgpu::Buffer,
    }
}

impl Dispatcher{
    pub fn new_direct(dims: &[u32; 3], wgsize: &[u32; 3]) -> Self { 
        let mut n_workgroups = [0, 0, 0];
        for i in 0..3 {
            n_workgroups[i] = (dims[i] + wgsize[i] - 1) / wgsize[i];
        }
        Self::Direct(n_workgroups)
    }

    pub fn new_indirect(device: &wgpu::Device, default: wgpu::util::DispatchIndirect) -> Self {
        let dispatcher = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: None,
            contents: default.as_bytes(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });

        let resetter = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: None,
            contents: default.as_bytes(),
            usage: wgpu::BufferUsages::COPY_SRC
        });
        
        Self::Indirect{
            dispatcher,
            resetter,
        }
    }
    
    pub fn reset_indirect(&self, encoder: &mut wgpu::CommandEncoder){
        match self{
            Self::Indirect { dispatcher, resetter } => {
                encoder.copy_buffer_to_buffer(resetter, 0, dispatcher, 0, resetter.size())
            },
            Self::Direct(_) => {}
        }
    }

    pub fn get_buffer(&self) -> &wgpu::Buffer{
        match self{
            Self::Indirect { dispatcher, .. } => { dispatcher },
            Self::Direct(_) => panic!("Tried to get buffer of a direct dispatcher.")
        }
    }
}

pub struct FullComputePass{
    pub bindgroup: wgpu::BindGroup,
    pub dispatcher: Option<Rc<Dispatcher>>,
    pub pipeline: Rc<wgpu::ComputePipeline>,
}


impl FullComputePass{
    pub fn execute_with_dispatcher(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        push_constants: &[u8],
        dispatcher: &Dispatcher,
    ){
        let mut cpass = encoder.begin_compute_pass(&Default::default());
        cpass.set_bind_group(0, &self.bindgroup, &[]);
        cpass.set_pipeline(&self.pipeline);
        if push_constants.len() > 0{
            cpass.set_push_constants(0, push_constants);
        }
        match dispatcher{
            Dispatcher::Direct(ref wg_n) => {
                cpass.dispatch_workgroups(wg_n[0], wg_n[1], wg_n[2]);
            },
            Dispatcher::Indirect { ref dispatcher, .. } => {
                cpass.dispatch_workgroups_indirect(dispatcher, 0);
            }
        }
        
    }
    
    pub fn execute(&self, encoder: &mut wgpu::CommandEncoder, push_constants: &[u8]){
        self.execute_with_dispatcher(encoder, push_constants, self.dispatcher.as_ref().unwrap().as_ref());
    }

    pub fn reset_indirect(&self, encoder: &mut wgpu::CommandEncoder){
        match self.dispatcher{
            Some(ref dispatcher) => {
                dispatcher.reset_indirect(encoder);
            },
            None => {}
        }
    }
}

impl<const N: usize> SeparableConvolution<N>{
    pub fn new<'a>(
        device: &wgpu::Device,
        pipeline: &(Rc<wgpu::ComputePipeline>, wgpu::BindGroupLayout, Option<Rc<Dispatcher>>),
        input_buffer: &wgpu::Buffer,
        output_buffer: &wgpu::Buffer,
        temp_buffer: &wgpu::Buffer,
        additional_buffers: impl IntoIterator<Item = &'a wgpu::Buffer>,
    ) -> Self{
        let mut bind_group_entries_output_first = vec![
            wgpu::BindGroupEntry{
                binding: 0,
                resource: output_buffer.as_entire_binding(),
            },
            
            wgpu::BindGroupEntry{
                binding: 1,
                resource: temp_buffer.as_entire_binding(),
            },
        ];
        let mut bind_group_entries_temp_first = vec![
            wgpu::BindGroupEntry{
                binding: 0,
                resource: temp_buffer.as_entire_binding(),
            },

            wgpu::BindGroupEntry{
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
        ];

        let first_out = if N % 2 == 0{
            temp_buffer
        } else {
            output_buffer
        };
        
        let mut bind_group_entries_input = vec![
            wgpu::BindGroupEntry{
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry{
                binding: 1,
                resource: first_out.as_entire_binding(),
            },
        ];

        for (i, additional_buffer) in additional_buffers.into_iter().enumerate(){
            bind_group_entries_output_first.push(
                wgpu::BindGroupEntry{
                    binding: i as u32 + 2,
                    resource: additional_buffer.as_entire_binding(),
                }
            );

            bind_group_entries_temp_first.push(
                wgpu::BindGroupEntry{
                    binding: i as u32 + 2,
                    resource: additional_buffer.as_entire_binding(),
                }
            );

            bind_group_entries_input.push(
                wgpu::BindGroupEntry{
                    binding: i as u32 + 2,
                    resource: additional_buffer.as_entire_binding(),
                }
            );
        }

        let bind_group_output_first = device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: None,
            layout: &pipeline.1,
            entries: &bind_group_entries_output_first[..],
        });

        let bind_group_temp_first = device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: None,
            layout: &pipeline.1,
            entries: &bind_group_entries_temp_first[..],
        });

        let bind_group_input = device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: None,
            layout: &pipeline.1,
            entries: &bind_group_entries_input[..],
        });


        let full_output_first = FullComputePass{
            pipeline: Rc::clone(&pipeline.0),
            bindgroup: bind_group_output_first,
            dispatcher: pipeline.2.as_ref().map(|d| Rc::clone(d)),
        };

        let full_temp_first = FullComputePass{
            pipeline: Rc::clone(&pipeline.0),
            bindgroup: bind_group_temp_first,
            dispatcher: pipeline.2.as_ref().map(|d| Rc::clone(d)),
        };

        let full_input = FullComputePass{
            pipeline: Rc::clone(&pipeline.0),
            bindgroup: bind_group_input,
            dispatcher: pipeline.2.as_ref().map(|d| Rc::clone(d)),
        };
        
        let base = if N % 2 == 0{
            [full_temp_first, full_output_first]
        } else{
            [full_output_first, full_temp_first]
        };

        Self{
            passes: base,
            input_pass: full_input,
        }
    }

    pub fn execute(&self, encoder: &mut wgpu::CommandEncoder, push_constants: &[u8]){
        let input = std::iter::once(&self.input_pass);
        for (dim, pass) in input.chain(self.passes.iter().cycle().take(N-1)).enumerate(){
            let push_constants = Vec::from_iter(unsafe{ any_as_u8_slice(&(dim as u32)).iter().chain(push_constants.iter()).cloned() });
            pass.execute(encoder, &push_constants[..]);
        }
    }
}


pub struct Laplace<const N: usize>{
    pass: SeparableConvolution<N>,
}



impl<const N: usize> Laplace<N>{
    pub fn new<'a>(
        device: &wgpu::Device,
        pipeline: &(Rc<wgpu::ComputePipeline>, wgpu::BindGroupLayout, Option<Rc<Dispatcher>>),
        input_buffer: &wgpu::Buffer,
        output_buffer: &'a wgpu::Buffer,
        temp1: &wgpu::Buffer,
        temp2: &wgpu::Buffer,
        additional_buffers: impl IntoIterator<Item = &'a wgpu::Buffer>,
    ) -> Self{
        let additional_buffers = [output_buffer].into_iter().chain(additional_buffers.into_iter());
        let sep = SeparableConvolution::<N>::new(
            device,
            pipeline,
            input_buffer,
            temp1,
            temp2,
            additional_buffers
        );

        Self{
            pass: sep,
        }
    }

    pub fn execute(&self, encoder: &mut wgpu::CommandEncoder, push_constants: &[u8]){

        for diff_dim in 0..(N as u32){
            let push_constants = Vec::from_iter(unsafe{ any_as_u8_slice(&(diff_dim)).iter().chain(push_constants.iter()).cloned() });
            self.pass.execute(encoder, &push_constants[..]);
        }
    }
}


pub struct MeanArray{
    // outplace: FullComputePass,
    inplace: FullComputePass,
    last_reduction: FullComputePass,
    size: u32,
    workgroup_size1d: [u32; 3],
    sqrt_mean: bool,
}

impl MeanArray{
    fn new(
        device: &wgpu::Device,
        inplace_sum_pipeline: &(Rc<wgpu::ComputePipeline>, wgpu::BindGroupLayout, Option<Rc<Dispatcher>>),
        // outplace_sum_pipeline: &(Rc<wgpu::ComputePipeline>, wgpu::BindGroupLayout, Option<Rc<Dispatcher>>),
        mean_pipeline: &(Rc<wgpu::ComputePipeline>, wgpu::BindGroupLayout, Option<Rc<Dispatcher>>),
        // frame_buffer: &wgpu::Buffer,
        temp_buffer: &wgpu::Buffer,
        mean_buffer: &wgpu::Buffer,
        n_elements_buffer: &wgpu::Buffer,
        size: u32,
        workgroup_size1d: &[u32; 3],
        sqrt_mean: bool,
    ) -> Self{
        // let mut outplace_bind_group_entries = [
        //     wgpu::BindGroupEntry{
        //         binding: 0,
        //         resource: frame_buffer.as_entire_binding(),
        //     },
            
        //     wgpu::BindGroupEntry{
        //         binding: 1,
        //         resource: temp_buffer.as_entire_binding(),
        //     },
        // ];
        let inplace_bind_group_entries = [
            wgpu::BindGroupEntry{
                binding: 0,
                resource: temp_buffer.as_entire_binding(),
            },
        ];
        
        let last_reduction_bind_group_entries = [
            wgpu::BindGroupEntry{
                binding: 0,
                resource: temp_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry{
                binding: 1,
                resource: mean_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry{
                binding: 2,
                resource: n_elements_buffer.as_entire_binding(),
            },
        ];
        // let outplace_bindgroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
        //     label: None,
        //     layout: &outplace_sum_pipeline.1,
        //     entries: &outplace_bind_group_entries[..],
        // });
        
        let inplace_bindgroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &inplace_sum_pipeline.1,
            entries: &inplace_bind_group_entries[..],
        });

        let last_reduction_bindgroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &mean_pipeline.1,
            entries: &last_reduction_bind_group_entries[..],
        });

        // let outplace = FullComputePass{
        //     pipeline: Rc::clone(&outplace_sum_pipeline.0),
        //     bindgroup: outplace_bindgroup,
        //     dispatcher: None,
        // };

        let inplace = FullComputePass{
            pipeline: Rc::clone(&inplace_sum_pipeline.0),
            bindgroup: inplace_bindgroup,
            dispatcher: None,
        };
        
        let last_reduction = FullComputePass{
            pipeline: Rc::clone(&mean_pipeline.0),
            bindgroup: last_reduction_bindgroup,
            dispatcher: None,
        };
        
        Self{
            // outplace,
            inplace,
            size,
            workgroup_size1d: workgroup_size1d.clone(),
            last_reduction,
            sqrt_mean,
        }
    }
    
    
    // pub fn execute(&self, encoder: &mut wgpu::CommandEncoder, state: &GpuState, staging_buffer: &wgpu::Buffer){
    pub fn execute(&self, encoder: &mut wgpu::CommandEncoder){
        let mut to_start = self.size;
        // to_start += 7;
        // to_start /= 8;
        while to_start > 24{
            to_start += 7;
            to_start /= 8;
            let dispatcher = Dispatcher::new_direct(&[to_start, 1, 1], &self.workgroup_size1d);
            self.inplace.execute_with_dispatcher(encoder, unsafe { any_as_u8_slice(&(to_start, self.sqrt_mean as u32)) }, &dispatcher);
            // let to_print = vec![
            //     &state.common_buffers.std_buffer,
            //     &state.common_buffers.temp_buffer,
            //     ];
            // std::fs::write("testing/mean_pic/shape0.dump", bytemuck::cast_slice(&[1u32, 0]));
            // std::fs::write("testing/mean_pic/shape1.dump", bytemuck::cast_slice(&[state.dims[0]*state.dims[1], 0]));
            // inspect_through_state(
            //     &to_print[..],
            //     staging_buffer,
            //     encoder,
            //     "testing/mean_pic",
            //     state,
            // );
        }
        let dispatcher = Dispatcher::Direct([1, 1, 1]);
        self.last_reduction.execute_with_dispatcher(encoder, unsafe { any_as_u8_slice(&(to_start, self.sqrt_mean as u32)) }, &dispatcher);
    }
}

pub struct MeanArrayOutplace{
    outplace: FullComputePass,
    meanarray: MeanArray,
    size: u32,
}

impl MeanArrayOutplace{
    fn new(
        device: &wgpu::Device,
        inplace_sum_pipeline: &(Rc<wgpu::ComputePipeline>, wgpu::BindGroupLayout, Option<Rc<Dispatcher>>),
        outplace_sum_pipeline: &(Rc<wgpu::ComputePipeline>, wgpu::BindGroupLayout, Option<Rc<Dispatcher>>),
        mean_pipeline: &(Rc<wgpu::ComputePipeline>, wgpu::BindGroupLayout, Option<Rc<Dispatcher>>),
        frame_buffer: &wgpu::Buffer,
        temp_buffer: &wgpu::Buffer,
        mean_buffer: &wgpu::Buffer,
        n_elements_buffer: &wgpu::Buffer,
        size: u32,
        workgroup_size1d: &[u32; 3],
        sqrt_mean: bool,
    ) -> Self{
        let meanarray = MeanArray::new(
            device,
            inplace_sum_pipeline,
            mean_pipeline,
            // frame_buffer,
            temp_buffer,
            mean_buffer,
            n_elements_buffer,
            (size + 7) / 8,
            workgroup_size1d,
            sqrt_mean,
        );
        let outplace_bind_group_entries = [
            wgpu::BindGroupEntry{
                binding: 0,
                resource: frame_buffer.as_entire_binding(),
            },
            
            wgpu::BindGroupEntry{
                binding: 1,
                resource: temp_buffer.as_entire_binding(),
            },
        ];
        let outplace_bindgroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &outplace_sum_pipeline.1,
            entries: &outplace_bind_group_entries[..],
        });
        
        let outplace = FullComputePass{
            pipeline: Rc::clone(&outplace_sum_pipeline.0),
            bindgroup: outplace_bindgroup,
            dispatcher: None,
        };

        
        Self{
            outplace,
            meanarray,
            size,
        }
    }
    
    
    pub fn execute(&self, encoder: &mut wgpu::CommandEncoder){
    // pub fn execute(&self, encoder: &mut wgpu::CommandEncoder, state: &GpuState, staging_buffer: &wgpu::Buffer){
        let mut to_start = self.size;
        to_start += 7;
        to_start /= 8;
        let dispatcher = Dispatcher::new_direct(&[to_start, 1, 1], &self.meanarray.workgroup_size1d);
        self.outplace.execute_with_dispatcher(encoder, unsafe { any_as_u8_slice(&(to_start)) }, &dispatcher);
        // self.outplace.execute_with_dispatcher(encoder, unsafe { any_as_u8_slice(&(to_start, self.meanarray.sqrt_mean as u32)) }, &dispatcher);
        
        self.meanarray.execute(encoder);
        // self.meanarray.execute(encoder, state, staging_buffer);
        // let to_print = vec![
        //     &state.common_buffers.std_buffer,
        //     &state.common_buffers.temp_buffer,
        //     ];
        // std::fs::write("testing/mean_pic/shape0.dump", bytemuck::cast_slice(&[1u32, 0]));
        // std::fs::write("testing/mean_pic/shape1.dump", bytemuck::cast_slice(&[state.dims[0]*state.dims[1], 0]));
        // inspect_through_state(
        //     &to_print[..],
        //     staging_buffer,
        //     encoder,
        //     "testing/mean_pic",
        //     state,
        // );
    }
}

pub struct StdArray{
    mean_pass1: MeanArrayOutplace,
    mean_pass2: MeanArray,
    std_pass: FullComputePass,
}

impl StdArray{
    pub fn new(
        device: &wgpu::Device,
        inplace_sum_pipeline: &(Rc<wgpu::ComputePipeline>, wgpu::BindGroupLayout, Option<Rc<Dispatcher>>),
        outplace_sum_pipeline: &(Rc<wgpu::ComputePipeline>, wgpu::BindGroupLayout, Option<Rc<Dispatcher>>),
        mean_pipeline: &(Rc<wgpu::ComputePipeline>, wgpu::BindGroupLayout, Option<Rc<Dispatcher>>),
        std_pipeline: &(Rc<wgpu::ComputePipeline>, wgpu::BindGroupLayout, Option<Rc<Dispatcher>>),
        frame_buffer: &wgpu::Buffer,
        temp_buffer: &wgpu::Buffer,
        std_buffer: &wgpu::Buffer,
        n_elements_buffer: &wgpu::Buffer,
        size: u32,
        workgroup_size1d: &[u32; 3],
    ) -> Self{
        let mean_pass1 = MeanArrayOutplace::new(
            device,
            inplace_sum_pipeline,
            outplace_sum_pipeline,
            mean_pipeline,
            frame_buffer,
            temp_buffer,
            std_buffer,
            n_elements_buffer,
            size,
            workgroup_size1d,
            false,
        );
        
        let mean_pass2 = MeanArray::new(
            device,
            inplace_sum_pipeline,
            // outplace_sum_pipeline,
            mean_pipeline,
            // temp_buffer,
            temp_buffer,
            std_buffer,
            n_elements_buffer,
            size,
            workgroup_size1d,
            true,
        );

        let std_bind_group_entries = [
            wgpu::BindGroupEntry{
                binding: 0,
                resource: frame_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry{
                binding: 1,
                resource: temp_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry{
                binding: 2,
                resource: std_buffer.as_entire_binding(),
            },
        ];
        
        let std_bindgroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &std_pipeline.1,
            entries: &std_bind_group_entries[..],
        });
        
        let std_pass = FullComputePass{
            pipeline: Rc::clone(&std_pipeline.0),
            bindgroup: std_bindgroup,
            dispatcher: Some(Rc::new(Dispatcher::new_direct(&[size, 1, 1], workgroup_size1d))),
        };
        
        Self{
            mean_pass1,
            mean_pass2,
            std_pass,
        }
    }

    pub fn execute(&self, encoder: &mut wgpu::CommandEncoder){
    // pub fn execute(&self, encoder: &mut wgpu::CommandEncoder, state: &GpuState, staging_buffer: &wgpu::Buffer){
        self.mean_pass1.execute(encoder);
        // self.mean_pass1.execute(encoder, state, staging_buffer);
        self.std_pass.execute(encoder, bytemuck::cast_slice(&[self.mean_pass2.size]));
        self.mean_pass2.execute(encoder);
        // self.mean_pass2.execute(encoder, state, staging_buffer);
        // let to_print = vec![
        //     &state.common_buffers.std_buffer,
        //     &state.common_buffers.temp_buffer,
        //     ];
        // std::fs::write("testing/mean_pic/shape0.dump", bytemuck::cast_slice(&[1u32, 0]));
        // std::fs::write("testing/mean_pic/shape1.dump", bytemuck::cast_slice(&[2, state.dims[0], state.dims[1], 0]));
        // inspect_through_state(
        //     &to_print[..],
        //     staging_buffer,
        //     encoder,
        //     "testing/mean_pic",
        //     state,
        // );
    }
}

pub fn inspect_through_state<P: AsRef<std::path::Path>>(
    buffers_to_inspect: &[&wgpu::Buffer],
    mappable_buffer: &wgpu::Buffer,
    // queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    // mut encoder: wgpu::CommandEncoder,
    // device: &wgpu::Device,
    file_path: P,
    state: &GpuState,
    ) -> ! {
    inspect_buffers(
        buffers_to_inspect,
        mappable_buffer,
        &state.queue,
        encoder,
        &state.device,
        file_path,
    )
}


pub fn inspect_buffers<P: AsRef<std::path::Path>>(
    buffers_to_inspect: &[&wgpu::Buffer],
    mappable_buffer: &wgpu::Buffer,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    // mut encoder: wgpu::CommandEncoder,
    device: &wgpu::Device,
    file_path: P,
    ) -> ! {
    
    let path = file_path.as_ref().to_owned();
    let encoder = std::mem::replace(encoder, device.create_command_encoder(&Default::default()));
    queue.submit(Some(encoder.finish()));
    
    for (i, &buffer) in buffers_to_inspect.iter().enumerate(){
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.clear_buffer(mappable_buffer, 0, None);
        encoder.copy_buffer_to_buffer(buffer, 0, mappable_buffer, 0, buffer.size());
        queue.submit(Some(encoder.finish()));
        let slice = mappable_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range()[..].to_vec();
        let mut path = path.clone();
        path.push(format!("dump{}.bin", i));
        std::fs::write(&path, &data[..buffer.size() as usize]).unwrap();
        drop(slice);
        mappable_buffer.unmap();
    }

    panic!("intended panic")
}



pub fn infer_compute_bindgroup_layout(device: &wgpu::Device, source: &str) -> wgpu::BindGroupLayout{
    let re = regex::Regex::new(r"@binding\((?P<idx>\d+)\)\s*var<(?P<type>.*?)>").unwrap();

    let mut entries = Vec::new();
    for capture in re.captures_iter(source){
        let idx: u32 = capture.name("idx").expect("Regex failed parse at binding idx").as_str().parse().unwrap();
        let ty = capture.name("type").expect("Regex failed parse at binding type").as_str();
        let ty = match ty{
            "uniform" => wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            "storage, read" => wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            "storage, read_write" => wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            _ => panic!("Unrecognized binding type: {}", ty)
        };
        entries.push(
            wgpu::BindGroupLayoutEntry{
                binding: idx,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty,
                count: None,
            }
        );
    }
    let bindgrouplayout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
        label: None,
        entries: &entries[..]
    });

    bindgrouplayout
    
}