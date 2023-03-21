use std::{num::NonZeroU32, mem};

use eframe::egui_wgpu::RenderState;
use eframe::wgpu;
use ndarray::ArrayView2;
use anyhow::*;
// use crate::colormaps::MAPS;
use wgpu::util::DeviceExt;

use crate::texture;
// use bytemuck::Pod;


pub struct Texture {
    pub texture: eframe::wgpu::Texture,
    pub view: eframe::wgpu::TextureView,
    pub sampler: eframe::wgpu::Sampler,
    pub cmap: eframe::wgpu::Buffer,
    pub minmax: eframe::wgpu::Buffer,
}

impl Texture {
    pub fn from_image(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        img: &ArrayView2<f32>,
        label: Option<&str>,
        cmap: [f32; 120],
        cpu_minmax: &[f32; 2],
    ) -> Result<Self> {
        let rgba = img.as_slice().unwrap();
        let dimensions = img.shape();
        let dimensions = (dimensions[1] as u32, dimensions[0] as u32);

        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            bytemuck::cast_slice(rgba),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: NonZeroU32::new(4 * dimensions.0),
                rows_per_image: NonZeroU32::new(dimensions.1),
            },
            size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let cmap = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: None,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            contents: bytemuck::cast_slice(&cmap),
        });

        // let img_min = img.iter().fold(f32::INFINITY, |acc, ele| {
        //     match acc.partial_cmp(ele){
        //         Some(std::cmp::Ordering::Less) => {
        //             acc
        //         },
        //         Some(std::cmp::Ordering::Equal) => {
        //             acc
        //         },
        //         Some(std::cmp::Ordering::Greater) => {
        //             *ele
        //         },
        //         None => acc
        //     }
        // });
        // let img_max = img.iter().fold(f32::NEG_INFINITY, |acc, ele| {
        //     match acc.partial_cmp(ele){
        //         Some(std::cmp::Ordering::Less) => {
        //             *ele
        //         },
        //         Some(std::cmp::Ordering::Equal) => {
        //             acc
        //         },
        //         Some(std::cmp::Ordering::Greater) => {
        //             acc
        //         },
        //         None => acc
        //     }
        // });

        // let cpu_minmax = [img_min, img_max];
        
        let minmax = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: None,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            contents: bytemuck::cast_slice(cpu_minmax),
        });

        Ok(Self {
            texture,
            view,
            sampler,
            cmap,
            minmax
        })
    }
}


#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
}

impl Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}


fn gen_vertices(bounding: &egui::Rect) -> [Vertex; 4]{
    [
        Vertex {
            position: [-1., 1., 0.0],
            tex_coords: [bounding.min.x, bounding.min.y],
        }, // A
        Vertex {
            position: [-1., -1., 0.0],
            tex_coords: [bounding.min.x, bounding.max.y],
        }, // B
        Vertex {
            position: [1., 1., 0.0],
            tex_coords: [bounding.max.x, bounding.min.y],
        }, // C
        Vertex {
            position: [1., -1., 0.0],
            tex_coords: [bounding.max.x, bounding.max.y],
        }, // D
    ]
}

const INDICES: &[u16] = &[0, 1, 2, 1, 3, 2];

pub struct ColormapRenderResources {
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    // NEW!
    #[allow(dead_code)]
    diffuse_texture: texture::Texture,
    diffuse_bind_group: wgpu::BindGroup,
	// frame_provider: (Box<dyn gpu_tracking::decoderiter::FrameProvider<
	// 	Frame = Vec<f32>,
	// 	FrameIter = Box<dyn Iterator<Item = Result<Vec<f32>, Error>>>
	// >>, [u32; 2])
}

impl ColormapRenderResources {
    pub fn new(wgpu_render_state: &RenderState, frame_view: &ArrayView2<f32>, cmap: [f32; 120], minmax: &[f32; 2]) -> Self {
        // let diffuse_bytes = include_bytes!("happy-tree.png");
        // let frame = provider.get_frame(0).unwrap();
        // let frame = Array::from_shape_vec([dims[0] as usize, dims[1] as usize], frame).unwrap();
        // let frame_view = frame.view();

        // let wgpu_render_state = cc.wgpu_render_state.as_ref()?;
        let device = wgpu_render_state.device.clone();
        let queue = wgpu_render_state.queue.clone();
        
        let diffuse_texture =
            texture::Texture::from_image(&device, &queue, frame_view, None, cmap, minmax).unwrap();

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let diffuse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(
                        wgpu::BufferBinding{
                            buffer: &diffuse_texture.cmap,
                            offset: 0,
                            size: None,
                        }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(
                        wgpu::BufferBinding{
                            buffer: &diffuse_texture.minmax,
                            offset: 0,
                            size: None,
                        }),
                },
            ],
            label: Some("diffuse_bind_group"),
        });

        // let shader_str = std::fs::read_to_string("src/shader.wgsl").unwrap();
        let shader_str = include_str!("texture.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_str.into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&texture_bind_group_layout],
                push_constant_ranges: &[],
            });

        
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu_render_state.target_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::REPLACE,
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::POLYGON_MODE_LINE
                // or Features::POLYGON_MODE_POINT
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            // If the pipeline will be used with a multiview render pass, this
            // indicates how many array layers the attachments will have.
            multiview: None,
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&gen_vertices(&egui::Rect::from_x_y_ranges(0.0..=1.0, 0.0..=1.0))),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });
        let num_indices = INDICES.len() as u32;

        Self {
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            diffuse_texture,
            diffuse_bind_group,
        }
    }

    pub fn set_cmap(&mut self, queue: &wgpu::Queue, cmap: &[f32; 120]){
        queue.write_buffer(&self.diffuse_texture.cmap, 0, bytemuck::cast_slice(cmap))
    }

	pub fn paint<'rp>(&'rp self, render_pass: &mut wgpu::RenderPass<'rp>){
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
	}

    pub fn resize(&mut self, queue: &wgpu::Queue, bounding: &egui::Rect){
        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&gen_vertices(bounding)))
    }

    pub fn get_min(frame: &ArrayView2<f32>) -> f32{
        frame.iter().fold(f32::INFINITY, |acc, ele| {
            match acc.partial_cmp(ele){
                Some(std::cmp::Ordering::Less) => {
                    acc
                },
                Some(std::cmp::Ordering::Equal) => {
                    acc
                },
                Some(std::cmp::Ordering::Greater) => {
                    *ele
                },
                None => acc
            }
        })
    }
    
    pub fn get_max(frame: &ArrayView2<f32>) -> f32{
        frame.iter().fold(f32::NEG_INFINITY, |acc, ele| {
            match acc.partial_cmp(ele){
                Some(std::cmp::Ordering::Less) => {
                    *ele
                },
                Some(std::cmp::Ordering::Equal) => {
                    acc
                },
                Some(std::cmp::Ordering::Greater) => {
                    acc
                },
                None => acc
            }
        })
    }

    pub fn get_minmax(frame: &ArrayView2<f32>) -> [f32; 2]{
        frame.iter().fold([f32::INFINITY, f32::NEG_INFINITY], |acc, ele| {
            let min = match acc[0].partial_cmp(ele){
                Some(std::cmp::Ordering::Less) => {
                    acc[0]
                },
                Some(std::cmp::Ordering::Equal) => {
                    acc[0]
                },
                Some(std::cmp::Ordering::Greater) => {
                    *ele
                },
                None => acc[0]
            };
            let max = match acc[1].partial_cmp(ele){
                Some(std::cmp::Ordering::Less) => {
                    *ele
                },
                Some(std::cmp::Ordering::Equal) => {
                    acc[1]
                },
                Some(std::cmp::Ordering::Greater) => {
                    acc[1]
                },
                None => acc[1]
            };
            [min, max]
        })
    }
    
	pub fn update_texture(&mut self, queue: &wgpu::Queue, frame: &ArrayView2<f32>, minmax: &[f32; 2]){
		let dimensions = frame.shape();
		
        let size = wgpu::Extent3d {
            width: dimensions[1] as u32,
            height: dimensions[0] as u32,
            depth_or_array_layers: 1,
        };
		queue.write_buffer(&self.diffuse_texture.minmax, 0, bytemuck::cast_slice(minmax));
		
        queue.write_texture(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &self.diffuse_texture.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            bytemuck::cast_slice(frame.as_slice().unwrap()),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: NonZeroU32::new(4 * dimensions[1] as u32),
                rows_per_image: NonZeroU32::new(dimensions[0] as u32),
            },
            size,
        );

	}
}
