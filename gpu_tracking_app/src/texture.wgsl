// Vertex shader

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
    out.clip_position = vec4<f32>(model.position, 1.0);
    return out;
}

// Fragment shader

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

@group(0) @binding(2)
var<uniform> cmap: array<vec4<f32>, 30>;

@group(0) @binding(3)
var<uniform> minmax: vec2<f32>;

fn interpolate_cmap(t: f32) -> vec4<f32>{
    let val = t * 29.;
    let ind = i32(val);
    let little_t = val - f32(ind);
    return mix(cmap[ind], cmap[ind + 1], little_t);
} 

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var sample = textureSample(t_diffuse, s_diffuse, in.tex_coords);
    let t = clamp((sample.x - minmax[0]) / (minmax[1] - minmax[0]), 0., 1.);
    let out = interpolate_cmap(t);
    return out;
}
