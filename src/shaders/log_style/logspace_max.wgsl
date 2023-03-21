// struct Shape{
//     nrows: u32,
//     ncols: u32,
// }

struct PushConstants{
    edge: i32,
    radius: f32,
}

struct ParticleLocation{
    x: i32,
    y: i32,
    r: f32,
    log_space: f32,
}

struct WorkgroupSize {
    x: atomic<u32>,
    y: atomic<u32>,
    z: atomic<u32>,
}

// @group(0) @binding(0)
// var<uniform> shape: Shape;

@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(1)
var<storage, read> bottom: array<f32>;
// var<storage, read> bottom: array<vec2<f32>>;

@group(0) @binding(2)
var<storage, read> middle: array<f32>;
// var<storage, read> middle: array<vec2<f32>>;

@group(0) @binding(3)
var<storage, read> top: array<f32>;
// var<storage, read> top: array<vec2<f32>>;

@group(0) @binding(4)
var<storage, read_write> n_particles: atomic<u32>;

@group(0) @binding(5)
var<storage, read_write> particles: array<ParticleLocation>;

// @group(0) @binding(6)
// var<storage, read_write> global_max: atomic<i32>;

@group(0) @binding(7)
var<storage, read_write> wg_size: WorkgroupSize;

@group(0) @binding(8)
var<storage, read> std_pic: f32;

@group(0) @binding(9)
var<storage, read> processed: array<f32>;

var<push_constant> pc: PushConstants;

fn is_max_in_plane(i: i32, j: i32, plane: u32, center: f32, row_transform: i32, col_transform: i32, pic_dims: vec2<i32>) -> bool{
    let stride = pic_dims[1];
    for (var loop_i = -1; loop_i < 2; loop_i++){
        for (var loop_j = -1; loop_j < 2; loop_j++){
            if (loop_i == 0 && loop_j == 0){
                continue;
            }
            let x = i + loop_i;
            let y = j + loop_j;
            if (x < 0 || x >= params.pic_nrows || y < 0 || y >= params.pic_ncols){
                return false;
            }
            
            let idx = (((x + row_transform) % pic_dims[0]) * stride + (y + col_transform) % pic_dims[1]);
            var neighbor: f32;
            if plane == 0u{
                // neighbor = bottom[idx][0];
                neighbor = bottom[idx];
            }
            else if plane == 1u{
                // neighbor = middle[idx][0];
                neighbor = middle[idx];
            }
            else if plane == 2u{
                // neighbor = top[idx][0];
                neighbor = top[idx];
            }
            if (neighbor >= center){
                return false;
            }
        }
    }
    return true;
}

@compute @workgroup_size(_)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= u32(params.pic_nrows) || global_id.y >= u32(params.pic_ncols)) {
        return;
    }
    
    let i = i32(global_id[0]);
    let j = i32(global_id[1]);
    
    // let row_transform = (i32(shape.nrows) - params.pic_nrows) / 2;
    // let col_transform = (i32(shape.ncols) - params.pic_ncols) / 2;
    // let row_transform = (i32(shape.nrows) - (params.pic_nrows + 1) / 2);
    // let col_transform = (i32(shape.ncols) - (params.pic_ncols + 1) / 2);
    let row_transform = 0;
    let col_transform = 0;
    // let pic_dims = vec2<i32>(i32(shape.nrows), i32(shape.ncols));
    let pic_dims = vec2<i32>(params.pic_nrows, params.pic_ncols);

    let stride = pic_dims[1];

    // let center = middle[((i + row_transform) % pic_dims[0]) * stride + ((j + col_transform) % pic_dims[1])][0];
    let flat_idx = ((i + row_transform) % pic_dims[0]) * stride + ((j + col_transform) % pic_dims[1]);
    // let center = middle[flat_idx][0];
    let center = middle[flat_idx];

    if processed[flat_idx] < std_pic * params.snr * params.rough_snr_factor{
        return;
    }
    
    if (center <= 0.0){
        return;
    }

    let is_max = is_max_in_plane(i, j, 1u, center, row_transform, col_transform, pic_dims);
    if !is_max {
        return;
    }

    if (pc.edge != -1){
        let is_max = is_max_in_plane(i, j, 0u, center, row_transform, col_transform, pic_dims);
        if !is_max {
            return;
        }
    }
    if (pc.edge != 1){
        let is_max = is_max_in_plane(i, j, 2u, center, row_transform, col_transform, pic_dims);
        if !is_max {
            return;
        }
    }

    let idx = atomicAdd(&n_particles, 1u);
    // atomicMax(&global_max, i32(center));
    particles[idx] = ParticleLocation(i, j, pc.radius, center);
    if ((idx % _workgroup1d_) == 0u){
      atomicAdd(&wg_size.x, 1u);
    }
}