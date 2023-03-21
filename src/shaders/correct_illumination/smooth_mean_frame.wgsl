struct PushConstants{
  dim: u32,
  sigma: f32,
  n_frames: f32,
}

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

var<push_constant> pc: PushConstants;


@compute @workgroup_size(_)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if (global_id.x >= u32(params.pic_nrows) || global_id.y >= u32(params.pic_ncols)) {
    return;
  }
  var gauss_sum = 0.0;
  var norm_sum = 0.0;
  let dim = pc.dim;
  let sigma = pc.sigma;
  let sigma2 = sigma * sigma;
  let rint = i32(4.0 * sigma + 0.5);
  var u: i32;
  var v: i32;
  var u_stride: i32;
  var u_bound: i32;
  if dim == 0u {
    u = i32(global_id.x);
    v = i32(global_id.y);
    u_stride = params.pic_ncols;
    u_bound = params.pic_nrows;
  } else {
    u = i32(global_id.y);
    v = i32(global_id.x) * params.pic_ncols;
    u_stride = 1;
    u_bound = params.pic_ncols;
  }

  let pic_idx_base = u * u_stride + v;
  for (var i: i32 = -rint; i <= rint; i = i + 1) {
    let x = f32(i);
    let pic_u = u + i;
    if (pic_u < 0) || (pic_u >= u_bound) {
      continue;
    }
    let pic_idx = pic_idx_base + i * u_stride;
    let x2 = x * x;
	let gauss = exp(-x2 / (2.0 * sigma2));
	gauss_sum += gauss * input[pic_idx];
    norm_sum += gauss;
  }
  
  // let sqrt2pi = 2.5066282746310002;
  // let norm = 1./(sqrt2pi*sigma*pc.n_frames);
  var norm = norm_sum;
  if (dim == 0u){
    norm *= pc.n_frames;
  }
  let result = gauss_sum / norm;
  output[pic_idx_base] = result;
}
