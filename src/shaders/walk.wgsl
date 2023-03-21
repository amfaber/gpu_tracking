struct ParticleLocation{
    x: i32,
    y: i32,
    r: f32,
    log_space: f32,
}

// struct Shape{
//     nrows: u32,
//     ncols: u32,
// }

struct ResultRow{
  x: f32,
  y: f32,
  mass: f32,
  r: f32,
  max_intensity: f32,
  Rg: f32,
  raw_mass: f32,
  signal: f32,
  ecc: f32,
  count: f32,
}

struct WalkResult{
  x: f32,
  y: f32,
  mass: f32,
  max: f32,
  count: f32,
}

struct WorkgroupSize {
    x: atomic<u32>,
    y: atomic<u32>,
    z: atomic<u32>,
}

@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(1)
var<storage, read> processed_buffer: array<f32>;

@group(0) @binding(2)
var<storage, read> particles: array<ParticleLocation>;

@group(0) @binding(3)
var<storage, read> n_particles: u32;

@group(0) @binding(4)
var<storage, read_write> n_particles_filtered: atomic<u32>;

@group(0) @binding(5)
var<storage, read_write> results: array<ResultRow>;

@group(0) @binding(6)
var<storage, read> raw_frame: array<f32>;

@group(0) @binding(7)
var<storage, read> image_std: f32;

@group(0) @binding(8)
var<storage, read_write> next_dispatch: WorkgroupSize;

fn get_center(u: i32, v: i32, rint: i32, r2: f32, transform: vec2<i32>) -> WalkResult{
  // let rint = (kernel_rows - 1) / 2;
  // let r = f32(rint);
  // let r2 = r * r;
  var result: WalkResult;
  for (var i: i32 = -rint; i <= rint; i = i + 1) {
    let x = f32(i);
    let x2 = x*x;
    for (var j: i32 = -rint; j <= rint; j = j + 1) {
      let y = f32(j);
      let y2 = y*y;
      var mask = x2 + y2;
      if (mask > r2) {
        continue;
      }
      let pic_u = u + i;
      let pic_v = v + j;
      if (pic_u < 0 || pic_u >= params.pic_nrows || pic_v < 0 || pic_v >= params.pic_ncols) {
        continue;
      }
      // //_feat_LoG let pic_u = (pic_u + transform[0]) % i32(shape.nrows);
      // //_feat_LoG let pic_v = (pic_v + transform[1]) % i32(shape.ncols);
      let pic_idx = pic_u * params.pic_ncols + pic_v;
      let data = processed_buffer[pic_idx];

      result.x += data * x;
      result.y += data * y;
      result.mass += data;
      result.count += 1.0;
      result.max = max(result.max, data);
    }
  }
  result.x /= result.mass;
  result.y /= result.mass;
  return result;
}

fn walk(argpicuv: vec2<i32>, r: f32, transform: vec2<i32>) -> WalkResult {
  var picuv = argpicuv;
  var changed = false;
  var result: WalkResult;

  // var radiusu: i32;
  // var radiusv: i32;
  // var kernel_rows: i32;
  // var kernel_cols: i32;
  // if r > 0.0 {
  //   radiusu = i32(ceil(r));
  //   radiusv = i32(ceil(r));
  //   kernel_rows = radiusu * 2 + 1;
  //   kernel_cols = radiusv * 2 + 1;

  // } else {
  //   radiusu = params.circle_nrows / 2;
  //   radiusv = params.circle_ncols / 2;
  //   kernel_rows = params.circle_nrows;
  //   kernel_cols = params.circle_ncols;
  // }
  
  var rint: i32;
  var r2: f32;
  if r <= 0.0{
    rint = (params.circle_nrows - 1) / 2;
    let r = f32(rint);
    r2 = r*r;
  } else {
    rint = i32(ceil(r));
    r2 = r*r;
  }
  let margin = rint;

  for (var i: u32 = 0u; i < params.max_iterations; i = i + 1u) {
    // idx = u * params.pic_ncols + v;
    result = get_center(picuv[0], picuv[1], rint, r2, transform);
    changed = false;
    if ((result.x > params.threshold) && (picuv[0] < params.pic_nrows - 1 - margin)) {
      picuv[0] += 1;
      changed = true;
    } else if (result.x < -params.threshold && picuv[0] > margin) {
      picuv[0] -= 1;
      changed = true;
    }
    if ((result.y > params.threshold) && (picuv[1] < params.pic_ncols - 1 - margin)) {
      picuv[1] += 1;
      changed = true;
    } else if ((result.y < -params.threshold) && (picuv[1] > margin)) {
      picuv[1] -= 1;
      changed = true;
    }
    if (!changed){
      break;
    }
  }
  // let final_coords = WalkResult(
  //   f32(picuv[0]) + result.x, // + 1000. * f32(changed),
  //   f32(picuv[1]) + result.y, // + 1000. * f32(changed),
  //   result.mass,
  //   result.count,
  //   );
  result.x += f32(picuv[0]);
  result.y += f32(picuv[1]);
  return result;
}

@compute @workgroup_size(_)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if (global_id.x >= n_particles) {
    return;
  }
  // let idx = global_id.x * u32(params.pic_ncols) + global_id.y;
  let part_idx = global_id.x;
  let particle_location = particles[part_idx];
  let picuv = vec2<i32>(particle_location.x, particle_location.y);

  let pad_offset = vec2<i32>(0, 0);
  // //_feat_LoG let pad_offset = vec2<i32>((i32(shape.nrows) - params.pic_nrows) / 2, (i32(shape.ncols) - params.pic_ncols));

  let r = particle_location.r;
  // let log_space = particle_location.log_space;

  let result = walk(picuv, r, pad_offset);

  //_feat_varcheck var kernel_cols: i32;
  //_feat_varcheck var kernel_rows: i32;
  //_feat_varcheck if r > 0.0 {
  //_feat_varcheck   kernel_rows = i32(ceil(r)) * 2 + 1;
  //_feat_varcheck   kernel_cols = i32(ceil(r)) * 2 + 1;
  //_feat_varcheck } else {
  //_feat_varcheck   kernel_rows = params.preprocess_nrows;
  //_feat_varcheck   kernel_cols = params.preprocess_ncols;
  //_feat_varcheck }
  //_feat_varcheck let varcheck = variance_check(final_coords[0], final_coords[1], kernel_rows, kernel_cols);

  var minmass: f32;
  if params.minmass > 0.0{
    minmass = params.minmass;
  } else {
    minmass = params.minmass_snr * result.count * image_std * params.rough_snr_factor;
  }

  // if (final_coords[2] > minmass && varcheck) {





  if result.mass >= minmass{
    var row: ResultRow; // auto initializes to 0.0 for all fields.
    row.x = result.x;
    row.y = result.y;
    row.mass = result.mass;
    row.y = result.y;
    row.count = result.count;
    row.max_intensity = result.max;
    if (r > 0.0){
      row.r = r;
    }
    let part_id = atomicAdd(&n_particles_filtered, 1u);
	if ((part_id % _workgroup1d_) == 0u){
		atomicAdd(&next_dispatch.x, 1u);
	}
    results[part_id] = row;
  }



  // if (final_coords[2] > minmass) {
  //   let part_id = atomicAdd(&n_particles_filtered, 1u);
  //   let n_res = 9u;
  //   results[part_id * n_res + 0u] = final_coords[0];
  //   results[part_id * n_res + 1u] = final_coords[1];
  //   results[part_id * n_res + 2u] = final_coords[2];
  //   if r>0.0 {
  //     results[part_id * n_res + 3u] = r;
  //     results[part_id * n_res + 4u] = log_space;
  //   }

  // }





  // let pic_size = u32(params.pic_nrows * params.pic_ncols);
  // if (is_max(i32(global_id.x), i32(global_id.y), params.dilation_nrows, params.dilation_ncols)) {
  //   // results[idx] = 1.;
  //   let walk_out = walk(i32(global_id.x), i32(global_id.y));
  //   if (walk_out[2] > params.minmass){
  //     results[idx + pic_size * 0u] = walk_out[2];
  //     results[idx + pic_size * 1u] = walk_out[0];
  //     results[idx + pic_size * 2u] = walk_out[1];
  //   }
  // }
}