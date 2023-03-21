@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(1)
var<storage, read> processed_buffer: array<f32>;

@group(0) @binding(2)
var<storage, read_write> max_rows: array<f32>;

@compute @workgroup_size(_)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>){
  let u = i32(global_id.x);
  let v = i32(global_id.y);
  if (u < 0 || u >= params.pic_nrows || v < 0 || v >= params.pic_ncols) {
    return;
  }

  let kernel_rows = params.dilation_nrows;
  let kernel_cols = params.dilation_ncols;
  let center = processed_buffer[u * params.pic_ncols + v];
  var pic_v = v - i32(f32(kernel_cols) / 2. - 0.5);
  var pic_idx: i32;
  var maximum = center;
  for (var j: i32 = 0; j < kernel_cols; j = j + 1) {
    if (pic_v >= 0 && pic_v < params.pic_ncols) {
      pic_idx = u * params.pic_ncols + pic_v;
      maximum = max(maximum, processed_buffer[pic_idx]);
    }
    pic_v += 1;
  }
  max_rows[u * params.pic_ncols + v] = maximum;
}