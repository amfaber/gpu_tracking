@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(2)
var<storage, read> temp: array<vec2<f32>>;

@group(0) @binding(3)
var<storage, read_write> processed_buffer: array<f32>;


@compute @workgroup_size(_)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if (global_id.x >= u32(params.pic_nrows) || global_id.y >= u32(params.pic_ncols)) {
    return;
  }
  let u = i32(global_id.x);
  let v = i32(global_id.y);
  var gauss_sum = 0.0;
  var constant_sum = 0.0;
  let kernel_rows = params.preprocess_nrows;
  let kernel_cols = params.preprocess_ncols;
  let sigma2: f32 = params.sigma2;
  let rint = i32(kernel_rows / 2);
  // let start_u = u - kernel_rows / 2;
  var pic_u: i32;
  var pic_idx: i32;
  var do_gauss: bool;
  var gauss_norm = 0.0;
  for (var i: i32 = -rint; i <= rint; i = i + 1) {
    let x = f32(i);
    pic_u = u + i;
    do_gauss = true;
    if (pic_u < 0){
        // pic_u = -pic_u + -1;
        pic_u = 0;
        do_gauss = false;
      }
    else if (pic_u >= i32(params.pic_nrows)){
        // pic_u = 2 * i32(params.pic_nrows) - pic_u - 1;
        pic_u = params.pic_nrows - 1;
        do_gauss = false;
      }
    pic_idx = pic_u * params.pic_ncols + v;
    let gauss = exp(-x*x / (2.0 * sigma2));
    gauss_norm += gauss;
    if (do_gauss){
      gauss_sum += gauss * temp[pic_idx][0];
      // gauss_sum += gauss1d[i] * temp[pic_idx][0];
    }
    constant_sum += temp[pic_idx][1];
  }
  
  let result = gauss_sum/gauss_norm - constant_sum/f32(kernel_rows * kernel_cols);
  //_feat_truncate_preprocessed if (result < 1./255.){
    //_feat_truncate_preprocessed processed_buffer[u * params.pic_ncols + v] = 0.0;
  //_feat_truncate_preprocessed }
  //_feat_truncate_preprocessed else{
    processed_buffer[u * params.pic_ncols + v] = result;
  //_feat_truncate_preprocessed }
}

