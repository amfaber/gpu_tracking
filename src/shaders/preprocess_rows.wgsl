@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(1)
var<storage, read> picture: array<f32>;

// @group(0) @binding(2)
// var<storage, read> gauss1d: array<f32>;

@group(0) @binding(3)
var<storage, read_write> temp: array<vec2<f32>>;


@compute @workgroup_size(_)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if (global_id.x >= u32(params.pic_nrows) || global_id.y >= u32(params.pic_ncols)) {
    return;
  }
  let u = i32(global_id.x);
  let v = i32(global_id.y);
  var gauss_sum = 0.0;
  var constant_sum = 0.0;
  let kernel_cols = params.preprocess_ncols;
  let sigma2 = params.sigma2;
  let rint = kernel_cols / 2;
  var pic_v: i32;
  var pic_idx: i32;
  var do_gauss: bool;
  var gauss_norm = 0.0;
  for (var i: i32 = -rint; i <= rint; i = i + 1) {
    let x = f32(i);
    pic_v = v + i;
    do_gauss = true;
    if (pic_v < 0){
        // pic_v = -pic_v + -1;
        pic_v = 0;
        do_gauss = false;
      }
    else if (pic_v >= i32(params.pic_ncols)){
        // pic_v = 2 * i32(params.pic_ncols) - pic_v - 1;
        pic_v = params.pic_ncols - 1;
        do_gauss = false;
      }
    pic_idx = u * params.pic_ncols + pic_v;

    let gauss = exp(-x*x / (2.0 * sigma2));
    gauss_norm += gauss;
    if (do_gauss){
      gauss_sum += gauss * picture[pic_idx];
    }
    constant_sum += picture[pic_idx];
  }

  temp[u * params.pic_ncols + v] = vec2(gauss_sum/gauss_norm, constant_sum);
}