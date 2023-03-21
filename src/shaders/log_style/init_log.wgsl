struct Shape{
    nrows: u32,
    ncols: u32,
}

struct PushConstants{
    sigma: f32,
    shape: Shape,
}

@group(0) @binding(0)
var<uniform> shape: Shape;

@group(0) @binding(1)
var<storage, read_write> buffer: array<vec2<f32>>;


var<push_constant> pc: PushConstants;


@compute @workgroup_size(_)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= pc.shape.nrows || global_id.y >= pc.shape.ncols) {
        return;
    }
    let offset = vec2<u32>((shape.nrows - pc.shape.nrows) / 2u, (shape.ncols - pc.shape.ncols) / 2u);
    let iu32 = global_id[0] + offset[0];
    let ju32 = global_id[1] + offset[1];
    let i = i32(iu32);
    let j = i32(ju32);

    let sigma = pc.sigma;
    let nrows = shape.nrows;
    let ncols = shape.ncols;
    
    // let sqrt2pi = 2.5066282746310002;
    let pi = 3.141592653589793;

    let x = f32(i - i32(nrows)/2);
    let y = f32(j - i32(ncols)/2);
    let r2 = x*x + y*y;
    let sigma2 = sigma*sigma;

    let idx = iu32 * ncols + ju32;
    // let log = 1./(sigma*sqrt2pi)*exp(-0.5*(x*x+y*y) / (sigma * sigma));
    let log = 1./(pi*sigma2)*(1.-r2/(2.*sigma2))*exp(-0.5*r2 / sigma2);
    buffer[idx] = vec2<f32>(log, 0.);

}