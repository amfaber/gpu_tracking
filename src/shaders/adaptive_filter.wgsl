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

struct WorkgroupSize {
    x: atomic<u32>,
    y: atomic<u32>,
    z: atomic<u32>,
}

@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(1)
var<storage, read> input: array<ResultRow>;

@group(0) @binding(2)
var<storage, read_write> output: array<ResultRow>;

@group(0) @binding(3)
var<storage, read_write> results: array<ResultRow>;

@group(0) @binding(4)
var<uniform> image_std: f32;

@group(0) @binding(5)
var<uniform> current_n: u32;

@group(0) @binding(6)
var<storage, read_write> new_n: atomic<u32>;

@group(0) @binding(7)
var<storage, read_write> result_n: atomic<u32>;

@group(0) @binding(8)
var<storage, read_write> dispatcher: WorkgroupSize; 

@group(0) @binding(9)
var<storage, read_write> dispatcher_results: WorkgroupSize; 

@group(0) @binding(10)
var<storage, read_write> frame: array<f32>;

@group(0) @binding(11)
var<storage, read_write> counter: atomic<u32>;

fn set_nan(row: ResultRow){
	var rint: i32;
	
	if row.r > 0.0 {
		rint = i32(ceil(row.r));
	} else {
		rint = (params.circle_nrows - 1) / 2;
	}
	let u = i32(round(row.x));
	let v = i32(round(row.y));
	let r = f32(rint);
	let r2 = r * r;
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
			let pic_idx = pic_u * params.pic_ncols + pic_v;
			frame[pic_idx] = bitcast<f32>(4294967295u);
		}
	}
}

@compute @workgroup_size(_)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
	if global_id.x >= current_n{
		return;
	}
	var row = input[global_id.x];
	
	var minmass: f32;
	if params.minmass > 0.0{
		minmass = params.minmass;
	} else {
		minmass = params.minmass_snr * row.count * image_std;
	}
	
	if ((row.mass < minmass) || (row.max_intensity < params.snr * image_std)){
	// if (row.mass < minmass){
		let id = atomicAdd(&new_n, 1u);
		if ((id % _workgroup1d_) == 0u){
			atomicAdd(&dispatcher.x, 1u);
		}
		output[id] = row;
	} else {
		let id = atomicAdd(&result_n, 1u);
		if ((id % _workgroup1d_) == 0u){
			atomicAdd(&dispatcher_results.x, 1u);
		}
		set_nan(row);
		atomicSub(&counter, u32(row.count));
		results[id] = row;
	}
	// var dummyrow: ResultRow;
	// dummyrow.x = minmass;
	// dummyrow.y = 500.0;
	// output[0] = dummyrow;
	// results[0] = dummyrow;
	
}
