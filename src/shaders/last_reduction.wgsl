
struct PushConstants{
    n_elements: u32,
	doing_variance: u32,
}

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: f32;

@group(0) @binding(2)
var<storage, read> mean_divisor: u32;

var<push_constant> pc: PushConstants;

@compute @workgroup_size(_)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>){
	// Not checking if we are outside the alloted amount of work, as only 1 invocation of this function should ever be started.
	// i.e. workgroup_size(1, 1, 1), dispatcher(1, 1, 1)
    var mean = 0.0;

	for (var i = 0u; i < pc.n_elements; i++){
		mean += input[i];
	}
	if (pc.doing_variance == 1u){
		mean /= f32(mean_divisor - 1u);
		mean = sqrt(mean);
	} else {
		mean /= f32(mean_divisor);
	}
    output = mean;
}

