struct PushConstants{
	arraylen: u32,
}

// @group(0) @binding(0)
// var<uniform> params: Params;

@group(0) @binding(0)
var<storage, read> frame: array<f32>;

@group(0) @binding(1)
var<storage, read_write> acc: array<f32>;

var<push_constant> pc: PushConstants;

@compute @workgroup_size(_)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>){
	if (global_id.x >= pc.arraylen) {
		return;
	}
	let flat_idx = global_id.x;
	acc[flat_idx] += frame[flat_idx];
}
