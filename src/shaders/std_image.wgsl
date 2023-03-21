struct PushConstants{
    total_elements: u32,
}

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@group(0) @binding(2)
var<storage, read> mean: f32;

var<push_constant> pc: PushConstants;

@compute @workgroup_size(_)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>){
    if (global_id.x >= pc.total_elements) {
        return;
    }

    let flat_idx = global_id.x;
    let datum = input[flat_idx];
    if (bitcast<u32>(datum) != 4294967295u){
        let diff = datum - mean;
        output[flat_idx] = diff * diff;
    } else {
        output[flat_idx] = 0.0;
    }
}

