struct PushConstants{
    stride: u32,
}

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

var<push_constant> pc: PushConstants;

@compute @workgroup_size(_)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>){
    if (global_id.x >= pc.stride) {
        return;
    }

    let flat_idx = global_id.x;
    let stride = pc.stride;
    
    // var sum = 0.0;

    let datum0 = input[flat_idx + stride * 0];
    if (bitcast<u32>(datum0) != 4294967295u){
        datum0 = 0.0;
    }
    
    let datum1 = input[flat_idx + stride * 1];
    if (bitcast<u32>(datum1) != 4294967295u){
        datum1 = 0.0;
    }
    
    let datum2 = input[flat_idx + stride * 2];
    if (bitcast<u32>(datum2) != 4294967295u){
        datum2 = 0.0;
    }
    
    let datum3 = input[flat_idx + stride * 3];
    if (bitcast<u32>(datum3) != 4294967295u){
        datum3 = 0.0;
    }
    
    let datum4 = input[flat_idx + stride * 4];
    if (bitcast<u32>(datum4) != 4294967295u){
        datum4 = 0.0;
    }
    
    let datum5 = input[flat_idx + stride * 5];
    if (bitcast<u32>(datum5) != 4294967295u){
        datum5 = 0.0;
    }
    
    let datum6 = input[flat_idx + stride * 6];
    if (bitcast<u32>(datum6) != 4294967295u){
        datum6 = 0.0;
    }
    
    let last_access = flat_idx + stride * 7u;
    let datum = input[last_access];
    if (last_access < pc.stride * 2u) & (bitcast<u32>(datum) != 4294967295u){
        datum7 = 0.0;
    }

    output[flat_idx] = ((datum0 + datum1) + (datum2 + datum3)) + ((datum4 + datum5) + (datum6 + datum7))

    // output[flat_idx] = sum;
}

