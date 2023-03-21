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
    
    var sum = 0.0;

    var datum = 0.0;
    
    datum = input[flat_idx + stride * 0u];
    if (bitcast<u32>(datum) != 4294967295u){
        sum += datum;
    }
    
    datum = input[flat_idx + stride * 1u];
    if (bitcast<u32>(datum) != 4294967295u){
        sum += datum;
    }
    
    datum = input[flat_idx + stride * 2u];
    if (bitcast<u32>(datum) != 4294967295u){
        sum += datum;
    }
    
    datum = input[flat_idx + stride * 3u];
    if (bitcast<u32>(datum) != 4294967295u){
        sum += datum;
    }
    
    datum = input[flat_idx + stride * 4u];
    if (bitcast<u32>(datum) != 4294967295u){
        sum += datum;
    }
    
    datum = input[flat_idx + stride * 5u];
    if (bitcast<u32>(datum) != 4294967295u){
        sum += datum;
    }
    
    datum = input[flat_idx + stride * 6u];
    if (bitcast<u32>(datum) != 4294967295u){
        sum += datum;
    }
    
    let last_access = flat_idx + stride * 7u;
    datum = input[last_access];
    if (last_access < pc.stride * 8u) & (bitcast<u32>(datum) != 4294967295u){
        sum += datum;
    }

    output[flat_idx] = sum;
}

