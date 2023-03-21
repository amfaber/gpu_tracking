struct PushConstants{
    stride: u32,
}

@group(0) @binding(1)
var<storage, read> input: array<f32>;

@group(0) @binding(2)
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

    datum = input[flat_idx];
    if ((datum == datum) != (datum != datum)){
        sum += datum;
    }
    
    datum = input[flat_idx + stride];
    if ((datum == datum) != (datum != datum)){
        sum += datum;
    }
    
    // let datum = input[flat_idx];
    // if ((datum == datum) != (datum != datum)){
    //     sum += datum;
    // }

    output[flat_idx] = sum;
}

