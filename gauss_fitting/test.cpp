float2 offset = float2(params.x * center.x - 0.5f * (params.x - 1.0f),
params.x * center.y - 0.5f * (params.x - 1.0f));  float4 O = f4texRECT(Operator, offset); 