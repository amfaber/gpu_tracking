use crate::my_dtype;

pub struct Kernel{
    pub data: Vec<my_dtype>,
    pub size: [u32; 2],
}

impl Kernel{
    pub fn new(data: Vec<my_dtype>, size: [u32; 2]) -> Self{
        if data.len() != (size[0] * size[1]) as usize{
            panic!("Kernel size does not match data length");
        }
        Self{
            data,
            size,
        }
    }

    pub fn normalize(&mut self){
        let sum = self.data.iter().sum::<my_dtype>();
        self.data = self.data.iter().map(|&x| x / sum).collect();
    }

    pub fn gaussian(sigma: my_dtype) -> Self{
        let radius = (3. * sigma).ceil() as u32;
        let size = [2 * radius + 1, 2 * radius + 1];
        let mut data = Vec::with_capacity((size[0] * size[1]) as usize);
        for i in 0..size[0]{
            for j in 0..size[1]{
                let x = i as my_dtype - radius as my_dtype;
                let y = j as my_dtype - radius as my_dtype;
                let val = (-(x.powi(2) + y.powi(2)) / (2. * sigma.powi(2))).exp();
                data.push(val);
            }
        }
        let mut kernel = Self::new(data, size);
        kernel.normalize();
        kernel
    }

    pub fn tp_gaussian(sigma: my_dtype, truncate: f32) -> Self{
        let radius = (truncate * sigma + 0.5) as u32;
        let size = [2 * radius + 1, 2 * radius + 1];
        let mut data = Vec::with_capacity((size[0] * size[1]) as usize);
        for i in 0..size[0]{
            for j in 0..size[1]{
                let x = i as my_dtype - radius as my_dtype;
                let y = j as my_dtype - radius as my_dtype;
                let val = (-(x.powi(2) + y.powi(2)) / (2. * sigma.powi(2))).exp();
                data.push(val);
            }
        }
        let mut kernel = Self::new(data, size);
        kernel.normalize();
        kernel
    }

    pub fn composite_kernel(sigma: my_dtype, size: [u32; 2]) -> Self{
        let mut data = Vec::with_capacity((size[0] * size[1]) as usize);
        let radius = size[0] / 2;
        let gauss_radius = ((4. * sigma) + 0.5) as i32;
        for i in 0..size[0]{
            for j in 0..size[1]{
                let x = i as i32 - radius as i32;
                let y = j as i32 - radius as i32;
                if x.abs() <= gauss_radius && y.abs() <= gauss_radius{
                    let x = x as my_dtype;
                    let y = y as my_dtype;
                    let val = (-(x.powi(2) + y.powi(2)) / (2. * sigma.powi(2))).exp();
                    data.push(val);
                }
                else{
                    data.push(0.);
                }
            }
        }
        let mut kernel = Self::new(data, size);
        kernel.normalize();
        
        for dat in kernel.data.iter_mut(){
            *dat -= 1. / (size[0] * size[1]) as my_dtype;
        }
        kernel
    }

    pub fn gauss_1d(sigma: my_dtype, size: u32) -> Self{
        let mut data = Vec::with_capacity(size as usize);
        let radius = size / 2;
        for i in 0..size{
            let x = i as my_dtype - radius as my_dtype;
            let val = (-(x.powi(2)) / (2. * sigma.powi(2))).exp();
            data.push(val);
        }
        let mut kernel = Self::new(data, [size, 1]);
        kernel.normalize();
        kernel
    }

    pub fn rolling_average(size: [u32; 2]) -> Self{
        let data = vec![1. / (size[0] * size[1]) as my_dtype; (size[0] * size[1]) as usize];
        let size = size;
        Self::new(data, size)
    }

    pub fn circle_mask_with_size(radius: u32, size: u32) -> Self{
        // let size = [2 * radius + 1, 2 * radius + 1];
        let mut data = Vec::with_capacity((size * size) as usize);
        for i in 0..size{
            for j in 0..size{
                let x = i as my_dtype - radius as my_dtype;
                let y = j as my_dtype - radius as my_dtype;
                let val = if x.powi(2) + y.powi(2) <= (radius as my_dtype).powi(2) {1.} else {0.};
                data.push(val);
            }
        }
        Self::new(data, [size, size])
    }
    pub fn circle_mask(radius: u32) -> Self{
        let size = [2 * radius + 1, 2 * radius + 1];
        let mut data = Vec::with_capacity((size[0] * size[1]) as usize);
        for i in 0..size[0]{
            for j in 0..size[1]{
                let x = i as my_dtype - radius as my_dtype;
                let y = j as my_dtype - radius as my_dtype;
                let val = x.powi(2) + y.powi(2);
                let val = if val <= (radius as my_dtype).powi(2) {val} else {0.};
                data.push(val);
            }
        }
        Self::new(data, size)
    }
}

pub fn circle_inds(radius: my_dtype) -> (Vec<[i32; 2]>, usize){
    let radius_bound = radius.ceil() as i32;
    let mut inds = Vec::new();
    let mut middle_most = -1;
    for i in -radius_bound..radius_bound+1{
        for j in -radius_bound..radius_bound+1{
            if (i.pow(2) + j.pow(2)) as f32 <= radius.powi(2){
                inds.push([i, j]);
            }
            if i == 0 && j == 0{
                middle_most = inds.len() as i32 - 1;
            }
        }
    }
    // let inds = Array2::from_shape_vec((inds.len() / 2, 2), inds).unwrap();
    (inds, middle_most as usize)
}

pub fn annulus_inds(outer_radius: my_dtype, inner_radius: my_dtype) -> Vec<[i32; 2]>{
    let bounds = outer_radius.ceil() as i32;
    let mut inds = Vec::new();
    let or2 = outer_radius.powi(2);
    let ir2 = inner_radius.powi(2);
    for i in -bounds..bounds+1{
        for j in -bounds..bounds+1{
            let r2 = (i.pow(2) + j.pow(2)) as f32;
            if r2 <= or2 && r2 >= ir2{
                inds.push([i, j]);
                // dbg!(i, j);
            }
        }
    }
    inds
}

pub fn r2_in_circle(radius: i32) -> Vec<my_dtype>{
    let mut output = Vec::new();
    for x in -radius..radius+1{
        for y in -radius..radius+1{
            let r2 = x.pow(2) + y.pow(2);
            if r2 <= radius.pow(2){
                output.push(r2 as my_dtype);
            }
        }
    }
    output
}

pub fn sin_in_circle(radius: i32) -> Vec<my_dtype>{
    let mut output = Vec::new();
    for x in -radius..radius+1{
        for y in -radius..radius+1{
            let r2 = x.pow(2) + y.pow(2);
            if r2 <= radius.pow(2){
                output.push((2.*(y as my_dtype).atan2(x as my_dtype)).sin());
            }
        }
    }
    output
}

pub fn cos_in_circle(radius: i32) -> Vec<my_dtype>{
    let mut output = Vec::new();
    for x in -radius..radius+1{
        for y in -radius..radius+1{
            let r2 = x.pow(2) + y.pow(2);
            if r2 <= radius.pow(2){
                output.push((2.*(y as my_dtype).atan2(x as my_dtype)).cos());
            }
        }
    }
    output
}