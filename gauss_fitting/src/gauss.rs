#![allow(warnings)]
use argmin::{
    core::{CostFunction, Executor, Error, Gradient},
    solver::{
        neldermead::NelderMead,
        quasinewton::BFGS,
        quasinewton::LBFGS,
        linesearch::MoreThuenteLineSearch,
    },
};
use argmin_testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative};
use nalgebra::{Vector2, Matrix2, Vector5};
use ndarray::{ArrayView2, array, Array};
use rayon::prelude::*;


struct GaussianFit<'a>{
	data: ArrayView2<'a, f32>,
	xy: Vec<Vector2<f32>>,
}

impl<'a> GaussianFit<'a>{
	fn new(data: ArrayView2<'a, f32>) -> Self{
		let [i_offset, j_offset] = if let &[h, w] = data.shape(){
			[(h as f32) / 2. - 0.5, (w as f32) / 2.0 - 0.5]
		} else { panic!() };
		let xy: Vec<Vector2<f32>> = data.indexed_iter().map(|((i, j), _)|{
			let i = i as f32;
			let j = j as f32;
			Vector2::new(i - i_offset, j - j_offset)
		}).collect();
		
		Self{
			data,
			xy,
		}
	}
}

// Param
// 0 x
// 1 y
// 2 sigma
// 3 A
// 4 B

fn loss(param: &Vector5<f32>, datum: Option<&f32>, xy: Option<&Vector2<f32>>, sigma2: f32) -> Option<f32>{
	let exponent = -(param.xy() - xy?).norm_squared()/sigma2;
	Some(((exponent).exp() * param[3] + param[4] - datum?).powi(2))
}

impl<'a> CostFunction for GaussianFit<'a>{
    type Param = Vector5<f32>;

    type Output = f32;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
		let mu = param.xy();
		let sigma2 = param[2].powi(2);
		// let mut part1 = 0.;
		// let mut part2 = 0.;
		// let mut part3 = 0.;
		// let mut part4 = 0.;
		
		// let mut i = 0;
		// loop{
		// 	part1 += match loss(param, self.data.as_slice().unwrap().get(i), self.xy.get(i), sigma2){
		// 		Some(val) => val,
		// 		None => break,
		// 	};
		// 	part2 += match loss(param, self.data.as_slice().unwrap().get(i + 1), self.xy.get(i + 1), sigma2){
		// 		Some(val) => val,
		// 		None => break,
		// 	};
		// 	part3 += match loss(param, self.data.as_slice().unwrap().get(i + 2), self.xy.get(i + 2), sigma2){
		// 		Some(val) => val,
		// 		None => break,
		// 	};
		// 	part4 += match loss(param, self.data.as_slice().unwrap().get(i + 3), self.xy.get(i + 3), sigma2){
		// 		Some(val) => val,
		// 		None => break,
		// 	};
		// 	i += 4;
		// }
		
		// Ok(
		// 	part1 + part2 + part3 + part4
		// )
		
		Ok(self.data.iter().zip(self.xy.iter()).map(|(&datum, &xy)|{
			let exponent = -(mu - xy).norm_squared()/sigma2;
			((exponent).exp() * param[3] + param[4] - datum).powi(2)
		}).sum::<f32>())
    }
}

impl<'a> Gradient for GaussianFit<'a>{
	type Param = Vector5<f32>;

	type Gradient  = Vector5<f32>;

	fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, Error> {
		let mu = param.xy();
		let sigma2 = param[2].powi(2);
		let sigma3 = sigma2 * param[2];
		Ok(self.data.iter().zip(self.xy.iter()).map(|(&datum, &xy)|{
			let r2 = (mu - xy).norm_squared();
			let e = (-r2 / sigma2).exp();
			let eA = e * param[3] * 2.;
			// let full = eA + param[4];
			let data_diff = 2.0 * (eA + param[4] - datum);
			
			let dx = eA * -(mu.x - xy.x);
			let dy = eA * -(mu.y - xy.y);
			let ds = eA * r2/sigma3;
			let dA = e;
			let dB = 1.0;
			
			let out = Vector5::new(dx, dy, ds, dA, dB);
			out.component_mul(&Vector5::from_element(data_diff))
			
		}).sum::<Vector5<f32>>())
	}

}

fn rand_perturb() -> Vector5<f32>{
	let mut out = Vector5::new_random();
	for mut_ref in out.iter_mut(){
		*mut_ref = 1.0 - (*mut_ref - 0.5) / 2.5
	}
	out
}

pub fn lbfgs_test(){
	let data = array![
		[0., 0., 1., 0., 0.],
		[0., 1., 3., 1., 0.],
		[1., 3., 8., 3., 1.],
		[0., 1., 3., 1., 0.],
		[0., 0., 1., 0., 0.],
	];
	
	let cost = GaussianFit::new(data.view());
	let line = MoreThuenteLineSearch::new();
	let guess: Vector5<f32> = Vector5::new(1., -1., 7., 1., 1.);
	let solver = LBFGS::new(line, 7);
	let res = Executor::new(cost, solver)
		.configure(|state| {
			state
				.param(guess)
				.max_iters(60)
		})
		.run()
		.unwrap();
	
	dbg!(res.state.time);
	dbg!(res.state.param);
	dbg!(res.state.iter);
	dbg!(res.state.counts);
	dbg!(res.state.cost);
}

#[profiling::function]
pub fn test(){
	// let data = array![
	// 	[0., 0., 1., 0., 0.],
	// 	[0., 1., 3., 1., 0.],
	// 	[1., 3., 8., 3., 1.],
	// 	[0., 1., 3., 1., 0.],
	// 	[0., 0., 1., 0., 0.],
	// ];
	let data = array![
		[9.64374924e-22, 7.81441095e-18, 8.56954216e-15, 1.27183282e-12,
	        2.55454451e-11, 6.94397193e-11, 2.55454451e-11, 1.27183282e-12,
	        8.56954216e-15, 7.81441095e-18, 9.64374924e-22],
	       [7.81441095e-18, 6.33208277e-14, 6.94397193e-11, 1.03057681e-08,
		        2.06996886e-07, 5.62675874e-07, 2.06996886e-07, 1.03057681e-08,
		        6.94397193e-11, 6.33208277e-14, 7.81441095e-18],
	       [8.56954216e-15, 6.94397193e-11, 7.61498987e-08, 1.13016470e-05,
		        2.26999649e-04, 6.17049020e-04, 2.26999649e-04, 1.13016470e-05,
		        7.61498987e-08, 6.94397193e-11, 8.56954216e-15],
	       [1.27183282e-12, 1.03057681e-08, 1.13016470e-05, 1.67731314e-03,
		        3.36897350e-02, 9.15781944e-02, 3.36897350e-02, 1.67731314e-03,
		        1.13016470e-05, 1.03057681e-08, 1.27183282e-12],
	       [2.55454451e-11, 2.06996886e-07, 2.26999649e-04, 3.36897350e-02,
		        6.76676416e-01, 1.83939721e+00, 6.76676416e-01, 3.36897350e-02,
		        2.26999649e-04, 2.06996886e-07, 2.55454451e-11],
	       [6.94397193e-11, 5.62675874e-07, 6.17049020e-04, 9.15781944e-02,
		        1.83939721e+00, 5.00000000e+00, 1.83939721e+00, 9.15781944e-02,
		        6.17049020e-04, 5.62675874e-07, 6.94397193e-11],
	       [2.55454451e-11, 2.06996886e-07, 2.26999649e-04, 3.36897350e-02,
		        6.76676416e-01, 1.83939721e+00, 6.76676416e-01, 3.36897350e-02,
		        2.26999649e-04, 2.06996886e-07, 2.55454451e-11],
	       [1.27183282e-12, 1.03057681e-08, 1.13016470e-05, 1.67731314e-03,
		        3.36897350e-02, 9.15781944e-02, 3.36897350e-02, 1.67731314e-03,
		        1.13016470e-05, 1.03057681e-08, 1.27183282e-12],
	       [8.56954216e-15, 6.94397193e-11, 7.61498987e-08, 1.13016470e-05,
		        2.26999649e-04, 6.17049020e-04, 2.26999649e-04, 1.13016470e-05,
		        7.61498987e-08, 6.94397193e-11, 8.56954216e-15],
	       [7.81441095e-18, 6.33208277e-14, 6.94397193e-11, 1.03057681e-08,
		        2.06996886e-07, 5.62675874e-07, 2.06996886e-07, 1.03057681e-08,
		        6.94397193e-11, 6.33208277e-14, 7.81441095e-18],
	       [9.64374924e-22, 7.81441095e-18, 8.56954216e-15, 1.27183282e-12,
		        2.55454451e-11, 6.94397193e-11, 2.55454451e-11, 1.27183282e-12,
		        8.56954216e-15, 7.81441095e-18, 9.64374924e-22]];
	
	let guess: Vector5<f32> = Vector5::new(1., -1., 7., 1., 1.);
	// let init = vec![
	// 	guess,
	// 	guess.component_mul(&rand_perturb()),
	// 	guess.component_mul(&rand_perturb()),
	// 	guess.component_mul(&rand_perturb()),
	// 	guess.component_mul(&rand_perturb()),
	// 	guess.component_mul(&rand_perturb()),
	// ];
	
    let now = std::time::Instant::now();
    // (0..1_000).into_par_iter().for_each(|_|{
    // (0..1_000).for_each(|_|{
		let init = vec![
			Vector5::new_random(),
			Vector5::new_random(),
			Vector5::new_random(),
			Vector5::new_random(),
			Vector5::new_random(),
			Vector5::new_random(),
		];

		let solver = NelderMead::new(init);

		let cost = GaussianFit::new(data.view());

		let res = Executor::new(cost, solver)
			.configure(|state| state.max_iters(500)).run().unwrap();
    // });
    dbg!(now.elapsed());

	dbg!(res.state.time);
	dbg!(res.state.param);
	dbg!(res.state.iter);
	dbg!(res.state.counts);
	dbg!(res.state.cost);
}


