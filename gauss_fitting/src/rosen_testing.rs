#![allow(warnings)]
use argmin::{
    core::{CostFunction, Error, Executor, Gradient},
    solver::{
        linesearch::MoreThuenteLineSearch, neldermead::NelderMead, quasinewton::BFGS,
        quasinewton::LBFGS,
    },
};
use argmin_testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative};
use nalgebra::{Matrix2, Vector2};

struct Rosenbrock {
    a: f64,
    b: f64,
}

impl CostFunction for Rosenbrock {
    type Param = Vector2<f64>;

    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        Ok(rosenbrock_2d(param.as_slice(), self.a, self.b))
    }
}

impl Gradient for Rosenbrock {
    type Param = Vector2<f64>;

    type Gradient = Vector2<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, Error> {
        Ok(Vector2::from_iterator(
            rosenbrock_2d_derivative(param.as_slice(), self.a, self.b).into_iter(),
        ))
    }
}

pub fn nelder_mead() {
    let init = vec![
        Vector2::new(-1.0f64, 3.0),
        Vector2::new(2.0, 1.5),
        Vector2::new(2.0, -1.0),
    ];
    let solver = NelderMead::new(init);

    let cost = Rosenbrock { a: 1., b: 100. };

    let res = Executor::new(cost, solver)
        .configure(|state| state.max_iters(100))
        .run()
        .ok();
}

// pub fn bfgs(){
//     let cost = Rosenbrock{ a: 1.0, b: 100.0 };
//     let line = MoreThuenteLineSearch::new().with_c(1e-4, 0.9).unwrap();
//     let solver = BFGS::new(line);
//     let init_param = Vector2::new(1.2, 1.0);
//     // let init_hess = Matrix2::<f64>::identity();
//     let res = Executor::new(cost, solver);
//         // .configure(|state|
//         //     state
//         //         .param(init_param)
//         //         // .inv_hessian(init_hess)
//         //         .max_iters(60))
//         // .run()
//         // .unwrap();
// }
