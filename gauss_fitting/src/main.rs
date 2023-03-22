use gauss_fitting;
// use rayon::prelude::*;

fn main() {
    // let now = std::time::Instant::now();
    // (0..100_000).into_par_iter().for_each(|_|{
    //     gauss_fitting::rosen_testing::nelder_mead()
    // });
    // dbg!(now.elapsed());

    gauss_fitting::gauss::test();
    // gauss_fitting::gauss::lbfgs_test();

    
}
