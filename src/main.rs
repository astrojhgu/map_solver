#![allow(unused_imports)]
#![allow(non_snake_case)]
extern crate linear_solver;
extern crate map_solver;
use num_complex::{Complex64, Complex};
use num_traits::{Float, NumCast, FloatConst, NumAssign};
use scorus::linear_space::type_wrapper::LsVec;
use linear_solver::io::RawMM;
use map_solver::madam::MappingProblem;
use map_solver::mcmc_func::ps_model;
use ndarray::array;
use map_solver::mcmc_func::DT;
use map_solver::mcmc::Problem;
use rand::{thread_rng, Rng};
use rand_distr::StandardNormal;
use rand::distributions::Distribution;
use fftn::ifft;
use map_solver::noise::gen_noise;



fn main() {

    let mut rng=thread_rng();

    let ptr_mat=RawMM::<f64>::from_file("ptr_mat.mtx").to_sparse();
    let tod=RawMM::<f64>::from_file("cheat_vis.mtx").to_array1();
    let ntod=tod.len();

    

    let psp=vec![2.0, 0.1, 0.001, -0.5];
    //let noise=gen_noise(ntod, &psp, &mut rng, DT);

    //for x in noise{
    //    println!("{}", x);
    //}
}
