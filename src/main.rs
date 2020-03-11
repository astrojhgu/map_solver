#![allow(unused_imports)]
#![allow(non_snake_case)]
extern crate linear_solver;
extern crate map_solver;
use fftn::ifft;
use linear_solver::io::RawMM;
use map_solver::madam::MappingProblem;
use map_solver::mcmc::Problem;
use map_solver::mcmc_func::ps_model;
use map_solver::mcmc_func::DT;
use map_solver::noise::gen_noise;
use map_solver::utils::deflatten_order_f;
use map_solver::utils::flatten_order_f;
use ndarray::array;
use num_complex::{Complex, Complex64};
use num_traits::{Float, FloatConst, NumAssign, NumCast};
use rand::distributions::Distribution;
use rand::{thread_rng, Rng};
use rand_distr::StandardNormal;
use scorus::linear_space::type_wrapper::LsVec;

fn main() {
    let mx = array![1, 2, 3, 4, 5, 6];
    println!("{}", mx);
    println!("{}", deflatten_order_f(mx.view(), 2, 3));
    let mx1 = flatten_order_f(deflatten_order_f(mx.view(), 2, 3).view());
    println!("{}", mx1);
}
