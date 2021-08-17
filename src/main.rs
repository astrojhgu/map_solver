extern crate map_solver;

use rand::{thread_rng, Rng};
use rand_distr::StandardNormal;

use ndarray::{Array1, ArrayView1};
use ndarray::{array};
use num_complex::Complex64;
//use fftn::fft;
//use fftn::ifft;
//use num_traits::identities::Zero;
use linear_solver::io::RawMM;
use map_solver::mcmc2d_func::DT;
use map_solver::mcmc2d_func::{logprob_ana, logprob_ana_grad};
use map_solver::noise::gen_noise_2d;
use map_solver::utils::flatten_order_f;

fn main() {
    let mut input=ndarray::Array2::<Complex64>::eye(4);
    let mut output=ndarray::Array2::<Complex64>::zeros((4,4));
    input[(1,2)]=Complex64::new(1., 0.);
    map_solver::utils::fft2(input.view_mut(), output.view_mut());
    println!("{:?}", output);
}
