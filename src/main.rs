extern crate map_solver;

use rand::{thread_rng, Rng};
use rand_distr::StandardNormal;

use ndarray::{Array1, ArrayView1};
//use num_complex::Complex64;
//use fftn::fft;
//use fftn::ifft;
//use num_traits::identities::Zero;
use linear_solver::io::RawMM;
use map_solver::mcmc2d_func::DT;
use map_solver::mcmc2d_func::{logprob_ana, logprob_ana_grad};
use map_solver::noise::gen_noise_2d;
use map_solver::utils::flatten_order_f;

fn main() {
}
