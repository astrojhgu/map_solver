extern crate map_solver;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use rand::Rng;
use rand::thread_rng;
use rand_distr::StandardNormal;
use std::fs::File;

use scorus::linear_space::type_wrapper::LsVec;
use scorus::mcmc::ensemble_sample::sample_pt as emcee_pt;
use scorus::mcmc::ensemble_sample::UpdateFlagSpec;
use scorus::mcmc::hmc::naive::{sample, sample_ensemble_pt as hmc_sample, HmcParam};
use scorus::mcmc::utils::swap_walkers;
use ndarray::Array1;

use linear_solver::io::RawMM;
use map_solver::mcmc2d::Problem;
use map_solver::noise::gen_noise_2d;

use map_solver::utils::SSFlag;
use map_solver::utils::{combine_ss, split_ss};
use map_solver::mcmc2d_func::{logprob_ana, ln_likelihood, mvn_ln_pdf};
use map_solver::pl_ps::PlPs;
use map_solver::utils::{transpose, flatten};


const L: usize = 5;
const NSTEPS: usize = 10;

fn main() {
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        println!("ctrl+C pressed, terminating...");
        r.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    let mut rng = thread_rng();

    let ptr_mat = RawMM::<f64>::from_file("ideal_data/ptr_32_ch.mtx").to_sparse();
    let tod = transpose(RawMM::<f64>::from_file("ideal_data/tod_32_ch.mtx").to_array2().view());
    let n_t = tod.ncols();
    let n_ch = tod.nrows();
    let tod = flatten(tod.view());
    let answer = flatten(
        transpose(RawMM::<f64>::from_file("ideal_data/answer_32_ch.mtx")
            .to_array2().view())
            .view(),
    );

    let ft_min = 1.0 / (n_t as f64 * 2.0);
    let fch_min = 1.0 / n_ch as f64;
    let (a_t, ft_0, alpha_t) = (3.0, ft_min * 20_f64, -1.);
    let (fch_0, alpha_ch) = (fch_min * 5_f64, -1.);
    let b = 0.1;

    let nx = ptr_mat.cols();
    //let answer=vec![0.0; answer.len()];
    let psp = vec![1.0, 0.1, -1.0, 0.1, -1.0, 0.0];
    let mut q = if let Ok(_f) = File::open("dump.mtx") {
        println!("dumped file found, loading...");
        LsVec(RawMM::from_file("dump.mtx").to_array1().to_vec())
    } else {
        println!("dumped file not found, use default values");
        
        let x: Vec<_> = answer.iter().chain(psp.iter()).cloned().collect();
        LsVec(x)
    };
    let psm=PlPs{};
    let mut problem = Problem::empty(n_t, n_ch, psm);

    let psd_param = vec![a_t, ft_0, alpha_t, fch_0, alpha_ch, b];

    
    let noise2d = gen_noise_2d(n_t, n_ch, &psd_param, &mut rng, 2.0) * 0.2;

    RawMM::from_array2(noise2d.view()).to_file("noise.mtx");

    let noise = flatten(noise2d.view());
    let total_tod = &tod + &noise;
    problem = problem.with_obs(total_tod.as_slice().unwrap(), &ptr_mat);


    let flag_psp:Vec<_>= (0..q.0.len())
    .map(|x| if x < nx { SSFlag::Fixed } else { SSFlag::Free })
    .collect();

    let (q_psp, q_rest)=split_ss(&q, &flag_psp);
    let lp_f=problem.get_logprob(&q_rest);

    println!("{}", lp_f(&LsVec(q_psp)));
}
