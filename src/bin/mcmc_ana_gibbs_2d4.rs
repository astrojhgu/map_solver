extern crate map_solver;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use rand::Rng;
use rand::thread_rng;
use rand_distr::StandardNormal;
use std::fs::File;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;

use scorus::linear_space::type_wrapper::LsVec;
use scorus::mcmc::ensemble_sample::sample_pt as emcee_pt;
use scorus::mcmc::ensemble_sample::UpdateFlagSpec;
use scorus::mcmc::hmc::naive::{sample, sample_ensemble_pt as hmc_sample, HmcParam};
use scorus::mcmc::utils::swap_walkers;
use ndarray::{Array1, ArrayView1};

use linear_solver::io::RawMM;
use map_solver::mcmc2d::Problem;
use map_solver::noise::gen_noise_2d;

use map_solver::utils::SSFlag;
use map_solver::utils::{combine_ss, split_ss};
use map_solver::mcmc2d_func::{logprob_ana, ln_likelihood, mvn_ln_pdf};
use map_solver::pl_ps::PlPs;
use map_solver::utils::{transpose, flatten};
use map_solver::ps_model::PsModel;

const L: usize = 5;
const NSTEPS: usize = 10;

fn main() {
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

    let ft: Vec<_> = (0..(n_t as isize + 1) / 2)
            .chain(-(n_t as isize) / 2..0)
            .map(|i| i as f64 * ft_min)
            .collect();
    let fch: Vec<_> = (0..(n_ch as isize + 1) / 2)
            .chain(-(n_ch as isize) / 2..0)
            .map(|i| i as f64 * fch_min)
            .collect();

    
    let nx = ptr_mat.cols();
    //let answer=vec![0.0; answer.len()];
    
    let psm=PlPs{};

    //let mut q=LsVec(RawMM::<f64>::from_file("q0.mtx").to_array1().to_vec());
    //println!("{:?}", q);
    let mut problem = Problem::empty(n_t, n_ch, psm);
    for i in 0..16{
        println!("{}",i);
        let noise_file_name=format!("noise_{}.mtx", i);
        //let noise=RawMM::<f64>::from_file(noise_file_name.as_str()).to_array1();
        let noise=flatten(gen_noise_2d(n_t, n_ch, &[10., 0.01, -1., 0.05, -1., 1.], &mut rng, 2.0).view());
        RawMM::from_array1(noise.view()).to_file(noise_file_name.as_str());
        let total_tod=&tod+&noise;
        problem=problem.with_obs(total_tod.as_slice().unwrap(), &ptr_mat);
    }

    //let mut psp=vec![10.0, 0.01, -1.0, 0.05, -1.0, 1.0];
    let mut psp=vec![30.0, 0.0001, -0.0, 0.0005, -0.0, 0.0];
    
    let beta_list:Vec<_>=(0..4).map(|i| 0.5_f64.powi(i)).collect();
    let mut x=problem.guess().to_vec();
    problem.sample_psp(&x, &mut psp, 20, &beta_list, 500, &mut rng);
    println!("{:?}", psp);
    problem.solve_map(&mut x, &psp);
    problem.sample_psp(&x, &mut psp, 20, &beta_list, 500, &mut rng);
    println!("{:?}", psp);
    problem.solve_map(&mut x, &psp);

}
