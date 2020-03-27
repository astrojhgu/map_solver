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

    let mut q=problem.guess().to_vec();
    //let mut psp=vec![10.0, 0.01, -1.0, 0.05, -1.0, 1.0];
    let mut psp=vec![30.0, 0.01, -0.0, 0.05, -0.0, 0.0];
    q.append(&mut psp);
    let mut q=LsVec(q);

    let flag_psp:Vec<_>= (0..q.0.len())
    .map(|x| if x < nx { SSFlag::Fixed } else { SSFlag::Free })
    .collect();

    let (q_psp, q_rest)=split_ss(&q, &flag_psp);
    let lp_f=problem.get_logprob(&q_rest);
    let lp_g=problem.get_logprob_grad(&q_rest);

    let mut ensemble:Vec<_>=(0..80).map(|i|{
        if i==0{
            LsVec(q_psp.clone())
        }else{
            LsVec(q_psp.iter().map(|x: &f64| *x+0.01*rng.sample::<f64, StandardNormal>(StandardNormal)).collect())
        }
        
    }).collect();

    let beta_list:Vec<_>=(0..4).map(|i| 0.5_f64.powi(i)).collect();
    let nbeta=beta_list.len();
    let n_per_beta=ensemble.len()/nbeta;

    let mut lp:Vec<_>=ensemble.par_iter().enumerate().map(|(i, x)| {
        println!("{}", i);
        lp_f(x)}).collect();
    let mut ufs=UpdateFlagSpec::All;
    let mut epsilon = vec![0.003; beta_list.len()];
    //let param=HmcParam::quick_adj(0.75);
    let param = HmcParam::new(0.75, 0.05);


    for i in 0..10000{
        if i%10==0{
            swap_walkers(&mut ensemble, &mut lp, &mut rng, &beta_list);
        }
        let old_lp=lp.clone();
        emcee_pt(&lp_f, &mut ensemble, &mut lp, &mut rng, 2.0, &mut ufs, &beta_list);
        let mut emcee_accept_cnt=vec![0; nbeta];
        for (k, (l1, l2)) in lp.iter().zip(old_lp.iter()).enumerate(){
            if l1!=l2{
                emcee_accept_cnt[k/n_per_beta]+=1;
            }
        }

        //let hmc_accept_cnt=scorus::mcmc::hmc::naive::sample_ensemble_pt(&lp_f, &lp_g, &mut ensemble, &mut lp, &mut rng, &mut epsilon, &beta_list, L, &param);


        
        let mut max_i=0;
        let mut max_lp=std::f64::NEG_INFINITY;
        for (j, &x) in lp.iter().enumerate().take(n_per_beta){
            if x>max_lp{
                max_lp=x;
                max_i=j;
            }
        }

        eprintln!("{} {:?} {}", max_i, &(ensemble[max_i].0)[..], lp[max_i]);
        //eprintln!("{:?} {:?} {:?}",emcee_accept_cnt,  epsilon, hmc_accept_cnt);
        eprintln!("{:?}",emcee_accept_cnt);
        let q=combine_ss(&ensemble[max_i], &q_rest);
        RawMM::from_array1(ArrayView1::from(&q)).to_file("dump.mtx");
    }
}
