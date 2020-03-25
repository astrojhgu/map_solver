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
use map_solver::utils::flatten_order_f;
use map_solver::utils::SSFlag;
use map_solver::utils::{combine_ss, split_ss};
use map_solver::pl_ps::PlPs;

const L: usize = 5;
const NSTEPS: usize = 10;

fn main() {
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    //ctrlc::set_handler(move || {
    //    println!("ctrl+C pressed, terminating...");
    //    r.store(false, Ordering::SeqCst);
    //})
    //.expect("Error setting Ctrl-C handler");

    let mut rng = thread_rng();

    let ptr_mat = RawMM::<f64>::from_file("ideal_data/ptr_32_ch.mtx").to_sparse();
    let tod = RawMM::<f64>::from_file("ideal_data/tod_32_ch.mtx").to_array2();
    let n_t = tod.nrows();
    let n_ch = tod.ncols();
    let tod = flatten_order_f(tod.view());
    let answer = flatten_order_f(
        RawMM::<f64>::from_file("ideal_data/answer_32_ch.mtx")
            .to_array2()
            .view(),
    );

    let ft_min = 1.0 / (n_t as f64 * 2.0);
    let fch_min = 1.0 / n_ch as f64;
    let (a_t, ft_0, alpha_t) = (3.0, ft_min * 20_f64, -1.);
    let (fch_0, alpha_ch) = (fch_min * 5_f64, -1.);
    let b = 0.1;

    let nx = ptr_mat.cols();
    //let answer=vec![0.0; answer.len()];
    let mut q = if let Ok(_f) = File::open("dump.mtx") {
        println!("dumped file found, loading...");
        LsVec(RawMM::from_file("dump.mtx").to_array1().to_vec())
    } else {
        println!("dumped file not found, use default values");
        let psp = vec![0.1, 0.1, 0.0, 0.1, 0.0, 0.0];
        let x: Vec<_> = answer.iter().chain(psp.iter()).cloned().collect();
        LsVec(x)
    };

    let psm=PlPs{};

    let mut problem = Problem::empty(n_t, n_ch, psm);

    let psd_param = vec![a_t, ft_0, alpha_t, fch_0, alpha_ch, b];

    for _i in 0..1 { 
        let noise = gen_noise_2d(n_t, n_ch, &psd_param, &mut rng, 2.0) * 0.2;
        let noise = flatten_order_f(noise.view());
        let total_tod = &tod + &noise;
        problem = problem.with_obs(total_tod.as_slice().unwrap(), &ptr_mat);
    }

    let flag_psp:Vec<_>= (0..q.0.len())
    .map(|x| if x < nx { SSFlag::Fixed } else { SSFlag::Free })
    .collect();

    let (q_psp, q_rest)=split_ss(&q, &flag_psp);
    println!("{:?}", q_psp);

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

    let lp_f=problem.get_logprob(&q_rest);
    let mut lp:Vec<_>=ensemble.par_iter().enumerate().map(|(i, x)| {
        println!("{}", i);
        lp_f(x)}).collect();

    eprintln!("{:?}", lp);
    let mut ufs=UpdateFlagSpec::All;
    
    for i in 0..10000{
        emcee_pt(&lp_f, &mut ensemble, &mut lp, &mut rng, 2.0, &mut ufs, &beta_list);
        let mut max_i=0;
        let mut max_lp=std::f64::NEG_INFINITY;
        for (j, &x) in lp.iter().enumerate().take(n_per_beta){
            if x>max_lp{
                max_lp=x;
                max_i=j;
            }
        }
        eprintln!("{} {:?} {}", max_i, ensemble[max_i], lp[max_i]);
        let q=combine_ss(&ensemble[max_i], &q_rest);
        RawMM::from_array1(ArrayView1::from(&q)).to_file("dump.mtx");
    }
    
    println!("{:?}", lp);

    /*

    let mut epsilon_p = 0.003;
    let mut epsilon_s = 0.003;
    //let param=HmcParam::quick_adj(0.75);
    let mut param = HmcParam::new(0.75, 0.05);

    for i in 0..1_000_000 {
        if i > 1000 {
            param = HmcParam::slow_adj(0.75);
        }
        let mut accept_cnt_p = 0;
        let mut accept_cnt_s = 0;

        let mut cnt_p = 0;
        let mut cnt_s = 0;

        {
            //let flags:Vec<_>=(0..q.0.len()).map(|x| x < nx).collect();
            let flags: Vec<_> = (0..q.0.len())
                .map(|x| if x < nx { SSFlag::Free } else { SSFlag::Fixed })
                .collect();
            //let mut q1=LsVec(q.0.iter().take(nx).cloned().collect::<Vec<_>>());
            let (q1, q_rest) = split_ss(&q, &flags);
            let mut q1 = LsVec(q1);

            let lp = problem.get_logprob(&q_rest);
            let lp_grad = problem.get_logprob_grad(&q_rest);

            let mut lp_value = lp(&q1);
            let mut lp_grad_value = lp_grad(&q1);
            //println!("{:?} {}", q1.0.len(), nx);
            eprint!("x");
            for _j in 0..NSTEPS {
                let accepted = sample(
                    &lp,
                    &lp_grad,
                    &mut q1,
                    &mut lp_value,
                    &mut lp_grad_value,
                    &mut rng,
                    &mut epsilon_s,
                    L,
                    &param,
                );
                if accepted {
                    eprint!(".");
                    accept_cnt_s += 1;
                } else {
                    eprint!(" ")
                }
                cnt_s += 1;
            }
            eprintln!("$");
            //q=LsVec(q1.iter().chain(psp.iter()).cloned().collect::<Vec<_>>());
            q = LsVec(combine_ss(&q1, &q_rest));

            let mean_value = q1.0.iter().sum::<f64>() / nx as f64;
            if i % 10 == 0 {
                println!(
                    "{} {:.3} {:.8} {:.5} {:e}",
                    i,
                    accept_cnt_s as f64 / cnt_s as f64,
                    epsilon_s,
                    lp_value,
                    mean_value
                );
            }
        }

        {
            //sample p
            let flags: Vec<_> = (0..q.0.len())
                .map(|x| if x >= nx { SSFlag::Free } else { SSFlag::Fixed })
                .collect();
            let (q1, q_rest) = split_ss(&q, &flags);
            let mut q1 = LsVec(q1);

            let lp = problem.get_logprob(&q_rest);
            let lp_grad = problem.get_logprob_grad(&q_rest);

            let mut lp_value = lp(&q1);
            let mut lp_grad_value = lp_grad(&q1);
            eprint!("p");
            for _j in 0..NSTEPS {
                let accepted = sample(
                    &lp,
                    &lp_grad,
                    &mut q1,
                    &mut lp_value,
                    &mut lp_grad_value,
                    &mut rng,
                    &mut epsilon_p,
                    L,
                    &param,
                );
                if accepted {
                    eprint!(".");
                    accept_cnt_p += 1;
                } else {
                    eprint!(" ");
                }
                cnt_p += 1;
            }
            eprintln!("$");
            q = LsVec(combine_ss(&q1, &q_rest));
            if i % 10 == 0 {
                println!(
                    "{} {:.3} {:.8} {:.5}  {:?}",
                    i,
                    accept_cnt_p as f64 / cnt_p as f64,
                    epsilon_p,
                    lp_value,
                    q1.0
                );
            }
        }
        if i % 10 == 0 {
            RawMM::from_array1(Array1::from(q.0.clone()).view()).to_file("dump.mtx");
        }

        if !running.load(Ordering::SeqCst) {
            println!("{:?}", &q.0[nx..]);
            RawMM::from_array1(Array1::from(q.0).view()).to_file("dump.mtx");
            break;
        }
    }
    */
}
