extern crate map_solver;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use rand::{thread_rng, Rng};
use rand_distr::StandardNormal;

use std::fs::File;


use scorus::linear_space::traits::IndexableLinearSpace;
use scorus::linear_space::traits::InnerProdSpace;
use scorus::mcmc::hmc::naive::sample;
use scorus::mcmc::hmc::naive::HmcParam;
use scorus::linear_space::type_wrapper::LsVec;

use ndarray::{Array1, Array2, Array, array, ArrayView1};
use num_complex::Complex64;
use fftn::fft;
use fftn::ifft;
use num_traits::identities::Zero;
use map_solver::noise::gen_noise_2d;
use map_solver::utils::flatten_order_f;
use map_solver::mcmc2d::Problem;
use map_solver::utils::{split_ss, combine_ss};
use linear_solver::io::RawMM;
use linear_solver::utils::sp_mul_a1;

const L:usize=1;
const nsteps:usize=5;

fn main(){
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        println!("ctrl+C pressed, terminating...");
        r.store(false, Ordering::SeqCst);
    }).expect("Error setting Ctrl-C handler");



    let mut rng=thread_rng();

    let ptr_mat=RawMM::<f64>::from_file("ideal_data/ptr_32_ch.mtx").to_sparse();
    let tod=RawMM::<f64>::from_file("ideal_data/tod_32_ch.mtx").to_array2();
    let n_t=tod.nrows();
    let n_ch=tod.ncols();
    let tod=flatten_order_f(tod.view());
    let answer=flatten_order_f(RawMM::<f64>::from_file("ideal_data/answer_32_ch.mtx").to_array2().view());
    

    let ft_min=1.0/(n_t as f64*2.0);
    let fch_min=1.0/n_ch as f64;
    let (a_t, ft_0, alpha_t)=(3.0, ft_min*20 as f64, -1.);
    let (fch_0, alpha_ch)=(fch_min*5 as f64, -1.);
    let b=0.1;
    
    
    let nx=ptr_mat.cols();
    //let answer=vec![0.0; answer.len()];
    let mut q=if let Ok(f)=File::open("dump.mtx"){
        println!("dumped file found, loading...");
        LsVec(RawMM::from_file("dump.mtx").to_array1().to_vec())
    }else{
        println!("dumped file not found, use default values");
        let psp=vec![0.1, 0.1, 0.0, 0.1, 0.0, 0.0];
        let x:Vec<_>=answer.iter().chain(psp.iter()).cloned().collect();
        LsVec(x)
    };

    let mut problem=Problem::empty(n_t, n_ch);

    let psd_param=vec![a_t, ft_0, alpha_t, fch_0, alpha_ch, b];

    for i in 0..16{
        let noise=gen_noise_2d(n_t, n_ch, &psd_param, &mut rng, 2.0)*0.2;
        let noise=flatten_order_f(noise.view());
        let total_tod=&tod+&noise;
        problem=problem.with_obs(total_tod.as_slice().unwrap(), &ptr_mat);    
    }
    
    let mut accept_cnt=0;
    let mut cnt=0;
    let mut epsilon=0.003;
    let mut epsilon_p=0.003;
    let mut epsilon_s=0.003;
    //let param=HmcParam::quick_adj(0.75);
    let mut param=HmcParam::new(0.75, 0.05);

    for i in 0..1000000 {
        if i>1000{
            param=HmcParam::slow_adj(0.75);
        }
        let mut accept_cnt_p=0;
        let mut accept_cnt_s=0;
        let mut accept_cnt=0;
        let mut cnt_p=0;
        let mut cnt_s=0;
        let mut cnt=0;
        {
            let flags:Vec<_>=(0..q.0.len()).map(|x| x < nx).collect();
            //let mut q1=LsVec(q.0.iter().take(nx).cloned().collect::<Vec<_>>());
            let (q1, q_rest)=split_ss(&q, &flags);
            let mut q1=LsVec(q1);

            let lp=problem.get_logprob(&q_rest);
            let lp_grad=problem.get_logprob_grad(&q_rest);

            let mut lp_value=lp(&q1);
            let mut lp_grad_value=lp_grad(&q1);
            //println!("{:?} {}", q1.0.len(), nx);

            for j in 0..nsteps{
                let accepted=sample(&lp, &lp_grad, &mut q1, &mut lp_value, &mut lp_grad_value, &mut rng, &mut epsilon_s, L, &param);
                if accepted{
                    accept_cnt_s+=1;
                }
                cnt_s+=1;    
            }
            //q=LsVec(q1.iter().chain(psp.iter()).cloned().collect::<Vec<_>>());
            q=LsVec(combine_ss(&q1, &q_rest));

            let mean_value=q1.0.iter().sum::<f64>()/nx as f64;
            if i%10==0{
                println!("{} {:.3} {:.8} {:.5} {:e}",i, accept_cnt_s as f64/cnt_s as f64, epsilon_s, lp_value,mean_value);
            }
        }
        
        {//sample p
            let flags:Vec<_>=(0..q.0.len()).map(|x| x >= nx).collect();
            let (q1, q_rest)=split_ss(&q, &flags);
            let mut q1=LsVec(q1);

            let lp=problem.get_logprob(&q_rest);
            let lp_grad=problem.get_logprob_grad(&q_rest);

            let mut lp_value=lp(&q1);
            let mut lp_grad_value=lp_grad(&q1);

            for j in 0..nsteps{
                let accepted=sample(&lp, &lp_grad, &mut q1, &mut lp_value, &mut lp_grad_value, &mut rng, &mut epsilon_p, L, &param);
                if accepted{
                    accept_cnt_p+=1;
                }
                cnt_p+=1;    
            }

            q=LsVec(combine_ss(&q1, &q_rest));
            if i%10==0{
                println!("{} {:.3} {:.8} {:.5}  {:?}",i, accept_cnt_p as f64/cnt_p as f64, epsilon_p, lp_value, q1.0);
            }

        }
        if i%10==0{
            RawMM::from_array1(Array1::from(q.0.clone()).view()).to_file("dump.mtx");
        }

        if !running.load(Ordering::SeqCst) {
            println!("{:?}", &q.0[nx..]);
            RawMM::from_array1(Array1::from(q.0.clone()).view()).to_file("dump.mtx");
            break;
        }
    }
}
