extern crate map_solver;

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
use map_solver::mcmc_func::{circulant_matrix, dft_matrix, circulant_det, cov2psd, psd2cov_mat, ln_xsx, dhalf_ln_xsx_dx, dhalf_ln_xsx_dp, dhalf_lndet_dps, mvn_ln_pdf, mvn_ln_pdf_grad, ln_likelihood, ln_det_sigma, ln_likelihood_grad, logprob_ana, logprob_ana_grad, FMAX};
use map_solver::noise::gen_noise;
use map_solver::mcmc::Problem;
use linear_solver::io::RawMM;
use linear_solver::utils::sp_mul_a1;

const L:usize=1;
const nsteps:usize=5;

fn main(){
    let mut rng=thread_rng();

    let ptr_mat=RawMM::<f64>::from_file("ptr_mat.mtx").to_sparse();
    let tod=RawMM::<f64>::from_file("cheat_vis.mtx").to_array1();

    let answer=if let Ok(_f)=File::open("solution.mtx"){
        RawMM::<f64>::from_file("solution.mtx").to_array1()
    }else{
        println!("{}", ptr_mat.cols());
        Array1::zeros(ptr_mat.cols())
    };
    
    
    

    let ntod=ptr_mat.rows();
    let nx=ptr_mat.cols();
    //let answer=vec![0.0; answer.len()];
    let psp=vec![0.1, 0.1, 1.0/30.0, 0.0];
    let x:Vec<_>=answer.iter().chain(psp.iter()).cloned().collect();
    let mut q=LsVec(x);

    let noise:Array1<f64>=tod.map(|_| {
        let f:f64=rng.sample(StandardNormal);
        f
    });

    let psp=vec![20.0, 2.0, 0.001, -1.0];
    let mut problem=Problem::empty();

    for i in 0..4{
        let noise=Array1::from(gen_noise(ntod, &psp, &mut rng, map_solver::mcmc_func::DT));
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
        
        {//sample p
            let sky=q.0.iter().take(nx).cloned().collect::<Vec<_>>();
            let mut q1=LsVec(q.0.iter().skip(nx).cloned().collect::<Vec<_>>());

            let lp=problem.get_logprob_psp(&q);
            let lp_grad=problem.get_logprob_grad_psp(&q);

            let mut lp_value=lp(&q1);
            let mut lp_grad_value=lp_grad(&q1);

            for j in 0..nsteps{
                let accepted=sample(&lp, &lp_grad, &mut q1, &mut lp_value, &mut lp_grad_value, &mut rng, &mut epsilon_p, L, &param);
                if accepted{
                    accept_cnt_p+=1;
                }
                cnt_p+=1;    
            }

            q=LsVec(sky.iter().chain(q1.iter()).cloned().collect::<Vec<_>>());
            if i%10==0{
                println!("{} {:.3} {:.8} {:.5}  {:?}",i, accept_cnt_p as f64/cnt_p as f64, epsilon_p, lp_value, q1.0);
            }

        }
        {
            let psp=q.0.iter().skip(nx).cloned().collect::<Vec<_>>();
            let mut q1=LsVec(q.0.iter().take(nx).cloned().collect::<Vec<_>>());

            let lp=problem.get_logprob_sky(&q);
            let lp_grad=problem.get_logprob_grad_sky(&q);

            let mut lp_value=lp(&q1);
            let mut lp_grad_value=lp_grad(&q1);

            for j in 0..nsteps{
                let accepted=sample(&lp, &lp_grad, &mut q1, &mut lp_value, &mut lp_grad_value, &mut rng, &mut epsilon_s, L, &param);
                if accepted{
                    accept_cnt_s+=1;
                }
                cnt_s+=1;    
            }
            q=LsVec(q1.iter().chain(psp.iter()).cloned().collect::<Vec<_>>());

            let mean_value=q1.0.iter().sum::<f64>()/nx as f64;
            if i%10==0{
                println!("{} {:.3} {:.8} {:.5} {:e}",i, accept_cnt_s as f64/cnt_s as f64, epsilon_s, lp_value,mean_value);
            }
        }
    }
}
