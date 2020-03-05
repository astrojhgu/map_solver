extern crate map_solver;

use rand::{thread_rng, Rng};
use rand_distr::StandardNormal;

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
use map_solver::utils::{circulant_matrix, dft_matrix, circulant_det, cov2psd, psd2cov_mat, ln_xsx, dhalf_ln_xsx_dx, dhalf_ln_xsx_dp, dhalf_lndet_dps, mvn_ln_pdf, mvn_ln_pdf_grad, ps_mirror, ps_mirror_t, ln_likelihood, ln_det_sigma, ln_likelihood_grad, logprob, logprob_grad};
use linear_solver::io::RawMM;
use linear_solver::utils::sp_mul_a1;

fn main(){
    let mut rng=thread_rng();

    let ptr_mat=RawMM::<f64>::from_file("ptr_mat.mtx").to_sparse();
    let tod=RawMM::<f64>::from_file("cheat_vis.mtx").to_array1();
    let answer=RawMM::<f64>::from_file("solution.mtx").to_array1();
    
    let noise:Array1<f64>=tod.map(|_| {
        let f:f64=rng.sample(StandardNormal);
        f
    });

    let total_tod=&tod+&noise;

    let ntod=ptr_mat.rows();
    let nx=ptr_mat.cols();
    let npsd=ntod/2+1;
    let smooth_param=2.0;
    let psd=vec![1.0; tod.len()/2+1];

    let lp=|x: &LsVec<f64, Vec<f64>>|{
        ln_likelihood(x, total_tod.as_slice().unwrap(),&psd,  &ptr_mat)
    };

    let lp_grad=|x: &LsVec<f64, Vec<f64>>|{
        let (dx, _dp)=ln_likelihood_grad(x, total_tod.as_slice().unwrap(), &psd, &ptr_mat);
        LsVec(dx)
    };
    
    //let x:Vec<_>=answer.to_vec();
    let x=vec![0.0; answer.len()];
    let mut q=LsVec(x);
    let mut lp_value=lp(&q);
    let mut lp_grad_value=lp_grad(&q);

    let dx=LsVec(q.0.iter().map(|_|{
        let f:f64=rng.sample(StandardNormal);
        f*0.0001
    }).collect::<Vec<_>>());

    let diff=dx.dot(&lp_grad_value);
    let q2=&q+&dx;
    let lp_value2=lp(&q2);
    println!("{} {}", diff, lp_value2-lp_value);
    
    let mut accept_cnt=0;
    let mut cnt=0;
    let mut epsilon=0.005;
    //let param=HmcParam::quick_adj(0.75);
    let param=HmcParam::new(0.75, 0.1);
    for i in 0..1000 {
        let accepted=sample(&lp, &lp_grad, &mut q, &mut lp_value, &mut lp_grad_value, &mut rng, &mut epsilon, 20, &param);
        if accepted{
            accept_cnt+=1;
        }
        cnt+=1;
        if i%1==0{
            let g=lp_grad_value.dot(&lp_grad_value);
            
            let dx=LsVec(q.0.iter().map(|_|{
                let f:f64=rng.sample(StandardNormal);
                f*0.00001
            }).collect::<Vec<_>>());
            let diff=dx.dot(&lp_grad_value);
            let q2=&q+&dx;
            let lp_value2=lp(&q2);

            println!("{} {:.3} {:.3} {:.5} {} {} {} {} {} {} {}",i, if accepted {1} else {0}, accept_cnt as f64/cnt as f64, epsilon, lp_value, g, diff, lp_value2-lp_value, diff/(lp_value2-lp_value), diff.abs()/lp_value.abs(), q[10]);
        }
    }

    let param=HmcParam::slow_adj(0.75);
    for i in 0..1000000 {
        let accepted=sample(&lp, &lp_grad, &mut q, &mut lp_value, &mut lp_grad_value, &mut rng, &mut epsilon, 20, &param);
        if accepted{
            accept_cnt+=1;
        }
        cnt+=1;
        if i%100==0{
            let g=lp_grad_value.dot(&lp_grad_value);
            
            let dx=LsVec(q.0.iter().map(|_|{
                let f:f64=rng.sample(StandardNormal);
                f*0.00001
            }).collect::<Vec<_>>());
            let diff=dx.dot(&lp_grad_value);
            let q2=&q+&dx;
            let lp_value2=lp(&q2);

            println!("{} {:.3} {:.3} {:.5} {} {} {} {} {} {} {}",i, if accepted {1} else {0}, accept_cnt as f64/cnt as f64, epsilon, lp_value, g, diff, lp_value2-lp_value, diff/(lp_value2-lp_value), diff.abs()/lp_value.abs(), q[10]);
        }
    }
}
