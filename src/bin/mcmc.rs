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
use map_solver::mcmc_func::{circulant_matrix, dft_matrix, circulant_det, cov2psd, psd2cov_mat, ln_xsx, dhalf_ln_xsx_dx, dhalf_ln_xsx_dp, dhalf_lndet_dps, mvn_ln_pdf, mvn_ln_pdf_grad, ps_mirror, ps_mirror_t, ln_likelihood, ln_det_sigma, ln_likelihood_grad, logprob, logprob_grad};
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
    let lp=|x: &LsVec<f64, Vec<f64>>|{
        let sky:Vec<_>=x.0.iter().take(nx).cloned().collect();
        let psd:Vec<_>=x.0.iter().skip(nx).cloned().collect();
        logprob(&sky, &psd, total_tod.as_slice().unwrap(), &ptr_mat, smooth_param)
    };

    let lp_grad=|x: &LsVec<f64, Vec<f64>>|{
        let sky:Vec<_>=x.0.iter().take(nx).cloned().collect();
        let psd:Vec<_>=x.0.iter().skip(nx).cloned().collect();
        let (gx, gp)=logprob_grad(&sky, &psd, total_tod.as_slice().unwrap(), &ptr_mat, smooth_param);
        LsVec(gx.iter().chain(gp.iter()).cloned().collect::<Vec<_>>())
    };
    
    let psd=vec![1.1; tod.len()/2+1];
    let x:Vec<_>=answer.iter().chain(psd.iter()).cloned().collect();
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
    let param=HmcParam::quick_adj(0.75);
    for i in 0..1000 {
        let accepted=sample(&lp, &lp_grad, &mut q, &mut lp_value, &mut lp_grad_value, &mut rng, &mut epsilon, 20, &param);
        if accepted{
            accept_cnt+=1;
        }
        cnt+=1;
        if i%1==0{
            let g=lp_grad_value.dot(&lp_grad_value);
            let psd:Vec<_>=q.0.iter().skip(nx).collect();
            let psd_g: Vec<_>=lp_grad_value.0.iter().skip(nx).collect();

            let dx=LsVec(q.0.iter().map(|_|{
                let f:f64=rng.sample(StandardNormal);
                f*0.00001
            }).collect::<Vec<_>>());
            let diff=dx.dot(&lp_grad_value);
            let q2=&q+&dx;
            let lp_value2=lp(&q2);

            println!("{} {:.3} {:.3} {:.5} {:.3} {:.3} {:.3} {:.3} {:.3} {:.3} {:.3} {:.3} {} {} {} {}",i, if accepted {1} else {0}, accept_cnt as f64/cnt as f64, epsilon, lp_value, g, psd[0], psd[10], psd[20], psd_g[0], psd_g[10], psd_g[20], diff, lp_value2-lp_value, diff/(lp_value2-lp_value), diff.abs()/lp_value.abs());
        }
    }

    let param=HmcParam::slow_adj(0.75);
    for i in 0..10000000 {
        let accepted=sample(&lp, &lp_grad, &mut q, &mut lp_value, &mut lp_grad_value, &mut rng, &mut epsilon, 20, &param);
        if accepted{
            accept_cnt+=1;
        }
        cnt+=1;
        if i%1==0{
            let g=lp_grad_value.dot(&lp_grad_value);
            let psd:Vec<_>=q.0.iter().skip(nx).collect();
            let psd_g: Vec<_>=lp_grad_value.0.iter().skip(nx).collect();

            let dx=LsVec(q.0.iter().map(|_|{
                let f:f64=rng.sample(StandardNormal);
                f*0.0001
            }).collect::<Vec<_>>());
            let diff=dx.dot(&lp_grad_value);
            let q2=&q+&dx;
            let lp_value2=lp(&q2);

            println!("{} {:.3} {:.3} {:.5} {:.3} {:.3} {:.3} {:.3} {:.3} {:.3} {:.3} {:.3} {} {} {} {}",i, if accepted {1} else {0}, accept_cnt as f64/cnt as f64, epsilon, lp_value, g, psd[0], psd[10], psd[20], psd_g[0], psd_g[10], psd_g[20], diff, lp_value2-lp_value, diff/(lp_value2-lp_value), diff.abs()/lp_value.abs());
        }
    }
}
