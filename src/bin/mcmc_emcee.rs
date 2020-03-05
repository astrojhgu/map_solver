extern crate map_solver;

use rand::{thread_rng, Rng};
use rand_distr::StandardNormal;

use scorus::linear_space::traits::IndexableLinearSpace;
use scorus::linear_space::traits::InnerProdSpace;
use scorus::mcmc::ensemble_sample::sample;
use scorus::linear_space::type_wrapper::LsVec;
use scorus::mcmc::ensemble_sample::UpdateFlagSpec;

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
    let mut ufs=UpdateFlagSpec::Prob(0.5);
    let mut ensemble=(0..64).map(|_|{
        let dx:Vec<_>=q.0.iter().map(|_|{
            let f:f64=rng.sample(StandardNormal);
            f*0.01
        }).collect();
        let dx=LsVec(dx);
        &q+&dx
    }).collect::<Vec<_>>();
    let mut lp_list=ensemble.iter().map(|x|lp(x)).collect::<Vec<_>>();

    for i in 0..10000000 {
        sample(&lp, &mut ensemble, &mut lp_list, &mut rng, 2.0, &mut ufs, 32);
        if i%100==0{
            println!("{} {}", i, lp_list[0]);

        }
    }
}
