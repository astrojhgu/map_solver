extern crate map_solver;

use rand::{thread_rng, Rng};
use rand_distr::StandardNormal;

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
    let psd=vec![15.; tod.len()/2+1];
    let dx_std=0.00001;
    let lp0=logprob(answer.as_slice().unwrap(), &psd, total_tod.as_slice().unwrap(), &ptr_mat, 1.0);
    println!("{}", lp0);


    let dx=answer.map(|_|{
        let f:f64=rng.sample(StandardNormal);
        f*dx_std
    });

    let dp:Vec<_>=psd.iter().map(|_|{
        let f: f64=rng.sample(StandardNormal);
        f*dx_std
    }).collect();

    let answer2=&answer+&dx;
    let psd2=&ArrayView1::from(&psd)+&ArrayView1::from(&dp);

    let lp1=logprob(answer2.as_slice().unwrap(), psd2.as_slice().unwrap(), total_tod.as_slice().unwrap(), &ptr_mat, 1.0);

    let (gx, gp)=logprob_grad(answer.as_slice().unwrap(), psd2.as_slice().unwrap(), total_tod.as_slice().unwrap(), &ptr_mat, 1.0);

    let diff=&ArrayView1::from(&gx).dot(&dx) + &ArrayView1::from(&gp).dot(&ArrayView1::from(&dp));

    println!("{:e} {:e} {:e}", diff, lp1-lp0, diff-(lp1-lp0));

}
