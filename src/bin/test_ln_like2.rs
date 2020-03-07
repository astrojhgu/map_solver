extern crate map_solver;

use rand::{thread_rng, Rng};
use rand_distr::StandardNormal;

use ndarray::{Array1, Array2, Array, array, ArrayView1};
use num_complex::Complex64;
use fftn::fft;
use fftn::ifft;
use num_traits::identities::Zero;
use map_solver::mcmc_func::{circulant_matrix, dft_matrix, circulant_det, cov2psd, psd2cov_mat, ln_xsx, dhalf_ln_xsx_dx, dhalf_ln_xsx_dp, dhalf_lndet_dps, mvn_ln_pdf, mvn_ln_pdf_grad, ps_mirror, ps_mirror_t, ln_likelihood, ln_det_sigma, ln_likelihood_grad, logprob_ana, logprob_ana_grad};
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
    let psd=vec![0.96; tod.len()/2+1];


    let psd_param=vec![1.0, 0.001, 2.0, -1.0];

    let lp1=logprob_ana(answer.as_slice().unwrap(), &psd_param, total_tod.as_slice().unwrap(), &ptr_mat);
    

    
    let dx_sigma=0.00000001;

    for i in 0..100{
        let dx:Array1<f64>=answer.map(|_|{
            let f:f64=rng.sample(StandardNormal);
            f*dx_sigma
        });
    
        let dp:Vec<f64>=psd_param.iter().map(|_|
        {
            let f:f64=rng.sample(StandardNormal);
            f*dx_sigma  
        }
        ).collect();
    
        let answer2=&answer+&dx;
        let psd_param2:Vec<_>=psd_param.iter().zip(dp.iter()).map(|(x,y)| x+y).collect();

        let dp=Array1::from_vec(dp);
    
        let lp2=logprob_ana(answer2.as_slice().unwrap(), &psd_param2, total_tod.as_slice().unwrap(), &ptr_mat);
    
        let diff=lp2-lp1;
        let (gx, gp)=logprob_ana_grad(answer.as_slice().unwrap(), &psd_param, total_tod.as_slice().unwrap(), &ptr_mat);
        let diff2=ArrayView1::from(&gx).dot(&dx)+ArrayView1::from(&gp).dot(&dp);
        println!("{} {} {}",diff, diff2, (diff2-diff).abs());
    }
}
