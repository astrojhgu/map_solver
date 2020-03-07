extern crate map_solver;

use rand::{thread_rng, Rng};
use rand_distr::StandardNormal;

use scorus::opt::cg::cg_iter;
use scorus::opt::tolerance::Tolerance;
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

    let ntod=ptr_mat.rows();
    let nx=ptr_mat.cols();
    let npsd=ntod/2+1;
    let smooth_param=2.0;
    let fobj=|x: &LsVec<f64, Vec<f64>>|{
        let sky:Vec<_>=x.0.iter().take(nx).cloned().collect();
        let pps:Vec<_>=x.0.iter().skip(nx).cloned().collect();
        
        -logprob_ana(&sky, &pps, total_tod.as_slice().unwrap(), &ptr_mat)
    };

    let grad=|x: &LsVec<f64, Vec<f64>>|{
        let sky:Vec<_>=x.0.iter().take(nx).cloned().collect();
        let pps:Vec<_>=x.0.iter().skip(nx).map(|&p|{p.abs()}).collect();
        let (gx, gp)=logprob_ana_grad(&sky, &pps, total_tod.as_slice().unwrap(), &ptr_mat);
        LsVec(gx.iter().chain(gp.iter()).map(|&x|{-x}).collect::<Vec<_>>())
    };

    let pps=vec![0.1, 0.1, 5.0, 0.0];

    let x:Vec<_>=answer.iter().chain(pps.iter()).cloned().collect();
    let mut x=LsVec(x);
    let mut g=grad(&x);
    let mut d=&g*(-1.0);
    let mut fret=fobj(&x);

    for i in 0..1000000{
        cg_iter(&fobj, &grad, &mut x, &mut d, &mut g, &mut fret, Tolerance::Abs(0.00001));
        println!("{} {} {}",i, fret, g.dot(&g));
    }
}
