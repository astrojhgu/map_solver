extern crate map_solver;

use rand::{thread_rng, Rng};
use rand_distr::StandardNormal;

use ndarray::{Array1, Array2, Array, array, ArrayView1};
use num_complex::Complex64;
use fftn::fft;
use fftn::ifft;
use map_solver::utils::{flatten_order_f, deflatten_order_f};
use num_traits::identities::Zero;
use map_solver::mcmc2d_func::{ln_likelihood, ln_det_sigma, ln_likelihood_grad};
use map_solver::mcmc2d_func::DT;
use map_solver::mcmc2d_func::ps_model;
use linear_solver::io::RawMM;
use linear_solver::utils::sp_mul_a1;

fn main(){
    let mut rng=thread_rng();

    let ptr_mat=RawMM::<f64>::from_file("ideal_data/ptr_32_ch.mtx").to_sparse();
    let tod=RawMM::<f64>::from_file("ideal_data/tod_32_ch.mtx").to_array2();
    let n_t=tod.nrows();
    let n_ch=tod.ncols();
    let tod=flatten_order_f(tod.view());
    let answer=flatten_order_f(RawMM::<f64>::from_file("ideal_data/answer_32_ch.mtx").to_array2().view());
    
    let noise:Array1<f64>=tod.map(|_| {
        let f:f64=rng.sample(StandardNormal);
        f
    });

    let total_tod=&tod+&noise;


    let ft_min=1.0/(n_t as f64*DT);
    let fch_min=1.0/n_ch as f64;
    let ft:Vec<_>=(0..(n_t as isize+1)/2).chain(-(n_t as isize)/2..0).map(|i| i as f64 * ft_min).collect();
    let fch:Vec<_>=(0..(n_ch as isize+1)/2).chain(-(n_ch as isize)/2..0).map(|i| i as f64 * fch_min).collect();
    let (a_t, ft_0, alpha_t)=(3.0, ft_min*2 as f64, -1.0);
    let (fch_0, alpha_ch)=(ft_min*2 as f64, -1.0);
    let b=0.1;
    let psd=ps_model(&ft, &fch, a_t, ft_0, alpha_t, fch_0, alpha_ch, b, 1e-9, 1e-9);

    let lp1=ln_likelihood(answer.as_slice().unwrap(), total_tod.as_slice().unwrap(), psd.view(), &ptr_mat, n_t, n_ch);
    

    
    let dx_sigma=0.00000001;

    for i in 0..10{
        let dx:Array1<f64>=answer.map(|_|{
            let f:f64=rng.sample(StandardNormal);
            f*dx_sigma
        });
    
        let dp=psd.map(|_|
        {
            let f:f64=rng.sample(StandardNormal);
            f*dx_sigma  
        }
        );

        let dp_1d=flatten_order_f(dp.view());
    
        let answer2=&answer+&dx;
        let psd2=&psd+&dp;
        //let dp=Array1::from_vec(dp);
    
        let lp2=ln_likelihood(answer2.as_slice().unwrap(), total_tod.as_slice().unwrap(), psd2.view(), &ptr_mat, n_t, n_ch);
    
        let diff=lp2-lp1;
        let (gx, gp)=ln_likelihood_grad(answer.as_slice().unwrap(), total_tod.as_slice().unwrap(), psd.view(), &ptr_mat, n_t, n_ch);

        let diff2=ArrayView1::from(&gx).dot(&dx)+ArrayView1::from(&gp).dot(&dp_1d);
        println!("{} {} {}",diff, diff2, (diff2-diff).abs());
    }
}
