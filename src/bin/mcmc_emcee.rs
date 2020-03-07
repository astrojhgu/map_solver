extern crate map_solver;

use rand::{thread_rng, Rng};
use rand_distr::StandardNormal;

use std::fs::File;


use scorus::linear_space::traits::IndexableLinearSpace;
use scorus::linear_space::traits::InnerProdSpace;
use scorus::mcmc::ensemble_sample::sample;
use scorus::mcmc::ensemble_sample::UpdateFlagSpec;
use scorus::linear_space::type_wrapper::LsVec;

use ndarray::{Array1, Array2, Array, array, ArrayView1};
use num_complex::Complex64;
use fftn::fft;
use fftn::ifft;
use num_traits::identities::Zero;
use map_solver::mcmc_func::{circulant_matrix, dft_matrix, circulant_det, cov2psd, psd2cov_mat, ln_xsx, dhalf_ln_xsx_dx, dhalf_ln_xsx_dp, dhalf_lndet_dps, mvn_ln_pdf, mvn_ln_pdf_grad, ps_mirror, ps_mirror_t, ln_likelihood, ln_det_sigma, ln_likelihood_grad, logprob_ana, logprob_ana_grad, FMAX};
use map_solver::mcmc::Problem;
use linear_solver::io::RawMM;
use linear_solver::utils::sp_mul_a1;

const L:usize=2;

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
    let pps=vec![0.1, 0.1, 1.0/30.0, 0.0];
    let x:Vec<_>=answer.iter().chain(pps.iter()).cloned().collect();
    let mut q=LsVec(x);

    let noise=Array1::from_vec((0..tod.len()).map(|_|{
        let f:f64=rng.sample(StandardNormal);
        f
    }).collect::<Vec<_>>());
    let total_tod=&tod+&noise;
    let problem=Problem::new(total_tod.as_slice().unwrap(), &ptr_mat);
    let noise=Array1::from_vec((0..tod.len()).map(|_|{
        let f:f64=rng.sample(StandardNormal);
        f
    }).collect::<Vec<_>>());
    let total_tod=&tod+&noise;
    let problem=problem.with_obs(total_tod.as_slice().unwrap(), &ptr_mat);
    let noise=Array1::from_vec((0..tod.len()).map(|_|{
        let f:f64=rng.sample(StandardNormal);
        f
    }).collect::<Vec<_>>());
    let total_tod=&tod+&noise;
    let problem=problem.with_obs(total_tod.as_slice().unwrap(), &ptr_mat);
    let noise=Array1::from_vec((0..tod.len()).map(|_|{
        let f:f64=rng.sample(StandardNormal);
        f
    }).collect::<Vec<_>>());
    let total_tod=&tod+&noise;
    let problem=problem.with_obs(total_tod.as_slice().unwrap(), &ptr_mat);
    
    //.with_obs(total_tod.as_slice().unwrap(), &ptr_mat);
    
    let mut accept_cnt=0;
    let mut cnt=0;
    let mut epsilon=0.003;
    let mut epsilon_p=0.003;
    let mut epsilon_s=0.003;
    let mut accept_cnt_p=0;
    let mut accept_cnt_s=0;
    let mut accept_cnt=0;
    let mut cnt_p=0;
    let mut cnt_s=0;
    let mut cnt=0;
    //let param=HmcParam::quick_adj(0.75);

    let lp_f=problem.get_logprob();

    let mut ensemble=Vec::new();
    for i in 0..32{
        let q1:Vec<_>=q.0.iter().map(|&x: &f64|->f64{
            let f:f64=rng.sample(StandardNormal);
            x+0.01*f
        }).collect::<Vec<f64>>();
        ensemble.push(LsVec(q1));
    }

    let mut lp:Vec<_>=ensemble.iter().map(|x|{
        lp_f(x)
    }).collect();

    let mut uf=UpdateFlagSpec::Prob(0.1);

    let mut rng1=thread_rng();
    let mut i=0;
    let mut update_func1=move ||{
        let mut result=vec![false;nx+4];
        
        for i in 0..nx{
            result[i]=rng1.gen_range(0.0, 1.0)<0.5;
        }
    
        i+=1;
        result
    };
    let mut update_func2=move ||{
        let mut result=vec![false;nx+4];
        for i in nx..nx+4{
            result[i]=true;
        }
        result
    };
    for i in 0..100000 {
        let mut uf=
        if i<10000{
            UpdateFlagSpec::Func(&mut update_func1)
        }else if i%2==0{
            UpdateFlagSpec::Func(&mut update_func2)
        }else{
            UpdateFlagSpec::Func(&mut update_func1)
        };
        

        sample(&lp_f, &mut ensemble, &mut lp, &mut rng, 2.0, &mut uf, 32);
        let mean_value=ensemble[0].iter().take(nx).sum::<f64>()/nx as f64;
        let ps_param:Vec<_>=ensemble[0].iter().skip(nx).collect();
        let lp_value=lp[0];
        if i%100==0{
            println!("{} {:.5} {:e} {:?}",i, lp_value,mean_value, ps_param);    
        }


    }
}
