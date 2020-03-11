#![allow(unused_imports)]
#![allow(non_snake_case)]
extern crate linear_solver;
extern crate map_solver;
use fftn::ifft;
use linear_solver::io::RawMM;
use ndarray::{array, ArrayView1};
use num_complex::{Complex, Complex64};
use num_traits::{Float, FloatConst, NumAssign, NumCast};
use rand::distributions::Distribution;
use rand::{thread_rng, Rng};
use rand_distr::StandardNormal;
use scorus::linear_space::type_wrapper::LsVec;

use map_solver::madam::MappingProblem;
use map_solver::mcmc::Problem;
use map_solver::mcmc2d_func::DT;
use map_solver::mcmc2d_func::{
    dhalf_ln_xsx_dp, ln_likelihood_grad, logprob_ana_grad, mvn_ln_pdf, mvn_ln_pdf_grad,
};
use map_solver::mcmc2d_func::{PS_E, PS_W};
use map_solver::noise::gen_noise_2d;
use map_solver::utils::deflatten_order_f;
use map_solver::utils::flatten_order_f;

use map_solver::noise::noise2d_psd;
use map_solver::noise::white2d;
fn main() {
    let mut rng = thread_rng();
    let psp = [1.0, 0.001, -1.0, 0.001, -1.0, 0.1];
    let n_t = 2700;
    let n_ch = 32;
    let tod = RawMM::<f64>::from_file("ideal_data/tod_32_ch.mtx").to_array2();
    let tod = flatten_order_f(tod.view());

    let ptr_mat = RawMM::<f64>::from_file("ideal_data/ptr_32_ch.mtx").to_sparse();

    let answer = RawMM::<f64>::from_file("ideal_data/answer_32_ch.mtx").to_array2();
    let answer = flatten_order_f(answer.view());

    let noise_psd = noise2d_psd(n_t, n_ch, &psp, 2.0);
    RawMM::from_array2(noise_psd.view()).to_file("psd.mtx");
    let noise = gen_noise_2d(n_t, n_ch, &psp, &mut rng, 2.0);
    let total_tod = &tod + &flatten_order_f(noise.view());
    RawMM::from_array2(noise.view()).to_file("noise.mtx");
    let (gx, gp) = ln_likelihood_grad(
        answer.as_slice().unwrap(),
        total_tod.as_slice().unwrap(),
        noise_psd.view(),
        &ptr_mat,
        n_t,
        n_ch,
    );
    //println!("psd:{:?}", noise_psd[(100,10)]);
    println!("gx {:?} {:?}", gx[100], gp[100]);
    let gx = ndarray::Array1::from(gx);
    let gp = ndarray::Array1::from(gp);
    RawMM::from_array1(gp.view()).to_file("gp_ll.mtx");
    RawMM::from_array1(gx.view()).to_file("gx_ll.mtx");

    let (gx, gp) = logprob_ana_grad(
        answer.as_slice().unwrap(),
        &psp,
        total_tod.as_slice().unwrap(),
        &ptr_mat,
        n_t,
        n_ch,
    );
    let gx = ndarray::Array1::from(gx);
    let gp = ndarray::Array1::from(gp);
    RawMM::from_array1(gp.view()).to_file("gp.mtx");
    RawMM::from_array1(gx.view()).to_file("gx.mtx");
    let ft_min = 1.0 / (n_t as f64 * 2.0);
    let fch_min = 1.0 / n_ch as f64;
    let ft: Vec<_> = (0..(n_t as isize + 1) / 2)
        .chain(-(n_t as isize) / 2..0)
        .map(|i| i as f64 * ft_min)
        .collect();
    let fch: Vec<_> = (0..(n_ch as isize + 1) / 2)
        .chain(-(n_ch as isize) / 2..0)
        .map(|i| i as f64 * fch_min)
        .collect();

    let a_t = psp[0];
    let ft_0 = psp[1];
    let alpha_t = psp[2];

    let fch_0 = psp[3];
    let alpha_ch = psp[4];

    let b = psp[5];

    let m = map_solver::mcmc2d_func::dps_model_dalpha_t(
        &ft, &fch, a_t, ft_0, alpha_t, fch_0, alpha_ch, b, PS_W, PS_E,
    );
    RawMM::from_array2(m.view()).to_file("diff.mtx");

    RawMM::from_array1(ArrayView1::from(&ft)).to_file("ft.mtx");
    RawMM::from_array1(ArrayView1::from(&fch)).to_file("fch.mtx");
}
