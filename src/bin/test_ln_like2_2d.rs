extern crate map_solver;

use rand::{thread_rng, Rng};
use rand_distr::StandardNormal;

use ndarray::{Array1, ArrayView1};
//use num_complex::Complex64;
//use fftn::fft;
//use fftn::ifft;
//use num_traits::identities::Zero;
use linear_solver::io::RawMM;
use map_solver::mcmc2d_func::DT;
use map_solver::mcmc2d_func::{logprob_ana, logprob_ana_grad};
use map_solver::noise::gen_noise_2d;
use map_solver::utils::flatten_order_f;
use map_solver::ps_model::PsModel;
use map_solver::pl_ps::PlPs;
use map_solver::utils::{flatten, transpose};

fn main() {
    let mut rng = thread_rng();

    let ptr_mat = RawMM::<f64>::from_file("ideal_data/ptr_32_ch.mtx").to_sparse();
    let tod = transpose(RawMM::<f64>::from_file("ideal_data/tod_32_ch.mtx").to_array2().view());
    let n_t = tod.ncols();
    let n_ch = tod.nrows();
    let tod = flatten(tod.view());
    let answer = flatten(
        transpose(RawMM::<f64>::from_file("ideal_data/answer_32_ch.mtx")
            .to_array2().view())
            .view(),
    );

    let ft_min = 1.0 / (n_t as f64 * DT);
    let fch_min = 1.0 / n_ch as f64;
    let (a_t, ft_0, alpha_t) = (3.0, ft_min * 20_f64, -1.);
    let (fch_0, alpha_ch) = (fch_min * 5_f64, -1.);
    let b = 0.1;
    let ft: Vec<_> = (0..(n_t as isize + 1) / 2)
        .chain(-(n_t as isize) / 2..0)
        .map(|i| i as f64 * ft_min)
        .collect();
    let fch: Vec<_> = (0..(n_ch as isize + 1) / 2)
        .chain(-(n_ch as isize) / 2..0)
        .map(|i| i as f64 * fch_min)
        .collect();

    let psd_param = vec![a_t, ft_0, alpha_t, fch_0, alpha_ch, b];

    let noise = gen_noise_2d(n_t, n_ch, &psd_param, &mut rng, 2.0) * 0.2;
    let noise = flatten_order_f(noise.view());
    let total_tod = &tod + &noise;
    let psm=PlPs{};
    let lp1 = logprob_ana(
        answer.as_slice().unwrap(),
        &psd_param,
        total_tod.as_slice().unwrap(),
        &ptr_mat,
        &ft, 
        &fch,
        &psm,
    );

    let dx_sigma = 1e-5;

    for _i in 0..100 {
        let dx: Array1<f64> = answer.map(|_| {
            let f: f64 = rng.sample(StandardNormal);
            f * dx_sigma
        });

        let dp: Vec<f64> = psd_param
            .iter()
            .map(|_| {
                let f: f64 = rng.sample(StandardNormal);
                f * dx_sigma
            })
            .collect();

        let answer2 = &answer + &dx;
        let psd_param2: Vec<_> = psd_param
            .iter()
            .zip(dp.iter())
            .map(|(x, y)| x + y)
            .collect();

        let dp = Array1::from(dp);

        let lp2 = logprob_ana(
            answer2.as_slice().unwrap(),
            &psd_param2,
            total_tod.as_slice().unwrap(),
            &ptr_mat,
            &ft,
            &fch,
            &psm
        );

        let diff = lp2 - lp1;
        let (gx, gp) = logprob_ana_grad(
            answer.as_slice().unwrap(),
            &psd_param,
            total_tod.as_slice().unwrap(),
            &ptr_mat,
            &ft,
            &fch,
            &psm
        );
        println!("gx {} gp {}", gx.len(), gp.len());
        let diff2 = ArrayView1::from(&gx).dot(&dx) + ArrayView1::from(&gp).dot(&dp);
        println!(
            "{} {} {} {} {}",
            lp2,
            lp1,
            diff,
            diff2,
            2.0 * (diff2 - diff).abs() / (diff.abs() + diff2.abs())
        );
    }
}
