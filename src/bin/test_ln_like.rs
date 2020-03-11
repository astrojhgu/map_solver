extern crate map_solver;

use rand::{thread_rng, Rng};
use rand_distr::StandardNormal;

use ndarray::{Array1, ArrayView1};

use linear_solver::io::RawMM;
use map_solver::mcmc_func::{ln_likelihood, ln_likelihood_grad};

fn main() {
    let mut rng = thread_rng();

    let ptr_mat = RawMM::<f64>::from_file("ptr_mat.mtx").to_sparse();
    let tod = RawMM::<f64>::from_file("cheat_vis.mtx").to_array1();
    let answer = RawMM::<f64>::from_file("solution.mtx").to_array1();

    let noise: Array1<f64> = tod.map(|_| {
        let f: f64 = rng.sample(StandardNormal);
        f
    });

    let total_tod = &tod + &noise;
    let psd = vec![0.96; tod.len()];

    let lp1 = ln_likelihood(
        answer.as_slice().unwrap(),
        total_tod.as_slice().unwrap(),
        &psd,
        &ptr_mat,
    );

    let dx_sigma = 0.000_000_01;

    for _i in 0..10 {
        let dx: Array1<f64> = answer.map(|_| {
            let f: f64 = rng.sample(StandardNormal);
            f * dx_sigma
        });

        let dp: Vec<f64> = psd
            .iter()
            .map(|_| {
                let f: f64 = rng.sample(StandardNormal);
                f * dx_sigma
            })
            .collect();

        let answer2 = &answer + &dx;
        let psd2: Vec<_> = psd.iter().zip(dp.iter()).map(|(x, y)| x + y).collect();
        let dp = Array1::from(dp);

        let lp2 = ln_likelihood(
            answer2.as_slice().unwrap(),
            total_tod.as_slice().unwrap(),
            &psd2,
            &ptr_mat,
        );

        let diff = lp2 - lp1;
        let (gx, gp) = ln_likelihood_grad(
            answer.as_slice().unwrap(),
            total_tod.as_slice().unwrap(),
            &psd,
            &ptr_mat,
        );
        let diff2 = ArrayView1::from(&gx).dot(&dx) + ArrayView1::from(&gp).dot(&dp);
        println!("{} {} {}", diff, diff2, (diff2 - diff).abs());
    }
}
