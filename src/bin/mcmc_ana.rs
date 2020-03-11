extern crate map_solver;

use rand::{thread_rng, Rng};
use rand_distr::StandardNormal;

use std::fs::File;

use scorus::linear_space::traits::InnerProdSpace;
use scorus::linear_space::type_wrapper::LsVec;
use scorus::mcmc::hmc::naive::sample;
use scorus::mcmc::hmc::naive::HmcParam;

use ndarray::Array1;

use map_solver::mcmc_func::{logprob_ana, logprob_ana_grad, FMAX};

use linear_solver::io::RawMM;

fn main() {
    let mut rng = thread_rng();

    let ptr_mat = RawMM::<f64>::from_file("ptr_mat.mtx").to_sparse();
    let tod = RawMM::<f64>::from_file("cheat_vis.mtx").to_array1();

    let answer = if let Ok(_f) = File::open("solution.mtx") {
        RawMM::<f64>::from_file("solution.mtx").to_array1()
    } else {
        println!("{}", ptr_mat.cols());
        Array1::zeros(ptr_mat.cols())
    };

    let noise: Array1<f64> = tod.map(|_| {
        let f: f64 = rng.sample(StandardNormal);
        f
    });

    let total_tod = &tod + &noise;

    //let ntod=ptr_mat.rows();
    let nx = ptr_mat.cols();

    let lp = |x: &LsVec<f64, Vec<f64>>| {
        let sky: Vec<_> = x.0.iter().take(nx).cloned().collect();
        let pps: Vec<_> = x.0.iter().skip(nx).cloned().collect();
        logprob_ana(&sky, &pps, total_tod.as_slice().unwrap(), &ptr_mat)
    };

    let lp_grad = |x: &LsVec<f64, Vec<f64>>| {
        let sky: Vec<_> = x.0.iter().take(nx).cloned().collect();
        let pps: Vec<_> = x.0.iter().skip(nx).cloned().collect();
        let (gx, gp) = logprob_ana_grad(&sky, &pps, total_tod.as_slice().unwrap(), &ptr_mat);
        LsVec(gx.iter().chain(gp.iter()).cloned().collect::<Vec<_>>())
    };
    //let answer=vec![0.0; answer.len()];
    let pps = vec![0.1, 0.1, FMAX / 10.0, 0.0];
    let x: Vec<_> = answer.iter().chain(pps.iter()).cloned().collect();
    let mut q = LsVec(x);
    let mut lp_value = lp(&q);
    let mut lp_grad_value = lp_grad(&q);

    let dx = LsVec(
        q.0.iter()
            .map(|_| {
                let f: f64 = rng.sample(StandardNormal);
                f * 0.0001
            })
            .collect::<Vec<_>>(),
    );

    let diff = dx.dot(&lp_grad_value);
    let q2 = &q + &dx;
    let lp_value2 = lp(&q2);
    println!("{} {}", diff, lp_value2 - lp_value);

    let mut accept_cnt = 0;
    let mut cnt = 0;
    let mut epsilon = 0.00003;
    //let param=HmcParam::quick_adj(0.75);
    let param = HmcParam::new(0.75, 0.05);
    for i in 0..10000 {
        let accepted = sample(
            &lp,
            &lp_grad,
            &mut q,
            &mut lp_value,
            &mut lp_grad_value,
            &mut rng,
            &mut epsilon,
            20,
            &param,
        );
        if accepted {
            accept_cnt += 1;
        }
        cnt += 1;
        if i % 100 == 0 {
            let g = lp_grad_value.dot(&lp_grad_value);
            let pps: Vec<_> = q.0.iter().skip(nx).collect();
            //let pps_g: Vec<_>=lp_grad_value.0.iter().skip(nx).collect();

            let dx = LsVec(
                q.0.iter()
                    .map(|_| {
                        let f: f64 = rng.sample(StandardNormal);
                        f * 0.00001
                    })
                    .collect::<Vec<_>>(),
            );
            let diff = dx.dot(&lp_grad_value);
            let q2 = &q + &dx;
            let lp_value2 = lp(&q2);
            let mean_value = q.0.iter().take(nx).sum::<f64>() / nx as f64;

            println!(
                "{} {} {:.3} {:.6} {:.5} {:.3} {:?} {:.3} {:.3} {:.3} {:.3} {:e}",
                i,
                if accepted { 1 } else { 0 },
                accept_cnt as f64 / cnt as f64,
                epsilon,
                lp_value,
                g,
                pps,
                diff,
                lp_value2 - lp_value,
                diff / (lp_value2 - lp_value),
                diff.abs() / lp_value.abs(),
                mean_value
            );
        }
    }

    let param = HmcParam::slow_adj(0.75);
    for i in 0..10_000_000 {
        let accepted = sample(
            &lp,
            &lp_grad,
            &mut q,
            &mut lp_value,
            &mut lp_grad_value,
            &mut rng,
            &mut epsilon,
            20,
            &param,
        );
        if accepted {
            accept_cnt += 1;
        }
        cnt += 1;
        if i % 100 == 0 {
            let g = lp_grad_value.dot(&lp_grad_value);
            let pps: Vec<_> = q.0.iter().skip(nx).collect();
            //let pps_g: Vec<_>=lp_grad_value.0.iter().skip(nx).collect();

            let dx = LsVec(
                q.0.iter()
                    .map(|_| {
                        let f: f64 = rng.sample(StandardNormal);
                        f * 0.00001
                    })
                    .collect::<Vec<_>>(),
            );
            let diff = dx.dot(&lp_grad_value);
            let q2 = &q + &dx;
            let lp_value2 = lp(&q2);

            let mean_value = q.0.iter().take(nx).sum::<f64>() / nx as f64;

            println!(
                "{} {} {:.3} {:.6} {:.5} {:.3} {:?} {:.3} {:.3} {:.3} {:.3} {:e}",
                i,
                if accepted { 1 } else { 0 },
                accept_cnt as f64 / cnt as f64,
                epsilon,
                lp_value,
                g,
                pps,
                diff,
                lp_value2 - lp_value,
                diff / (lp_value2 - lp_value),
                diff.abs() / lp_value.abs(),
                mean_value
            );
        }
    }
}
