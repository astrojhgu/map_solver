extern crate map_solver;

use rand::thread_rng;

use scorus::linear_space::type_wrapper::LsVec;
use scorus::mcmc::hmc::naive::sample;
use scorus::mcmc::hmc::naive::HmcParam;

use ndarray::Array1;

use linear_solver::io::RawMM;
use map_solver::mcmc::Problem;
use map_solver::noise::gen_noise;

const L: usize = 1;
const NSTEPS: usize = 5;

fn main() {
    let mut rng = thread_rng();

    let ptr_mat = RawMM::<f64>::from_file("ptr_mat.mtx").to_sparse();
    let tod = RawMM::<f64>::from_file("cheat_vis.mtx").to_array1();

    let answer = Array1::zeros(ptr_mat.cols());

    let ntod = ptr_mat.rows();
    let nx = ptr_mat.cols();
    //let answer=vec![0.0; answer.len()];
    let psp = vec![0.1, 0.1, 1.0 / 30.0, 0.0];
    let x: Vec<_> = answer.iter().chain(psp.iter()).cloned().collect();
    let mut q = LsVec(x);

    let psp = vec![20.0, 2.0, 0.001, -1.0];
    let mut problem = Problem::empty();

    for _i in 0..20 {
        let noise = Array1::from(gen_noise(ntod, &psp, &mut rng, map_solver::mcmc_func::DT));
        let total_tod = &tod + &noise;
        problem = problem.with_obs(total_tod.as_slice().unwrap(), &ptr_mat);
    }

    let mut epsilon_p = 0.003;
    let mut epsilon_s = 0.003;
    //let param=HmcParam::quick_adj(0.75);
    let mut param = HmcParam::new(0.75, 0.05);

    for i in 0..1_000_000 {
        if i > 1000 {
            param = HmcParam::slow_adj(0.75);
        }
        let mut accept_cnt_p = 0;
        let mut accept_cnt_s = 0;
        let mut cnt_p = 0;
        let mut cnt_s = 0;

        {
            //sample p
            let sky = q.0.iter().take(nx).cloned().collect::<Vec<_>>();
            let mut q1 = LsVec(q.0.iter().skip(nx).cloned().collect::<Vec<_>>());

            let lp = problem.get_logprob_psp(&q);
            let lp_grad = problem.get_logprob_grad_psp(&q);

            let mut lp_value = lp(&q1);
            let mut lp_grad_value = lp_grad(&q1);

            for _j in 0..NSTEPS {
                let accepted = sample(
                    &lp,
                    &lp_grad,
                    &mut q1,
                    &mut lp_value,
                    &mut lp_grad_value,
                    &mut rng,
                    &mut epsilon_p,
                    L,
                    &param,
                );
                if accepted {
                    accept_cnt_p += 1;
                }
                cnt_p += 1;
            }

            q = LsVec(sky.iter().chain(q1.iter()).cloned().collect::<Vec<_>>());
            if i % 10 == 0 {
                println!(
                    "{} {:.3} {:.8} {:.5}  {:?}",
                    i,
                    accept_cnt_p as f64 / cnt_p as f64,
                    epsilon_p,
                    lp_value,
                    q1.0
                );
            }
        }
        {
            let psp = q.0.iter().skip(nx).cloned().collect::<Vec<_>>();
            let mut q1 = LsVec(q.0.iter().take(nx).cloned().collect::<Vec<_>>());

            let lp = problem.get_logprob_sky(&q);
            let lp_grad = problem.get_logprob_grad_sky(&q);

            let mut lp_value = lp(&q1);
            let mut lp_grad_value = lp_grad(&q1);

            for _j in 0..NSTEPS {
                let accepted = sample(
                    &lp,
                    &lp_grad,
                    &mut q1,
                    &mut lp_value,
                    &mut lp_grad_value,
                    &mut rng,
                    &mut epsilon_s,
                    L,
                    &param,
                );
                if accepted {
                    accept_cnt_s += 1;
                }
                cnt_s += 1;
            }
            q = LsVec(q1.iter().chain(psp.iter()).cloned().collect::<Vec<_>>());

            let mean_value = q1.0.iter().sum::<f64>() / nx as f64;
            if i % 10 == 0 {
                println!(
                    "{} {:.3} {:.8} {:.5} {:e}",
                    i,
                    accept_cnt_s as f64 / cnt_s as f64,
                    epsilon_s,
                    lp_value,
                    mean_value
                );
            }
        }
    }
}
