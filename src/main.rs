#![allow(unused_imports)]
#![allow(non_snake_case)]
extern crate linear_solver;
extern crate map_solver;

use scorus::linear_space::type_wrapper::LsVec;
use linear_solver::io::RawMM;
use map_solver::madam::MappingProblem;
use map_solver::utils::ps_model;
use ndarray::array;
use map_solver::utils::DT;
use map_solver::mcmc::Problem;

fn main() {
    let p=Problem{
        tod: vec![],
        ptr_mat: vec![]
    };

    let b=p.get_logprob();
    let c=p.get_logprob();

    println!("{}", b(&LsVec(vec![1.,2.,3.])));
}
