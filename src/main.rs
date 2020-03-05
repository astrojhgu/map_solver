#![allow(unused_imports)]
#![allow(non_snake_case)]
extern crate linear_solver;
extern crate map_solver;
use linear_solver::io::RawMM;
use map_solver::madam::MappingProblem;
use map_solver::utils::ps_model;
use ndarray::array;
fn main() {
    let n=16;
    for (i, f) in (0..(n+1)/2).chain(-n/2..0).enumerate(){
        println!("{} {}", i, ps_model(f, 3.0_f64.sqrt(), 2., 2.0, -1.0, 0.0001, 0.0001));
    }

    
}
