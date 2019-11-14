#![allow(unused_imports)]
#![allow(non_snake_case)]
extern crate linear_solver;
extern crate map_solver;
use linear_solver::io::RawMM;
use map_solver::madam::MappingProblem;
use ndarray::array;
fn main() {
    let a = array![
        [1., 2., 3., 4., 5., 6., 7., 8.],
        [2., 3., 4., 5., 6., 7., 8., 1.],
        [3., 4., 5., 6., 7., 8., 1., 2.],
        [4., 5., 6., 7., 8., 1., 2., 3.],
        [5., 6., 7., 8., 1., 2., 3., 4.],
        [6., 7., 8., 1., 2., 3., 4., 5.],
        [7., 8., 1., 2., 3., 4., 5., 6.],
        [8., 1., 2., 3., 4., 5., 6., 7.],
    ];

    println!("{:?}", a);

    let b = map_solver::utils::rfft2(a.view());
    println!("{:?}", b);

    let c = map_solver::utils::irfft2(b.view());
    println!("{:?}", c);
}
