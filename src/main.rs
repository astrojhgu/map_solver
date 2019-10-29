#![allow(non_snake_case)]
extern crate map_solver;
extern crate linear_solver;


use linear_solver::io::RawMM;
use map_solver::madam::MappingProblem;
fn main() {
    let ptr_mat=RawMM::<f64>::from_file("A_no_conv.mtx").to_sparse();
    let cm_white=RawMM::<f64>::from_file("nnt.mtx").to_array1();
    let cm_pink=RawMM::<f64>::from_file("cm_pink.mtx").to_array1();
    let F=RawMM::<f64>::from_file("F.mtx").to_array2();

    let tod=RawMM::<f64>::from_file("tod.mtx").to_array1();
    let mp=MappingProblem::new(ptr_mat, cm_white, cm_pink, F, tod);
    //let a=mp.solve_a(3);
    //println!("{:?}", a);
    let x=mp.solve_sky(3, 30);
    RawMM::from_array1(x.view()).to_file("solution.mtx");
}
    
