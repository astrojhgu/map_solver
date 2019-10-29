#![allow(non_snake_case)]

use map_solver::utils::deconv;
use clap::{App, Arg};
use ndarray::{ArrayView1, Array1};
use linear_solver::io::RawMM;
use linear_solver::utils::sp_mul_a1;
use linear_solver::minres::agmres::AGmresState;
use map_solver::brute::MappingProblem;

fn main(){
    let matches=App::new("solve map making problem with noise model")
        .arg(Arg::with_name("pointing matrix")
            .short("p")
            .long("pointing")
            .value_name("pointing_matrix")
            .takes_value(true)
            .help("pointing matrix in matrix market format")
            .required(true)
        )
        .arg(Arg::with_name("tod data")
            .short("t")
            .long("tod")
            .value_name("tod data")
            .takes_value(true)
            .help("tod data")
            .required(true)
        )
        .arg(Arg::with_name("output")
            .short("o")
            .long("out")
            .takes_value(true)
            .value_name("outfile")
            .required(true)
            .help("output file name")
        )
        .arg(Arg::with_name("noise covariance matrix")
            .short("n")
            .long("noise")
            .value_name("noise covariance matrix")
            .takes_value(true)
            .help("noise covariance matrix")
            .required(true)
        ).get_matches();
        //.arg(Arg::with_name("noise spectrum"))


    let scan=RawMM::<f64>::from_file(matches.value_of("pointing matrix").unwrap()).to_sparse();
    let tod=RawMM::<f64>::from_file(matches.value_of("tod data").unwrap()).to_array1();
    
    let noise=RawMM::<f64>::from_file(matches.value_of("noise covariance matrix").unwrap()).to_array1();

    let mp=MappingProblem::new(scan, noise, tod);
    let x=mp.solve_sky(30);

    RawMM::from_array1(x.view()).to_file(matches.value_of("output").unwrap());

}
