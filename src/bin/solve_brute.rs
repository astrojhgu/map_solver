#![allow(non_snake_case)]
#![allow(unused_imports)]

use clap::{App, Arg};
use linear_solver::io::RawMM;
use map_solver::brute_mo::MappingProblem as BruteSolver;
use map_solver::naive_mo::MappingProblem as NaiveSolver;
use ndarray::Array1;
fn main() {
    let matches = App::new("solve map making problem with noise model")
        .arg(
            Arg::with_name("pointing matrix")
                .short("p")
                .long("pointing")
                .value_name("pointing_matrix")
                .takes_value(true)
                .help("pointing matrix in matrix market format")
                .required(true),
        )
        .arg(
            Arg::with_name("tod data")
                .short("t")
                .long("tod")
                .value_name("tod data")
                .takes_value(true)
                .help("tod data")
                .required(true),
        )
        .arg(
            Arg::with_name("output")
                .short("o")
                .long("out")
                .takes_value(true)
                .value_name("outfile")
                .required(true)
                .help("output file name"),
        )
        .arg(
            Arg::with_name("noise covariance matrix")
                .short("n")
                .long("noise")
                .value_name("noise covariance matrix")
                .takes_value(true)
                .help("noise covariance matrix")
                .required(true),
        )
        .arg(
            Arg::with_name("output resid")
                .short("r")
                .long("resid")
                .takes_value(true)
                .value_name("resid file")
                .required(false)
                .help("output resid"),
        )
        .arg(
            Arg::with_name("tol")
                .short("l")
                .long("tol")
                .takes_value(true)
                .required(false)
                .value_name("TOL")
                .help("tol, default value: 1e-12"),
        )
        .arg(
            Arg::with_name("m_max")
                .short("a")
                .long("mmax")
                .takes_value(true)
                .required(false)
                .value_name("m_max")
                .help("m_max param for the solver"),
        )
        .arg(
            Arg::with_name("init")
                .short("i")
                .long("init")
                .takes_value(true)
                .required(false)
                .value_name("initial guess")
                .help("initial guess"),
        )
        .get_matches();
    //.arg(Arg::with_name("noise spectrum"))

    let scan = RawMM::<f64>::from_file(matches.value_of("pointing matrix").unwrap()).to_sparse();
    let tod = RawMM::<f64>::from_file(matches.value_of("tod data").unwrap()).to_array1();

    let corr_noise =
        RawMM::<f64>::from_file(matches.value_of("noise covariance matrix").unwrap()).to_array1();

    let tol = matches
        .value_of("tol")
        .or(Some("1e-15"))
        .unwrap()
        .parse::<f64>()
        .unwrap();
    let m_max = matches
        .value_of("m_max")
        .or(Some("50"))
        .unwrap()
        .parse::<usize>()
        .unwrap();

    let x = if matches.is_present("init") {
        RawMM::<f64>::from_file(matches.value_of("init").unwrap()).to_array1()
    } else {
        let mp = NaiveSolver::new(vec![scan.clone()], vec![tod.clone()])
            .with_tol(tol)
            .with_m_max(m_max);
        mp.solve_sky()
    };

    let mp = BruteSolver::new(vec![scan], vec![corr_noise], vec![tod])
        .with_tol(tol)
        .with_m_max(m_max)
        .with_init_value(x);
    let x = mp.solve_sky(
        10000,
        Some(&mut |x| {
            RawMM::from_array1(x.view()).to_file(matches.value_of("output").unwrap());
        }),
    );

    RawMM::from_array1(x.view()).to_file(matches.value_of("output").unwrap());

    if matches.is_present("output resid") {
        let resid = &mp.tod[0] - &mp.apply_ptr_mat(x.view());
        RawMM::from_array1(resid.view()).to_file(matches.value_of("output resid").unwrap());
    }
}
