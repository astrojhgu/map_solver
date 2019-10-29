use clap::{App, Arg};
use linear_solver::io::RawMM;
use ndarray::{Array2};
use map_solver::MappingProblem;
use fitsimg::write_img;

pub fn main(){
    let matches=App::new("solve")
        .arg(Arg::with_name("tod")
            .short("t")
            .long("tod")
            .takes_value(true)
            .value_name("tod")
            .required(true)
            .help("tod data")
        )
        .arg(Arg::with_name("ptr_matrix")
            .short("p")
            .long("pm")
            .takes_value(true)
            .value_name("pointing matrix")
            .required(true)
            .help("pointing matrix")
        )
        .arg(Arg::with_name("nnt")
            .short("n")
            .long("nnt")
            .takes_value(true)
            .value_name("nnt")
            .required(true)
            .help("nnt of the white part of noise, only the diagonal part is supplied as a Nx1 matrix")
        )
        .arg(Arg::with_name("pink_coeff")
            .short("k")
            .long("pc")
            .takes_value(true)
            .value_name("coeff file")
            .required(true)
            .help("the cov of the pink part of the noise is supplied in the form of the circulant matrix coefficient, i.e., the first row of the matrix, under the assumption that the noise is stationary")
        )
        .arg(Arg::with_name("F")
            .short("F")
            .long("bf")
            .takes_value(true)
            .value_name("base function file")
            .required(true)
            .help("the base functions are supplied as a matrix file, each column of which is the base function in time domain")
        )
        .arg(Arg::with_name("output")
        .short("o")
        .long("out")
        .takes_value(true)
        .value_name("outfile")
        .required(true)
        .help("output fits file name")
        ).get_matches();


    let tod=RawMM::<f64>::from_file(matches.value_of("tod").unwrap()).to_array1();
    let ptr_mat=RawMM::<f64>::from_file(matches.value_of("ptr_matrix").unwrap()).to_sparse();
    let cov_white=RawMM::<f64>::from_file(matches.value_of("nnt").unwrap()).to_array1();
    let cov_coeff_pink=RawMM::<f64>::from_file(matches.value_of("pink_coeff").unwrap()).to_array1();
    let base_func=RawMM::<f64>::from_file(matches.value_of("F").unwrap()).to_array2();
    
    let mp=MappingProblem::new(ptr_mat, cov_white, cov_coeff_pink, base_func, tod);

    let x=mp.solve_sky(3, 30);
    
    RawMM::from_array1(x.view()).to_file(matches.value_of("output").unwrap());
}