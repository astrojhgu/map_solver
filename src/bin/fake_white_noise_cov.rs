use clap::{App, Arg};
use linear_solver::io::RawMM;
use ndarray::Array1;

pub fn main() {
    let matches = App::new("generate fourier base functions")
        .arg(
            Arg::with_name("tod")
                .short("t")
                .long("tod")
                .required(true)
                .takes_value(true)
                .value_name("tod")
                .help("tod file"),
        )
        .arg(
            Arg::with_name("sigma")
                .short("s")
                .long("sigma")
                .required(true)
                .takes_value(true)
                .value_name("sigma")
                .help("sigma"),
        )
        .arg(
            Arg::with_name("output")
                .short("o")
                .long("out")
                .required(true)
                .takes_value(true)
                .value_name("out name")
                .help("output file name"),
        )
        .get_matches();

    let tod = RawMM::<f64>::from_file(matches.value_of("tod").unwrap()).to_array1();

    let sigma = matches.value_of("sigma").unwrap().parse::<f64>().unwrap();

    let nt = tod.len();

    let result: Array1<_> = (0..nt).map(|_| sigma.powi(2)).collect();

    RawMM::from_array1(result.view()).to_file(matches.value_of("output").unwrap());
}
