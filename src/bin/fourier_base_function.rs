use clap::{App, Arg};
use linear_solver::io::RawMM;
use ndarray::Array2;
use num_traits::float::FloatConst;

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
            Arg::with_name("num_of_bf")
                .short("n")
                .long("num")
                .required(true)
                .takes_value(true)
                .value_name("num")
                .help("number of base functions"),
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
    let n = matches
        .value_of("num_of_bf")
        .unwrap()
        .parse::<usize>()
        .unwrap();

    let nt = tod.len();

    let mut result = Array2::<f64>::zeros((nt, n));

    for i in 0..nt {
        let t = (i as f64 / nt as f64) * 2.0 * f64::PI();
        for j in 0..n {
            let k = (j + 1) / 2 + 1;
            let y = if j % 2 == 0 {
                (t * k as f64).sin()
            } else {
                (t * k as f64).cos()
            };
            result[(i, j)] = y;
        }
    }

    RawMM::from_array2(result.view()).to_file(matches.value_of("output").unwrap());
}
