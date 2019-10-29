use clap::{App, Arg};
use linear_solver::io::RawMM;
use ndarray::{Array2};
use map_solver::MappingProblem;
use fitsimg::write_img;

pub fn main(){
    let matches=App::new("solve")
    .arg(Arg::with_name("solution")
        .short("s")
        .long("sol")
        .takes_value(true)
        .value_name("solution")
        .required(true)
        .help("solved value for each pix")
    )
    .arg(Arg::with_name("pix_idx")
        .short("x")
        .long("pixels")
        .takes_value(true)
        .value_name("pixel indices")
        .required(true)
        .help("The pixels to be solved, should be supplied as a Nx2 matrix file, with the number of rows equals to the number of columns of the pointing matrix")
    )
    .arg(Arg::with_name("output")
        .short("o")
        .long("out")
        .takes_value(true)
        .value_name("outfile")
        .required(true)
        .help("output fits file name")
    )
    .get_matches();

    let x=RawMM::<f64>::from_file(matches.value_of("solution").unwrap()).to_array1();

    let pix_idx=RawMM::<isize>::from_file(matches.value_of("pix_idx").unwrap()).to_array2();

    let ilist=pix_idx.column(0);
    let jlist=pix_idx.column(1);

    let i_max=*pix_idx.column(0).iter().max().unwrap();
    let i_min=*pix_idx.column(0).iter().min().unwrap();
    let j_max=*pix_idx.column(1).iter().max().unwrap();
    let j_min=*pix_idx.column(1).iter().min().unwrap();

    let mut image=Array2::<f64>::zeros([(i_max-i_min+1) as usize, (j_max-j_min+1) as usize]);
    
    for (&v, (&i, &j)) in x.iter().zip(ilist.iter().zip(jlist.iter())){
        image[((i_max-i) as usize, (j-j_min) as usize)]=v;
    }

    write_img(matches.value_of("output").unwrap().to_string(), &image.into_dyn());
}