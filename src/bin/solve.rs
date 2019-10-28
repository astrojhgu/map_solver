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
        .arg(Arg::with_name("pix_idx")
            .short("x")
            .long("pixels")
            .takes_value(true)
            .value_name("pixel indices")
            .required(true)
            .help("The pixels to be solved, should be supplied as a Nx2 matrix file, with the number of rows equals to the number of columns of the pointing matrix")
        ).get_matches();
    let tod=RawMM::<f64>::from_file(matches.value_of("tod").unwrap()).to_array1();
    let ptr_mat=RawMM::<f64>::from_file(matches.value_of("ptr_matrix").unwrap()).to_sparse();
    let cov_white=RawMM::<f64>::from_file(matches.value_of("nnt").unwrap()).to_array1();
    let cov_coeff_pink=RawMM::<f64>::from_file(matches.value_of("pink_coeff").unwrap()).to_array1();
    let base_func=RawMM::<f64>::from_file(matches.value_of("F").unwrap()).to_array2();
    let pix_idx=RawMM::<isize>::from_file(matches.value_of("pix_idx").unwrap()).to_array2();

    let mp=MappingProblem::new(ptr_mat, cov_white, cov_coeff_pink, base_func, tod);

    let x=mp.solve_sky(3, 30);
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

    write_img("map.fits".to_string(), &image.into_dyn());
}