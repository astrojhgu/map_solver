use map_solver::deconv;
use num_complex::Complex64;
use clap::{App, Arg};
use fitsimg::write_img;
use ndarray::{ArrayView1, Array1, Array2};
use linear_solver::io::RawMM;
use linear_solver::utils::sp_mul_a1;
use linear_solver::minres::agmres::AGmresState;
use linear_solver::minres::gmres::GmresState;
use linear_solver::bicgstab::BiCGStabState;

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
        .arg(Arg::with_name("noise spectrum")
            .short("n")
            .long("noise")
            .value_name("noise spectrum")
            .takes_value(true)
            .help("noise spectrum, length must be 1/2 of tod data")
            .required(true)
        ).get_matches();
        //.arg(Arg::with_name("noise spectrum"))


    let scan=RawMM::<f64>::from_file(matches.value_of("pointing matrix").unwrap()).to_sparse();
    let tod=RawMM::<f64>::from_file(matches.value_of("tod data").unwrap()).to_array1();
    
    let noise=RawMM::<f64>::from_file(matches.value_of("noise spectrum").unwrap()).to_array1();

    let mut rfft=chfft::RFft1D::<f64>::new(noise.len());
    let fnoise=rfft.forward(noise.as_slice().unwrap());

    //let noise1=deconv(tod.as_slice().unwrap(), noise.as_slice().unwrap());

    println!("{:?}", tod.shape());
    let ata=&scan.transpose_view()*&scan;
    
    let b=sp_mul_a1(&scan.transpose_view(), Array1::from_vec(deconv(tod.as_slice().unwrap(), fnoise.as_slice())).view());

    let A = |x: ArrayView1<f64>| -> Array1<f64> {
        //a.dot(&x.to_owned())
        //sp_mul_a1(&ata, x)
        sp_mul_a1(&scan.transpose_view(), Array1::from_vec(deconv(sp_mul_a1(&scan, x).as_slice().unwrap(), fnoise.as_slice())).view())
    };
    
    let M = |x: ArrayView1<f64>| -> Array1<f64> { x.to_owned() };

    let x=Array1::<f64>::zeros(b.len());

    let tol=1e-12;
    let mut ags = AGmresState::<f64>::new(&A, x.view(), b.view(), &M, 30, 1, 1, 0.4, tol);
    //let mut ags = GmresState::<f64>::new(&A, x.view(), b.view(), &M, 30, tol);
    //let mut ags=BiCGStabState::new(&A, x.view(), b.view(), 1e-25);

    let mut cnt = 0;
    //while !ags.converged {
    loop{
        cnt += 1;
        if cnt % 1 == 0 {
            println!("{}", ags.resid);
            //println!("{}", delta);
        }
        if ags.converged{
            break;
        }
        ags.next(&A, &M);
        //ags.next(&A);
    }

    RawMM::from_array1(ags.x.view()).to_file(matches.value_of("output").unwrap());

}
