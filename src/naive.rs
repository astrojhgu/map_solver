use linear_solver::minres::agmres::AGmresState;
use std::iter::FromIterator;
use sprs::CsMat;
use linear_solver::utils::{sp_mul_a1, sp_mul_a2};
use ndarray::{Array2, Array1, ArrayView1};
use crate::utils::deconv;

use crate::utils::{circmat_x_mat, diag2csmat, csmat2diag};


pub struct MappingProblem{
    pub ptr_mat: CsMat<f64>,
    pub tod: Array1<f64>,
}


impl MappingProblem{
    pub fn new(ptr_mat: CsMat<f64>, 
    tod: Array1<f64>,
    )->MappingProblem{
        //circmat_x_mat(cm_pink.as_slice().unwrap(), F.view());
        MappingProblem{
            ptr_mat, 
            tod
        }
    }

    pub fn apply_ptr_mat(&self, x: ArrayView1<f64>)->Array1<f64>{
        sp_mul_a1(&self.ptr_mat, x)
    }


    pub fn solve_sky(&self, m_max: usize)-> Array1<f64>{
        let ata=&self.ptr_mat.transpose_view()*&self.ptr_mat;
        let A = |x: ArrayView1<f64>| -> Array1<f64> {
            sp_mul_a1(&ata, x)
        };
    
        let M = |x: ArrayView1<f64>| -> Array1<f64> { x.to_owned() };

        let b=sp_mul_a1(&self.ptr_mat.transpose_view(), self.tod.view());

        let x=Array1::<f64>::zeros(b.len());
        let tol=1e-12;
        let mut ags = AGmresState::<f64>::new(&A, x.view(), b.view(), &M, 30, 1, 1, 0.4, tol);

        let mut cnt = 0;
    //while !ags.converged {
        loop{
            cnt += 1;
            if cnt % 10 == 0 {
                println!("{}", ags.resid);
                //println!("{}", delta);
            }
            if ags.converged{
                break;
            }
            ags.next(&A, &M);
            //ags.next(&A);
        }

        ags.x
    }
}
