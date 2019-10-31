use crate::utils::deconv;
use linear_solver::minres::agmres::AGmresState;
use linear_solver::utils::sp_mul_a1;
use ndarray::{Array1, ArrayView1};
use sprs::CsMat;

pub struct MappingProblem {
    pub ptr_mat: CsMat<f64>,
    pub cov_mat: Array1<f64>, //assuming the noise is stationary
    pub tod: Array1<f64>,
}

impl MappingProblem {
    pub fn new(ptr_mat: CsMat<f64>, cov_mat: Array1<f64>, tod: Array1<f64>) -> MappingProblem {
        //circmat_x_mat(cm_pink.as_slice().unwrap(), F.view());
        MappingProblem {
            ptr_mat,
            cov_mat,
            tod,
        }
    }

    pub fn apply_ptr_mat(&self, x: ArrayView1<f64>) -> Array1<f64> {
        sp_mul_a1(&self.ptr_mat, x)
    }

    pub fn solve_sky(&self, m_max: usize) -> Array1<f64> {
        let mut rfft = chfft::RFft1D::<f64>::new(self.cov_mat.len());
        let fnoise = rfft.forward(self.cov_mat.as_slice().unwrap());

        let A = |x: ArrayView1<f64>| -> Array1<f64> {
            sp_mul_a1(
                &self.ptr_mat.transpose_view(),
                Array1::from(deconv(
                    sp_mul_a1(&self.ptr_mat, x).as_slice().unwrap(),
                    fnoise.as_slice(),
                ))
                .view(),
            )
        };

        let M = |x: ArrayView1<f64>| -> Array1<f64> { x.to_owned() };

        let b = sp_mul_a1(
            &self.ptr_mat.transpose_view(),
            Array1::from(deconv(self.tod.as_slice().unwrap(), fnoise.as_slice())).view(),
        );

        let x = Array1::<f64>::zeros(b.len());
        let tol = 1e-15;
        let mut ags = AGmresState::<f64>::new(&A, x.view(), b.view(), &M, m_max, 1, 1, 0.4, tol);

        let mut cnt = 0;
        //while !ags.converged {
        loop {
            cnt += 1;
            if cnt % 10 == 0 {
                println!("{}", ags.resid);
                //println!("{}", delta);
            }
            if ags.converged {
                break;
            }
            ags.next(&A, &M);
            //ags.next(&A);
        }

        ags.x
    }
}
