use linear_solver::minres::agmres::AGmresState;
use linear_solver::utils::sp_mul_a1;
use ndarray::{Array1, ArrayView1};
use sprs::CsMat;

pub struct MappingProblem {
    pub ptr_mat: CsMat<f64>,
    pub tod: Array1<f64>,
    pub tol: f64,
    pub m_max: usize,
}

impl MappingProblem {
    pub fn new(ptr_mat: CsMat<f64>, tod: Array1<f64>) -> MappingProblem {
        //circmat_x_mat(cm_pink.as_slice().unwrap(), F.view());
        MappingProblem {
            ptr_mat,
            tod,
            tol: 1e-12,
            m_max: 30,
        }
    }

    pub fn with_tol(mut self, tol: f64) -> MappingProblem {
        self.tol = tol;
        self
    }

    pub fn with_m_max(mut self, m_max: usize) -> MappingProblem {
        self.m_max = m_max;
        self
    }

    pub fn apply_ptr_mat(&self, x: ArrayView1<f64>) -> Array1<f64> {
        sp_mul_a1(&self.ptr_mat, x)
    }

    pub fn solve_sky(&self) -> Array1<f64> {
        let ata = &self.ptr_mat.transpose_view() * &self.ptr_mat;
        let A = |x: ArrayView1<f64>| -> Array1<f64> { sp_mul_a1(&ata, x) };

        let b = sp_mul_a1(&self.ptr_mat.transpose_view(), self.tod.view());

        let x = Array1::<f64>::zeros(b.len());
        let tol = self.tol;
        let m_max = self.m_max;
        let mut ags =
            AGmresState::<f64, f64>::new(&A, x.view(), b.view(), None, m_max, 1, 1, 0.4, tol);

        let mut cnt = 0;
        //while !ags.converged {
        loop {
            cnt += 1;
            if cnt % 10 == 0 {
                println!("a={}", ags.resid);
                //println!("{}", delta);
            }
            if ags.converged {
                break;
            }
            ags.next(&A, None);
            //ags.next(&A);
        }

        ags.x
    }
}
