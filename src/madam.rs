use linear_solver::minres::agmres::AGmresState;
use linear_solver::utils::{sp_mul_a1, sp_mul_a2};
use ndarray::{Array1, Array2, ArrayView1};
use sprs::CsMat;
use std::iter::FromIterator;

use crate::utils::{circmat_x_mat, csmat2diag, diag2csmat};

pub struct MappingProblem {
    pub ptr_mat: CsMat<f64>,
    pub cm_white: Array1<f64>,
    pub cm_alpha: Array2<f64>,
    pub F: Array2<f64>,
    pub tod: Array1<f64>,
}

impl MappingProblem {
    pub fn new(
        ptr_mat: CsMat<f64>,
        cm_white: Array1<f64>,
        cm_pink: Array1<f64>,
        F: Array2<f64>,
        tod: Array1<f64>,
    ) -> MappingProblem {
        //circmat_x_mat(cm_pink.as_slice().unwrap(), F.view());
        let ftf_inv = Array2::from_diag(&Array1::from_iter(
            F.t().dot(&F).diag().iter().map(|&x| 1.0 / x),
        ));

        let ftf_inv_ft = ftf_inv.dot(&F.t());

        let ca = ftf_inv_ft.dot(&circmat_x_mat(cm_pink.as_slice().unwrap(), ftf_inv_ft.t()));
        let ca = (&ca + &ca.t()) / 2.0; //should be symm, otherwise, it is caused numeric round off err

        //println!("{:?}", ca);
        MappingProblem {
            ptr_mat,
            cm_white,
            cm_alpha: ca,
            F,
            tod,
        }
    }

    pub fn calc_Z(&self) -> CsMat<f64> {
        let cm_white_inv_diag: Vec<_> = self.cm_white.iter().map(|&x| 1.0 / x).collect();
        let cm_white_inv = diag2csmat(&cm_white_inv_diag);
        //&self.ptr_mat.transpose_view()*&cm_white_inv;
        let q = &self.ptr_mat.transpose_view() * &(&cm_white_inv * &self.ptr_mat);
        let q = diag2csmat(
            &csmat2diag(&q)
                .iter()
                .map(|x| {
                    let b = 1.0 / x;
                    assert!(b.is_finite());
                    b
                })
                .collect::<Vec<f64>>(),
        );
        let ppcppc = &(&self.ptr_mat * &(&q * &self.ptr_mat.transpose_view())) * &cm_white_inv;
        let I = CsMat::eye(self.ptr_mat.rows());
        &I - &ppcppc
    }

    pub fn gen_a_equation(&self) -> (Array2<f64>, Array1<f64>) {
        let z = self.calc_Z();
        let cm_white_inv_diag: Vec<_> = self.cm_white.iter().map(|&x| 1.0 / x).collect();
        let cm_white_inv = diag2csmat(&cm_white_inv_diag);
        let cm_inv_z = &cm_white_inv * &z;
        let cm_inv_z_y = sp_mul_a1(&cm_inv_z, self.tod.view());
        let lhs = self
            .cm_alpha
            .dot(&self.F.t().dot(&sp_mul_a2(&cm_inv_z, self.F.view())))
            + Array2::<f64>::eye(self.F.ncols());
        let rhs = self.cm_alpha.dot(&self.F.t().dot(&cm_inv_z_y));
        (lhs, rhs)
    }

    pub fn solve_a(&self, m_max: usize) -> Array1<f64> {
        let (a, b) = self.gen_a_equation();
        let A = |x: ArrayView1<f64>| -> Array1<f64> {
            //a.dot(&x.to_owned())
            a.dot(&x)
        };

        let x = Array1::<f64>::zeros(b.len());
        //println!("{:?}", a);
        //println!("{:?}", b);

        let mut ags = AGmresState::new(&A, x.view(), b.view(), None, m_max, 1, 1, 0.4, 1e-6);
        let mut cnt = 0;
        while !ags.converged {
            cnt += 1;
            if cnt % 100 == 0 {
                println!("{}", ags.resid);
            }
            ags.next(&A, None);
        }
        ags.x
    }

    pub fn apply_ptr_mat(&self, x: ArrayView1<f64>) -> Array1<f64> {
        sp_mul_a1(&self.ptr_mat, x)
    }

    pub fn gen_sky_equation(&self, m_max_a: usize) -> (CsMat<f64>, Array1<f64>) {
        let cm_white_inv_diag: Vec<_> = self.cm_white.iter().map(|&x| 1.0 / x).collect();
        let cm_white_inv = diag2csmat(&cm_white_inv_diag);
        let a = self.solve_a(m_max_a);
        let lhs = &self.ptr_mat.transpose_view() * &(&cm_white_inv * &self.ptr_mat);
        //let rhs=&self.ptr_mat.transpose_view()*&(&cm_white_inv*&(&self.tod-self.F.dot(&a)));
        let y_fa = &self.tod - &self.F.dot(&a);
        let rhs = sp_mul_a1(
            &(&self.ptr_mat.transpose_view() * &cm_white_inv),
            y_fa.view(),
        );
        (lhs, rhs)
    }

    pub fn solve_sky(&self, m_max_a: usize, m_max_sky: usize) -> Array1<f64> {
        let (a, b) = self.gen_sky_equation(m_max_a);
        let A = |x: ArrayView1<f64>| -> Array1<f64> {
            //a.dot(&x.to_owned())
            sp_mul_a1(&a, x)
        };

        let x = Array1::<f64>::zeros(b.len());
        let mut ags = AGmresState::new(&A, x.view(), b.view(), None, m_max_sky, 1, 1, 0.4, 1e-6);
        let mut cnt = 0;
        while !ags.converged {
            cnt += 1;
            if cnt % 100 == 0 {
                println!("{}", ags.resid);
            }
            ags.next(&A, None);
        }
        ags.x
    }
}
