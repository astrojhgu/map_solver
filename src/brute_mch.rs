#![allow(unused_imports)]
#![allow(clippy::modulo_one)]

use crate::utils::deconv2;
use crate::utils::rfft2;
use linear_solver::bicgstab::BiCGStabState;
use linear_solver::minres::agmres::AGmresState;
use linear_solver::minres::gmres::GmresState;
use linear_solver::utils::sp_mul_a1;
use ndarray::ShapeBuilder;
use ndarray::{s, Array1, Array2, ArrayView1};
use sprs::CsMat;

pub struct MappingProblem {
    pub ptr_mat: Vec<CsMat<f64>>,
    pub noise_cov: Array2<f64>, //assuming the noise is stationary
    pub tod: Array2<f64>,       //[nch, ntime]
    pub tol: f64,
    pub m_max: usize,
    pub x: Option<Array2<f64>>,
}

impl MappingProblem {
    pub fn new(
        ptr_mat: Vec<CsMat<f64>>,
        noise_cov: Array2<f64>,
        tod: Array2<f64>,
    ) -> MappingProblem {
        //circmat_x_mat(cm_pink.as_slice().unwrap(), F.view());
        MappingProblem {
            ptr_mat,
            noise_cov,
            tod,
            tol: 1e-12,
            m_max: 50,
            x: None,
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

    pub fn with_init_value(mut self, x: Array2<f64>) -> MappingProblem {
        self.x = Some(x);
        self
    }

    pub fn apply_ptr_mat(&self, x: ArrayView1<f64>) -> Array1<f64> {
        //sp_mul_a1(&self.ptr_mat, x)
        let mut result = Array1::zeros(self.ptr_mat.iter().map(|x| x.rows()).sum::<usize>());

        let mut rcnt = 0;
        let mut ccnt = 0;
        for p in self.ptr_mat.iter() {
            result
                .slice_mut(s![rcnt..rcnt + p.rows()])
                .assign(&sp_mul_a1(p, x.slice(s![ccnt..ccnt + p.cols()]).view()));
            rcnt += p.rows();
            ccnt += p.cols();
        }
        result
    }

    pub fn apply_ptr_mat_t(&self, x: ArrayView1<f64>) -> Array1<f64> {
        //sp_mul_a1(&self.ptr_mat, x)
        let mut result = Array1::zeros(self.ptr_mat.iter().map(|x| x.cols()).sum::<usize>());

        let mut rcnt = 0;
        let mut ccnt = 0;
        for p in self.ptr_mat.iter() {
            result
                .slice_mut(s![ccnt..ccnt + p.cols()])
                .assign(&sp_mul_a1(
                    &p.transpose_view(),
                    x.slice(s![rcnt..rcnt + p.rows()]).view(),
                ));
            rcnt += p.rows();
            ccnt += p.cols();
        }
        result
    }

    pub fn solve_sky(&self, mut cb: Option<&mut dyn FnMut(&Array1<f64>)>) -> Array1<f64> {
        //let mut rfft = chfft::RFft1D::<f64>::new(self.noise_cov.len());
        //let fnoise = rfft.forward(self.noise_cov.as_slice().unwrap());
        let fnoise = rfft2(self.noise_cov.view());
        let ntime = self.tod.ncols();
        let nch = self.tod.nrows();

        let A: Box<dyn Fn(ArrayView1<f64>) -> Array1<f64>> =
            Box::new(|x: ArrayView1<f64>| -> Array1<f64> {
                self.apply_ptr_mat_t(
                    deconv2(
                        self.apply_ptr_mat(x)
                            .into_shape((nch, ntime))
                            .unwrap()
                            .view(),
                        fnoise.view(),
                    )
                    .into_shape(ntime * nch)
                    .unwrap()
                    .view(),
                )
            });

        //let M = |x: ArrayView1<f64>| -> Array1<f64> { x.to_owned() };

        let b = self.apply_ptr_mat_t(
            deconv2(self.tod.view(), fnoise.view())
                .into_shape((ntime * nch,))
                .unwrap()
                .view(),
        );

        let x = if let Some(ref x) = self.x {
            x.clone().into_shape((ntime * nch,)).unwrap()
        } else {
            Array1::<f64>::zeros(b.len())
        };
        let tol = self.tol;
        let m_max = self.m_max;
        let mut ags =
            AGmresState::<f64, f64>::new(&A, x.view(), b.view(), None, m_max, 1, 1, 0.4, tol);

        //let mut ags = GmresState::<f64>::new(&A, x.view(), b.view(), &M, m_max, tol);

        //let mut ags=BiCGStabState::new(&A, x.view(), b.view(), tol);
        let mut cnt = 0;
        //while !ags.converged {
        loop {
            cnt += 1;
            if cnt % 1 == 0 {
                println!("b={}", ags.resid);
                println!(
                    "{}",
                    ags.calc_resid(&A, &b)
                        .iter()
                        .map(|&x| { x.powi(2) })
                        .sum::<f64>()
                );
                //println!("{}", delta);
            }
            if ags.converged {
                break;
            }
            ags.next(&A, None);
            if let Some(ref mut f) = cb {
                f(&ags.x);
            }
            //ags.next(&A);
            //ags.next(&A);
        }

        let resid = &b - &(A(ags.x.view()));
        eprintln!(
            "resid = {:?}",
            resid.iter().map(|&x| { x.powi(2) }).sum::<f64>().sqrt()
        );
        ags.x
    }
}
