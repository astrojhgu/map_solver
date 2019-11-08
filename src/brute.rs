use crate::utils::deconv;
use linear_solver::minres::agmres::AGmresState;
use linear_solver::minres::gmres::GmresState;
use linear_solver::bicgstab::BiCGStabState;
use linear_solver::utils::sp_mul_a1;
use ndarray::{Array1, ArrayView1};
use sprs::CsMat;

pub struct MappingProblem {
    pub ptr_mat: CsMat<f64>,
    pub cov_mat: Array1<f64>, //assuming the noise is stationary
    pub tod: Array1<f64>,
    pub tol: f64,
    pub m_max: usize, 
    pub x: Option<Array1<f64>>, 
}

impl MappingProblem {
    pub fn new(ptr_mat: CsMat<f64>, cov_mat: Array1<f64>, tod: Array1<f64>) -> MappingProblem {
        //circmat_x_mat(cm_pink.as_slice().unwrap(), F.view());
        MappingProblem {
            ptr_mat,
            cov_mat,
            tod,
            tol: 1e-12,
            m_max: 50,
            x: None
        }
    }

    pub fn with_tol(mut self, tol: f64)->MappingProblem{
        self.tol=tol;
        self
    }

    pub fn with_m_max(mut self, m_max: usize)-> MappingProblem{
        self.m_max=m_max;
        self
    }

    pub fn with_init_value(mut self, x: Array1<f64>)->MappingProblem{
        self.x=Some(x);
        self
    }

    pub fn apply_ptr_mat(&self, x: ArrayView1<f64>) -> Array1<f64> {
        sp_mul_a1(&self.ptr_mat, x)
    }

    pub fn solve_sky(&self, mut cb: Option<&mut dyn FnMut(&Array1<f64>)>) -> Array1<f64> {
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

        //let M = |x: ArrayView1<f64>| -> Array1<f64> { x.to_owned() };

        let b = sp_mul_a1(
            &self.ptr_mat.transpose_view(),
            Array1::from(deconv(self.tod.as_slice().unwrap(), fnoise.as_slice())).view(),
        );

        let x = if let Some(ref x)=self.x{
            x.clone()
        }else{
            Array1::<f64>::zeros(b.len())
        };
        let tol = self.tol;
        let m_max=self.m_max;
        let mut ags = AGmresState::<f64>::new(&A, x.view(), b.view(), None, m_max, 1, 1, 0.4, tol);

        //let mut ags = GmresState::<f64>::new(&A, x.view(), b.view(), &M, m_max, tol);

        //let mut ags=BiCGStabState::new(&A, x.view(), b.view(), tol);
        let mut cnt = 0;
        //while !ags.converged {
        loop {
            cnt += 1;
            if cnt % 1 == 0 {
                println!("b={}", ags.resid);
                println!("{}", ags.calc_resid(&A, &b).iter().map(|&x|{x.powi(2)}).sum::<f64>());
                //println!("{}", delta);
            }
            if ags.converged {
                break;
            }
            ags.next(&A, None);
            if let Some(ref mut f)=cb{
                f(&ags.x);
            }
            //ags.next(&A);
            //ags.next(&A);
        }

        let resid=&b-&(A(ags.x.view()));
        eprintln!("resid = {:?}", resid.iter().map(|&x|{x.powi(2)}).sum::<f64>().sqrt());
        ags.x
    }
}
