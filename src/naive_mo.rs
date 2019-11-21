use linear_solver::minres::agmres::AGmresState;
use linear_solver::utils::sp_mul_a1;
use ndarray::{Array1, ArrayView1,s};
use sprs::CsMat;

pub struct MappingProblem {
    pub ptr_mat: Vec<CsMat<f64>>,
    pub tod: Vec<Array1<f64>>,
    pub tol: f64,
    pub m_max: usize,
}

impl MappingProblem {
    pub fn new(ptr_mat: Vec<CsMat<f64>>, tod: Vec<Array1<f64>>) -> MappingProblem {
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
        //sp_mul_a1(&self.ptr_mat, x)
        let mut result = Array1::zeros(self.ptr_mat.iter().map(|x|{
            x.rows()
        }).sum::<usize>());

        assert!(x.len()==self.ptr_mat[0].cols());
        let mut rcnt = 0;
        //let mut ccnt = 0;
        for p in self.ptr_mat.iter() {
            result
                .slice_mut(s![rcnt..rcnt + p.rows()])
                .assign(&sp_mul_a1(p, x.view()));
            rcnt += p.rows();
        }
        result
    }

    pub fn apply_ptr_mat_t(&self, x: ArrayView1<f64>)->Array1<f64>{
        let mut result =  Array1::zeros(self.ptr_mat[0].cols());
        println!("l:{:?}", self.ptr_mat[0].cols());
        let mut ccnt=0;
        for p in self.ptr_mat.iter() {
            result+=&sp_mul_a1(&p.transpose_view(), x.slice(s![ccnt..ccnt+p.rows()]));
            ccnt+=p.rows();
        }
        result
    }

    pub fn solve_sky(&self) -> Array1<f64> {
        let A = |x: ArrayView1<f64>| -> Array1<f64> { 
            self.apply_ptr_mat_t(self.apply_ptr_mat(x).view())};

        let concated_tod=Array1::from(self.tod.iter().map(|x|{x.to_vec()}).flatten().collect::<Vec<f64>>());

        let b=self.apply_ptr_mat_t(concated_tod.view());

        let x = Array1::<f64>::zeros(b.len());
        let tol = self.tol;
        let m_max = self.m_max;
        let mut ags = AGmresState::<f64>::new(&A, x.view(), b.view(), None, m_max, 1, 1, 0.4, tol);

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
