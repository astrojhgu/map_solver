use sprs::CsMat;
use mcmc2d_func::{logprob_ana, logprob_ana_grad};
use scorus::linear_space::type_wrapper::LsVec;
use rayon::iter::ParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::IndexedParallelIterator;
use crate::{mcmc2d_func};
use crate::utils::{combine_ss, split_ss};

pub struct Problem{
    pub tod: Vec<Vec<f64>>,
    pub ptr_mat: Vec<CsMat<f64>>,
    pub n_t: usize, 
    pub n_ch: usize,
}

impl Problem{
    pub fn empty(n_t: usize, n_ch: usize)->Problem{
        Problem{
            tod: Vec::new(),
            ptr_mat: Vec::new(),
            n_t, 
            n_ch,
        }
    }

    pub fn new(tod: &[f64], ptr_mat: &CsMat<f64>, n_t: usize, n_ch: usize)->Problem{
        let tod:Vec<_>=tod.iter().cloned().collect();
        Problem{
            tod: vec![tod],
            ptr_mat: vec![ptr_mat.clone()],
            n_t, 
            n_ch,
        }
    }

    pub fn with_obs(mut self, tod: &[f64], ptr_mat: &CsMat<f64>)->Problem{
        self.tod.push(tod.iter().cloned().collect());
        self.ptr_mat.push(ptr_mat.clone());
        self
    }

    pub fn get_logprob<'a>(&'a self, q0: &[Option<f64>])->impl Fn(&LsVec<f64, Vec<f64>>)->f64+'a + std::marker::Sync+std::clone::Clone{
        let nx=self.ptr_mat[0].cols();
        let q0:Vec<_>=q0.iter().cloned().collect();
        move |p:&LsVec<f64, Vec<f64>>|{
            let p=combine_ss(p, &q0);
            let sky=p.iter().take(nx).cloned().collect::<Vec<f64>>();
            let psp=p.iter().skip(nx).cloned().collect::<Vec<f64>>();
            assert_eq!(psp.len(),6);
            self.ptr_mat.par_iter().zip(self.tod.par_iter()).map(|(p, t)|{
                //logprob_ana(&sky, &psp, t, p)
                logprob_ana(&sky, &psp, t, p, self.n_t, self.n_ch)
            }).sum::<f64>()
        }
    }

    pub fn get_logprob_grad<'a>(&'a self, q0: &[Option<f64>])->impl Fn(&LsVec<f64, Vec<f64>>)->LsVec<f64, Vec<f64>>+'a+ std::marker::Sync+std::clone::Clone{
        let nx=self.ptr_mat[0].cols();
        let q0:Vec<_>=q0.iter().cloned().collect();
        let flag:Vec<_>=q0.iter().map(|x| x.is_none()).collect();
        move |p1: &LsVec<f64, Vec<f64>>|{
            let p=combine_ss(p1, &q0);
            let sky=p.iter().take(nx).cloned().collect::<Vec<f64>>();
            let psp=p.iter().skip(nx).cloned().collect::<Vec<f64>>();
            assert_eq!(psp.len(),6);

            /*
            let (gx, gp)=self.ptr_mat.iter().zip(self.tod.iter()).map(|(p, t)|{
                logprob_ana_grad(&sky, &psp, t, p)
            }).fold((LsVec(vec![0.0_f64; sky.len()]), LsVec(vec![0.0_f64; psp.len()])), |a,b|{
                (&a.0+&LsVec(b.0), &a.1+&LsVec(b.1))
            });*/
            
            let grads:Vec<_>=self.ptr_mat.par_iter().zip(self.tod.par_iter()).map(|(p, t)|{
                //logprob_ana_grad(&sky, &psp, t, p)
                logprob_ana_grad(&sky, &psp, t, p, self.n_t, self.n_ch)
            }).collect();
            let (gx, gp)=grads.into_iter()
                        .fold((LsVec(vec![0.0_f64; sky.len()]), LsVec(vec![0.0_f64; psp.len()])), |a,b|{
                (&a.0+&LsVec(b.0), &a.1+&LsVec(b.1))
            });
            let g=gx.0.into_iter().chain(gp.0.into_iter()).collect::<Vec<_>>();
            let (g,_)=split_ss(&g, &flag);
            assert_eq!(g.len(), p1.len());
            LsVec(g)
        }
    }


}
