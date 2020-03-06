use crate::utils;
use sprs::CsMat;
use ndarray::{Array1};
use std::ops::{Add, Sub, Mul};
use utils::{logprob_ana, logprob_ana_grad};
use scorus::linear_space::type_wrapper::LsVec;
use scorus::linear_space::traits::InnerProdSpace;

pub struct Problem{
    pub tod: Vec<Vec<f64>>,
    pub ptr_mat: Vec<CsMat<f64>>,
}

impl Problem{
    pub fn new(tod: &[f64], ptr_mat: &CsMat<f64>)->Problem{
        let tod:Vec<_>=tod.iter().cloned().collect();
        Problem{
            tod: vec![tod],
            ptr_mat: vec![ptr_mat.clone()],
        }
    }

    pub fn with_obs(mut self, tod: &[f64], ptr_mat: &CsMat<f64>)->Problem{
        self.tod.push(tod.iter().cloned().collect());
        self.ptr_mat.push(ptr_mat.clone());
        self
    }

    pub fn get_logprob<'a>(&'a self)->impl Fn(&LsVec<f64, Vec<f64>>)->f64+'a{
        let nx=self.ptr_mat[0].cols();
        move |p:&LsVec<f64, Vec<f64>>|{
            let sky=p.0.iter().take(nx).cloned().collect::<Vec<f64>>();
            let psp=p.0.iter().skip(nx).cloned().collect::<Vec<f64>>();
            assert_eq!(psp.len(),4);
            self.ptr_mat.iter().zip(self.tod.iter()).map(|(p, t)|{
                logprob_ana(&sky, &psp, t, p)
            }).sum::<f64>()
        }
    }

    pub fn get_logprob_grad<'a>(&'a self)->impl Fn(&LsVec<f64, Vec<f64>>)->LsVec<f64, Vec<f64>>+'a{
        let nx=self.ptr_mat[0].cols();
        move |p: &LsVec<f64, Vec<f64>>|{
            let sky=p.0.iter().take(nx).cloned().collect::<Vec<f64>>();
            let psp=p.0.iter().skip(nx).cloned().collect::<Vec<f64>>();
            assert_eq!(psp.len(),4);

            let (gx, gp)=self.ptr_mat.iter().zip(self.tod.iter()).map(|(p, t)|{
                logprob_ana_grad(&sky, &psp, t, p)
            }).fold((LsVec(vec![0.0_f64; sky.len()]), LsVec(vec![0.0_f64; psp.len()])), |a,b|{
                (&a.0+&LsVec(b.0), &a.1+&LsVec(b.1))
            });
            LsVec(gx.0.into_iter().chain(gp.0.into_iter()).collect::<Vec<_>>())
        }
    }

    pub fn get_logprob_sky<'a>(&'a self, q: &[f64])->impl Fn(&LsVec<f64, Vec<f64>>)->f64+'a{
        let nx=self.ptr_mat[0].cols();
        let psp:Vec<_>=q.iter().skip(nx).cloned().collect();
        assert_eq!(psp.len(),4);
        
        move |sky:&LsVec<f64, Vec<f64>>|{
            assert_eq!(sky.len(), nx);            
            self.ptr_mat.iter().zip(self.tod.iter()).map(|(p, t)|{
                logprob_ana(&sky, &psp, t, p)
            }).sum::<f64>()
        }
    }

    pub fn get_logprob_grad_sky<'a>(&'a self, q: &[f64])->impl Fn(&LsVec<f64, Vec<f64>>)->LsVec<f64, Vec<f64>>+'a{
        let nx=self.ptr_mat[0].cols();
        let psp:Vec<_>=q.iter().skip(nx).cloned().collect();
        assert_eq!(psp.len(),4);
        move |sky: &LsVec<f64, Vec<f64>>|{
            assert_eq!(sky.len(), nx);
            let (gx, _)=self.ptr_mat.iter().zip(self.tod.iter()).map(|(p, t)|{
                logprob_ana_grad(&sky, &psp, t, p)
            }).fold((LsVec(vec![0.0_f64; sky.len()]), LsVec(vec![0.0_f64; psp.len()])), |a,b|{
                (&a.0+&LsVec(b.0), &a.1+&LsVec(b.1))
            });
            gx
        }
    }

    pub fn get_logprob_psp<'a>(&'a self, q: &[f64])->impl Fn(&LsVec<f64, Vec<f64>>)->f64+'a{
        let nx=self.ptr_mat[0].cols();
        let sky:Vec<_>=q.iter().take(nx).cloned().collect();
        move |psp:&LsVec<f64, Vec<f64>>|{
            assert_eq!(psp.len(), 4);            
            self.ptr_mat.iter().zip(self.tod.iter()).map(|(p, t)|{
                logprob_ana(&sky, psp, t, p)
            }).sum::<f64>()
        }
    }

    pub fn get_logprob_grad_psp<'a>(&'a self, q: &[f64])->impl Fn(&LsVec<f64, Vec<f64>>)->LsVec<f64, Vec<f64>>+'a{
        let nx=self.ptr_mat[0].cols();
        let sky:Vec<_>=q.iter().take(nx).cloned().collect();
        move |psp: &LsVec<f64, Vec<f64>>|{
            assert_eq!(sky.len(), nx);
            let (_gx, gp)=self.ptr_mat.iter().zip(self.tod.iter()).map(|(p, t)|{
                logprob_ana_grad(&sky, psp, t, p)
            }).fold((LsVec(vec![0.0_f64; sky.len()]), LsVec(vec![0.0_f64; psp.len()])), |a,b|{
                (&a.0+&LsVec(b.0), &a.1+&LsVec(b.1))
            });
            gp
        }
    }
}
