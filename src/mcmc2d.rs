use crate::{mcmc2d_func};
use sprs::CsMat;
use mcmc2d_func::{logprob_ana, logprob_ana_grad};
use scorus::linear_space::type_wrapper::LsVec;
use rayon::iter::ParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::IndexedParallelIterator;
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

    pub fn get_logprob<'a>(&'a self, negate: bool)->impl Fn(&LsVec<f64, Vec<f64>>)->f64+'a + std::marker::Sync+std::clone::Clone{
        let nx=self.ptr_mat[0].cols();
        move |p:&LsVec<f64, Vec<f64>>|{
            let sky=p.0.iter().take(nx).cloned().collect::<Vec<f64>>();
            let psp=p.0.iter().skip(nx).cloned().collect::<Vec<f64>>();
            assert_eq!(psp.len(),6);
            self.ptr_mat.par_iter().zip(self.tod.par_iter()).map(|(p, t)|{
                //logprob_ana(&sky, &psp, t, p)
                logprob_ana(&sky, &psp, t, p, self.n_t, self.n_ch)
            }).sum::<f64>()*if negate{-1.0}else{1.0}
            /*
            self.ptr_mat.iter().zip(self.tod.iter()).map(|(p, t)|{
                logprob_ana(&sky, &psp, t, p)
            }).sum::<f64>()            */
        }
    }

    pub fn get_logprob_grad<'a>(&'a self, negate: bool)->impl Fn(&LsVec<f64, Vec<f64>>)->LsVec<f64, Vec<f64>>+'a+ std::marker::Sync+std::clone::Clone{
        let nx=self.ptr_mat[0].cols();
        move |p: &LsVec<f64, Vec<f64>>|{
            let sky=p.0.iter().take(nx).cloned().collect::<Vec<f64>>();
            let psp=p.0.iter().skip(nx).cloned().collect::<Vec<f64>>();
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
            &LsVec(gx.0.into_iter().chain(gp.0.into_iter()).collect::<Vec<_>>())*if negate{-1.0}else{1.0}
        }
    }

    pub fn get_logprob_sky<'a>(&'a self, q: &[f64], negate: bool)->impl Fn(&LsVec<f64, Vec<f64>>)->f64+'a+ std::marker::Sync+std::clone::Clone{
        let nx=self.ptr_mat[0].cols();
        let psp:Vec<_>=q.iter().skip(nx).cloned().collect();
        assert_eq!(psp.len(),6);
        
        move |sky:&LsVec<f64, Vec<f64>>|{
            assert_eq!(sky.len(), nx);            
            self.ptr_mat.par_iter().zip(self.tod.par_iter()).map(|(p, t)|{
                logprob_ana(&sky, &psp, t, p, self.n_t, self.n_ch)
            }).sum::<f64>()*if negate{-1.0}else{1.0}
        }
    }

    pub fn get_logprob_grad_sky<'a>(&'a self, q: &[f64], negate: bool)->impl Fn(&LsVec<f64, Vec<f64>>)->LsVec<f64, Vec<f64>>+'a+ std::marker::Sync+std::clone::Clone{
        let nx=self.ptr_mat[0].cols();
        let psp:Vec<_>=q.iter().skip(nx).cloned().collect();
        assert_eq!(psp.len(),6);
        move |sky: &LsVec<f64, Vec<f64>>|{
            assert_eq!(sky.len(), nx);
            let grads:Vec<_>=self.ptr_mat.par_iter().zip(self.tod.par_iter()).map(|(p, t)|{
                logprob_ana_grad(&sky, &psp, t, p, self.n_t, self.n_ch)
            }).collect();
            let (gx, _)=grads.into_iter().fold((LsVec(vec![0.0_f64; sky.len()]), LsVec(vec![0.0_f64; psp.len()])), |a,b|{
                (&a.0+&LsVec(b.0), &a.1+&LsVec(b.1))
            });
            &gx*if negate{-1.0}else{1.0}
        }
    }

    pub fn get_logprob_psp<'a>(&'a self, q: &[f64], negate: bool)->impl Fn(&LsVec<f64, Vec<f64>>)->f64+'a+ std::marker::Sync+std::clone::Clone{
        let nx=self.ptr_mat[0].cols();
        let sky:Vec<_>=q.iter().take(nx).cloned().collect();
        move |psp:&LsVec<f64, Vec<f64>>|{
            assert_eq!(psp.len(), 6);
            self.ptr_mat.par_iter().zip(self.tod.par_iter()).map(|(p, t)|{
                logprob_ana(&sky, psp, t, p, self.n_t, self.n_ch)
            }).sum::<f64>()*if negate{-1.0}else{1.0}
        }
    }

    pub fn get_logprob_grad_psp<'a>(&'a self, q: &[f64], negate: bool)->impl Fn(&LsVec<f64, Vec<f64>>)->LsVec<f64, Vec<f64>>+'a+ std::marker::Sync+std::clone::Clone{
        let nx=self.ptr_mat[0].cols();
        let sky:Vec<_>=q.iter().take(nx).cloned().collect();
        move |psp: &LsVec<f64, Vec<f64>>|{
            assert_eq!(sky.len(), nx);
            let grads:Vec<_>=self.ptr_mat.par_iter().zip(self.tod.par_iter()).map(|(p, t)|{
                logprob_ana_grad(&sky, psp, t, p, self.n_t, self.n_ch)
            }).collect();
            let (_gx, gp)=grads.into_iter().fold((LsVec(vec![0.0_f64; sky.len()]), LsVec(vec![0.0_f64; psp.len()])), |a,b|{
                (&a.0+&LsVec(b.0), &a.1+&LsVec(b.1))
            });
            &gp*if negate{-1.0}else{1.0}
        }
    }
}
