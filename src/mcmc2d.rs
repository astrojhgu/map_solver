#![allow(clippy::too_many_arguments)]
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use scorus::linear_space::type_wrapper::LsVec;
use sprs::CsMat;
use ndarray::{Array1, ArrayView1};
use linear_solver::utils::{sp_mul_a1};
use crate::utils::{combine_ss, split_ss, SSFlag};
use crate::mcmc2d_func::{logprob_ana, logprob_ana_grad};
use crate::ps_model::PsModel;
pub struct Problem<P:PsModel+Send+Sync>
{
    pub tod: Vec<Vec<f64>>,
    pub ptr_mat: Vec<CsMat<f64>>,
    pub n_t: usize,
    pub n_ch: usize,
    pub ft: Vec<f64>,
    pub fch: Vec<f64>,
    pub dt: f64,
    pub psm: P,
}

impl<P> Problem<P> 
where P: PsModel+Sync+Send
{
    pub fn empty(n_t: usize, n_ch: usize, psm: P) -> Problem<P> {
        let dt=2.0;
        let ft_min = 1.0 / (dt * n_t as f64);
        let fch_min = 1.0 / n_ch as f64;
        let ft: Vec<_> = (0..(n_t as isize + 1) / 2)
            .chain(-(n_t as isize) / 2..0)
            .map(|i| i as f64 * ft_min)
            .collect();
        let fch: Vec<_> = (0..(n_ch as isize + 1) / 2)
            .chain(-(n_ch as isize) / 2..0)
            .map(|i| i as f64 * fch_min)
            .collect();

        Problem {
            tod: Vec::new(),
            ptr_mat: Vec::new(),
            n_t,
            n_ch,
            dt,
            ft, 
            fch,
            psm,
        }
    }

    pub fn new(tod: &[f64], ptr_mat: &CsMat<f64>, n_t: usize, n_ch: usize, psm: P) -> Problem<P> {
        let tod: Vec<_> = tod.to_vec();
        let dt=2.0;
        let ft_min = 1.0 / (dt * n_t as f64);
        let fch_min = 1.0 / n_ch as f64;
        let ft: Vec<_> = (0..(n_t as isize + 1) / 2)
            .chain(-(n_t as isize) / 2..0)
            .map(|i| i as f64 * ft_min)
            .collect();
        let fch: Vec<_> = (0..(n_ch as isize + 1) / 2)
            .chain(-(n_ch as isize) / 2..0)
            .map(|i| i as f64 * fch_min)
            .collect();
        Problem {
            tod: vec![tod],
            ptr_mat: vec![ptr_mat.clone()],
            n_t,
            n_ch,
            dt,
            psm,
            ft, 
            fch,
        }
    }

    pub fn with_obs(mut self, tod: &[f64], ptr_mat: &CsMat<f64>) -> Problem<P> {
        self.tod.push(tod.to_vec());
        self.ptr_mat.push(ptr_mat.clone());
        self
    }

    pub fn guess(&self)->Array1<f64>{
        let mut ptp=&self.ptr_mat[0].transpose_view()*&self.ptr_mat[0];
        let mut pty=sp_mul_a1(&self.ptr_mat[0].transpose_view(), ArrayView1::from(&self.tod[0]));
        for (p, t) in self.ptr_mat.iter().zip(self.tod.iter()).skip(1){
            ptp=&ptp+&(&p.transpose_view()*p);
            pty=&pty+&(sp_mul_a1(&p.transpose_view(), ArrayView1::from(t)));
        }
        for (&x, (i, j)) in ptp.iter() {
            assert_eq!(i,j);
            //result[(i)] = result[(i)] + x * b[(j)];
            pty[i]/=x;
        }
        pty
    }

    pub fn get_logprob<'a>(
        &'a self,
        q0: &[Option<f64>],
    ) -> impl Fn(&LsVec<f64, Vec<f64>>) -> f64 + 'a + std::marker::Sync + std::clone::Clone {
        let nx = self.ptr_mat[0].cols();
        let q0: Vec<_> = q0.to_vec();
        move |p| {
            let p = combine_ss(p, &q0);
            let sky = p.iter().take(nx).cloned().collect::<Vec<f64>>();
            let psp = p.iter().skip(nx).cloned().collect::<Vec<f64>>();
            assert_eq!(psp.len(), 6);
            self.ptr_mat
                .par_iter()
                .zip(self.tod.par_iter())
                .map(|(p, t)| {
                    //logprob_ana(&sky, &psp, t, p)
                    logprob_ana(&sky, &psp, t, p, &self.ft, &self.fch, &self.psm)
                })
                .sum::<f64>()
        }
    }

    pub fn get_logprob_grad<'a>(
        &'a self,
        q0: &[Option<f64>],
    ) -> impl Fn(&LsVec<f64, Vec<f64>>) -> LsVec<f64, Vec<f64>>
           + 'a
           + std::marker::Sync
           + std::clone::Clone {
        let nx = self.ptr_mat[0].cols();
        let q0: Vec<_> = q0.to_vec();
        //let flag:Vec<_>=q0.iter().map(|x| x.is_none()).collect();
        let flags: Vec<_> = q0
            .iter()
            .map(|x| {
                if x.is_none() {
                    SSFlag::Free
                } else {
                    SSFlag::Fixed
                }
            })
            .collect();
        move |p1| {
            let p = combine_ss(p1, &q0);
            let sky = p.iter().take(nx).cloned().collect::<Vec<f64>>();
            let psp = p.iter().skip(nx).cloned().collect::<Vec<f64>>();
            assert_eq!(psp.len(), 6);

            /*
            let (gx, gp)=self.ptr_mat.iter().zip(self.tod.iter()).map(|(p, t)|{
                logprob_ana_grad(&sky, &psp, t, p)
            }).fold((LsVec(vec![0.0_f64; sky.len()]), LsVec(vec![0.0_f64; psp.len()])), |a,b|{
                (&a.0+&LsVec(b.0), &a.1+&LsVec(b.1))
            });*/

            let grads: Vec<_> = self
                .ptr_mat
                .par_iter()
                .zip(self.tod.par_iter())
                .map(|(p, t)| {
                    //logprob_ana_grad(&sky, &psp, t, p)
                    logprob_ana_grad(&sky, &psp, t, p, &self.ft, &self.fch, &self.psm)
                })
                .collect();
            let (gx, gp) = grads.into_iter().fold(
                (
                    LsVec(vec![0.0_f64; sky.len()]),
                    LsVec(vec![0.0_f64; psp.len()]),
                ),
                |a, b| (&a.0 + &LsVec(b.0), &a.1 + &LsVec(b.1)),
            );
            let g = gx.0.into_iter().chain(gp.0.into_iter()).collect::<Vec<_>>();
            let (g, _) = split_ss(&g, &flags);
            assert_eq!(g.len(), p1.len());
            LsVec(g)
        }
    }
}
