#![allow(clippy::too_many_arguments)]
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use scorus::linear_space::type_wrapper::LsVec;
use scorus::mcmc::ensemble_sample::sample_pt as emcee_pt;
use scorus::mcmc::ensemble_sample::UpdateFlagSpec;
use scorus::mcmc::utils::swap_walkers;

use rand::Rng;
use rand_distr::StandardNormal;


use sprs::CsMat;
use num_complex::Complex64;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1};
use crate::utils::{fft2, ifft2};
use linear_solver::utils::{sp_mul_a1};
use linear_solver::minres::agmres;
use linear_solver::io::RawMM;



use crate::utils::{combine_ss, split_ss, SSFlag, flatten, deflatten};
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

pub fn build_b1(ptr_mat: &CsMat<f64>, psd: ArrayView2<f64>, tod: ArrayView1<f64>, n_t: usize, n_ch: usize)->Array1<f64>{
    let tod2d=deflatten(tod, n_ch, n_t);
    let mut x=tod2d.map(|&x| Complex64::from(x));
    let mut X = Array2::zeros((n_ch, n_t));
    fft2(x.view_mut(), X.view_mut());
    X=&X/&psd;
    ifft2(X.view_mut(), x.view_mut());
    let x=x.map(|x| x.re);
    let x=flatten(x.view());
    sp_mul_a1(&ptr_mat.transpose_view(), x.view())
}

pub fn func_a1(ptr_mat: &CsMat<f64>, psd: ArrayView2<f64>, x: ArrayView1<f64>, n_t: usize, n_ch: usize)->Array1<f64>{
    //println!("x:{:?}", deflatten(sp_mul_a1(ptr_mat, x).view(), n_t, n_ch));
    let mut x_c=deflatten(sp_mul_a1(ptr_mat, x).view(), n_ch, n_t).map(|&x| Complex64::from(x));//x_c=Px
    let mut X = Array2::zeros((n_ch, n_t));
    fft2(x_c.view_mut(), X.view_mut());
    X=&X/&psd;
    ifft2(X.view_mut(), x_c.view_mut());
    let nax=flatten(x_c.map(|x1| x1.re).view());//nax=N^-1Px
    sp_mul_a1(&ptr_mat.transpose_view(), nax.view())
}

pub fn build_b(ptr_mat: &[CsMat<f64>], psd: ArrayView2<f64>, tod: &[Vec<f64>], n_t: usize, n_ch: usize)->Array1<f64>{
    let mut result=build_b1(&ptr_mat[0], psd.view(), ArrayView1::from(&tod[0]), n_t, n_ch);
    for (pm1, tod1) in ptr_mat.iter().zip(tod.iter()).skip(1){
        result=&result+&build_b1(pm1, psd.view(), ArrayView1::from(tod1), n_t, n_ch);
    }
    result
}

pub fn func_a(ptr_mat: &[CsMat<f64>], psd: ArrayView2<f64>, x: ArrayView1<f64>, n_t: usize, n_ch: usize)->Array1<f64>{
    let mut result=func_a1(&ptr_mat[0], psd.view(), x.view(), n_t, n_ch);
    for p in ptr_mat.iter().skip(1){
        result=&result+&func_a1(p, psd.view(), x.view(), n_t, n_ch);
    }
    result
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

    pub fn solve_map(&self, x0: &mut [f64], psp: &[f64]){
        
        let psd=self.psm.value(&self.ft, &self.fch, psp);
        let b=build_b(&self.ptr_mat, psd.view(), &self.tod, self.n_t, self.n_ch);
        let A=|x: ArrayView1<f64>|{func_a(&self.ptr_mat, psd.view(), x, self.n_t, self.n_ch)};

        let mut solver=agmres::AGmresState::new(&A, ArrayView1::from(x0 as &[f64]), b.view(), None, 50, 20, 1, 0.4, 1e-9);
        while !solver.converged {
            println!("{} {}", solver.tol, solver.resid);
            solver.next(&A, None);
        }
        ArrayViewMut1::from(x0).assign(&solver.x);
    }

    pub fn sample_psp<U>(&self, x: &[f64], psp0: &mut [f64], nwalkers_per_beta: usize, beta_list: &[f64], niter: usize, rng: &mut U)
    where U: Rng
    {
        let psp0_vec=psp0.to_vec();
        let nbeta=beta_list.len();
        let nwalkers=nwalkers_per_beta*nbeta;
        let mut ensemble:Vec<_>=(0..nwalkers).map(|i|{
            if i==0{
                LsVec(psp0_vec.clone())
            }else{
                LsVec(psp0_vec.iter().map(|x: &f64| *x+0.01*rng.sample::<f64, StandardNormal>(StandardNormal)).collect())
            }
            
        }).collect();
        let nx=self.ptr_mat[0].cols();
        assert_eq!(x.len(), nx);
        assert_eq!(psp0.len(), self.psm.nparams());
        let mut q=x.to_vec();
        for &q1 in psp0.iter(){
            q.push(q1);
        }

        let flag_psp:Vec<_>= (0..q.len())
            .map(|x| if x < nx { SSFlag::Fixed } else { SSFlag::Free })
            .collect();

        let (q_psp, q_rest)=split_ss(&q, &flag_psp);
        let lp_f=self.get_logprob(&q_rest);
        
        let mut lp:Vec<_>=ensemble.par_iter().enumerate().map(|(i, x)| {
            println!("{}", i);
            lp_f(x)}).collect();
        
        let mut ufs=UpdateFlagSpec::All;
        let mut max_lp_all=std::f64::NEG_INFINITY;
        let mut optimal_psp=Vec::new();
        for i in 0..niter{
            if i%10==0{
                swap_walkers(&mut ensemble, &mut lp, rng, beta_list);
            }
            let old_lp=lp.clone();
            emcee_pt(&lp_f, &mut ensemble, &mut lp, rng, 2.0, &mut ufs, beta_list);
            let mut emcee_accept_cnt=vec![0; nbeta];
            for (k, (l1, l2)) in lp.iter().zip(old_lp.iter()).enumerate(){
                if l1!=l2{
                    emcee_accept_cnt[k/nwalkers_per_beta]+=1;
                }
            }
            
            let mut max_i=0;
            let mut max_lp=std::f64::NEG_INFINITY;
            for (j, &x) in lp.iter().enumerate().take(nwalkers_per_beta){
                if x>max_lp{
                    max_lp=x;
                    max_i=j;
                }
            }

            if max_lp>max_lp_all{
                max_lp_all=max_lp;
                optimal_psp=ensemble[max_i].0.clone();
            }
            eprintln!("{} {} {:?} {}",i, max_i, &(ensemble[max_i].0)[..], lp[max_i]);
            eprintln!("{:?}",emcee_accept_cnt);
            let q=combine_ss(&ensemble[max_i], &q_rest);
            //RawMM::from_array1(ArrayView1::from(&q)).to_file("dump.mtx");
        }
        ArrayViewMut1::from(psp0).assign(&ArrayView1::from(&optimal_psp));
    }
}
