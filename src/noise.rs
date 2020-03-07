use num_complex::Complex;
use num_traits::{Float, FloatConst, NumCast, NumAssign};
use fftn::ifft;
use rand::Rng;
use rand_distr::Distribution;
use rand_distr::StandardNormal;

use crate::mcmc_func::ps_model;

pub fn gen_noise<T, U>(ntod: usize,psp: &[T], rng: &mut U, dt: T)->Vec<T>
where T: Float+NumCast+FloatConst+ std::fmt::Debug + fftn::FFTnum + std::convert::From<u32>+NumAssign,
StandardNormal: Distribution<T>,
U: Rng,
{
    let fmin=T::one()/(dt*<T as NumCast>::from(ntod).unwrap());
    let sigma=(0..(ntod+2)/2).map(|i|{
        if i>0{
            Complex::<T>::new(rng.sample(StandardNormal),rng.sample(StandardNormal))
        }else{
            let f:T=rng.sample(StandardNormal);
            Complex::<T>::new(f, T::zero())
        }
    }
    ).collect::<Vec<_>>();
    
    let mut psd=(0..(ntod as isize+1)/2).chain(-(ntod as isize)/2..0).map(|i| {
        let mut x=sigma[i.abs() as usize]*ps_model(<T as NumCast>::from(i).unwrap()*fmin, psp[0], psp[1], psp[2], psp[3],T::from_f64(1e-6).unwrap(), T::from_f64(1e-9).unwrap());
        if i<0{
            x=x.conj();
        }
        x
    }).collect::<Vec<_>>();

    let mut noise=vec![Complex::<T>::new(T::zero(), T::zero()); psd.len()];
    ifft(&mut psd, &mut noise);
    noise.into_iter().map(|x|x.re).collect()
}
