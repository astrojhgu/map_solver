use crate::mcmc2d_func::ps_model as ps_model_2d;
use crate::mcmc2d_func::{PS_E, PS_W};
use crate::mcmc_func::ps_model;
//use fftn::{fft2, ifft, ifft2};
use crate::utils::{fft2, ifft2, fft, ifft};
use ndarray::Array1;
use ndarray::Array2;
use num_complex::Complex;
use num_traits::{Float, FloatConst, NumAssign, NumCast, One, Zero, cast::FromPrimitive, };
use rand::Rng;
use rand_distr::Distribution;
use rand_distr::StandardNormal;
type T=f64;
pub fn gen_noise<U>(ntod: usize, psp: &[T], rng: &mut U, dt: T) -> Vec<T>
where
/*
    T: Float
        + NumCast
        + FloatConst
        + std::fmt::Debug
        + fftn::FFTnum
        + std::convert::From<u32>
        + NumAssign,
        */
    StandardNormal: Distribution<T>,
    U: Rng,
{
    let fmin = T::one() / (dt * <T as NumCast>::from(ntod).unwrap());
    let sigma = (0..(ntod + 2) / 2)
        .map(|i| {
            if i > 0 {
                Complex::<T>::new(rng.sample(StandardNormal), rng.sample(StandardNormal))
            } else {
                let f: T = rng.sample(StandardNormal);
                Complex::<T>::new(f, T::zero())
            }
        })
        .collect::<Vec<_>>();

    let mut psd = (0..(ntod as isize + 1) / 2)
        .chain(-(ntod as isize) / 2..0)
        .map(|i| {
            let mut x = sigma[i.abs() as usize]
                * ps_model(
                    <T as NumCast>::from(i).unwrap() * fmin,
                    psp[0],
                    psp[1],
                    psp[2],
                    psp[3],
                    T::from_f64(PS_W).unwrap(),
                    T::from_f64(PS_E).unwrap(),
                );
            if i < 0 {
                x = x.conj();
            }
            x
        })
        .map(|x| x.sqrt())
        .collect::<Vec<_>>();

    let mut noise = vec![Complex::<T>::new(T::zero(), T::zero()); psd.len()];
    ifft(&mut psd, &mut noise);
    noise.into_iter().map(|x| x.re).collect()
}

pub fn noise2d_psd(n_t: usize, n_ch: usize, psp: &[T], dt: T) -> Array2<T>
where
/*
    T: Float
        + NumCast
        + FloatConst
        + std::fmt::Debug
        + fftn::FFTnum
        + std::convert::From<u32>
        + NumAssign,*/
{
    let ft_min = T::one() / (T::from_usize(n_t).unwrap() * dt);
    let fch_min = T::one() / T::from_usize(n_ch).unwrap();
    let ft: Vec<_> = (0..(n_t as isize + 1) / 2)
        .chain(-(n_t as isize) / 2..0)
        .map(|i| T::from_isize(i).unwrap() * ft_min)
        .collect();
    let fch: Vec<_> = (0..(n_ch as isize + 1) / 2)
        .chain(-(n_ch as isize) / 2..0)
        .map(|i| T::from_isize(i).unwrap() * fch_min)
        .collect();

    let a_t = psp[0];
    let ft_0 = psp[1];
    let alpha_t = psp[2];

    let fch_0 = psp[3];
    let alpha_ch = psp[4];

    let b = psp[5];

    ps_model_2d(
        &ft,
        &fch,
        a_t,
        ft_0,
        alpha_t,
        fch_0,
        alpha_ch,
        b,
        T::from_f64(1e-9).unwrap(),
        T::from_f64(1e-9).unwrap(),
    )
}

pub fn white1d<U>(n_t: usize, rng: &mut U, sigma: T) -> Array1<T>
where
/*
    T: Float
        + NumCast
        + FloatConst
        + std::fmt::Debug
        + fftn::FFTnum
        + std::convert::From<u32>
        + NumAssign,*/
    StandardNormal: Distribution<T>,
    U: Rng,
{
    let mut white = Array1::<T>::zeros((n_t,));
    for i in 0..n_t {
        white[i] = rng.sample::<T, StandardNormal>(StandardNormal) * sigma;
    }
    white
}

pub fn white2d<U>(n_t: usize, n_ch: usize, rng: &mut U, sigma: T) -> Array2<T>
where/*
    T: Float
        + NumCast
        + FloatConst
        + std::fmt::Debug
        + fftn::FFTnum
        + std::convert::From<u32>
        + NumAssign,*/
    StandardNormal: Distribution<T>,
    U: Rng,
{
    let mut white = Array2::<T>::zeros((n_t, n_ch));
    for i in 0..n_t {
        for j in 0..n_ch {
            white[(i, j)] = rng.sample::<T, StandardNormal>(StandardNormal) * sigma;
        }
    }
    white
}

pub fn gen_noise_2d<U>(n_t: usize, n_ch: usize, psp: &[T], rng: &mut U, dt: T) -> Array2<T>
where/*
    T: Float
        + NumCast
        + FloatConst
        + std::fmt::Debug
        + fftn::FFTnum
        + std::convert::From<u32>
        + NumAssign,*/
    StandardNormal: Distribution<T>,
    U: Rng,
{
    assert_eq!(psp.len(), 6);
    let mut white = Array2::zeros((n_t, n_ch));
    for i in 0..n_t {
        for j in 0..n_ch {
            white[(i, j)] = Complex::new(rng.sample(StandardNormal), T::zero());
        }
    }

    let mut fwhite = Array2::zeros((n_t, n_ch));
    fft2(white.view_mut(), fwhite.view_mut());

    let ft_min = T::one() / (T::from_usize(n_t).unwrap() * dt);
    let fch_min = T::one() / T::from_usize(n_ch).unwrap();
    let ft: Vec<_> = (0..(n_t as isize + 1) / 2)
        .chain(-(n_t as isize) / 2..0)
        .map(|i| T::from_isize(i).unwrap() * ft_min)
        .collect();
    let fch: Vec<_> = (0..(n_ch as isize + 1) / 2)
        .chain(-(n_ch as isize) / 2..0)
        .map(|i| T::from_isize(i).unwrap() * fch_min)
        .collect();

    let a_t = psp[0];
    let ft_0 = psp[1];
    let alpha_t = psp[2];

    let fch_0 = psp[3];
    let alpha_ch = psp[4];

    let b = psp[5];

    let psm = ps_model_2d(
        &ft,
        &fch,
        a_t,
        ft_0,
        alpha_t,
        fch_0,
        alpha_ch,
        b,
        T::from_f64(1e-9).unwrap(),
        T::from_f64(1e-9).unwrap(),
    )
    .map(|&x| x.sqrt());
    fwhite = &fwhite * &psm;
    let mut result = Array2::zeros((n_t, n_ch));
    ifft2(fwhite.view_mut(), result.view_mut());
    result.map(|&x| x.re)
}
