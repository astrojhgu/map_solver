#![allow(clippy::too_many_arguments)]
#![allow(unused_imports)]
#![allow(clippy::many_single_char_names)]
use std::default::Default;

use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;

//use rustfft::{FFTnum, FFTplanner};
use crate::mathematica::{arctan, log, powerf, poweri};
use fftn::{fft2, ifft2, Complex, FFTnum};
use linear_solver::utils::{sp_mul_a1, sp_mul_a2};
use ndarray::{azip, s, Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2};
use num_traits::{Float, FloatConst, NumAssign, Zero};
use scorus::linear_space::traits::InnerProdSpace;
use scorus::linear_space::type_wrapper::LsVec;
use sprs::CsMat;
//use crate::

use crate::utils::{deflatten_order_f, flatten_order_f};
const PS_W: f64 = 0.0001;
const PS_E: f64 = 0.00001;
pub const DT: f64 = 2.0;
pub const FMAX: f64 = 0.5 / DT;

pub fn dft2d_matrix<T>(M: usize, N: usize, forward: bool) -> Array2<Complex<T>>
where
    T: Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>,
{
    let mut result = Array2::zeros((M * N, M * N));
    let two = T::one() + T::one();
    let ang = if forward {
        -two * T::PI()
    } else {
        two * T::PI()
    };
    let w = Complex::from_polar(&T::one(), &ang);
    let norm = T::from_usize(M * N).unwrap().sqrt();
    for m in 0..M {
        for n in 0..N {
            for k in 0..M {
                for l in 0..N {
                    let p = n * M + m;
                    let q = l * M + k;
                    let f = T::from_usize(k * m).unwrap() / T::from_usize(M).unwrap()
                        + T::from_usize(l * n).unwrap() / T::from_usize(N).unwrap();
                    result[(q, p)] = w.powf(f) / norm;
                }
            }
        }
    }
    result
}

pub fn smooth_step<T>(x: T, w: T) -> T
where
    T: Float + FloatConst,
{
    let two = T::one() + T::one();
    (T::PI() / two + arctan((x) / w)) / T::PI()
}

pub fn smooth_step_prime<T>(x: T, w: T) -> T
where
    T: Float + FloatConst,
{
    w / (T::PI() * (w.powi(2) + x.powi(2)))
}

pub fn pl<T>(f: T, a: T, f0: T, alpha: T, w: T, e: T) -> T
where
    T: Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>,
{
    let two = T::one() + T::one();
    let f = f.abs();
    let a2 = a.powi(2);
    let y = ((e + f.powi(2)) / (e + f0.powi(2))).powf(alpha / two);
    let s = smooth_step(f - f0, w);
    a2 * y * s + a2 * (T::one() - s)
}

pub fn ps_model<T>(
    ft: &[T],
    fch: &[T],
    a_t: T,
    ft_0: T,
    alpha_t: T,
    fch_0: T,
    alpha_ch: T,
    b: T,
    w: T,
    e: T,
) -> Array2<T>
where
    T: Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>,
{
    let a_ch = T::one();
    let x = ft
        .par_iter()
        .map(|&f| pl(f, a_t, ft_0, alpha_t, w, e))
        .collect::<Vec<_>>();
    let y = fch
        .par_iter()
        .map(|&f| pl(f, a_ch, fch_0, alpha_ch, w, e))
        .collect::<Vec<_>>();
    let b2 = b.powi(2);
    let mut result = Array2::<T>::zeros((ft.len(), fch.len()));
    for i in 0..ft.len() {
        for j in 0..fch.len() {
            result[(i, j)] = x[i] * y[j] + b2;
        }
    }
    result
}

pub fn dpl_da<T>(f: T, a: T, f0: T, alpha: T, w: T, e: T) -> T
where
    T: Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>,
{
    let two = T::one() + T::one();
    let f = f.abs();
    let s = smooth_step(f - f0, w);
    let y = ((e + f.powi(2)) / (e + f0.powi(2))).powf(alpha / two);
    (T::one() + (y - T::one()) * s) * two * a
}

pub fn dps_model_da_t<T>(
    ft: &[T],
    fch: &[T],
    a_t: T,
    ft_0: T,
    alpha_t: T,
    fch_0: T,
    alpha_ch: T,
    _b: T,
    w: T,
    e: T,
) -> Array2<T>
where
    T: Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>,
{
    let a_ch = T::one();
    let x = ft
        .par_iter()
        .map(|&f| dpl_da(f, a_t, ft_0, alpha_t, w, e))
        .collect::<Vec<_>>();
    let y = fch
        .par_iter()
        .map(|&f| pl(f, a_ch, fch_0, alpha_ch, w, e))
        .collect::<Vec<_>>();
    //let b2 = b.powi(2);
    let mut result = Array2::<T>::zeros((ft.len(), fch.len()));
    for i in 0..ft.len() {
        for j in 0..fch.len() {
            result[(i, j)] = x[i] * y[j];
        }
    }
    result
}

pub fn dps_model_da_ch<T>(
    ft: &[T],
    fch: &[T],
    a_t: T,
    ft_0: T,
    alpha_t: T,
    fch_0: T,
    alpha_ch: T,
    _b: T,
    w: T,
    e: T,
) -> Array2<T>
where
    T: Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>,
{
    let a_ch = T::one();
    let x = ft
        .par_iter()
        .map(|&f| pl(f, a_t, ft_0, alpha_t, w, e))
        .collect::<Vec<_>>();
    let y = fch
        .par_iter()
        .map(|&f| dpl_da(f, a_ch, fch_0, alpha_ch, w, e))
        .collect::<Vec<_>>();
    //let b2 = b.powi(2);
    let mut result = Array2::<T>::zeros((ft.len(), fch.len()));
    for i in 0..ft.len() {
        for j in 0..fch.len() {
            result[(i, j)] = x[i] * y[j];
        }
    }
    result
}

pub fn dpl_df0<T>(f: T, a: T, f0: T, alpha: T, w: T, e: T) -> T
where
    T: Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>,
{
    let two = T::one() + T::one();
    let f = f.abs();
    //let s=smooth_step(f-f0, w);
    let y = ((e + f.powi(2)) / (e + f0.powi(2))).powf(alpha / two);
    //let a2=a.powi(2);
    a.powi(2)
        * (-((alpha * f0 * y * smooth_step(f - f0, w)) / (e + f0.powi(2)))
            - (-T::one() + ((e + f.powi(2)) / (e + f0.powi(2))).powf(alpha / two))
                * smooth_step_prime(f - f0, w))
}

pub fn dps_model_df0_t<T>(
    ft: &[T],
    fch: &[T],
    a_t: T,
    ft_0: T,
    alpha_t: T,
    fch_0: T,
    alpha_ch: T,
    _b: T,
    w: T,
    e: T,
) -> Array2<T>
where
    T: Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>,
{
    let a_ch = T::one();
    let x = ft
        .par_iter()
        .map(|&f| dpl_df0(f, a_t, ft_0, alpha_t, w, e))
        .collect::<Vec<_>>();
    let y = fch
        .par_iter()
        .map(|&f| pl(f, a_ch, fch_0, alpha_ch, w, e))
        .collect::<Vec<_>>();
    //let b2 = b.powi(2);
    let mut result = Array2::<T>::zeros((ft.len(), fch.len()));
    for i in 0..ft.len() {
        for j in 0..fch.len() {
            result[(i, j)] = x[i] * y[j];
        }
    }
    result
}

pub fn dps_model_df0_ch<T>(
    ft: &[T],
    fch: &[T],
    a_t: T,
    ft_0: T,
    alpha_t: T,
    fch_0: T,
    alpha_ch: T,
    _b: T,
    w: T,
    e: T,
) -> Array2<T>
where
    T: Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>,
{
    let a_ch = T::one();
    let x = ft
        .par_iter()
        .map(|&f| pl(f, a_t, ft_0, alpha_t, w, e))
        .collect::<Vec<_>>();
    let y = fch
        .par_iter()
        .map(|&f| dpl_df0(f, a_ch, fch_0, alpha_ch, w, e))
        .collect::<Vec<_>>();
    let _b2 = b.powi(2);
    let mut result = Array2::<T>::zeros((ft.len(), fch.len()));
    for i in 0..ft.len() {
        for j in 0..fch.len() {
            result[(i, j)] = x[i] * y[j];
        }
    }
    result
}

pub fn dpl_dalpha<T>(f: T, a: T, f0: T, alpha: T, w: T, e: T) -> T
where
    T: Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>,
{
    let two = T::one() + T::one();
    let f = f.abs();
    let y = (e + f.powi(2)) / (e + f0.powi(2));
    T::one() / two * a.powi(2) * y.powf(alpha / two) * y.ln() * smooth_step(f - f0, w)
}

pub fn dps_model_dalpha_t<T>(
    ft: &[T],
    fch: &[T],
    a_t: T,
    ft_0: T,
    alpha_t: T,
    fch_0: T,
    alpha_ch: T,
    _b: T,
    w: T,
    e: T,
) -> Array2<T>
where
    T: Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>,
{
    let a_ch = T::one();
    let x = ft
        .par_iter()
        .map(|&f| dpl_dalpha(f, a_t, ft_0, alpha_t, w, e))
        .collect::<Vec<_>>();
    let y = fch
        .par_iter()
        .map(|&f| pl(f, a_ch, fch_0, alpha_ch, w, e))
        .collect::<Vec<_>>();
    //let b2 = b.powi(2);
    let mut result = Array2::<T>::zeros((ft.len(), fch.len()));
    for i in 0..ft.len() {
        for j in 0..fch.len() {
            result[(i, j)] = x[i] * y[j];
        }
    }
    result
}

pub fn dps_model_dalpha_ch<T>(
    ft: &[T],
    fch: &[T],
    a_t: T,
    ft_0: T,
    alpha_t: T,
    fch_0: T,
    alpha_ch: T,
    _b: T,
    w: T,
    e: T,
) -> Array2<T>
where
    T: Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>,
{
    let a_ch = T::one();
    let x = ft
        .par_iter()
        .map(|&f| pl(f, a_t, ft_0, alpha_t, w, e))
        .collect::<Vec<_>>();
    let y = fch
        .par_iter()
        .map(|&f| dpl_dalpha(f, a_ch, fch_0, alpha_ch, w, e))
        .collect::<Vec<_>>();
    //let b2 = b.powi(2);
    let mut result = Array2::<T>::zeros((ft.len(), fch.len()));
    for i in 0..ft.len() {
        for j in 0..fch.len() {
            result[(i, j)] = x[i] * y[j];
        }
    }
    result
}

pub fn dps_model_db<T>(
    ft: &[T],
    fch: &[T],
    _a_t: T,
    _ft_0: T,
    _alpha_t: T,
    _fch_0: T,
    _alpha_ch: T,
    b: T,
    _w: T,
    _e: T,
) -> Array2<T>
where
    T: Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>,
{
    //let a_ch=T::one();
    let two = T::one() + T::one();
    let mut result = Array2::<T>::zeros((ft.len(), fch.len()));
    for i in 0..ft.len() {
        for j in 0..fch.len() {
            result[(i, j)] = two * b;
        }
    }
    result
}

pub fn ln_xsx<T>(x: ArrayView2<T>, psd: ArrayView2<T>) -> T
where
    T: Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>,
{
    //let two=T::one()+T::one();
    let n_t = x.nrows();
    let n_ch = x.ncols();
    let mut x_c = x.map(|&x1| Complex::new(x1, T::zero()));
    let mut X = Array2::zeros((n_t, n_ch));

    fft2(&mut x_c.view_mut(), &mut X.view_mut());

    X.iter()
        .zip(psd.iter())
        .map(|(y, &p)| y.norm_sqr() / p / T::from_usize(n_t * n_ch).unwrap())
        .fold(T::zero(), |x, y| x + y)
}

pub fn ln_det_sigma<T>(psd: ArrayView2<T>) -> T
where
    T: Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>,
{
    psd.iter().map(|x| x.ln()).fold(T::zero(), |x, y| x + y)
}

pub fn mvn_ln_pdf<T>(x: ArrayView2<T>, psd: ArrayView2<T>) -> T
where
    T: Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>,
{
    let two = T::one() + T::one();
    let n_t = x.nrows();
    let n_ch = x.ncols();

    -ln_xsx(x, psd)/two//-X^T PSD X
    -ln_det_sigma(psd)/two // -1/2*ln |sigma|
    -T::from_usize(n_t*n_ch).unwrap()/two*(two*T::PI()).ln() //k/2*ln(2pi)
}

pub fn dhalf_lndet_dps<T>(psd: ArrayView2<T>) -> Array2<T>
where
    T: Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>,
{
    let two = T::one() + T::one();
    psd.map(|&p| T::one() / p / two)
}

pub fn dhalf_ln_xsx_dx<T>(x: ArrayView2<T>, psd: ArrayView2<T>) -> Array2<T>
where
    T: Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>,
{
    let n_t = x.nrows();
    let n_ch = x.ncols();
    let mut x_c = x.map(|&x| Complex::<T>::from(x));
    let mut X = Array2::zeros((n_t, n_ch));
    fft2(&mut x_c.view_mut(), &mut X.view_mut());
    let mut px = X / psd;

    let mut fpx = Array2::zeros((n_t, n_ch));
    ifft2(&mut px.view_mut(), &mut fpx.view_mut());
    fpx.map(|&x| x.re)
}

pub fn dhalf_ln_xsx_dp<T>(x: ArrayView2<T>, psd: ArrayView2<T>) -> Array2<T>
where
    T: Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>,
{
    let n_t = x.nrows();
    let n_ch = x.ncols();
    let two = T::one() + T::one();
    let mut x_c = x.map(|&x| Complex::<T>::from(x));
    let mut X = Array2::zeros((n_t, n_ch));
    fft2(&mut x_c.view_mut(), &mut X.view_mut());
    //println!("psd={:?}", psd);
    let mut result = Array2::<T>::zeros((n_t, n_ch));
    azip!((r in &mut result, &x in &X, &p in &psd) *r=-x.norm_sqr()/p.powi(2)/two/T::from_usize(n_t*n_ch).unwrap());
    result
}

pub fn mvn_ln_pdf_grad<T>(x: ArrayView2<T>, psd: ArrayView2<T>) -> (Array2<T>, Array2<T>)
where
    T: Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>,
{
    let dydx = -dhalf_ln_xsx_dx(x, psd);
    let dydp = -dhalf_ln_xsx_dp(x, psd) - dhalf_lndet_dps(psd);
    (dydx, dydp)
}

pub fn ln_likelihood(
    x: &[f64],
    y: &[f64],
    psd: ArrayView2<f64>,
    ptr_mat: &CsMat<f64>,
    n_t: usize,
    n_ch: usize,
) -> f64 {
    let noise = &ArrayView1::from(y) - &sp_mul_a1(&ptr_mat, ArrayView1::from(x));

    let noise_2d = deflatten_order_f(noise.view(), n_t, n_ch);

    mvn_ln_pdf(noise_2d.view(), psd)
}

pub fn logprob_ana(
    x: &[f64],
    psp: &[f64],
    tod: &[f64],
    ptr_mat: &CsMat<f64>,
    n_t: usize,
    n_ch: usize,
) -> f64 {
    assert_eq!(psp.len(), 6);
    let a_t = psp[0];
    let ft_0 = psp[1];
    let alpha_t = psp[2];

    let fch_0 = psp[3];
    let alpha_ch = psp[4];

    let b = psp[5];

    let ft_min = 1.0 / (DT * n_t as f64);
    let fch_min = 1.0 / n_ch as f64;
    let ft: Vec<_> = (0..(n_t as isize + 1) / 2)
        .chain(-(n_t as isize) / 2..0)
        .map(|i| i as f64 * ft_min)
        .collect();
    let fch: Vec<_> = (0..(n_ch as isize + 1) / 2)
        .chain(-(n_ch as isize) / 2..0)
        .map(|i| i as f64 * fch_min)
        .collect();
    let psd = ps_model(
        &ft, &fch, a_t, ft_0, alpha_t, fch_0, alpha_ch, b, PS_W, PS_E,
    );

    if ft_0 > ft_min * (n_t / 2) as f64 || alpha_t < -3.0 || alpha_t > 1.0 {
        return -std::f64::INFINITY;
    }

    if fch_0 > fch_min * (n_ch / 2) as f64 || alpha_ch < -3.0 || alpha_ch > 1.0 {
        return -std::f64::INFINITY;
    }

    ln_likelihood(x, tod, psd.view(), ptr_mat, n_t, n_ch)
}

pub fn logprob_ana_grad(
    x: &[f64],
    psp: &[f64],
    tod: &[f64],
    ptr_mat: &CsMat<f64>,
    n_t: usize,
    n_ch: usize,
) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(psp.len(), 6);
    let a_t = psp[0];
    let ft_0 = psp[1];
    let alpha_t = psp[2];

    let fch_0 = psp[3];
    let alpha_ch = psp[4];

    let b = psp[5];

    let ft_min = 1.0 / (DT * n_t as f64);
    let fch_min = 1.0 / n_ch as f64;
    let ft: Vec<_> = (0..(n_t as isize + 1) / 2)
        .chain(-(n_t as isize) / 2..0)
        .map(|i| i as f64 * ft_min)
        .collect();
    let fch: Vec<_> = (0..(n_ch as isize + 1) / 2)
        .chain(-(n_ch as isize) / 2..0)
        .map(|i| i as f64 * fch_min)
        .collect();
    let psd = ps_model(
        &ft, &fch, a_t, ft_0, alpha_t, fch_0, alpha_ch, b, PS_W, PS_E,
    );

    let dpsd_da_t = dps_model_da_t(
        &ft, &fch, a_t, ft_0, alpha_t, fch_0, alpha_ch, b, PS_W, PS_E,
    );

    let dpsd_df0_t = dps_model_df0_t(
        &ft, &fch, a_t, ft_0, alpha_t, fch_0, alpha_ch, b, PS_W, PS_E,
    );
    let dpsd_df0_ch = dps_model_df0_ch(
        &ft, &fch, a_t, ft_0, alpha_t, fch_0, alpha_ch, b, PS_W, PS_E,
    );

    let dpsd_dalpha_t = dps_model_dalpha_t(
        &ft, &fch, a_t, ft_0, alpha_t, fch_0, alpha_ch, b, PS_W, PS_E,
    );
    let dpsd_dalpha_ch = dps_model_dalpha_ch(
        &ft, &fch, a_t, ft_0, alpha_t, fch_0, alpha_ch, b, PS_W, PS_E,
    );

    let dpsd_db = dps_model_db(
        &ft, &fch, a_t, ft_0, alpha_t, fch_0, alpha_ch, b, PS_W, PS_E,
    );

    //println!("aaa {} {}", psd.nrows(), psd.ncols());

    let (gx, gp) = ln_likelihood_grad(x, tod, psd.view(), ptr_mat, n_t, n_ch);
    let gp = LsVec(gp);
    let g_ps_param = vec![
        gp.dot(&LsVec(flatten_order_f(dpsd_da_t.view()).to_vec())),
        gp.dot(&LsVec(flatten_order_f(dpsd_df0_t.view()).to_vec())),
        gp.dot(&LsVec(flatten_order_f(dpsd_dalpha_t.view()).to_vec())),
        gp.dot(&LsVec(flatten_order_f(dpsd_df0_ch.view()).to_vec())),
        gp.dot(&LsVec(flatten_order_f(dpsd_dalpha_ch.view()).to_vec())),
        gp.dot(&LsVec(flatten_order_f(dpsd_db.view()).to_vec())),
    ];
    (gx, g_ps_param)
}

pub fn ln_likelihood_grad(
    x: &[f64],
    y: &[f64],
    psd: ArrayView2<f64>,
    ptr_mat: &CsMat<f64>,
    n_t: usize,
    n_ch: usize,
) -> (Vec<f64>, Vec<f64>) {
    let noise = &ArrayView1::from(y) - &sp_mul_a1(&ptr_mat, ArrayView1::from(x));

    let noise_2d = deflatten_order_f(noise.view(), n_t, n_ch);
    //println!("{} {}", noise_2d.nrows(), noise_2d.ncols());
    //println!("{} {}", psd.nrows(), psd.ncols());

    let (dlnpdn, dlnpdp) = mvn_ln_pdf_grad(noise_2d.view(), psd);

    let dlnpdn = flatten_order_f(dlnpdn.view());
    let dlnpdp = flatten_order_f(dlnpdp.view());

    let dlnpdx = (-sp_mul_a1(&ptr_mat.transpose_view(), ArrayView1::from(&dlnpdn))).to_vec();

    (dlnpdx, dlnpdp.to_vec())
}

pub fn psd2cov<T>(x: &[T], n_t: usize, n_ch: usize) -> Array2<T>
where
    T: Float
        + FloatConst
        + NumAssign
        + std::fmt::Debug
        + FFTnum
        + From<u32>
        + std::default::Default,
{
    //let n=x.len();
    let x_c = Array1::from(x.iter().map(|&x| Complex::<T>::from(x)).collect::<Vec<_>>());
    let mut x_c = deflatten_order_f(x_c.view(), n_t, n_ch);

    let mut X = Array2::zeros((n_t, n_ch));

    ifft2(&mut x_c.view_mut(), &mut X.view_mut());
    X.map(|&x| x.re)
}

pub fn circulant_matrix<T>(x: &[T]) -> Array2<T>
where
    T: Copy,
{
    let n = x.len();
    let mut result = unsafe { Array2::uninitialized((n, n)) };

    for i in 0..n {
        for j in 0..n {
            let k = (j as isize - i as isize) % n as isize;
            let k = if k < 0 { k + n as isize } else { k } as usize;
            result[(i, j)] = x[k];
        }
    }
    result
}

pub fn circulant_block_matrix<T>(x: &[ArrayView2<T>]) -> Array2<T>
where
    T: Copy,
{
    let n = x.len();
    let nc1 = x[0].ncols();
    let nr1 = x[0].nrows();
    let mut result = unsafe { Array2::uninitialized((nr1 * n, nc1 * n)) };

    for i in 0..n {
        for j in 0..n {
            let k = (j as isize - i as isize) % n as isize;
            let k = if k < 0 { k + n as isize } else { k } as usize;
            //result[(i,j)]=x[k];
            result
                .slice_mut(s![i * nr1..(i + 1) * nr1, j * nc1..(j + 1) * nc1])
                .assign(&x[k]);
        }
    }
    result
}

pub fn psd2cov_mat<T>(x: &[T], n_t: usize, n_ch: usize) -> Array2<T>
where
    T: Float
        + FloatConst
        + NumAssign
        + std::fmt::Debug
        + FFTnum
        + From<u32>
        + std::default::Default,
{
    let cov = flatten_order_f(psd2cov(x, n_t, n_ch).view()).to_vec();
    let xb = cov
        .chunks(n_t)
        .map(|x1| circulant_matrix(x1))
        .collect::<Vec<_>>();
    let xb1: Vec<_> = xb.iter().map(|x| x.view()).collect();
    circulant_block_matrix(&xb1)
}

#[cfg(test)]
mod tests {
    use linear_solver::io::RawMM;
    use num_traits::Float;
    use rand::{thread_rng, Rng};
    use rand_distr::StandardNormal;
    fn get_psd() -> ndarray::Array2<f64> {
        let n_t = 32;
        let n_ch = 16;
        let ft_min = 0.1;
        let fch_min = 1.0 / n_ch as f64;
        let ft: Vec<_> = (0..(n_t + 1) / 2)
            .chain(-n_t / 2..0)
            .map(|i| i as f64 * ft_min)
            .collect();
        let fch: Vec<_> = (0..(n_ch as isize + 1) / 2)
            .chain(-(n_ch as isize) / 2..0)
            .map(|i| i as f64 * fch_min)
            .collect();
        let (a_t, ft_0, alpha_t) = (3.0, ft_min * 2 as f64, -1.0);
        let (fch_0, alpha_ch) = (ft_min * 2 as f64, -1.0);
        let b = 0.1;
        super::ps_model(
            &ft, &fch, a_t, ft_0, alpha_t, fch_0, alpha_ch, b, 1e-9, 1e-9,
        )
    }

    fn get_psd_vec() -> (Vec<f64>, usize, usize) {
        let a = get_psd();
        let n_t = a.nrows();
        let n_ch = a.ncols();
        (super::flatten_order_f(a.view()).to_vec(), n_t, n_ch)
    }

    //#[test]
    fn test_xsx() {
        let (psd, n_t, n_ch) = get_psd_vec();
        //let n=psd.len();
        let cov_mat = super::psd2cov_mat(&psd, n_t, n_ch);
        RawMM::from_array2(cov_mat.view()).to_file("cov.mtx");
        let mut x = ndarray::Array::from(vec![0.0; n_t * n_ch]);
        x[100] = 1.0;
        let brute_force_answer = x.dot(&cov_mat.dot(&x));
        let x = super::deflatten_order_f(x.view(), n_t, n_ch);
        let psd = super::deflatten_order_f(ndarray::ArrayView1::from(&psd), n_t, n_ch);
        RawMM::from_array2(psd.view()).to_file("psd.mtx");
        let smart_answer = super::ln_xsx(x.view(), psd.view());
        //println!("{} {}", smart_answer, brute_force_answer);
        assert!((smart_answer - brute_force_answer).abs() < 1e-3);
    }

    #[test]
    fn test_dxsx_dx() {
        let psd = get_psd();
        let mut rng = thread_rng();
        let x = psd.map(|_| rng.sample(StandardNormal));

        let m = (4, 3);
        let mut x1 = x.clone();
        let dx = 1e-8;
        x1[m] += dx;
        let direct_g =
            (super::ln_xsx(x1.view(), psd.view()) - super::ln_xsx(x.view(), psd.view())) / dx;

        let half_g = super::dhalf_ln_xsx_dx(x.view(), psd.view())[m];
        //println!("xx={}", (half_g*2.0-direct_g));
        //println!("a1={}", half_g*2.0);
        //println!("b1={}", direct_g);
        println!("a: {} {}", half_g * 2.0, direct_g);
        assert!((half_g * 2.0 - direct_g).abs() < 1e-4);
    }

    #[test]
    fn test_dxsx_dp() {
        //println!("afdsfadsfsafs");
        let psd = get_psd();
        let mut rng = thread_rng();
        let x = psd.map(|_| rng.sample(StandardNormal));

        let m = (5, 5);
        let mut psd1 = psd.clone();
        let dp = 0.00000001;
        psd1[m] += dp;
        //psd1[3]+=dp;
        //println!("{:?}", psd1);
        let direct_g =
            (super::ln_xsx(x.view(), psd1.view()) - super::ln_xsx(x.view(), psd.view())) / dp;

        let half_g = super::dhalf_ln_xsx_dp(x.view(), psd.view())[m];
        //println!("half_g*2-direct_g={}", (half_g*2.0-direct_g));
        println!("half_g*2={}", half_g * 2.0);
        println!("direct_g={}", direct_g);
        //println!("a={}",super::ln_xsx(x.as_slice().unwrap(), &psd));
        //println!("b={}",super::ln_xsx(x.as_slice().unwrap(), &psd1));
        assert!((half_g * 2.0 - direct_g).abs() < 1e-5);
    }
    #[test]
    fn test_dlndet_dp() {
        let psd = get_psd();
        let m = (3, 2);
        let lndet1 = psd.iter().map(|x| x.ln()).fold(0.0, |x, y| x + y);
        let mut psd1 = psd.clone();
        let dp = 0.0000001;
        psd1[m] += dp;
        let lndet2 = psd1.iter().map(|x| x.ln()).fold(0.0, |x, y| x + y);
        let half_dlndet = super::dhalf_lndet_dps(psd.view())[m];
        let direct_dlndet = (lndet2 - lndet1) / dp;
        println!("a2: {} {}", direct_dlndet, 2.0 * half_dlndet);
        assert!((half_dlndet * 2.0 - direct_dlndet) < 1e-5);
    }

    #[test]
    fn test_dlp_dx() {
        let psd = get_psd();
        let mut rng = thread_rng();

        let x = psd.map(|_| rng.sample(StandardNormal));
        let m = (2, 3);
        let delta = 0.000001;
        let lp1 = super::mvn_ln_pdf(x.view(), psd.view());
        let mut x1 = x.clone();
        x1[m] += delta;
        let lp2 = super::mvn_ln_pdf(x1.view(), psd.view());
        //println!("{} ", (lp2-lp1)/delta);

        let (dydx, dydp) = super::mvn_ln_pdf_grad(x.view(), psd.view());

        println!("dlp dx: {} {}", (lp2 - lp1) / delta, dydx[m]);
        assert!(((lp2 - lp1) / delta - dydx[m]).abs() < 1e-5);
    }
    #[test]
    fn test_grad() {
        let mut rng = thread_rng();
        let psd = get_psd();
        //let psd=vec![2.0;5];
        let x = psd.map(|_| rng.sample(StandardNormal));

        let m = (3, 4);
        let delta = 0.00000001;
        let grad1 = super::mvn_ln_pdf_grad(x.view(), psd.view());

        let dlp_dx = -super::dhalf_ln_xsx_dx(x.view(), psd.view());

        let dlp_dp1 = -super::dhalf_ln_xsx_dp(x.view(), psd.view());
        let dlp_dp2 = -super::dhalf_lndet_dps(psd.view());
        for i in 0..psd.nrows() {
            for j in 0..psd.ncols() {
                //println!("mcmc2: {} {} {} {}", grad1.0[(i,j)], grad1.1[(i,j)], dlp_dx[(i,j)], dlp_dp1[(i,j)]+dlp_dp2[(i,j)]);
                assert!((grad1.0[(i, j)] - dlp_dx[(i, j)]).abs() < 1e-5);
                assert!((grad1.1[(i, j)] - dlp_dp1[(i, j)] - dlp_dp2[(i, j)]).abs() < 1e-5);
            }
        }
    }
    #[test]
    fn test_mvn_dlp_dp() {
        let mut rng = thread_rng();
        let psd = get_psd();
        //let psd=vec![2.0;5];
        let x = psd.map(|_| rng.sample(StandardNormal));

        let m = (3, 4);
        let delta = 0.00000001;

        let lp1a = super::ln_xsx(x.view(), psd.view()) / 2.0;
        let lp1b = super::ln_det_sigma(psd.view()) / 2.0;
        let lp1 = super::mvn_ln_pdf(x.view(), psd.view());

        let mut psd1 = psd.clone();
        psd1[m] += delta;

        let lp2a = super::ln_xsx(x.view(), psd1.view()) / 2.0;
        let lp2b = super::ln_det_sigma(psd1.view()) / 2.0;

        let lp2 = super::mvn_ln_pdf(x.view(), psd1.view());

        println!(
            "a2: {} {}",
            (lp2a - lp1a) / delta,
            super::dhalf_ln_xsx_dp(x.view(), psd.view())[m]
        );

        println!(
            "b2: {} {}",
            (lp2b - lp1b) / delta,
            super::dhalf_lndet_dps(psd.view())[m]
        );

        let (gx, gp) = super::mvn_ln_pdf_grad(x.view(), psd.view());
        println!("c2: {} {}", (lp2 - lp1) / delta, gp[m]);
        assert!(((lp2 - lp1) / delta - gp[m]).abs() < 1e-5);
    }

    #[test]
    fn test_mvn_dlp_dx() {
        let mut rng = thread_rng();
        let psd = get_psd();
        //let psd=vec![2.0;5];
        let x = psd.map(|_| rng.sample(StandardNormal));

        let m = (4, 5);
        let delta = 0.00000001;
        let lp1a = super::ln_xsx(x.view(), psd.view()) / 2.0;
        let lp1 = super::mvn_ln_pdf(x.view(), psd.view());

        let mut x1 = x.clone();
        x1[m] += delta;

        let lp2a = super::ln_xsx(x1.view(), psd.view()) / 2.0;
        let lp2 = super::mvn_ln_pdf(x1.view(), psd.view());

        println!(
            "A1: {} {}",
            (lp2a - lp1a) / delta,
            super::dhalf_ln_xsx_dx(x.view(), psd.view())[m]
        );

        let (gx, gp) = super::mvn_ln_pdf_grad(x.view(), psd.view());
        println!("C1: {} {}", (lp2 - lp1) / delta, gx[m]);
        assert!(((lp2 - lp1) / delta - gx[m]).abs() < 5e-5);
    }

    #[test]
    fn test_mvn_grad() {
        let mut rng = thread_rng();
        let psd = get_psd();
        //let psd=vec![2.0;5];
        let noise = psd.map(|_| rng.sample(StandardNormal));

        let dx_std = 0.0000000001;

        let lp1 = super::mvn_ln_pdf(noise.view(), psd.view());
        for m in 0..100 {
            let mut noise2 = noise.clone();
            let dx: ndarray::Array2<f64> = noise.map(|_| {
                let d: f64 = rng.sample(StandardNormal);
                d * dx_std
            });
            let noise2 = &noise + &dx;

            let dp = psd.map(|_| {
                let d: f64 = rng.sample(StandardNormal);
                d * dx_std
            });

            let psd2 = &psd + &dp;
            let lp2 = super::mvn_ln_pdf(noise2.view(), psd2.view());

            let (gx, gp) = super::mvn_ln_pdf_grad(noise.view(), psd.view());
            let diff = (&gx * &dx).sum() + (&gp * &dp).sum();

            //let diff=ndarray::ArrayView1::from(&gx).dot(&dx)+ndarray::ArrayView1::from(&gp).dot(&ndarray::ArrayView1::from(&dp));
            println!("xxx: {} {} {}", lp2 - lp1, diff, lp2 - lp1 - diff);
            assert!(lp2 - lp1 - diff < 1e-7);
            //println!("{} {} {}",lp2-lp1, diff,  lp2-lp1-diff);
            //println!("{} {} {}", (lp2-lp1)/dx, gx[m], ((lp2-lp1)/dx-gx[m]).abs());
        }
    }
}
