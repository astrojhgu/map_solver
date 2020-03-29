use num_traits::{One, Zero, FloatConst};
use ndarray::{Array1, Array2, array};
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use crate::ps_model::PsModel;
use crate::mathematica::{arctan, log, powerf, poweri};

type T=f64;
pub const PS_W: f64 = 1e-5;
pub const PS_E: f64 = 1e-5;

pub fn smooth_step(x: T, w: T) -> T
{
    let two = T::one() + T::one();
    (T::PI() / two + arctan((x) / w)) / T::PI()
}

pub fn smooth_step_prime(x: T, w: T) -> T
{
    w / (T::PI() * (w.powi(2) + x.powi(2)))
}

pub fn pl(f: T, a: T, f0: T, alpha: T, w: T, e: T) -> T
//where
//    T: Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>,
{
    let two = T::one() + T::one();
    let f = f.abs();
    let a2 = a.powi(2);
    let y = ((e + f.powi(2)) / (e + f0.powi(2))).powf(alpha / two);
    let s = smooth_step(f - f0, w);
    a2 * y * s + a2 * (T::one() - s)
}

pub fn ps_model(
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
//where
//    T: Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>,
{
    let a_ch = T::one();

    let x = fch
        .par_iter()
        .map(|&f| pl(f, a_ch, fch_0, alpha_ch, w, e))
        .collect::<Vec<_>>();
    let y = ft
        .par_iter()
        .map(|&f| pl(f, a_t, ft_0, alpha_t, w, e))
        .collect::<Vec<_>>();
    let b2 = b.powi(2);
    let mut result = Array2::<T>::zeros((fch.len(), ft.len()));
    for i in 0..fch.len() {
        for j in 0..ft.len() {
            result[(i, j)] = x[i] * y[j] + b2;
        }
    }
    result
}

pub fn dpl_da(f: T, a: T, f0: T, alpha: T, w: T, e: T) -> T
//where
//    T: Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>,
{
    let two = T::one() + T::one();
    let f = f.abs();
    let s = smooth_step(f - f0, w);
    let y = ((e + f.powi(2)) / (e + f0.powi(2))).powf(alpha / two);
    (T::one() + (y - T::one()) * s) * two * a
}

pub fn dps_model_da_t(
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
//where
//    T: Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>,
{
    let a_ch = T::one();
    let x = fch
        .par_iter()
        .map(|&f| pl(f, a_ch, fch_0, alpha_ch, w, e))
        .collect::<Vec<_>>();
    let y = ft
        .par_iter()
        .map(|&f| dpl_da(f, a_t, ft_0, alpha_t, w, e))
        .collect::<Vec<_>>();
    //let b2 = b.powi(2);
    let mut result = Array2::<T>::zeros((fch.len(), ft.len()));
    for i in 0..fch.len() {
        for j in 0..ft.len() {
            result[(i, j)] = x[i] * y[j];
        }
    }
    result
}

pub fn dps_model_da_ch(
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
//where
//    T: Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>,
{
    let a_ch = T::one();
    let x = fch
        .par_iter()
        .map(|&f| dpl_da(f, a_ch, fch_0, alpha_ch, w, e))
        .collect::<Vec<_>>();
    let y = ft
        .par_iter()
        .map(|&f| pl(f, a_t, ft_0, alpha_t, w, e))
        .collect::<Vec<_>>();
    
    //let b2 = b.powi(2);
    let mut result = Array2::<T>::zeros((fch.len(), ft.len()));
    for i in 0..fch.len() {
        for j in 0..ft.len() {
            result[(i, j)] = x[i] * y[j];
        }
    }
    result
}

pub fn dpl_df0(f: T, a: T, f0: T, alpha: T, w: T, e: T) -> T
//where
//    T: Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>,
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

pub fn dps_model_df0_t(
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
//where
//    T: Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>,
{
    let a_ch = T::one();
    let x = fch
        .par_iter()
        .map(|&f| pl(f, a_ch, fch_0, alpha_ch, w, e))
        .collect::<Vec<_>>();
    let y = ft
        .par_iter()
        .map(|&f| dpl_df0(f, a_t, ft_0, alpha_t, w, e))
        .collect::<Vec<_>>();
    //let b2 = b.powi(2);
    let mut result = Array2::<T>::zeros((fch.len(), ft.len()));
    for i in 0..fch.len() {
        for j in 0..ft.len() {
            result[(i, j)] = x[i] * y[j];
        }
    }
    result
}

pub fn dps_model_df0_ch(
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
//where
//    T: Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>,
{
    let a_ch = T::one();
    let x = fch
        .par_iter()
        .map(|&f| dpl_df0(f, a_ch, fch_0, alpha_ch, w, e))
        .collect::<Vec<_>>();
    let y = ft
        .par_iter()
        .map(|&f| pl(f, a_t, ft_0, alpha_t, w, e))
        .collect::<Vec<_>>();
        //let b2 = b.powi(2);
    let mut result = Array2::<T>::zeros((fch.len(), ft.len()));
    for i in 0..fch.len() {
        for j in 0..ft.len() {
            result[(i, j)] = x[i] * y[j];
        }
    }
    result
}

pub fn dpl_dalpha(f: T, a: T, f0: T, alpha: T, w: T, e: T) -> T
//where
//    T: Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>,
{
    let two = T::one() + T::one();
    let f = f.abs();
    let y = (e + f.powi(2)) / (e + f0.powi(2));
    T::one() / two * a.powi(2) * y.powf(alpha / two) * y.ln() * smooth_step(f - f0, w)
}

pub fn dps_model_dalpha_t(
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
//where
//    T: Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>,
{
    let a_ch = T::one();
    let x = fch
        .par_iter()
        .map(|&f| pl(f, a_ch, fch_0, alpha_ch, w, e))
        .collect::<Vec<_>>();
    let y = ft
        .par_iter()
        .map(|&f| dpl_dalpha(f, a_t, ft_0, alpha_t, w, e))
        .collect::<Vec<_>>();
        //let b2 = b.powi(2);
    let mut result = Array2::<T>::zeros((fch.len(), ft.len()));
    for i in 0..fch.len() {
        for j in 0..ft.len() {
            result[(i, j)] = x[i] * y[j];
        }
    }
    result
}

pub fn dps_model_dalpha_ch(
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
//where
//    T: Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>,
{
    let a_ch = T::one();
    let x = fch
        .par_iter()
        .map(|&f| dpl_dalpha(f, a_ch, fch_0, alpha_ch, w, e))
        .collect::<Vec<_>>();
    let y = ft
        .par_iter()
        .map(|&f| pl(f, a_t, ft_0, alpha_t, w, e))
        .collect::<Vec<_>>();
    //let b2 = b.powi(2);
    let mut result = Array2::<T>::zeros((fch.len(), ft.len()));
    for i in 0..fch.len() {
        for j in 0..ft.len() {
            result[(i, j)] = x[i] * y[j];
        }
    }
    result
}

pub fn dps_model_db(
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
//where
//    T: Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>,
{
    //let a_ch=T::one();
    let two = T::one() + T::one();
    let mut result = Array2::<T>::zeros((fch.len(), ft.len()));
    for i in 0..fch.len() {
        for j in 0..ft.len() {
            result[(i, j)] = two * b;
        }
    }
    result
}

pub fn ps_model_grad(ft: &[T],
    fch: &[T],
    a_t: T,
    ft_0: T,
    alpha_t: T,
    fch_0: T,
    alpha_ch: T,
    b: T,
    w: T,
    e: T)->Vec<Array2<T>>    
{
    let dpsd_da_t = dps_model_da_t(
        &ft, &fch, a_t, ft_0, alpha_t, fch_0, alpha_ch, b, w, e,
    );

    let dpsd_df0_t = dps_model_df0_t(
        &ft, &fch, a_t, ft_0, alpha_t, fch_0, alpha_ch, b, w, e,
    );
    let dpsd_df0_ch = dps_model_df0_ch(
        &ft, &fch, a_t, ft_0, alpha_t, fch_0, alpha_ch, b, w, e,
    );

    let dpsd_dalpha_t = dps_model_dalpha_t(
        &ft, &fch, a_t, ft_0, alpha_t, fch_0, alpha_ch, b, w, e,
    );
    let dpsd_dalpha_ch = dps_model_dalpha_ch(
        &ft, &fch, a_t, ft_0, alpha_t, fch_0, alpha_ch, b, w, e,
    );

    let dpsd_db = dps_model_db(
        &ft, &fch, a_t, ft_0, alpha_t, fch_0, alpha_ch, b, w, e,
    );

    vec![dpsd_da_t, dpsd_df0_t, dpsd_dalpha_t, dpsd_df0_ch, dpsd_dalpha_ch, dpsd_db]
}

#[derive(Clone, Copy)]
pub struct PlPs{
}



impl PsModel for PlPs{
    fn value(&self, ft: &[f64], fch: &[f64], psp:&[f64])->Array2<f64>{
        let a_t = psp[0];
        let ft_0 = psp[1];
        let alpha_t = psp[2];

        let fch_0 = psp[3];
        let alpha_ch = psp[4];

        let b = psp[5];
        ps_model(ft, fch, a_t, ft_0, alpha_t, fch_0, alpha_ch, b, PS_W, PS_E)
    }
    fn grad(&self, ft: &[f64], fch: &[f64], psp: &[f64])->Vec<Array2<f64>>{
        let a_t = psp[0];
        let ft_0 = psp[1];
        let alpha_t = psp[2];

        let fch_0 = psp[3];
        let alpha_ch = psp[4];

        let b = psp[5];
        ps_model_grad(ft, fch, a_t, ft_0, alpha_t, fch_0, alpha_ch, b, PS_W, PS_E)
    }
    fn nparams(&self)->usize{
        6
    }
    fn support(&self, ft: &[f64], fch: &[f64], psp: &[f64])->bool{
        let a_t = psp[0];
        let ft_0 = psp[1];
        let alpha_t = psp[2];

        let fch_0 = psp[3];
        let alpha_ch = psp[4];

        let b = psp[5];
        let n_t=ft.len();
        let n_ch=fch.len();
        let ft_min=ft[1];
        let fch_min=fch[1];

        
        if ft_0.abs() > ft_min * (n_t / 2) as f64 || alpha_t < -3.0 || alpha_t > 1.0 {
            return false;
        }    
        if fch_0.abs() > fch_min * (n_ch / 2) as f64 || alpha_ch < -3.0 || alpha_ch > 1.0 {
            return false;
        }
        true
    }

    fn boundaries(&self, ft: &[f64], fch: &[f64])->(Array1<f64>, Array1<f64>){
        let n_t=ft.len();
        let n_ch=fch.len();
        let ft_min=ft[1];
        let fch_min=fch[1];
        let ft_max=ft_min*(n_t/4) as f64;
        let fch_max=fch_min*(n_ch/4) as f64;
        (array![1e-10, ft_min, -3.0, fch_min, -3.0, 1e-10],
            array![1e0, ft_max, 0.0, fch_max, 0.0, 1e0])
    }

}
