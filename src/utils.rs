#![allow(unused_imports)]
#![allow(clippy::many_single_char_names)]
use std::default::Default;
//use rustfft::{FFTnum, FFTplanner};
use fftw::plan::C2CPlan64;
use fftw::plan::C2CPlan;
use fftw::types::{Flag, Sign};


use crate::mathematica::{arctan, log, powerf, poweri};

use num_complex::Complex;
//use fftn::{fft, ifft, Complex, FFTnum};
use linear_solver::utils::{sp_mul_a1, sp_mul_a2};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2};
use num_traits::{Float, FloatConst, NumAssign, Zero};
use scorus::linear_space::traits::InnerProdSpace;
use scorus::linear_space::type_wrapper::LsVec;
use sprs::CsMat;

type T=f64;

pub fn rfft(indata: &[T]) -> Vec<Complex<T>>
where
    //T: FFTnum + Float + From<u32> + std::fmt::Debug,
{
    let mut cindata: Vec<Complex<T>> = indata.iter().map(|&x| Complex::from(x)).collect();
    let mut result = vec![Complex::<T>::new(T::zero(), T::zero()); indata.len()];

    fft(&mut cindata, &mut result);
    result.truncate(indata.len() / 2 + 1);
    result
}

pub fn irfft(indata: &[Complex<T>]) -> Vec<T>
where
    //T: FFTnum + From<u32> + Float,
{
    let n = (indata.len() - 1) * 2;
    let mut cindata = Vec::<Complex<T>>::with_capacity(n);
    for &x in indata {
        cindata.push(x);
    }
    for i in indata.len()..n {
        cindata.push(indata[n - i].conj());
    }

    let mut result = vec![Complex::<T>::from(T::zero()); n];
    //fft.process(&mut cindata, &mut result);
    ifft(&mut cindata, &mut result);
    result.iter().map(|&x| x.re).collect()
}

pub fn rfft2(indata: ArrayView2<T>) -> Array2<Complex<T>>
where
    //T: FFTnum + From<u32> + Float,
{
    let h = indata.nrows();
    let w = indata.ncols();
    let mut cindata = Array2::<Complex<T>>::zeros((h, w));
    for i in 0..h {
        for j in 0..w {
            cindata[(i, j)] = indata[(i, j)].into();
        }
    }

    let mut result = Array2::<Complex<T>>::zeros((h, w));
    fft2(cindata.view_mut(), result.view_mut());
    result
}

pub fn irfft2(indata: ArrayView2<Complex<T>>) -> Array2<T>
where
    //T: FFTnum + From<u32> + Float + std::fmt::Debug,
{
    let h = indata.nrows();
    let w = indata.ncols();
    let mut cindata = Array2::<Complex<T>>::zeros((h, w));
    for i in 0..h {
        for j in 0..w {
            cindata[(i, j)] = indata[(i, j)];
        }
    }

    let mut result = Array2::<Complex<T>>::zeros((h, w));
    ifft2(cindata.view_mut(), result.view_mut());
    let mut rresult = Array2::<T>::zeros((h, w));
    for i in 0..h {
        for j in 0..w {
            rresult[(i, j)] = result[(i, j)].re
        }
    }
    rresult
}

pub fn circmat_x_vec(m: &[T], x: &[T]) -> Vec<T>
where
    //T: Float + FloatConst + NumAssign + std::fmt::Debug + Zero + FFTnum + From<u32>,
{
    //let mut fft = chfft::RFft1D::new(m.len());
    let a = rfft(m);
    let b = rfft(x);
    let c: Vec<_> = a.iter().zip(b.iter()).map(|(&a, &b)| a * b).collect();
    //fft.backward(&c)
    irfft(&c)
}

pub fn circmat_x_mat(m: &[T], x: ArrayView2<T>) -> Array2<T>
where
    //T: Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>,
{
    //let mut fft = chfft::RFft1D::new(m.len());
    let a = rfft(m);
    //let a = fft.forward(m);
    let mut result = Array2::<T>::zeros((x.nrows(), x.ncols()));

    for i in 0..x.ncols() {
        let b = rfft(x.column(i).to_owned().as_slice().unwrap());
        //let b = fft.forward(x.column(i).to_owned().as_slice().unwrap());
        let c: Vec<_> = a.iter().zip(b.iter()).map(|(&a, &b)| a * b).collect();
        let d = irfft(&c);
        //let d = fft.backward(&c);
        result.column_mut(i).assign(&Array1::from(d));
    }
    /*
    for (mut x, y1) in result.gencolumns_mut().into_iter().zip(x.gencolumns()){
        let b=fft.forward(y1.to_owned().as_slice().unwrap());
        let c:Vec<_>=a.iter().zip(b.iter()).map(|(&a, &b)|a*b).collect();
        let d=fft.backward(&c);
        x.assign(&Array1::from(d));
    }*/

    result
}

pub fn circmat_inv_x_mat(m: &[T], x: ArrayView2<T>) -> Array2<T>
where
    //T: Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>,
{
    //let mut fft = chfft::RFft1D::new(m.len());
    let a = rfft(m);
    //let a = fft.forward(m);
    let mut result = Array2::<T>::zeros((x.nrows(), x.ncols()));

    for i in 0..x.ncols() {
        let b = rfft(x.column(i).to_owned().as_slice().unwrap());
        //let b = fft.forward(x.column(i).to_owned().as_slice().unwrap());
        let c: Vec<_> = a.iter().zip(b.iter()).map(|(&a, &b)| b / a).collect();
        let d = irfft(&c);
        //let d = fft.backward(&c);
        result.column_mut(i).assign(&Array1::from(d));
    }
    result
}

pub fn diag2csmat<T>(diag: &[T]) -> CsMat<T>
where
    T: Copy,
{
    let n = diag.len();
    let idxptr: Vec<usize> = (0..=n).collect();
    let idx: Vec<usize> = (0..n).collect();
    let data = Vec::from(diag);
    CsMat::new((n, n), idxptr, idx, data)
}

pub fn csmat2diag<T>(mat: &CsMat<T>) -> Vec<T>
where
    T: Copy + Zero,
{
    let mut result = vec![T::zero(); mat.rows()];
    for (&v, (i, j)) in mat.iter() {
        assert!(i == j);
        result[i] = v
    }

    result
}

pub fn deconv(data: &[T], kernel: &[Complex<T>]) -> Vec<T>
where
    //T: Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>,
{
    //let mut rfft = chfft::RFft1D::<T>::new(data.len());
    let mut s = rfft(data);
    //let mut s = rfft.forward(data);
    assert!(s.len() == kernel.len());
    for i in 1..s.len() {
        if kernel[i].norm() != T::zero() {
            s[i] /= kernel[i];
        }
    }

    irfft(&s[..])
    //rfft.backward(&s[..])
}

pub fn flatten_order_f<T>(data: ArrayView2<T>) -> Array1<T>
where
    T: Copy + Default + Zero,
{
    let n = data.ncols() * data.nrows();
    //data.t().as_standard_layout().into_owned().into_shape((n,)).unwrap()
    let mut result = Array1::zeros((n,));
    let nr = data.nrows();
    for i in 0..data.nrows() {
        for j in 0..data.ncols() {
            result[j * nr + i] = data[(i, j)];
        }
    }
    result
}

pub fn deflatten_order_f<T>(data: ArrayView1<T>, nrows: usize, ncols: usize) -> Array2<T>
where
    T: Copy + Default + Zero,
{
    //data.into_shape((ncols,nrows)).unwrap().t().to_owned().as_standard_layout().into_owned()
    let mut result = Array2::zeros((nrows, ncols));
    for i in 0..nrows {
        for j in 0..ncols {
            result[(i, j)] = data[j * nrows + i];
        }
    }
    result
}

pub fn flatten<T>(data: ArrayView2<T>) -> Array1<T>
where
    T: Copy + Default + Zero,
{
    let n = data.ncols() * data.nrows();
    //data.t().as_standard_layout().into_owned().into_shape((n,)).unwrap()
    data.to_owned().into_shape((n,)).unwrap()
}

pub fn deflatten<T>(data: ArrayView1<T>, nrows: usize, ncols: usize) -> Array2<T>
where
    T: Copy + Default + Zero,
{
    data.to_owned().into_shape((nrows, ncols)).unwrap()
    //data.into_shape((ncols,nrows)).unwrap().t().to_owned().as_standard_layout().into_owned()
}


pub fn deconv2(data: ArrayView2<T>, kernel: ArrayView2<Complex<T>>) -> Array2<T>
where
    //T: Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>,
{
    let mut s = rfft2(data);
    assert!(s.shape() == kernel.shape());
    for i in 0..s.nrows() {
        for j in 0..s.ncols() {
            if i != 0 && j != 0 {
                s[(i, j)] /= kernel[(i, j)];
            }
        }
    }
    irfft2(s.view())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SSFlag {
    Fixed,
    Free,
}

pub fn split_ss<T>(x: &[T], flag: &[SSFlag]) -> (Vec<T>, Vec<Option<T>>)
where
    T: Copy,
{
    let mut y1 = Vec::new();
    let mut y2 = Vec::new();
    for (&x1, &f1) in x.iter().zip(flag.iter()) {
        match f1 {
            SSFlag::Free => {
                y1.push(x1);
                y2.push(None);
            }
            SSFlag::Fixed => {
                y2.push(Some(x1));
            }
        }
    }
    (y1, y2)
}

pub fn combine_ss<T>(x: &[T], y: &[Option<T>]) -> Vec<T>
where
    T: Copy,
{
    let mut n = 0;
    let mut result = Vec::new();
    for &y1 in y.iter() {
        if let Some(x1) = y1 {
            result.push(x1);
        } else {
            result.push(x[n]);
            n += 1;
        }
    }
    assert_eq!(n, x.len());
    result
}

pub fn fft2(mut in_data: ArrayViewMut2<Complex<f64>>, 
            mut out_data: ArrayViewMut2<Complex<f64>>){
    let mut plan=C2CPlan64::new(&[in_data.nrows(),in_data.ncols()], in_data.as_slice_mut().unwrap(), out_data.as_slice_mut().unwrap(), 
    Sign::Forward, Flag::ESTIMATE).unwrap();
    plan.c2c(&mut in_data.as_slice_mut().unwrap(), &mut out_data.as_slice_mut().unwrap());
    let n=Complex::<f64>::from((in_data.ncols() as f64*in_data.nrows() as f64).sqrt());
    out_data/=n;
}

pub fn ifft2(mut in_data: ArrayViewMut2<Complex<f64>>, 
    mut out_data: ArrayViewMut2<Complex<f64>>){
    let mut plan=C2CPlan64::new(&[in_data.nrows(),in_data.ncols()], in_data.as_slice_mut().unwrap(), out_data.as_slice_mut().unwrap(), 
    Sign::Backward, Flag::ESTIMATE).unwrap();
    plan.c2c(&mut in_data.as_slice_mut().unwrap(), &mut out_data.as_slice_mut().unwrap());
    let n=Complex::<f64>::from((in_data.ncols() as f64*in_data.nrows() as f64).sqrt());
    out_data/=n;
}

pub fn fft(in_data: &mut [Complex<f64>], 
    out_data: &mut [Complex<f64>]){
    let mut plan=C2CPlan64::new(&[in_data.len()], in_data, out_data, 
    Sign::Forward, Flag::ESTIMATE).unwrap();
    plan.c2c(in_data, out_data);
    let n=Complex::<f64>::from((in_data.len() as f64).sqrt());
    //out_data/=n;
    for x in out_data.iter_mut(){
        *x/=n;
    }
}

pub fn ifft(in_data: &mut [Complex<f64>], 
    out_data: &mut [Complex<f64>]){
    let mut plan=C2CPlan64::new(&[in_data.len()], in_data, out_data, 
    Sign::Backward, Flag::ESTIMATE).unwrap();
    plan.c2c(in_data, out_data);
    let n=Complex::<f64>::from((in_data.len() as f64).sqrt());
    //out_data/=n;
    for x in out_data.iter_mut(){
        *x/=n;
    }
}

pub fn transpose<T>(in_data: ArrayView2<T>)->Array2<T>
where T: Copy{
    let mut result = unsafe { Array2::uninitialized((in_data.ncols(), in_data.nrows())) };

    for i in 0..in_data.nrows(){
        for j in 0..in_data.ncols(){
            result[(j,i)]=in_data[(i,j)]
        }
    }
    result
}