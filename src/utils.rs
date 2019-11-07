#![allow(clippy::many_single_char_names)]

use num_complex::Complex;
use sprs::CsMat;
//use linear_solver::utils::{sp_mul_a1, sp_mul_a2};
use ndarray::{Array1, Array2, ArrayView2};

use num_traits::{Float, FloatConst, NumAssign, Zero};

pub fn circmat_x_vec<T>(m: &[T], x: &[T]) -> Vec<T>
where
    T: Float + FloatConst + NumAssign + std::fmt::Debug + Zero,
{
    let mut fft = chfft::RFft1D::new(m.len());
    let a = fft.forward(m);
    let b = fft.forward(x);
    let c: Vec<_> = a.iter().zip(b.iter()).map(|(&a, &b)| a * b).collect();
    fft.backward(&c)
}

pub fn circmat_x_mat<T>(m: &[T], x: ArrayView2<T>) -> Array2<T>
where
    T: Float + FloatConst + NumAssign + std::fmt::Debug,
{
    let mut fft = chfft::RFft1D::new(m.len());
    let a = fft.forward(m);
    let mut result = Array2::<T>::zeros((x.nrows(), x.ncols()));

    for i in 0..x.ncols() {
        let b = fft.forward(x.column(i).to_owned().as_slice().unwrap());
        let c: Vec<_> = a.iter().zip(b.iter()).map(|(&a, &b)| a * b).collect();
        let d = fft.backward(&c);
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

pub fn deconv<T>(data: &[T], kernel: &[Complex<T>]) -> Vec<T>
where
    T: Float + FloatConst + NumAssign + std::fmt::Debug,
{
    let mut rfft = chfft::RFft1D::<T>::new(data.len());
    let mut s = rfft.forward(data);
    assert!(s.len() == kernel.len());
    for i in 1..s.len() {
        s[i] /= kernel[i];
    }
    
    rfft.backward(&s[..])
}
