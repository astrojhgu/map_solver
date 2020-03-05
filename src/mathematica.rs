use num_traits::{Float,NumCast, FloatConst};
use fftn::{fft, ifft, Complex, FFTnum};
use std::fmt::Debug;

pub fn poweri<T>(x:T, n:i32)->T
where T:Float
{
    x.powi(n)
}

pub fn powerf<T>(x: T, y: T)->T
where T:Float
{
    x.powf(y)
}

pub fn arctan<T>(x: T)->T
where T:Float
{
    x.atan()
}

pub fn log<T>(x:T)->T
where T:Float{
    x.ln()
}
