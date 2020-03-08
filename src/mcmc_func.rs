#![allow(unused_imports)]
#![allow(clippy::many_single_char_names)]
use std::default::Default;
//use rustfft::{FFTnum, FFTplanner};
use scorus::linear_space::type_wrapper::LsVec;
use scorus::linear_space::traits::InnerProdSpace;
use fftn::{fft, ifft, Complex, FFTnum};
use sprs::CsMat;
use linear_solver::utils::{sp_mul_a1, sp_mul_a2};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2};
use num_traits::{Float, FloatConst, NumAssign, Zero};
use crate::mathematica::{arctan, powerf, poweri, log};


const PS_W:f64=1e-9;
const PS_E:f64=0.00001;
pub const DT:f64=2.0;
pub const FMAX:f64=0.5/DT;

pub fn dft_matrix<T>(n: usize, forward: bool)->Array2<Complex<T>>
where 
    T: Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32> ,{
    let mut result=Array2::zeros((n,n));
    let two=T::one()+T::one();
    let ang=if forward{
        -two*T::PI()/T::from_usize(n).unwrap()
    }else{
        two*T::PI()/T::from_usize(n).unwrap()
    };
    let w=Complex::from_polar(&T::one(), &ang);
    let norm=T::from_usize(n).unwrap().sqrt();
    for j in 0..n{
        for k in 0..n{
            result[(j,k)]=w.powi((j*k) as i32)/norm;
        }
    }
    result
}

pub fn circulant_matrix<T>(x: &[T])->Array2<T>
where 
    T: Copy{
    let n=x.len();
    let mut result=unsafe{Array2::uninitialized((n,n))};

    for i in 0..n{
        for j in 0..n{
            let k=(j as isize - i as isize)%n as isize;
            let k=if k<0{
                k+n as isize
            }else{
                k
            } as usize;
            result[(i,j)]=x[k];
        }
    }
    result
}

pub fn circulant_det<T>(x: &[T])->T
where 
T: Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>
{   
    let n=x.len();
    let mut y:Vec<_>=x.iter().map(|x|{Complex::<T>::from(x)}).collect();
    let mut y1=vec![Complex::<T>::zero(); n];
    fft(&mut y, &mut y1);
    y1.into_iter().map(|x| x.re).fold(T::one(), |x,y| x*y)
}

pub fn unit_step<T>(x:T)->T
where T: Float
{
    if x<T::zero(){
        T::zero()
    }else{
        T::one()
    }
}

pub fn smooth_step<T>(x:T, w:T)->T
where T: Float + FloatConst
{
    let two=T::one()+T::one();
    (T::PI()/two + arctan((x)/w))/T::PI()
}

pub fn smooth_step_prime<T>(x:T, w: T)->T
where T: Float + FloatConst{
    w/(T::PI()*(w.powi(2)+x.powi(2)))
}

pub fn ps_model<T>(f: T, a: T, b: T, f0: T, alpha: T, w: T, e: T)->T
where T:Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>
{
    let two=T::one()+T::one();
    let f=f.abs();
    let a2=a.powi(2);
    let b2=b.powi(2);
    let y=((e + f.powi(2))/(e + f0.powi(2))).powf(alpha/two);
    let s=smooth_step(f-f0, w);
    (a2*y+b2)*s+(a2+b2)*(T::one()-s)
}

pub fn dps_model_da<T>(f: T, a: T, b: T, f0: T, alpha: T, w: T, e: T)->T
where T:Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>
{
    let two=T::one()+T::one();
    let f=f.abs();
    let s=smooth_step(f-f0, w);
    let y=((e + f.powi(2))/(e + f0.powi(2))).powf(alpha/two);
    (T::one()+(y-T::one())*s)*two*a
}

pub fn dps_model_db<T>(f: T, a: T, b: T, f0: T, alpha: T, w: T, e: T)->T
where T:Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>{
    let two=T::one()+T::one();
    two*b
}

pub fn dps_model_df0<T>(f: T, a: T, b: T, f0: T, alpha: T, w: T, e: T)->T
where T:Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>{
    let two=T::one()+T::one();
    let f=f.abs();
    let s=smooth_step(f-f0, w);
    let y=((e + f.powi(2))/(e + f0.powi(2))).powf(alpha/two);
    let a2=a.powi(2);
    let b2=b.powi(2);
    a.powi(2)*(-((alpha*f0*y*smooth_step(f-f0,w))/(
    e + f0.powi(2))) - (-T::one() + ((e + f.powi(2))/(e + f0.powi(2))).powf(alpha/two)) *smooth_step_prime(f-f0,w))
}

pub fn dps_model_dalpha<T>(f: T, a: T, b: T, f0: T, alpha: T, w: T, e: T)->T
where T:Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>{
    let two=T::one()+T::one();
    let f=f.abs();
    let y= ((e + f.powi(2))/(e + f0.powi(2)));
    T::one()/two * a.powi(2)*y.powf(alpha/two)*y.ln()*smooth_step(f-f0,w)
}


pub fn ln_xsx<T>(x: &[T], psd: &[T])->T
where T:Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>
{
    let two=T::one()+T::one();
    let n=x.len();
    let mut x_c:Vec<_>=x.iter().map(|&x| Complex::<T>::from(x)).collect();
    let mut X=vec![Complex::<T>::zero(); n];
    fft(&mut x_c, &mut X);
    X.iter().zip(psd.iter()).map(|(y,&p)| {
        let r=y.norm_sqr()/p/T::from_usize(n).unwrap();
        r
    }).fold(T::zero(), |x, y|{x+y})
}

pub fn ln_det_sigma<T>(psd: &[T])->T
where T:Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>
{
    psd.iter().map(|x| x.ln()).fold(T::zero(), |x,y|x+y)
}

pub fn mvn_ln_pdf<T>(x: &[T], psd: &[T])->T
where T:Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>
{
    
    let two=T::one()+T::one();
    let n=x.len();

    -ln_xsx(x, psd)/two//-X^T PSD X
    -ln_det_sigma(psd)/two // -1/2*ln |sigma|
    -T::from_usize(n).unwrap()/two*(two*T::PI()).ln() //k/2*ln(2pi)
}

pub fn dhalf_lndet_dps<T>(psd: &[T])->Vec<T>
where T:Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>
{
    let two=T::one()+T::one();
    psd.iter().map(|&p| T::one()/p/two).collect()
}

pub fn dhalf_ln_xsx_dx<T>(x: &[T], psd: &[T])->Vec<T>
where T:Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>
{
    let n=x.len();
    let mut x_c:Vec<_>=x.iter().map(|&x| Complex::<T>::from(x)).collect();
    let mut X=vec![Complex::zero(); n];
    fft(&mut x_c, &mut X);
    let mut px:Vec<_>=psd.iter().zip(X.iter()).map(|(&p, &x)|{
        x/p
    }).collect();
    let mut fpx=vec![Complex::zero(); n];
    ifft(&mut px, &mut fpx);
    fpx.iter().map(|&x| x.re).collect()
}

pub fn dhalf_ln_xsx_dp<T>(x: &[T], psd: &[T])->Vec<T>
where T:Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>
{
    let n=x.len();
    let two=T::one()+T::one();
    let mut x_c:Vec<_>=x.iter().map(|&x| Complex::<T>::from(x)).collect();
    let mut X=vec![Complex::zero(); n];
    fft(&mut x_c, &mut X);
    //println!("psd={:?}", psd);
    X.iter().zip(psd.iter()).map(|(&x, &p)| -x.norm_sqr()/two/T::from_usize(n).unwrap()/p.powi(2)).collect()
}

pub fn mvn_ln_pdf_grad<T>(x: &[T], psd: &[T])->(Vec<T>, Vec<T>)
where T:Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>
{
    let dydx:Vec<_>=dhalf_ln_xsx_dx(x, psd).into_iter().map(|x|-x).collect();
    let dydp:Vec<_>=dhalf_ln_xsx_dp(x, psd).into_iter().zip(dhalf_lndet_dps(psd).into_iter()).map(|(dy1, dy2)|-dy1-dy2).collect();
    (dydx, dydp)
}

pub fn ln_likelihood(x: &[f64], y: &[f64], psd: &[f64], ptr_mat: &CsMat<f64>)->f64
{
    let nout=y.len();
    let ps1=ps_mirror(psd, nout);
    let noise=&ArrayView1::from(y)-&sp_mul_a1(&ptr_mat, ArrayView1::from(x));
    mvn_ln_pdf(noise.as_slice().unwrap(), &ps1)
}

pub fn smoothness(psd: &[f64], k:f64)->f64
{
    psd.windows(3).map(|p|{
        -(p[0].ln()+p[2].ln()-2.0*p[1].ln()).powi(2)/(2.0*k.powi(2))
    }).sum::<f64>()
}

pub fn delta(i: usize, j: usize)->f64{
    if i==j{
        1.0
    }else{
        0.0
    }
}

pub fn d_smoothness(y: &[f64], k: f64)->Vec<f64>{
    let mut result=vec![0.0; y.len()];

    for q in 0..y.len(){
        for i in 1..y.len()-1{
            result[q]=-(y[i+1].ln()+y[i-1].ln()-2.0*y[i].ln())*(
                delta(i+1, q)/y[i+1]
                +delta(i-1, q)/y[i-1]
                -2.0*delta(i, q)/y[i]
            )/k.powi(2);
        }
    }
    result
}

pub fn logprob(x: &[f64], psd: &[f64], tod: &[f64], ptr_mat: &CsMat<f64>, sp: f64)->f64{
    let s=smoothness(psd, sp);
    ln_likelihood(x, tod, psd, ptr_mat)+s
}

pub fn logprob_ana(x: &[f64], psp: &[f64], tod: &[f64], ptr_mat: &CsMat<f64>)->f64{
    assert_eq!(psp.len(), 4);
    let a=psp[0];
    let b=psp[1];
    let f0=psp[2];
    let alpha=psp[3];
    let ntod=tod.len() as isize;
    let fmin=1.0/(DT*ntod as f64);
    let psd=(0..(ntod+1)/2).chain(-ntod/2..0).map(|i| ps_model(i as f64*fmin, a, b, f0, alpha, PS_W, PS_E)+1e-9).collect::<Vec<_>>();
    if f0<fmin || f0>fmin*(ntod/2) as f64 || alpha< -3.0 || alpha >1.0{
        return -std::f64::INFINITY;
    }

    ln_likelihood(x, tod, &psd, ptr_mat)
}

pub fn logprob_grad(x: &[f64], psd: &[f64], tod: &[f64], ptr_mat: &CsMat<f64>, sp: f64)->(Vec<f64>, Vec<f64>){
    let (gx,gp1)=ln_likelihood_grad(x, tod, psd, ptr_mat);
    let gp2=d_smoothness(psd, sp);
    let gp:Vec<_>=gp1.iter().zip(gp2.iter()).map(|(a,b)| a+b).collect();
    (gx, gp)
    //(gx, gp1)
}

pub fn logprob_ana_grad(x: &[f64], psp: &[f64], tod: &[f64], ptr_mat: &CsMat<f64>)->(Vec<f64>, Vec<f64>){
    assert_eq!(psp.len(), 4);
    let a=psp[0];
    let b=psp[1];
    let f0=psp[2];
    let alpha=psp[3];

    let ntod=tod.len() as isize;
    let fmin=1.0/(DT*ntod as f64);
    let psd=(0..(ntod+1)/2).chain(-ntod/2..0).map(|i| ps_model(i as f64*fmin, a, b, f0, alpha, PS_W, PS_E)+1e-9).collect::<Vec<_>>();
    let dpsd_da=LsVec((0..(ntod+1)/2).chain(-ntod/2..0).map(|i| dps_model_da(i as f64*fmin, a, b, f0, alpha, PS_W, PS_E)).collect::<Vec<_>>());
    let dpsd_db=LsVec((0..(ntod+1)/2).chain(-ntod/2..0).map(|i| dps_model_db(i as f64*fmin, a, b, f0, alpha, PS_W, PS_E)).collect::<Vec<_>>());
    let dpsd_df0=LsVec((0..(ntod+1)/2).chain(-ntod/2..0).map(|i| dps_model_df0(i as f64*fmin, a, b, f0, alpha, PS_W, PS_E)).collect::<Vec<_>>());
    let dpsd_dalpha=LsVec((0..(ntod+1)/2).chain(-ntod/2..0).map(|i| dps_model_dalpha(i as f64*fmin, a, b, f0, alpha, PS_W, PS_E)).collect::<Vec<_>>());
    let (gx, gp)=ln_likelihood_grad(x, tod, &psd, ptr_mat);
    let gp=LsVec(gp);
    let g_ps_param=vec![
        gp.dot(&dpsd_da),
        gp.dot(&dpsd_db),
        gp.dot(&dpsd_df0),
        gp.dot(&dpsd_dalpha),
    ];
    (gx, g_ps_param)
}

pub fn ln_likelihood_grad(x: &[f64], y: &[f64], psd: &[f64], ptr_mat: &CsMat<f64>)->(Vec<f64>, Vec<f64>){
    let nout=y.len();
    let noise=&ArrayView1::from(y)-&sp_mul_a1(&ptr_mat, ArrayView1::from(x));
    let ps1=ps_mirror(psd, nout);
    let (dlnpdn, dlnpdp)=mvn_ln_pdf_grad(noise.as_slice().unwrap(), &ps1);

    let dlnpdx=(-sp_mul_a1(&ptr_mat.transpose_view(), ArrayView1::from(&dlnpdn))).to_vec();
    let dlnpdp=ps_mirror_t(&dlnpdp, psd.len());
    (dlnpdx, dlnpdp)
}

pub fn psd2cov<T>(x: &[T])->Vec<T>
where T:Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>
{
    let n=x.len();
    let mut x_c:Vec<_>=x.iter().map(|&x| Complex::<T>::from(x)).collect();
    let mut X=vec![Complex::<T>::zero();n];
    ifft(&mut x_c, &mut X);
    X.iter().map(|&x| x.re).collect()
}

pub fn cov2psd<T>(x: &[T])->Vec<T>
where T:Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>
{
    let n=x.len();
    let mut x_c:Vec<_>=x.iter().map(|&x| Complex::<T>::from(x)).collect();
    let mut X=vec![Complex::<T>::zero();n];
    fft(&mut x_c, &mut X);
    X.iter().map(|&x| x.re).collect()
}

pub fn psd2cov_mat<T>(x: &[T])->Array2<T>
where T:Float + FloatConst + NumAssign + std::fmt::Debug + FFTnum + From<u32>
{
    circulant_matrix(&psd2cov(x))
}

pub fn ps_mirror<T>(x: &[T], n_out: usize)->Vec<T>
where T:Copy
{
    let mut result=vec![x[0]; n_out];
    for i in 0..x.len(){
        result[i]=x[i];
    }
    for i in 0..(result.len()-x.len()){
        result[n_out-i-1]=x[1+i];
    }
    result
}

pub fn ps_mirror_t<T>(x: &[T], n_out:usize)->Vec<T>
where T: Float
{
    let mut result:Vec<T>=x.iter().cloned().take(n_out).collect();
    for i in 0..(x.len()-result.len()){
        result[i+1]=result[i+1]+x[x.len()-i-1];
    }
    result
}

#[cfg(test)]
mod tests {
    use num_traits::Float;
    use rand::{thread_rng, Rng};
    use rand_distr::StandardNormal;

    fn get_psd()->Vec<f64>{
        vec![1.0, 0.5, 0.3, 0.2, 0.2, 0.3, 0.5]
    }

    #[test]
    fn test_xsx() {
        let psd=get_psd();
        let n=psd.len();
        let cov_mat=super::psd2cov_mat(&psd);
        let mut x=ndarray::Array::from(vec![1.0; n]);
        let brute_force_answer=x.dot(&cov_mat.dot(&x));
        let smart_answer=super::ln_xsx(x.as_slice().unwrap(), &psd);
        assert!((smart_answer-brute_force_answer).abs()<1e-10);
    }

    #[test]
    fn test_dxsx_dx(){
        let psd=get_psd();
        let n=psd.len();
        let mut rng=thread_rng();
        let x:Vec<f64>=(0..n).map(|_|rng.sample(StandardNormal)).collect();
        let mut x=ndarray::Array1::from(x);
        let m=2;
        let mut x1=x.clone();
        let dx=0.00000001;
        x1[m]+=dx;
        let direct_g=(super::ln_xsx(x1.as_slice().unwrap(), &psd)-super::ln_xsx(x.as_slice().unwrap(), &psd))/dx;

        let half_g=super::dhalf_ln_xsx_dx(x.as_slice().unwrap(), &psd)[m];
        //println!("xx={}", (half_g*2.0-direct_g));
        //println!("a1={}", half_g*2.0);
        //println!("b1={}", direct_g);
        assert!((half_g*2.0-direct_g).abs()<1e-5);
    }

    #[test]
    fn test_dxsx_dp(){
        //println!("afdsfadsfsafs");
        let psd=get_psd();
        let n=psd.len();
        let mut rng=thread_rng();
        let x:Vec<f64>=(0..n).map(|_|rng.sample(StandardNormal)).collect();
        let x=ndarray::Array1::from(x);

        let m=1;
        let mut psd1=psd.clone();
        let dp=0.00000001;
        psd1[m]+=dp;
        //psd1[3]+=dp;
        //println!("{:?}", psd1);
        let direct_g=(super::ln_xsx(x.as_slice().unwrap(), &psd1)-super::ln_xsx(x.as_slice().unwrap(), &psd))/dp;

        let half_g=super::dhalf_ln_xsx_dp(x.as_slice().unwrap(), &psd)[m];
        //println!("half_g*2-direct_g={}", (half_g*2.0-direct_g));
        //println!("half_g*2={}", half_g*2.0);
        //println!("direct_g={}", direct_g);
        //println!("a={}",super::ln_xsx(x.as_slice().unwrap(), &psd));
        //println!("b={}",super::ln_xsx(x.as_slice().unwrap(), &psd1));
        assert!((half_g*2.0-direct_g).abs()<1e-5);
    }

    #[test]
    fn test_dlndet_dp(){
        let psd=get_psd();
        let n=psd.len();
        let m=2;
        let lndet1=psd.iter().map(|x| x.ln()).fold(0.0, |x,y|x+y);
        let mut psd1=psd.clone();
        let dp=0.0000001;
        psd1[m]+=dp;
        let lndet2=psd1.iter().map(|x| x.ln()).fold(0.0, |x,y|x+y);
        let half_dlndet=super::dhalf_lndet_dps(psd.as_slice())[m];
        let direct_dlndet=(lndet2-lndet1)/dp;
        //println!("a: {} {}", direct_dlndet, half_dlndet);
        assert!((half_dlndet*2.0-direct_dlndet)<1e-5);
    }

    #[test]
    fn test_dlp_dx(){
        let psd=get_psd();
        let n=psd.len();
        let mut x=ndarray::Array::from(vec![1.0; n]);
        let m=2;
        let delta=0.000001;
        let lp1=super::mvn_ln_pdf(x.as_slice().unwrap(), &psd);
        let mut x1=x.clone();
        x1[m]+=delta;
        let lp2=super::mvn_ln_pdf(x1.as_slice().unwrap(), &psd);
        //println!("{} ", (lp2-lp1)/delta);

        let (dydx, dydp)=super::mvn_ln_pdf_grad(x.as_slice().unwrap(), &psd);

        assert!(((lp2-lp1)/delta-dydx[m]).abs()<1e-5);
    }

    #[test]
    fn test_grad(){
        let mut rng=thread_rng();
        let psd=get_psd();
        //let psd=vec![2.0;5];
        let n=psd.len();
        let x:Vec<_>=(0..n).map(|_|rng.sample(StandardNormal)).collect();
        let x=ndarray::Array1::from(x);
    
        let m=2;
        let delta=0.00000001;
        let grad1=super::mvn_ln_pdf_grad(x.as_slice().unwrap(), &psd);
        let dlp_dx=super::dhalf_ln_xsx_dx(x.as_slice().unwrap(), &psd)
        .into_iter().map(|x|{-x}).collect::<Vec<_>>();
        let dlp_dp1=super::dhalf_ln_xsx_dp(x.as_slice().unwrap(), &psd)
        .into_iter().map(|x|{-x}).collect::<Vec<_>>();
        let dlp_dp2=super::dhalf_lndet_dps(&psd)
        .into_iter().map(|x|{-x}).collect::<Vec<_>>();
        for i in 0..n{
            println!("{} {} {} {}", grad1.0[i], grad1.1[i], dlp_dx[i], dlp_dp1[i]+dlp_dp2[i]);
            assert!((grad1.0[i]-dlp_dx[i]).abs()<1e-5);
            assert!((grad1.1[i]-dlp_dp1[i]-dlp_dp2[i]).abs()<1e-5);
        }
    }

    #[test]
    fn test_mvn_dlp_dp(){
        let mut rng=thread_rng();
        let psd=get_psd();
        //let psd=vec![2.0;5];
        let n=psd.len();
        let x:Vec<_>=(0..n).map(|_|rng.sample(StandardNormal)).collect();
        let x=ndarray::Array1::from(x);
    
        let m=2;
        let delta=0.00000001;
        let lp1a=super::ln_xsx(x.as_slice().unwrap(), &psd)/2.0;
        let lp1b=super::ln_det_sigma(&psd)/2.0;
        let lp1=super::mvn_ln_pdf(x.as_slice().unwrap(), &psd);

        let mut psd1=psd.clone();
        psd1[m]+=delta;    

        let lp2a=super::ln_xsx(x.as_slice().unwrap(), &psd1)/2.0;
        let lp2b=super::ln_det_sigma(&psd1)/2.0;
        
        let lp2=super::mvn_ln_pdf(x.as_slice().unwrap(), &psd1);


        println!("a: {} {}", (lp2a-lp1a)/delta, super::dhalf_ln_xsx_dp(x.as_slice().unwrap(), &psd)[m]);
        
        println!("b: {} {}", (lp2b-lp1b)/delta, super::dhalf_lndet_dps(&psd)[m]);

        let (gx, gp)=super::mvn_ln_pdf_grad(x.as_slice().unwrap(), &psd);
        println!("c: {} {}", (lp2-lp1)/delta, gp[m]);
        assert!(((lp2-lp1)/delta-gp[m]).abs()<1e-5);
    }

    #[test]
    fn test_mvn_dlp_dx(){
        let mut rng=thread_rng();
        let psd=get_psd();
        //let psd=vec![2.0;5];
        let n=psd.len();
        let x:Vec<_>=(0..n).map(|_|rng.sample(StandardNormal)).collect();
        let x=ndarray::Array1::from(x);
    
        let m=2;
        let delta=0.00000001;
        let lp1a=super::ln_xsx(x.as_slice().unwrap(), &psd)/2.0;
        let lp1=super::mvn_ln_pdf(x.as_slice().unwrap(), &psd);

        let mut x1=x.clone();
        x1[m]+=delta;    

        let lp2a=super::ln_xsx(x1.as_slice().unwrap(), &psd)/2.0;
        let lp2=super::mvn_ln_pdf(x1.as_slice().unwrap(), &psd);


        println!("a1: {} {}", (lp2a-lp1a)/delta, super::dhalf_ln_xsx_dx(x.as_slice().unwrap(), &psd)[m]);
        
        let (gx, gp)=super::mvn_ln_pdf_grad(x.as_slice().unwrap(), &psd);
        println!("c1: {} {}", (lp2-lp1)/delta, gx[m]);
        assert!(((lp2-lp1)/delta-gx[m]).abs()<1e-6);
    }

    #[test]
    fn test_mvn_grad(){
        let mut rng=thread_rng();
        let dx_std=0.0000000001;
        let noise:ndarray::Array1<f64>=ndarray::Array1::<f64>::zeros(1024).map(|_|rng.sample(StandardNormal));

        let psd=vec![1e0; noise.len()];

        let lp1=super::mvn_ln_pdf(noise.as_slice().unwrap(), &psd);
        for m in 0..100{
            let mut noise2=noise.clone();
            let dx:ndarray::Array1<f64>=noise.map(|_| {
                let d:f64=rng.sample(StandardNormal);
                d*dx_std}
            );
            let noise2=&noise+&dx;

            let dp: Vec<_>=psd.iter().map(|_|{
                let d: f64=rng.sample(StandardNormal);
                d*dx_std
            }).collect();

            let psd2=&ndarray::ArrayView1::from(&psd)+&ndarray::ArrayView1::from(&dp);
            let lp2=super::mvn_ln_pdf(noise2.as_slice().unwrap(), psd2.as_slice().unwrap());
            
            let (gx, gp)=super::mvn_ln_pdf_grad(noise.as_slice().unwrap(), &psd);
            let diff=ndarray::ArrayView1::from(&gx).dot(&dx)+ndarray::ArrayView1::from(&gp).dot(&ndarray::ArrayView1::from(&dp));
            assert!(lp2-lp1-diff<1e-7);
            println!("{} {} {}",lp2-lp1, diff,  lp2-lp1-diff);
            //println!("{} {} {}", (lp2-lp1)/dx, gx[m], ((lp2-lp1)/dx-gx[m]).abs());
        }
    }
}
