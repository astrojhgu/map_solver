use ndarray::Array2;
pub trait PsModel{
    fn value(&self, ft: &[f64], fch: &[f64], psp:&[f64])->Array2<f64>;
    fn grad(&self, ft: &[f64], fch: &[f64], psp: &[f64])->Vec<Array2<f64>>;
    fn priori(&self, ft: &[f64], fch: &[f64], psp: &[f64])->f64{
        0.0
    }
    fn nparams(&self)->usize;
}
