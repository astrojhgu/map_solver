use ndarray::{Array1,Array2};
pub trait PsModel{
    fn value(&self, ft: &[f64], fch: &[f64], psp:&[f64])->Array2<f64>;
    fn grad(&self, ft: &[f64], fch: &[f64], psp: &[f64])->Vec<Array2<f64>>;
    fn support(&self, ft: &[f64], fch: &[f64], psp: &[f64])->bool{
        true
    }
    fn boundaries(&self, ft:&[f64], fch: &[f64])->(Array1<f64>, Array1<f64>);
    fn nparams(&self)->usize;
}
