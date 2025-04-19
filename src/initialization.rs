use rand::{Rng, distr::Uniform};
use rand_distr::Normal;

#[derive(Clone, Copy)]
pub enum Initialization {
    UniformXavier,
    UniformHe,

    NormalXavier,
    NormalHe,
}

pub fn eval(function_type: Initialization, inputs: usize, outputs: usize, vec: &mut Vec<f32>) {
    match function_type {
        Initialization::UniformXavier => uniform_xavier_initialization(inputs, outputs, vec),
        Initialization::UniformHe => uniform_he_initialization(inputs, vec),
        
        Initialization::NormalXavier => normal_xavier_initialization(inputs, outputs, vec),
        Initialization::NormalHe => normal_he_initialization(inputs, vec),
    }
}

fn uniform_xavier_initialization(inputs: usize, outputs: usize, vec: &mut Vec<f32>) {
    let bound = (6.0 / (inputs as f32 + outputs as f32)).sqrt();
    
    let uniform = Uniform::new(-bound, bound).unwrap();
    let mut rng = rand::rng();
    
    for i in 0..vec.len() {
        vec[i] = rng.sample(&uniform);
    }
}

fn normal_xavier_initialization(inputs: usize, outputs: usize, vec: &mut Vec<f32>) {
    let bound = (2.0 / (inputs as f32 + outputs as f32)).sqrt();

    let normal = Normal::new(0.0, bound).unwrap();
    let mut rng = rand::rng();

    for i in 0..vec.len() {
        vec[i] = rng.sample(&normal);
    }
}

fn uniform_he_initialization(inputs: usize, vec: &mut Vec<f32>) {
    let bound = (6.0 / inputs as f32).sqrt();

    let uniform = Uniform::new(-bound, bound).unwrap();
    let mut rng = rand::rng();
    
    for i in 0..vec.len() {
        vec[i] = rng.sample(&uniform);
    }
}

fn normal_he_initialization(inputs: usize, vec: &mut Vec<f32>) {
    let bound = (2.0 / inputs as f32).sqrt();

    let normal = Normal::new(0.0, bound).unwrap();
    let mut rng = rand::rng();

    for i in 0..vec.len() {
        vec[i] = rng.sample(&normal);
    }
}