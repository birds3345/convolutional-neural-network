use serde::{Serialize, Deserialize};

#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum ErrorFunction {
    HalfMeanSquaredError,
    BinaryCrossEntropy,
}

pub fn eval(function_type: ErrorFunction, values: &Vec<f32>, expected: &Vec<f32>) -> f32 {
    match function_type {
        ErrorFunction::HalfMeanSquaredError => half_mean_squared(values, expected),
        ErrorFunction::BinaryCrossEntropy => binary_cross_entropy(values, expected),
    }
}

pub fn eval_derivative(function_type: ErrorFunction, i: usize, values: &Vec<f32>, expected: &Vec<f32>) -> f32 {
    match function_type {
        ErrorFunction::HalfMeanSquaredError => half_mean_squared_derivative(i, values, expected),
        ErrorFunction::BinaryCrossEntropy => binary_cross_entropy_derivative(i, values, expected),
    }
}

fn half_mean_squared(values: &Vec<f32>, expected: &Vec<f32>) -> f32 {
    let mut result: f32 = 0.0;

    for i in 0..values.len() {
        let diff = values[i] - expected[i];
        result += diff * diff;
    }

    result / values.len() as f32 * 0.5
}

fn binary_cross_entropy(values: &Vec<f32>, expected: &Vec<f32>) -> f32 {
    let mut result: f32 = 0.0;

    for i in 0..values.len() {
        let clamped_value = values[i].clamp(1e-12, 1.0 - 1e-12);
        result += expected[i] * clamped_value.ln() + (1.0 - expected[i]) * (1.0 - clamped_value).ln();
    }

    -result / values.len() as f32
}


fn half_mean_squared_derivative(i: usize, values: &Vec<f32>, expected: &Vec<f32>) -> f32 {
    (values[i] - expected[i]) / values.len() as f32
}

fn binary_cross_entropy_derivative(i: usize, values: &Vec<f32>, expected: &Vec<f32>) -> f32 {
    let clamped_value = values[i].clamp(1e-12, 1.0 - 1e-12);
    -(expected[i] / clamped_value - (1.0 - expected[i]) / (1.0 - clamped_value)) / values.len() as f32
}