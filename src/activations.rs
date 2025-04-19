use serde::{Serialize, Deserialize};

#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum ActivationFunction {
    Sigmoid,
    ReLU,
    LeakyReLU(f32),

    None,
}

pub fn eval(function_type: ActivationFunction, x: f32) -> f32 {
    match function_type {
        ActivationFunction::Sigmoid => sigmoid(x),
        ActivationFunction::ReLU => relu(x),
        ActivationFunction::LeakyReLU(slope) => leaky_relu(x, slope),

        ActivationFunction::None => x,
    }
}

pub fn eval_derivative(function_type: ActivationFunction, x: f32) -> f32 {
    match function_type {
        ActivationFunction::Sigmoid => sigmoid_derivative(x),
        ActivationFunction::ReLU => relu_derivative(x),
        ActivationFunction::LeakyReLU(slope) => leaky_relu_derivative(x, slope),

        ActivationFunction::None => 1.0,
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

fn leaky_relu(x: f32, slope: f32) -> f32 {
    x.max(x * slope)
}

fn sigmoid_derivative(x: f32) -> f32 {
    let res = sigmoid(x);

    res * (1.0 - res)
}

fn relu_derivative(x: f32) -> f32 {
    if x <= 0.0 {
        0.0
    } else {
        1.0
    }
}

fn leaky_relu_derivative(x: f32, slope: f32) -> f32 {
    if x < 0.0 {
        slope
    } else {
        1.0
    }
}