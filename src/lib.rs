pub use initialization::Initialization;
pub use activations::ActivationFunction;
pub use nn_error::ErrorFunction;

pub use pooling_layer::PoolingType;
pub use layer::Layer;

pub use neural_network::NeuralNetwork;

pub use errors::Error;


pub mod util;
pub mod errors;
pub mod initialization;
pub mod activations;

mod neural_network;
mod layer;
mod convolutional_layer;
mod fully_connected_layer;
mod pooling_layer;

mod nn_error;

#[cfg(test)]
mod tests;