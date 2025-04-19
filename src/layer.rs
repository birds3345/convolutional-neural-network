use crate::errors::Error;

use crate::convolutional_layer::ConvolutionalLayer;
use crate::fully_connected_layer::FullyConnectedLayer;
use crate::pooling_layer::{PoolingLayer, PoolingType};

use crate::initialization;
use crate::activations;

use serde::{Serialize, Deserialize};

pub(crate) trait LayerBase {
    fn forward_propagate(&self, next_layer: &mut Layer) -> Result<(), Error>;
    fn back_propagate(&mut self, previous_layer: &mut Layer) -> Result<(), Error>;
}

pub trait LearnableLayer: LayerBase {
    /// Initializes the weights and biases
    fn initialize(&mut self, func: initialization::Initialization) -> ();

    fn reset_gradients(&mut self) -> ();

    fn activate(&mut self, func: activations::ActivationFunction) -> ();
    fn back_activate(&mut self, func: activations::ActivationFunction) -> ();
}

#[derive(Clone, Serialize, Deserialize)]
pub enum Layer {
    Convolutional(ConvolutionalLayer),
    Pooling(PoolingLayer),
    FullyConnected(FullyConnectedLayer),
}

impl Layer {
    pub fn make_convolutional_layer(zero_padding: usize, stride: usize, kernel_size: usize, dimension: (usize, usize, usize), input_depth: usize) -> Layer {
        Layer::Convolutional(ConvolutionalLayer::new(zero_padding, stride, kernel_size, dimension, input_depth))
    }

    pub fn make_pooling_layer(pooling_type: PoolingType, zero_padding: usize, stride: usize, kernel_size: usize, dimension: (usize, usize, usize)) -> Layer {
        Layer::Pooling(PoolingLayer::new(pooling_type, zero_padding, stride, kernel_size, dimension))
    }

    pub fn make_fully_connected_layer(num_inputs: usize, num_neurons: usize) -> Layer {
        Layer::FullyConnected(FullyConnectedLayer::new(num_inputs, num_neurons))
    }
    // TODO: make this a separate layer for less memory consumption
    pub fn make_input_layer(zero_padding: usize, dimension: (usize, usize, usize)) -> Layer {
        Self::make_convolutional_layer(zero_padding, 0, 0, dimension, 0)
    }

    pub fn forward_propagate(&self, next_layer: &mut Layer) -> Result<(), Error> {
        match self {
            Layer::Convolutional(layer) => layer.forward_propagate(next_layer),
            Layer::Pooling(layer) => layer.forward_propagate(next_layer),
            Layer::FullyConnected(layer) => layer.forward_propagate(next_layer),
        }
    }

    pub fn back_propagate(&mut self, previous_layer: &mut Layer) -> Result<(), Error> {
        match self {
            Layer::Convolutional(layer) => layer.back_propagate(previous_layer),
            Layer::Pooling(layer) => layer.back_propagate(previous_layer),
            Layer::FullyConnected(layer) => layer.back_propagate(previous_layer),
        }
    }

    pub fn apply_gradients(&mut self, learning_rate: f32, momentum: f32, weight_decay: f32) -> () {
        match self {
            Layer::Convolutional(layer) => layer.apply_gradients(learning_rate, momentum, weight_decay),
            Layer::FullyConnected(layer) => layer.apply_gradients(learning_rate, momentum, weight_decay),

            _ => (),
        }
    }

    pub fn reset_gradients(&mut self) -> () {
        match self {
            Layer::Convolutional(layer) => layer.reset_gradients(),
            Layer::FullyConnected(layer) => layer.reset_gradients(),

            _ => (),
        }
    }

    pub fn activate(&mut self, func: activations::ActivationFunction) -> () {
        match self {
            Layer::Convolutional(layer) => layer.activate(func),
            Layer::FullyConnected(layer) => layer.activate(func),

            _ => (),
        }
    }

    pub fn backward_activate(&mut self, func: activations::ActivationFunction) -> () {
        match self {
            Layer::Convolutional(layer) => layer.back_activate(func),
            Layer::FullyConnected(layer) => layer.back_activate(func),

            _ => (),
        }
    }

    pub fn initialize(&mut self, func: initialization::Initialization) -> () {
        match self {
            Layer::Convolutional(layer) => layer.initialize(func),
            Layer::FullyConnected(layer) => layer.initialize(func),

            _ => (),
        }
    }
}