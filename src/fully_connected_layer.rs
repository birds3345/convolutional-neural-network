use crate::layer::{Layer, LayerBase, LearnableLayer};
use crate::errors::Error;
use crate::activations;
use crate::initialization;
use crate::nn_error;

use serde::de::{Deserialize, Visitor};
use serde::ser::{Serialize, SerializeStruct};

#[derive(Clone)]
pub struct FullyConnectedLayer {
    pub(crate) num_inputs: usize,
    pub(crate) weight_gradients: Vec<f32>,
    pub(crate) bias_gradients: Vec<f32>,
    
    num_neurons: usize,

    raw_values: Vec<f32>,
    back_activated_values: Vec<f32>,
    values: Vec<f32>,
    weights: Vec<f32>,
    biases: Vec<f32>,

    value_gradients: Vec<f32>,

    weight_velocity: Vec<f32>,
    bias_velocity: Vec<f32>,
}

impl FullyConnectedLayer {
    pub fn new(num_inputs: usize, num_neurons: usize) -> Self {
        Self {
            num_inputs,
            num_neurons,

            raw_values: vec![0.0; num_neurons],
            back_activated_values: vec![0.0; num_neurons],
            values: vec![0.0; num_neurons],
            weights: vec![0.0; num_inputs * num_neurons],
            biases: vec![0.0; num_neurons],

            value_gradients: vec![0.0; num_neurons],
            weight_gradients: vec![0.0; num_inputs * num_neurons],
            bias_gradients: vec![0.0; num_neurons],

            weight_velocity: vec![0.0; num_inputs * num_neurons],
            bias_velocity: vec![0.0; num_neurons],
        }
    }

    pub fn get_outputs(&self) -> Vec<f32> {
        self.values.clone()
    }

    pub fn get_error(&self, function_type: nn_error::ErrorFunction, expected: &Vec<f32>) -> Result<f32, Error> {
        if self.values.len() != expected.len() { return Err(Error::InvalidInput) };

        Ok(nn_error::eval(function_type, &self.values, expected))
    }

    pub fn calculate_output_gradients(&mut self, error_function_type: nn_error::ErrorFunction, expected: &Vec<f32>) -> Result<(), Error> {
        if self.values.len() != expected.len() { return Err(Error::InvalidInput) };

        for i in 0..expected.len() {
            self.value_gradients[i] = nn_error::eval_derivative(error_function_type, i, &self.values, expected);
        }

        Ok(())
    }

    pub fn apply_gradients(&mut self, learning_rate: f32, momentum: f32, weight_decay: f32) -> () {
        for i in 0..self.num_neurons {
            let vel = self.bias_velocity[i] * momentum + learning_rate * self.bias_gradients[i];
            self.bias_velocity[i] = vel;
            self.biases[i] -= vel;
        }

        for i in 0..(self.num_inputs * self.num_neurons) {
            let gradient = self.weight_gradients[i] + weight_decay * self.weights[i];
            let vel = self.weight_velocity[i] * momentum + learning_rate * gradient;
            self.weight_velocity[i] = vel;

            self.weights[i] -= vel;
        }
    }

    #[inline(always)]
    fn get_weight(&self, input: usize, neuron: usize) -> usize {
        neuron * self.num_inputs + input
    }

    pub(crate) fn feed_forward(&mut self, input: &Vec<f32>) -> () {
        for i in 0..self.num_neurons {
            let mut value = self.biases[i];

            for j in 0..self.num_inputs {
                value += input[j] * self.weights[self.get_weight(j, i)];
            }

            self.values[i] = value;
            self.raw_values[i] = value;
        }
    }

    pub(crate) fn feed_back(&mut self, input: &Vec<f32>, input_gradients: &mut Vec<f32>) -> () {
        input_gradients.fill(0.0);

        for i in 0..self.num_neurons {
            let derivative = self.back_activated_values[i];
            self.bias_gradients[i] += derivative;

            for j in 0..self.num_inputs {
                let index = self.get_weight(j, i);
                self.weight_gradients[index] += derivative * input[j];

                input_gradients[j] += derivative * self.weights[index];
            }
        }
    }
}

impl LearnableLayer for FullyConnectedLayer {
    fn activate(&mut self, func: activations::ActivationFunction) -> () {
        for i in 0..self.values.len() {
            self.values[i] = activations::eval(func, self.raw_values[i]);
        }
    }

    fn back_activate(&mut self, func: activations::ActivationFunction) {
        for i in 0..self.values.len() {
            self.back_activated_values[i] = activations::eval_derivative(func, self.raw_values[i]) * self.value_gradients[i];
        }
    }

    fn initialize(&mut self, func: initialization::Initialization) -> () {
        initialization::eval(func, self.num_inputs, self.num_neurons, &mut self.weights);
        initialization::eval(func, self.num_inputs, self.num_neurons, &mut self.biases);
    }

    fn reset_gradients(&mut self) -> () {
        for i in 0..self.bias_gradients.len() {
            self.bias_gradients[i] = 0.0;
        }

        for i in 0..self.weight_gradients.len() {
            self.weight_gradients[i] = 0.0;
        }
    }
}

impl LayerBase for FullyConnectedLayer {
    fn forward_propagate(&self, next_layer: &mut Layer) -> Result<(), Error> {
        match next_layer {
            Layer::FullyConnected(layer) => {
                if layer.num_inputs != self.num_neurons { return Err(Error::DimensionMismatch) }

                layer.feed_forward(&self.values);
            }

            _ => { return Err(Error::IncompatibleLayers) }
        }

        Ok(())
    }

    fn back_propagate(&mut self, previous_layer: &mut Layer) -> Result<(), Error> {
        match previous_layer {
            Layer::FullyConnected(layer) => {
                if layer.num_neurons != self.num_inputs { return Err(Error::DimensionMismatch) };

                self.feed_back(&layer.values, &mut layer.value_gradients);
            }

            Layer::Convolutional(layer) => {
                let dim = layer.dimension;
                if dim.0 * dim.1 * dim.2 != self.num_inputs { return Err(Error::DimensionMismatch) };

                self.feed_back(&layer.volume, &mut layer.volume_gradients);
            }
            
            Layer::Pooling(layer) => {
                let dim = layer.dimension;
                if dim.0 * dim.1 * dim.2 != self.num_inputs { return Err(Error::DimensionMismatch) };

                self.feed_back(&layer.volume, &mut layer.volume_gradients);
            }
        }

        Ok(())
    }
}

impl Serialize for FullyConnectedLayer {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("FullyConnectedLayer", 4)?;

        state.serialize_field("num_inputs", &self.num_inputs)?;
        state.serialize_field("num_neurons", &self.num_neurons)?;

        state.serialize_field("weights", &self.weights)?;
        state.serialize_field("biases", &self.biases)?;
        
        state.end()
    }
}

impl<'de> Deserialize<'de> for FullyConnectedLayer {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_struct("FullyConnectedLayer", &["num_inputs", "num_neurons", "weights", "biases"], FullyConnectedLayerVisitor)
    }
}

struct FullyConnectedLayerVisitor;
impl<'de> Visitor<'de> for FullyConnectedLayerVisitor {
    type Value = FullyConnectedLayer;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a FullyConnectedLayer struct")
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
        where
            M: serde::de::MapAccess<'de>,
    {
        let mut num_inputs = None;
        let mut num_neurons = None;

        let mut weights = None;
        let mut biases = None;

        while let Some(key) = map.next_key::<&str>()? {
            match key {
                "num_inputs" => {
                    if num_inputs.is_some() { return Err(serde::de::Error::duplicate_field("num_inputs")); };

                    num_inputs = Some(map.next_value()?);
                },

                "num_neurons" => {
                    if num_neurons.is_some() { return Err(serde::de::Error::duplicate_field("num_neurons")); };

                    num_neurons = Some(map.next_value()?);
                }

                "weights" => {
                    if weights.is_some() { return Err(serde::de::Error::duplicate_field("weights")); };

                    weights = Some(map.next_value()?);
                }

                "biases" => {
                    if biases.is_some() { return Err(serde::de::Error::duplicate_field("biases")); };

                    biases = Some(map.next_value()?);
                }

                _ => return Err(serde::de::Error::unknown_field(key, &["num_inputs", "num_neurons"])),
            }
        }

        let num_inputs = num_inputs.ok_or_else(|| serde::de::Error::missing_field("num_inputs"))?;
        let num_neurons = num_neurons.ok_or_else(|| serde::de::Error::missing_field("num_neurons"))?;

        let weights = weights.ok_or_else(|| serde::de::Error::missing_field("weights"))?;
        let biases = biases.ok_or_else(|| serde::de::Error::missing_field("biases"))?;

        let mut layer = FullyConnectedLayer::new(num_inputs, num_neurons);

        layer.weights = weights;
        layer.biases = biases;

        Ok(layer)
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        let num_inputs = seq.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
        let num_neurons = seq.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(1, &self))?;

        let weights = seq.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(2, &self))?;
        let biases = seq.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(3, &self))?;

        let mut layer = FullyConnectedLayer::new(num_inputs, num_neurons);
        
        layer.weights = weights;
        layer.biases = biases;

        Ok(layer)
    }
}