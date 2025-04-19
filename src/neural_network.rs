use crate::{ActivationFunction, Error, ErrorFunction, Initialization, Layer};

use serde::{Serialize, Deserialize, de::Visitor, ser::SerializeStruct};

#[derive(Clone)]
pub struct NeuralNetwork {
    layers: Vec<(Layer, ActivationFunction)>,
    error_function: ErrorFunction,
}

impl NeuralNetwork {
    pub fn new(error_function: ErrorFunction) -> Self {
        Self {
            layers: Vec::new(),
            error_function,
        }
    }

    pub fn set_input(&mut self, input: &Vec<f32>) -> Result<(), Error> {
        if self.layers.len() == 0 { return Err(Error::IncompatibleLayers) };

        match &mut self.layers[0] {
            (Layer::Convolutional(layer), _) => layer.set_volume(input),
            _ => Err(Error::IncompatibleLayers),
        }
    }

    pub fn forward_propagate(&mut self) -> Result<(), Error> {
        for i in 0..(self.layers.len() - 1) {
            let (slice1, slice2) = self.layers.split_at_mut(i + 1);

            slice1[i].0.forward_propagate(&mut slice2[0].0)?;
            slice2[0].0.activate(slice2[0].1);
        };

        Ok(())
    }

    pub fn back_propagate(&mut self, target_output: &Vec<f32>) -> Result<(), Error> {
        let last = self.layers.len() - 1;
        let (Layer::FullyConnected(_), _) = self.layers[last] else { return Err(Error::IncompatibleLayers) };

        if let (Layer::FullyConnected(ref mut layer), _) = self.layers[last] {
            layer.calculate_output_gradients(self.error_function, target_output)?;
        }

        for i in (1..self.layers.len()).rev() {
            let (slice1, slice2) = self.layers.split_at_mut(i);

            slice2[0].0.backward_activate(slice2[0].1);
            slice2[0].0.back_propagate(&mut slice1[i - 1].0)?;
        }

        Ok(())
    }

    /// starts a new batch and resets gradients
    pub fn start_batch(&mut self) -> () {
        for (layer, _) in &mut self.layers {
            layer.reset_gradients();
        }
    }

    /// ends the batch and applies the gradients
    pub fn end_batch(&mut self, sample_count: u8, learning_rate: f32, momentum: f32, weight_decay: f32) -> () {
        let new_learning_rate = learning_rate / sample_count as f32;

        for i in 1..self.layers.len() {
            self.layers[i].0.apply_gradients(new_learning_rate, momentum, weight_decay);
        }
    }

    pub fn get_error(&self, target_output: &Vec<f32>) -> Result<f32, Error> {
        let last = self.layers.len() - 1;
        if let (Layer::FullyConnected(ref layer), _) = self.layers[last] {
            return layer.get_error(self.error_function, target_output);
        };

        Err(Error::InvalidInput)
    }

    pub fn get_output(&self) -> Result<Vec<f32>, Error> {
        let last = self.layers.len() - 1;
        if let (Layer::FullyConnected(ref layer), _) = self.layers[last] {
            return Ok(layer.get_outputs());
        };

        Err(Error::InvalidInput)
    }

    pub fn initialize(&mut self, layer_index: usize, initialization_function: Initialization) -> Result<(), Error> {
        if layer_index >= self.layers.len() { return Err(Error::InvalidInput) };

        self.layers[layer_index].0.initialize(initialization_function);
        Ok(())
    }

    pub fn register_layer(&mut self, activation_function: ActivationFunction, layer: Layer) -> () {
        self.layers.push((layer, activation_function));
    }

    pub fn collect_gradients_mut(&mut self) -> Vec<&mut f32> {
        let mut result = Vec::new();

        for (layer, _) in &mut self.layers {
            match layer {
                Layer::Convolutional(layer) => {
                    result.extend(layer.kernel_gradients.iter_mut());
                    result.extend(layer.bias_gradients.iter_mut());
                },

                Layer::FullyConnected(layer) => {
                    result.extend(layer.weight_gradients.iter_mut());
                    result.extend(layer.bias_gradients.iter_mut());
                }

                _ => (),
            }
        }

        return result;
    }

    pub fn collect_gradients(&self) -> Vec<f32> {
        let mut result = Vec::new();

        for (layer, _) in &self.layers {
            match layer {
                Layer::Convolutional(layer) => {
                    result.extend(layer.kernel_gradients.iter());
                    result.extend(layer.bias_gradients.iter());
                },

                Layer::FullyConnected(layer) => {
                    result.extend(layer.weight_gradients.iter());
                    result.extend(layer.bias_gradients.iter());
                }

                _ => (),
            }
        }

        return result;
    }
}

impl Serialize for NeuralNetwork {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("NeuralNetwork", 2)?;
        
        state.serialize_field("layers", &self.layers)?;
        state.serialize_field("error_function", &self.error_function)?;

        state.end()
    }
}

impl<'de> Deserialize<'de> for NeuralNetwork {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_struct("NeuralNetwork", &["layers", "error_function"], NeuralNetworkVisitor)
    }
}

struct NeuralNetworkVisitor;
impl<'de> Visitor<'de> for NeuralNetworkVisitor {
    type Value = NeuralNetwork;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a NeuralNetwork struct")
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
        where
            M: serde::de::MapAccess<'de>,
    {
        let mut layers = None;
        let mut error_function = None;
        
        while let Some(key) = map.next_key::<&str>()? {
            match key {
                "layers" => {
                    if layers.is_some() { return Err(serde::de::Error::duplicate_field("layers")); };

                    layers = Some(map.next_value()?);
                },

                "error_function" => {
                    if error_function.is_some() { return Err(serde::de::Error::duplicate_field("error_function")); };

                    error_function = Some(map.next_value()?);
                }

                _ => return Err(serde::de::Error::unknown_field(key, &["layers", "error_function"])),
            }
        }

        let layers = layers.ok_or_else(|| serde::de::Error::missing_field("layers"))?;
        let error_function = error_function.ok_or_else(|| serde::de::Error::missing_field("error_function"))?;

        let mut neural_network = NeuralNetwork::new(error_function);
        neural_network.layers = layers;

        Ok(neural_network)
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        let layers = seq.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
        let error_function = seq.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(1, &self))?;

        let mut neural_network = NeuralNetwork::new(error_function);
        neural_network.layers = layers;

        Ok(neural_network)
    }
}