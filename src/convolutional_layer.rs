use crate::layer::{Layer, LayerBase, LearnableLayer};
use crate::errors::Error;
use crate::{activations, util};
use crate::initialization;

use serde::de::{Deserialize, Visitor};
use serde::ser::{Serialize, SerializeStruct};

#[derive(Clone)]
pub struct ConvolutionalLayer {
    pub(crate) stride: usize,
    pub(crate) kernel_size: usize,
    pub(crate) num_kernels: usize,

    pub(crate) dimension: (usize, usize, usize),

    pub(crate) volume: Vec<f32>,
    pub(crate) volume_gradients: Vec<f32>,
    
    pub(crate) bias_gradients: Vec<f32>,
    pub(crate) kernel_gradients: Vec<f32>,
    
    back_activated_volume: Vec<f32>,

    raw_volume: Vec<f32>,


    bias_velocity: Vec<f32>,
    kernel_velocity: Vec<f32>,

    zero_padding: usize,
    
    biases: Vec<f32>,
    kernel: Vec<f32>,

    input_depth: usize,
}

impl ConvolutionalLayer {
    pub fn new(zero_padding: usize, stride: usize, kernel_size: usize, dimension: (usize, usize, usize), input_depth: usize) -> Self {
        let (dimension_x, dimension_y, depth) = dimension;
        
        Self {
            stride,
            kernel_size,
            num_kernels: depth,

            dimension,

            volume: vec![0.0; dimension_x * dimension_y * depth],
            volume_gradients: vec![0.0; dimension_x * dimension_y * depth],

            back_activated_volume: vec![0.0; dimension_x * dimension_y * depth],

            bias_gradients: vec![0.0; depth],
            kernel_gradients: vec![0.0; kernel_size * kernel_size * input_depth * depth],

            bias_velocity: vec![0.0; depth],
            kernel_velocity: vec![0.0; kernel_size * kernel_size * input_depth * depth],
            
            raw_volume: vec![0.0; dimension_x * dimension_y * depth],

            zero_padding,
            
            biases: vec![0.0; depth],
            kernel: vec![0.0; kernel_size * kernel_size * input_depth * depth],
            
            input_depth,

        }
    }
    
    /// Data is packed in row major order and each depth is stored sequentially
    pub fn set_volume(&mut self, volume: &Vec<f32>) -> Result<(), Error> {
        if self.volume.len() != volume.len() { return Err(Error::DimensionMismatch) };

        self.volume.clear();
        self.volume.extend(volume);

        Ok(())
    }

    pub fn set_kernel(&mut self, kernel: Vec<f32>) -> Result<(), Error> {
        if self.kernel.len() != kernel.len() { return Err(Error::InvalidInput) };

        self.kernel.clear();
        self.kernel.extend(kernel);

        Ok(())
    }

    pub fn set_biases(&mut self, biases: Vec<f32>) -> Result<(), Error> {
        if self.biases.len() != biases.len() { return Err(Error::InvalidInput) };

        self.biases.clear();
        self.biases.extend(biases);

        Ok(())
    }

    pub fn apply_gradients(&mut self, learning_rate: f32, momentum: f32, weight_decay: f32) -> () {
        for i in 0..self.biases.len() {
            let vel = self.bias_velocity[i] * momentum + learning_rate * self.bias_gradients[i];
            self.bias_velocity[i] = vel;
            self.biases[i] -= vel;
        }

        for i in 0..self.kernel_gradients.len() {
            let gradient = self.kernel_gradients[i] + weight_decay * self.kernel[i];
            let vel = self.kernel_velocity[i] * momentum + learning_rate * gradient;
            self.kernel_velocity[i] = vel;

            self.kernel[i] -= vel;
        }
    }

    pub(crate) fn convolve(&mut self, input_dimension: (usize, usize, usize), volume: &Vec<f32>, zero_padding: usize) -> () {
        let (padded_input_x, padded_input_y) = (input_dimension.0 + zero_padding * 2, input_dimension.1 + zero_padding * 2);
        
        for k in 0..self.num_kernels {
            let mut o_x = 0;

            for x in (0..(padded_input_x - self.kernel_size + 1)).step_by(self.stride) {
                let mut o_y = 0;

                for y in (0..(padded_input_y - self.kernel_size + 1)).step_by(self.stride) {
                    let mut value: f32 = 0.0;

                    for z in 0..input_dimension.2 {
                        for kernel_y in 0..self.kernel_size {
                            for kernel_x in 0..self.kernel_size {
                                let index = util::query_zero_padded((x + kernel_x, y + kernel_y, z), input_dimension, zero_padding);

                                if let Some(ind) = index {
                                    value += volume[ind] * self.kernel[util::get_kernel_index((kernel_x, kernel_y, z, k), self.kernel_size, self.input_depth)];
                                }
                            }
                        }
                    }

                    let index = util::get_index((o_x, o_y, k), self.dimension);

                    let output =  value + self.biases[k];
                    self.raw_volume[index] = output;
                    self.volume[index] = output;

                    o_y += 1;
                }

                o_x += 1;
            }
        };
    }

    fn convolve_back(&mut self, input_dimension: (usize, usize, usize), volume: &Vec<f32>, volume_gradients: &mut Vec<f32>, zero_padding: usize) -> () {
        let (padded_input_x, padded_input_y) = (input_dimension.0 + zero_padding * 2, input_dimension.1 + zero_padding * 2);

        volume_gradients.fill(0.0);

        for k in 0..self.num_kernels {
            let mut o_x = 0;

            for x in (0..(padded_input_x - self.kernel_size + 1)).step_by(self.stride) {
                let mut o_y = 0;

                for y in (0..(padded_input_y - self.kernel_size + 1)).step_by(self.stride) {
                    let index = util::get_index((o_x, o_y, k), self.dimension);
                    let derivative = self.back_activated_volume[index];

                    // the derivative could be exactly zero if there are max pooling layers
                    if derivative == 0.0 {
                        o_y += 1;
                        continue;
                    }

                    for z in 0..input_dimension.2 {
                        for kernel_y in 0..self.kernel_size {
                            for kernel_x in 0..self.kernel_size {
                                let kernel_index = util::get_kernel_index((kernel_x, kernel_y, z, k), self.kernel_size, self.input_depth);
                                let input_volume_index = util::query_zero_padded((x + kernel_x, y + kernel_y, z), input_dimension, zero_padding);

                                if let Some(ind) = input_volume_index {
                                    self.kernel_gradients[kernel_index] += volume[ind] * derivative;
                                    volume_gradients[ind] += self.kernel[kernel_index] * derivative;
                                }
                            }
                        }
                    }
                    
                    self.bias_gradients[k] += derivative * input_dimension.2 as f32;
                    o_y += 1;
                }

                o_x += 1;
            }
        };
    }
}

impl LayerBase for ConvolutionalLayer {
    fn forward_propagate(&self, next_layer: &mut Layer) -> Result<(), Error> {
        match next_layer {
            Layer::Convolutional(layer) => {
                util::check_output_dimension(self.dimension,
                    layer.dimension,
                    self.zero_padding,
                    layer.num_kernels,
                    layer.kernel_size,
                    layer.stride
                )?;
                
                layer.convolve(self.dimension, &self.volume, self.zero_padding);
            }

            Layer::Pooling(layer) => {
                util::check_output_dimension(self.dimension,
                    layer.dimension,
                    0, // a pooling layer doesn't take padding into account
                    layer.dimension.2,
                    layer.kernel_size,
                    layer.stride
                )?;

                layer.convolve(self.dimension, &self.volume);
            }

            Layer::FullyConnected(layer) => {
                let dim = self.dimension;
                if dim.0 * dim.1 * dim.2 != layer.num_inputs { return Err(Error::DimensionMismatch) };

                layer.feed_forward(&self.volume);
            }
        }

        Ok(())
    }

    fn back_propagate(&mut self, previous_layer: &mut Layer) -> Result<(), Error> {
        match previous_layer {
            Layer::Convolutional(layer) => {
                util::check_output_dimension(layer.dimension,
                    self.dimension,
                    layer.zero_padding,
                    self.num_kernels,
                    self.kernel_size,
                    self.stride
                )?;

                self.convolve_back(layer.dimension, &layer.volume, &mut layer.volume_gradients, layer.zero_padding);
            }

            Layer::Pooling(layer) => {
                util::check_output_dimension(layer.dimension,
                    self.dimension,
                    layer.zero_padding,
                    self.num_kernels,
                    self.kernel_size,
                    self.stride
                )?;

                self.convolve_back(layer.dimension, &layer.volume, &mut layer.volume_gradients, layer.zero_padding);
            }

            _ => ()
        }

        Ok(())
    }
}

impl LearnableLayer for ConvolutionalLayer {
    fn initialize(&mut self, func: initialization::Initialization) -> () {
        let inputs =  self.input_depth * self.kernel_size * self.kernel_size;
        let outputs = self.num_kernels * self.kernel_size * self.kernel_size;
        
        initialization::eval(func, inputs, outputs, &mut self.kernel);
        initialization::eval(func, inputs, outputs, &mut self.biases);
    }

    fn activate(&mut self, func: activations::ActivationFunction) -> () {
        for i in 0..self.volume.len() {
            self.volume[i] = activations::eval(func, self.raw_volume[i]);
        }
    }

    fn back_activate(&mut self, func: activations::ActivationFunction) -> () {
        for i in 0..self.volume.len() {
            self.back_activated_volume[i] = activations::eval(func, self.raw_volume[i]) * self.volume_gradients[i];
        }
    }

    fn reset_gradients(&mut self) -> () {
        for i in 0..self.bias_gradients.len() {
            self.bias_gradients[i] = 0.0;
        }

        for i in 0..self.kernel_gradients.len() {
            self.kernel_gradients[i] = 0.0;
        }
    }
}

impl Serialize for ConvolutionalLayer {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("ConvolutionalLayer", 7)?;

        state.serialize_field("zero_padding", &self.zero_padding)?;
        state.serialize_field("stride", &self.stride)?;
        state.serialize_field("kernel_size", &self.kernel_size)?;
        state.serialize_field("dimension", &self.dimension)?;
        state.serialize_field("input_depth", &self.input_depth)?;

        state.serialize_field("kernel", &self.kernel)?;
        state.serialize_field("biases", &self.biases)?;
        
        state.end()
    }
}

impl<'de> Deserialize<'de> for ConvolutionalLayer {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_struct("ConvolutionalLayer", &["zero_padding", "stride", "kernel_size", "dimension", "input_depth", "kernel", "biases"], ConvolutionalLayerVisitor)
    }
}

struct ConvolutionalLayerVisitor;
impl<'de> Visitor<'de> for ConvolutionalLayerVisitor {
    type Value = ConvolutionalLayer;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a ConvolutionalLayer struct")
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
        where
            M: serde::de::MapAccess<'de>,
    {
        let mut zero_padding = None;
        let mut stride = None;
        let mut kernel_size = None;
        let mut dimension = None;
        let mut input_depth = None;

        let mut kernel = None;
        let mut biases = None;

        while let Some(key) = map.next_key::<&str>()? {
            match key {
                "zero_padding" => {
                    if zero_padding.is_some() { return Err(serde::de::Error::duplicate_field("zero_padding")); };

                    zero_padding = Some(map.next_value()?);
                },

                "stride" => {
                    if stride.is_some() { return Err(serde::de::Error::duplicate_field("stride")); };

                    stride = Some(map.next_value()?);
                },

                "kernel_size" => {
                    if kernel_size.is_some() { return Err(serde::de::Error::duplicate_field("kernel_size")); };

                    kernel_size = Some(map.next_value()?);
                },

                "dimension" => {
                    if dimension.is_some() { return Err(serde::de::Error::duplicate_field("dimension")); };

                    dimension = Some(map.next_value()?);
                },

                "input_depth" => {
                    if input_depth.is_some() { return Err(serde::de::Error::duplicate_field("input_depth")); };

                    input_depth = Some(map.next_value()?);
                },

                "kernel" => {
                    if kernel.is_some() { return Err(serde::de::Error::duplicate_field("kernel")); };

                    kernel = Some(map.next_value()?);
                },

                "biases" => {
                    if biases.is_some() { return Err(serde::de::Error::duplicate_field("biases")); };

                    biases = Some(map.next_value()?);
                },

                _ => return Err(serde::de::Error::unknown_field(key, &["zero_padding", "stride", "kernel_size", "dimension", "input_depth", "kernel", "biases"])),
            }
        }

        let zero_padding = zero_padding.ok_or_else(|| serde::de::Error::missing_field("zero_padding"))?;
        let stride = stride.ok_or_else(|| serde::de::Error::missing_field("stride"))?;
        let kernel_size = kernel_size.ok_or_else(|| serde::de::Error::missing_field("kernel_size"))?;
        let dimension = dimension.ok_or_else(|| serde::de::Error::missing_field("dimension"))?;
        let input_depth = input_depth.ok_or_else(|| serde::de::Error::missing_field("input_depth"))?;

        let kernel = kernel.ok_or_else(|| serde::de::Error::missing_field("kernel"))?;
        let biases = biases.ok_or_else(|| serde::de::Error::missing_field("biases"))?;

        let mut layer = ConvolutionalLayer::new(zero_padding, stride, kernel_size, dimension, input_depth);

        layer.kernel = kernel;
        layer.biases = biases;

        Ok(layer)
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        let zero_padding = seq.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
        let stride = seq.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(1, &self))?;
        let kernel_size = seq.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(2, &self))?;
        let dimension = seq.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(3, &self))?;
        let input_depth = seq.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(4, &self))?;

        let kernel = seq.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(5, &self))?;
        let biases = seq.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(6, &self))?;

        let mut layer = ConvolutionalLayer::new(zero_padding, stride, kernel_size, dimension, input_depth);
        
        layer.kernel = kernel;
        layer.biases = biases;

        Ok(layer)
    }
}