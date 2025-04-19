use crate::layer::{Layer, LayerBase};
use crate::errors::Error;
use crate::util;

use serde::{Serialize, Deserialize, de::Visitor, ser::SerializeStruct};

#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum PoolingType {
    Max,
    Average,
}

#[derive(Clone)]
pub struct PoolingLayer {
    pub(crate) zero_padding: usize,
    pub(crate) stride: usize,
    pub(crate) kernel_size: usize,
    
    pub(crate) dimension: (usize, usize, usize),

    pub(crate) volume: Vec<f32>,
    pub(crate) volume_gradients: Vec<f32>,

    pooling_type: PoolingType,
}

impl PoolingLayer {
    pub fn new(pooling_type: PoolingType, zero_padding: usize, stride: usize, kernel_size: usize, dimension: (usize, usize, usize)) -> Self {
        Self {
            pooling_type,

            zero_padding,
            stride,
            kernel_size,
            
            dimension,
            volume: vec![0.0; dimension.0 * dimension.1 * dimension.2],
            volume_gradients: vec![0.0; dimension.0 * dimension.1 * dimension.2],
        }
    }

    pub(crate) fn convolve(&mut self, input_dimension: (usize, usize, usize), volume: &Vec<f32>) -> () {
        let mut o_x = 0;

        let kernel_volume = 1.0 / (self.kernel_size as f32 * self.kernel_size as f32);

        // TODO: use zero padding?
        for x in (0..input_dimension.0 - self.kernel_size + 1).step_by(self.stride) {
            let mut o_y = 0;

            for y in (0..input_dimension.1 - self.kernel_size + 1).step_by(self.stride) {
                for z in 0..input_dimension.2 {
                    let mut value: f32 = 0.0;

                    match self.pooling_type {
                        PoolingType::Max => {
                            for kernel_y in 0..self.kernel_size {
                                for kernel_x in 0..self.kernel_size {
                                    let val = volume[util::get_index((x + kernel_x, y + kernel_y, z), input_dimension)];
                                    value = value.max(val);
                                }
                            }
                        },

                        PoolingType::Average => {
                            for kernel_y in 0..self.kernel_size {
                                for kernel_x in 0..self.kernel_size {
                                    let val = volume[util::get_index((x + kernel_x, y + kernel_y, z), input_dimension)];
                                    value += val;
                                }
                            }

                            value *= kernel_volume;
                        }
                    }

                    self.volume[util::get_index((o_x, o_y, z), self.dimension)] = value;
                }

                o_y += 1;
            }

            o_x += 1;
        }
    }

    fn convolve_back(&mut self, input_dimension: (usize, usize, usize), volume: &Vec<f32>, volume_gradients: &mut Vec<f32>) {
        volume_gradients.fill(0.0);

        let kernel_volume = 1.0 / (self.kernel_size as f32 * self.kernel_size as f32);

        let mut o_x = 0;

        for x in (0..input_dimension.0 - self.kernel_size + 1).step_by(self.stride) {
            let mut o_y = 0;

            for y in (0..input_dimension.1 - self.kernel_size + 1).step_by(self.stride) {
                for z in 0..input_dimension.2 {
                    let output_index = util::get_index((o_x, o_y, z), self.dimension);
                    
                    match self.pooling_type {
                        PoolingType::Max => {
                            let mut max_index = util::get_index((x, y, z), input_dimension);
                            let mut max_value = volume[max_index];

                            for kernel_y in 0..self.kernel_size {
                                for kernel_x in 0..self.kernel_size {
                                    let index = util::get_index((x + kernel_x, y + kernel_y, z), input_dimension);

                                    let val = volume[index];

                                    if val > max_value {
                                        max_index = index;
                                        max_value = val;
                                    }
                                }
                            }

                            volume_gradients[max_index] += self.volume_gradients[output_index];
                        },

                        PoolingType::Average => {
                            for kernel_y in 0..self.kernel_size {
                                for kernel_x in 0..self.kernel_size {
                                    let index = util::get_index((x + kernel_x, y + kernel_y, z), input_dimension);

                                    volume_gradients[index] += self.volume_gradients[output_index] * kernel_volume;
                                }
                            }
                        },
                    }
                }

                o_y += 1;
            }

            o_x += 1;
        }
    }
}

impl LayerBase for PoolingLayer {
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
                    0,
                    self.dimension.2,
                    self.kernel_size,
                    self.stride
                )?;

                self.convolve_back(layer.dimension, &layer.volume, &mut layer.volume_gradients);
            }

            Layer::Pooling(layer) => {
                util::check_output_dimension(layer.dimension,
                    self.dimension,
                    0,
                    self.dimension.2,
                    self.kernel_size,
                    self.stride
                )?;

                self.convolve_back(layer.dimension, &layer.volume, &mut layer.volume_gradients);
            }

            _ => (),
        }

        Ok(())
    }
}

impl Serialize for PoolingLayer {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("PoolingLayer", 5)?;

        state.serialize_field("pooling_type", &self.pooling_type)?;
        state.serialize_field("zero_padding", &self.zero_padding)?;
        state.serialize_field("stride", &self.stride)?;
        state.serialize_field("kernel_size", &self.kernel_size)?;
        state.serialize_field("dimension", &self.dimension)?;

        state.end()
    }
}

impl<'de> Deserialize<'de> for PoolingLayer {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_struct("PoolingLayer", &["pooling_type", "zero_padding", "stride", "kernel_size", "dimension"], PoolingLayerVisitor)
    }
}

struct PoolingLayerVisitor;
impl<'de> Visitor<'de> for PoolingLayerVisitor {
    type Value = PoolingLayer;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a PoolingLayer struct")
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
        where
            M: serde::de::MapAccess<'de>,
    {
        let mut pooling_type = None;
        let mut zero_padding = None;
        let mut stride = None;
        let mut kernel_size = None;
        let mut dimension = None;
        
        while let Some(key) = map.next_key::<&str>()? {
            match key {
                "pooling_type" => {
                    if pooling_type.is_some() { return Err(serde::de::Error::duplicate_field("pooling_type")); };

                    pooling_type = Some(map.next_value()?);
                },

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

                _ => return Err(serde::de::Error::unknown_field(key, &["zero_padding", "stride", "kernel_size", "dimension", "input_depth", "kernel", "biases"])),
            }
        }

        let pooling_type = pooling_type.ok_or_else(|| serde::de::Error::missing_field("pooling_type"))?;
        let zero_padding = zero_padding.ok_or_else(|| serde::de::Error::missing_field("zero_padding"))?;
        let stride = stride.ok_or_else(|| serde::de::Error::missing_field("stride"))?;
        let kernel_size = kernel_size.ok_or_else(|| serde::de::Error::missing_field("kernel_size"))?;
        let dimension = dimension.ok_or_else(|| serde::de::Error::missing_field("dimension"))?;

        Ok(PoolingLayer::new(pooling_type, zero_padding, stride, kernel_size, dimension))
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        let pooling_type = seq.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
        let zero_padding = seq.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(1, &self))?;
        let stride = seq.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(2, &self))?;
        let kernel_size = seq.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(3, &self))?;
        let dimension = seq.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(4, &self))?;

        Ok(PoolingLayer::new(
            pooling_type,
            zero_padding,
            stride,
            kernel_size,
            dimension,
        ))
    }
}