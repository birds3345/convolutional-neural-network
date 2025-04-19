use std::{fs, str::FromStr};

use image::{imageops::FilterType, GenericImageView};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct TrainingData {
    pub data: Vec<u8>,
    pub classification: String,
}

pub fn compile_images(directory_path: &str, classification: &str, output: &mut Vec<TrainingData>) -> Result<(), std::io::Error> {
    for entry in fs::read_dir(directory_path)? {
        let entry = entry.unwrap();
        let path = entry.path();

        if path.is_file() {
            if let Ok(img) = image::open(&path) {
                let resized_img = img.resize_exact(128, 128, FilterType::Triangle);
                let mut image_data = vec![0; 128 * 128 * 3];

                for (x, y, pixel) in resized_img.pixels() {
                    image_data[convolutional_neural_network::util::get_index((x as usize, y as usize, 0), (128, 128, 3))] = pixel.0[0];
                    image_data[convolutional_neural_network::util::get_index((x as usize, y as usize, 1), (128, 128, 3))] = pixel.0[1];
                    image_data[convolutional_neural_network::util::get_index((x as usize, y as usize, 2), (128, 128, 3))] = pixel.0[2];
                }

                output.push(TrainingData { data: image_data, classification: String::from_str(classification).unwrap() });
            }
        }
    };

    Ok(())
}