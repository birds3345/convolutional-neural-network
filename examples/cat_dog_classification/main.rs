mod image_compiler;
mod trainer;
mod trainer_parallel;
mod test;

use convolutional_neural_network::{ActivationFunction, ErrorFunction, Initialization, Layer, NeuralNetwork, PoolingType};
use std::fs;
use rand::seq::SliceRandom;

use image::{imageops::FilterType, GenericImageView};

pub fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 { return };

    match args[1].as_str() {
        "compile_images" => {
            if args.len() < 3 { return };
            
            let mut output: Vec<image_compiler::TrainingData> = Vec::new();

            for i in (4..args.len()).step_by(2) {
                let classification = args[i - 1].as_str();
                image_compiler::compile_images(args[i].as_str(), classification, &mut output).unwrap();
            }

            let mut rng = rand::rng();
            output.shuffle(&mut rng);

            println!("Compiler: collected images");

            let mut write = std::io::BufWriter::new(fs::OpenOptions::new()
                .write(true)
                .truncate(true)
                .create(true)
                .open(&args[2])
                .unwrap());

            bincode::serde::encode_into_std_write(
                &output,
                &mut write,
                bincode::config::standard()
            ).unwrap();
        }

        "create" => {
            if args.len() < 3 { return };

            let mut neural_net = NeuralNetwork::new(ErrorFunction::BinaryCrossEntropy);

            neural_net.register_layer(
                ActivationFunction::ReLU,
                Layer::make_input_layer(1, (128, 128, 3))
            );

            neural_net.register_layer(
                ActivationFunction::ReLU,
                Layer::make_convolutional_layer(0, 1, 3, (128, 128, 32), 3)
            );

            neural_net.register_layer(
                ActivationFunction::ReLU,
                Layer::make_pooling_layer(PoolingType::Max, 1, 2, 2, (64, 64, 32))
            );

            neural_net.register_layer(
                ActivationFunction::ReLU,
                Layer::make_convolutional_layer(0, 1, 3, (64, 64, 64), 32)
            );

            neural_net.register_layer(
                ActivationFunction::ReLU,
                Layer::make_pooling_layer(PoolingType::Max, 1, 2, 2, (32, 32, 64))
            );

            neural_net.register_layer(
                ActivationFunction::ReLU,
                Layer::make_convolutional_layer(0, 1, 3, (32, 32, 128), 64)
            );

            neural_net.register_layer(
                ActivationFunction::ReLU,
                Layer::make_pooling_layer(PoolingType::Max, 0, 2, 2, (16, 16, 128))
            );

            neural_net.register_layer(
                ActivationFunction::ReLU,
                Layer::make_fully_connected_layer(32768, 512)
            );

            neural_net.register_layer(
                ActivationFunction::Sigmoid,
                Layer::make_fully_connected_layer(512, 1)
            );

            neural_net.initialize(1, Initialization::NormalHe).unwrap();
            neural_net.initialize(3, Initialization::NormalHe).unwrap();
            neural_net.initialize(5, Initialization::NormalHe).unwrap();
            neural_net.initialize(7, Initialization::NormalHe).unwrap();
            neural_net.initialize(8, Initialization::NormalXavier).unwrap();

            let mut write = std::io::BufWriter::new(fs::OpenOptions::new()
                .write(true)
                .truncate(true)
                .create(true)
                .open(&args[2])
                .unwrap());
            
            bincode::serde::encode_into_std_write(
                &neural_net,
                &mut write,
                bincode::config::standard()
            ).unwrap();

            println!("Create: created new model");
        }

        "run" => {
            if args.len() < 4 { return };

            let mut read = std::io::BufReader::new(fs::File::open(&args[2]).unwrap());
            let mut neural_network: NeuralNetwork = bincode::serde::decode_from_std_read(&mut read, bincode::config::standard()).unwrap();

            println!("Run: loaded model");

            let input = image::open(&args[3]).unwrap().resize_exact(128, 128, FilterType::Triangle);
            let mut image_data = vec![0.0f32; 128 * 128 * 3];

            for (x, y, pixel) in input.pixels() {
                image_data[convolutional_neural_network::util::get_index((x as usize, y as usize, 0), (128, 128, 3))] = pixel.0[0] as f32 / 255.0;
                image_data[convolutional_neural_network::util::get_index((x as usize, y as usize, 1), (128, 128, 3))] = pixel.0[1] as f32 / 255.0;
                image_data[convolutional_neural_network::util::get_index((x as usize, y as usize, 2), (128, 128, 3))] = pixel.0[2] as f32 / 255.0;
            }

            neural_network.set_input(&image_data).unwrap();
            neural_network.forward_propagate().unwrap();

            let output = neural_network.get_output().unwrap();
            println!("Run: {}% dog, {}% cat, raw_output={}", output[0] * 100.0, (1.0 - output[0]) * 100.0, output[0]);
        }

        "test" => {
            if args.len() < 4 { return };

            let mut read = std::io::BufReader::new(fs::File::open(&args[2]).unwrap());
            let mut neural_network: NeuralNetwork = bincode::serde::decode_from_std_read(&mut read, bincode::config::standard()).unwrap();
            
            let mut read = std::io::BufReader::new(fs::File::open(&args[3]).unwrap());
            let images: Vec<image_compiler::TrainingData> = bincode::serde::decode_from_std_read(&mut read, bincode::config::standard()).unwrap();

            println!("Test: loaded model and images");
            
            test::test(&images, &mut neural_network);
        }

        "train" => {
            if args.len() < 8 { return };
            
            const PARALLEL: bool = true;

            let mut read = std::io::BufReader::new(fs::File::open(&args[2]).unwrap());
            let mut neural_network: NeuralNetwork = bincode::serde::decode_from_std_read(&mut read, bincode::config::standard()).unwrap();
            
            let mut read = std::io::BufReader::new(fs::File::open(&args[3]).unwrap());
            let images: Vec<image_compiler::TrainingData> = bincode::serde::decode_from_std_read(&mut read, bincode::config::standard()).unwrap();

            let start: usize = args[4].parse().unwrap();
            let learning_rate: f32 = args[5].parse().unwrap();
            let batch_size: usize = args[6].parse().unwrap();
            let epoches: usize = args[7].parse().unwrap();

            println!("Train: loaded model and images");

            for i in 0..epoches {
                if PARALLEL {
                    trainer_parallel::train(start, batch_size, learning_rate, args[2].clone(), &images, &mut neural_network);
                } else {
                    trainer::train(start, batch_size, learning_rate, args[2].clone(), &images, &mut neural_network);
                }
                println!("Completed epoch {}/{}", i + 1, epoches);
            }
        }

        _ => ()
    }
}