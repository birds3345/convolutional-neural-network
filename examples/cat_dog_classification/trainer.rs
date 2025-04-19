use convolutional_neural_network::NeuralNetwork;
use crate::image_compiler;

use std::fs;

pub fn train(
    start: usize,
    batch_size: usize,
    learning_rate: f32,
    path: String,
    images: &Vec<image_compiler::TrainingData>,
    neural_network: &mut NeuralNetwork
) {
    let total_batches = (images.len() + batch_size - 1) / batch_size;

    let batches = images.chunks(batch_size).enumerate().skip(start);

    for (batch_idx, batch) in batches {
        neural_network.start_batch();

        let mut error = 0.0f32;
        let mut correct = 0;
        let mut incorrect = 0;

        for image in batch {
            let mut image_data = vec![0.0f32; 128 * 128 * 3];
            
            for i in 0..image.data.len() {
                image_data[i] = image.data[i] as f32 / 255.0;
            }

            let expected: f32 = if image.classification == "cat" { 0.0 } else { 1.0 };
            let expected_vec = vec![expected];

            neural_network.set_input(&image_data).unwrap();
            neural_network.forward_propagate().unwrap();

            let err = neural_network.get_error(&expected_vec).unwrap();
            error += err;

            let output = neural_network.get_output().unwrap()[0];
            if (output > 0.5) == (expected > 0.5) {
                correct += 1;
            } else {
                incorrect += 1;
            }

            //println!("output={}, expected_output={}, error={}", output, expected_output, err);

            neural_network.back_propagate(&expected_vec).unwrap();
        }

        neural_network.end_batch(batch_size as u8, learning_rate, 0.9, 5e-4);
        
        println!(
            "Completed batch {}/{}, average_error={}, correct_vs_incorrect={}/{}",
            batch_idx + 1,
            total_batches,
            error / batch_size as f32,
            correct,
            incorrect
        );

        if (batch_idx + 1) % 20 == 0 {
            let mut writer = std::io::BufWriter::new(fs::OpenOptions::new()
                .write(true)
                .truncate(true)
                .create(true)
                .open(&path)
                .unwrap());

            bincode::serde::encode_into_std_write(
                &neural_network,
                &mut writer,
                bincode::config::standard()
            ).unwrap();

            println!("Saved neural network");
        }
    }

    let mut writer = std::io::BufWriter::new(fs::OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(&path)
        .unwrap());

    bincode::serde::encode_into_std_write(
        &neural_network,
        &mut writer,
        bincode::config::standard()
    ).unwrap();

    println!("Saved neural network");
}