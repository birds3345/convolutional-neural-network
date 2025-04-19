use convolutional_neural_network::NeuralNetwork;

use crate::image_compiler;

pub fn test(images: &Vec<image_compiler::TrainingData>, neural_network: &mut NeuralNetwork) {
    let mut average_error = 0.0f32;
    let mut correct = 0;
    let mut incorrect = 0;

    for image in images {
        let mut image_data = vec![0.0f32; 128 * 128 * 3];
                    
        for i in 0..image.data.len() {
            image_data[i] = image.data[i] as f32 / 255.0;
        }

        let expected_output: f32 = if image.classification == "cat" { 0.0 } else { 1.0 };
        let expected_output_vec = vec![expected_output];

        neural_network.set_input(&image_data).unwrap();
        neural_network.forward_propagate().unwrap();

        let error = neural_network.get_error(&expected_output_vec).unwrap();
        average_error += error;

        let output = neural_network.get_output().unwrap()[0];
        if (output > 0.5) == (expected_output > 0.5) {
            correct += 1;
        } else {
            incorrect += 1;
        }
    }

    println!("Test: correct_vs_incorrect={}/{}, percentage_correct={}%, average_error={}", correct, incorrect, correct as f32 / (correct + incorrect) as f32 * 100.0, average_error / images.len() as f32);
}