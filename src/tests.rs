use crate::*;

// TODO: add more tests

#[test]
fn convolutional_layer_forward_propagate()
{
    let mut layer1 = Layer::make_convolutional_layer(1, 0, 0, (3, 3, 2), 0);
    let mut layer2 = Layer::make_convolutional_layer(1, 1, 2, (4, 4, 1), 2);

    if let Layer::Convolutional(ref mut conv) = layer1 {
        conv.set_volume(&vec![1.0, 10.0, 2.0, 11.0, 3.0, 12.0, 4.0, 13.0, 5.0, 14.0, 6.0, 15.0, 7.0, 16.0, 8.0, 17.0, 9.0, 18.0]).expect("Set volume");
    }

    if let Layer::Convolutional(ref mut conv) = layer2 {
        conv.set_kernel(vec![1.0, 0.5, 0.5, 1.0, 0.5, 1.0, 1.0, 0.5]).expect("Set kernel");
        conv.set_biases(vec![0.5]).expect("Set biases");
    }

    layer1.forward_propagate(&mut layer2).expect("Forward propagation");

    if let Layer::Convolutional(ref mut conv) = layer2 {
        assert_eq!(conv.volume, vec![
            6.5,
            18.5,
            12.5,
            0.5,
            21.5,
            45.5,
            24.5,
            0.5,
            15.5,
            27.5,
            12.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
        ]);
    }
}