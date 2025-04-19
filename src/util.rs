use crate::errors::Error;

pub fn get_output_dimension(
    dimension: (usize, usize, usize),
    zero_padding: usize,
    num_kernels: usize,
    kernel_size: usize,
    stride: usize
) -> Option<(usize, usize, usize)> {

    if num_kernels == 0 ||
       kernel_size == 0 ||
       stride == 0 ||
       dimension.0 == 0 ||
       dimension.1 == 0 ||
       dimension.2 == 0
    { return None }

    let (x, y, _) = dimension;
    let (padded_x, padded_y) = (x + zero_padding * 2, y + zero_padding * 2);
    if kernel_size - 1 >= padded_x || kernel_size - 1 >= padded_y { return None };

    let (length_x, length_y) = (padded_x - kernel_size + 1, padded_y - kernel_size + 1);
    let (result_x, result_y) = ((length_x + stride - 1) / stride, (length_y + stride - 1) / stride);

    if result_x == 0 || result_y == 0 { return None };

    Some((result_x, result_y, num_kernels))
}

pub(crate) fn check_output_dimension(
    dimension: (usize, usize, usize),
    expected_dimension: (usize, usize, usize),
    zero_padding: usize,
    num_kernels: usize,
    kernel_size: usize,
    stride: usize
) -> Result<(), Error> {
    let output_dim =
        get_output_dimension(dimension,
            zero_padding,
            num_kernels,
            kernel_size,
            stride
        );
                
    if let Some(dim) = output_dim {
        if dim.0 != expected_dimension.0 ||
            dim.1 != expected_dimension.1 ||
            dim.2 != expected_dimension.2
        { return Err(Error::DimensionMismatch) };
                
    } else { return Err(Error::ImpossibleOutputDimension); };

    Ok(())
}

/// used to simulate zero padding without using extra memory
#[inline(always)]
pub(crate) fn query_zero_padded(position: (usize, usize, usize), input_dimension: (usize, usize, usize), zero_padding: usize) -> Option<usize> {
    let (x, y, z) = position;
            
    if x < zero_padding ||
        x >= input_dimension.0 - zero_padding ||
        y < zero_padding ||
        y >= input_dimension.1 - zero_padding
    { return None };

    Some(get_index((x - zero_padding, y - zero_padding, z), input_dimension))
}

#[inline(always)]
pub(crate) fn get_kernel_index(position: (usize, usize, usize, usize), kernel_size: usize, input_depth: usize) -> usize {
    let (x, y, z, kernel_index) = position;

    kernel_index * (kernel_size * kernel_size * input_depth)
    + z * (kernel_size * kernel_size)
    + y * kernel_size
    + x
}

#[inline(always)]
pub fn get_index(position: (usize, usize, usize), dimension: (usize, usize, usize)) -> usize {
    let (x, y, z) = position;
    let (_, dim_y, dim_z) = dimension;

    z + dim_z * (y + dim_y * x)
}