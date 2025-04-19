#[derive(Debug, Clone, Copy)]
pub enum Error {
    DimensionMismatch,
    IncompatibleLayers,

    ImpossibleOutputDimension,

    InvalidInput,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Error::DimensionMismatch => write!(f, "Layer dimensions do not match"),
            Error::IncompatibleLayers => write!(f, "Layers are incompatible or don't exist"),
            Error::ImpossibleOutputDimension => write!(f, "Output dimension is impossible"),
            Error::InvalidInput => write!(f, "Input arguments to this function are invalid"),
        }
    }
}

impl std::error::Error for Error {

}