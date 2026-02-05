#[cfg(not(feature = "gmp"))]
#[path = "b128.rs"]
mod imp;

#[cfg(feature = "gmp")]
#[path = "gmp.rs"]
mod imp;

pub(crate) use imp::*;
