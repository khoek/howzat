#![allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    unnecessary_transmutes
)]

#[cfg(feature = "f64")]
#[allow(clippy::missing_safety_doc, clippy::ptr_offset_with_cast)]
pub mod f64 {
    include!(concat!(env!("OUT_DIR"), "/bindings_f64.rs"));
}

#[cfg(feature = "gmp")]
#[allow(clippy::missing_safety_doc, clippy::ptr_offset_with_cast)]
pub mod gmpfloat {
    include!(concat!(env!("OUT_DIR"), "/bindings_gmpfloat.rs"));
}

#[cfg(feature = "gmprational")]
#[allow(clippy::missing_safety_doc, clippy::ptr_offset_with_cast)]
pub mod gmprational {
    include!(concat!(env!("OUT_DIR"), "/bindings_gmprational.rs"));
}
