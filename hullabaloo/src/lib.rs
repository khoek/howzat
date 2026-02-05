//! Engine-agnostic primitives and helpers for polyhedral experiments.
//!
//! This crate owns the shared core primitives used across solver backends:
//! - numeric traits (`calculo`)
//! - compact bitset types (`types`)
//! - matrix storage/builders (`matrix`)
//! - incidence/adjacency set-families (`set_family`)
//! - fast adjacency builders (`adjacency`)
//!
//! Solver engines (e.g. DD/LRS) live in separate crates and build on top of these primitives.

pub mod adjacency;
pub mod adjacency_list;
pub mod matrix;
pub mod set_family;
pub mod types;

pub use adjacency::AdjacencyListBuilder;
pub use adjacency_list::AdjacencyList;
pub mod drum;
pub mod geometrizable;
pub mod matroid;

pub use drum::{Drum, DrumBases, DrumPromotion, DrumSkin, PromotionError};
pub use geometrizable::Geometrizable;
pub use matroid::{CharacteristicPolynomial, LinearOrientedMatroid, MatroidError};
