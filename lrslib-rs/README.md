# lrslib-rs

Safe Rust bindings over `lrslib-sys`. Wraps lrslib's reverse-search routines
for H-/V-representation conversion without exposing raw pointers.

## Features

- **Safe wrappers**: RAII management of lrslib dictionary/data lifecycle.
- **Representation conversion**: Compute facets from vertices, vertices/rays from inequalities.
- **`gmp` feature**: Arbitrary precision via GMP (`gmp-mpfr-sys`). Without it, uses fixed-width
  `LRSLONG` (128-bit when supported).

## Example

```rust
use lrslib_rs::Polyhedron;

let poly = Polyhedron::from_vertices(&[vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]])?;
let facets = poly.facets()?;
```

## License

AGPL-3.0-only. See `LICENSE` for details.
