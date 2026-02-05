# cddlib-rs

Safe Rust bindings over `cddlib-sys`. Wraps cddlib's double-description
routines for convex polyhedra without exposing raw pointers.

## Features

- **Safe wrappers**: Matrices, polyhedra, and LP solutions with RAII lifetime management.
- **Multiple backends**: `f64`, `CddFloat`, `CddRational`---selectable via generics.
- **Familiar API**: Compute facets, generators, and widths using standard Rust types.

## Example

```rust
use cddlib_rs::{CddRational, Matrix, Polyhedron};

let verts = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];

// Inferred scalar type (`f64` here).
let poly = Polyhedron::from_vertex_rows(&verts)?;

// Convert to exact rationals.
let m_f64 = Matrix::from_vertex_rows(&verts)?;
let m_rat: Matrix<CddRational> = m_f64.convert()?;
let poly_exact = Polyhedron::from_matrix(&m_rat)?;

let facets = poly_exact.facets()?;
```

All backends enabled by default. `Matrix` and `Polyhedron` default to `f64`.

## License

AGPL-3.0-only. See `LICENSE` for details.
