# howzat

Dynamic double-description method for convex cones and polytopes. Small API,
pluggable numeric backends, predictable behavior.

## Features

- **Cone/polytope primitives**: Adjacency queries, tableau-based convex hull routines.
- **Backend-agnostic**: Works with `rug` (GMP) or `dashu` (pure-Rust) arbitrary precision.
- **Minimal footprint**: Few dependencies, optional tracing for debugging.

## Related crates

- `howzat-kit`: a higher-level runner API that selects common backends via a string backend spec.

## Example

```rust
use howzat::prelude::*;

fn main() -> Result<(), HowzatError> {
    // Triangle with vertices (0,0), (1,0), (0,1) in homogeneous generator form [1, x, y].
    let generators = vec![vec![1.0, 0.0, 0.0], vec![1.0, 1.0, 0.0], vec![1.0, 0.0, 1.0]];
    let matrix = LpMatrixBuilder::<f64, Generator>::from_rows(generators).build();
    let eps = f64::default_eps();

    let poly = PolyhedronOutput::builder(matrix)
        .cone_options(ConeOptions::default())
        .polyhedron_options(PolyhedronOptions {
            output_incidence: IncidenceOutput::Set,
            output_adjacency: AdjacencyOutput::List,
            ..Default::default()
        })
        .run_dd_with_eps(eps)?;

    assert_eq!(poly.output().representation(), RepresentationKind::Inequality);
    assert!(poly.adjacency().is_some());
    Ok(())
}
```

## License

AGPL-3.0-only. See `LICENSE` for details.
