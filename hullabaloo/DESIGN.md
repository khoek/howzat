# hullabaloo design

`hullabaloo` is a geometry construction crate: it generates vertex data but does not call any
polyhedral solver (cddlib/howzat/lrslib/etc).

## Geometrizable

Constructions are exposed as opaque types (e.g. `Drum<N>`) implementing:

- `Geometrizable` (`hullabaloo/src/geometrizable.rs`)

`Geometrizable` provides:

- `into_vertices(self) -> Vec<Vec<N>>` (the primary output; may be computed lazily)
- `into_matrix(self) -> Matrix<N>` (conversion helper only; rows are `[1, x1, x2, ...]`)

Callers are expected to pick and invoke a backend solver explicitly, using these conversions when
convenient.
