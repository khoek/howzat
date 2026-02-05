# howzat

User-friendly Python bindings for the Rust crates [`howzat`](https://crates.io/crates/howzat) and
[`howzat-kit`](https://crates.io/crates/howzat-kit) via PyO3.

## Usage

```python
import numpy as np
import howzat

verts = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

res = howzat.solve(verts)  # cached default backend: "howzat-dd[purify[snap]]:f64[eps[1e-12]]"
# res = howzat.Backend().solve(verts)  # same as above
# res = howzat.Backend("howzat-dd[purify[snap],adj[sparse]]:f64").solve(verts)  # force sparse adjacency graphs

verts_i64 = np.asarray([[0, 0], [1, 0], [0, 1]], dtype=np.int64)
res_exact = howzat.solve_exact(verts_i64)  # cached exact backend: "howzat-dd:gmprat"

print(res.facets, res.ridges)
print(res.facet_adjacency)
print(res.generators)    # numpy.ndarray[float64] (row-major, shape=(n, d+1))
print(res.inequalities)  # numpy.ndarray[float64] (row-major, shape=(m, d+1))
```

`input` must be a contiguous (C-order) 2D `numpy.ndarray`.

- `repr=howzat.Representation.Generator` (default): `input` has shape `(n, d)` (vertex coordinates).
- `repr=howzat.Representation.Inequality`: `input` has shape `(m, d+1)` (H-rep rows `(b, a...)`).

### Backends

The class `howzat.Backend(spec)` accepts backend spec strings, which specify the algorithm and
parameters of the computation pipeline.

At a high level, the syntax is: `KIND[OPTIONS][:SPEC]`. Some examples:
- `howzat-dd[purify[snap]]:f64[eps[1e-12]]` (default)
- `howzat-dd:f64`
- `howzat-dd:f64[eps[1e-12],max]-resolve[gmprat]`
- `cddlib:gmprational`
- `cddlib+hlbl:f64`
- `lrslib+hlbl` (defaults to `lrslib+hlbl:gmpint`)

Backend options (`[OPTIONS]`) are comma-separated and may include:
- `adj[dense]` / `adj[sparse]`: force dense bitset graphs vs adjacency lists (if supported natively)
- `purify[...]`: configure `howzat-dd` purifiers, e.g. `purify[snap]` or `purify[upsnap[gmprat]]`

If `adj[...]` is omitted and the backend supports both adjacency representations, `howzat-kit`
chooses dense unless the dense bitset representation would exceed `128MiB` (for either the vertex
graph, or the facet graph upper bound), in which case it chooses sparse.

Supported `KIND[:SPEC]` forms:
- `cddlib[:f64|gmpfloat|gmprational]`
- `cddlib+hlbl[:f64|gmpfloat|gmprational]`
- `howzat-dd[:PIPELINE]`
- `howzat-lrs[:rug|dashu]`
- `lrslib+hlbl[:gmpint]`
- `ppl+hlbl[:gmpint]`

### API

- `howzat.solve(input, repr=Representation.Generator) -> SolveResult`
  Convenience function which uses the default backend (`howzat-dd[purify[snap]]:f64[eps[1e-12]]`).
- `howzat.solve_exact(input, repr=Representation.Generator) -> SolveResult`
  Convenience function which uses the default exact backend (`howzat-dd:gmprat`).
- `howzat.Backend(spec: str | None = None)`
  Loads the specified backend; `None` selects the default.
- `Backend.solve(input, repr=Representation.Generator) -> SolveResult`
  Runs the backend.
- `Backend.solve_exact(input, repr=Representation.Generator) -> SolveResult`
  Runs the backend in exact mode (`int64`); errors if the backend is not exact.

#### SolveResult

A `SolveResult` has the fields:
- `spec: str` backend spec actually used
- `dimension: int` ambient dimension `d`
- `vertices: int` number of vertices `n`
- `facets: int` number of facets
- `ridges: int` number of ridges (edges in the facet adjacency / FR graph)
- `total_seconds: float` time spent inside the backend (seconds)
- `vertex_positions: list[list[float]] | None` vertex coordinates if the backend returned baseline geometry
- `vertex_adjacency: DenseGraph | AdjacencyList` vertex adjacency graph (dense or sparse)
- `facets_to_vertices: list[list[int]]` for each facet, the incident vertex indices
- `facet_adjacency: DenseGraph | AdjacencyList` facet adjacency graph (FR graph; dense or sparse)
- `generators`: V-representation coefficients (row-major, shape `(n, d+1)`)
- `inequalities`: H-representation coefficients (row-major, shape `(m, d+1)`)
- `fails: int` backend-specific failure count (pipeline dependent)
- `fallbacks: int` backend-specific fallback count (pipeline dependent)

For `solve_exact(...)`, the coefficient matrices are returned as `list[list[str]]` (exact strings).

Both graph types support:
- `node_count() -> int`
- `degree(node: int) -> int`
- `contains(node: int, neighbor: int) -> bool`
- `neighbors(node: int) -> list[int]`

## Install

From PyPI:

```bash
python -m pip install howzat
```

Force a local source build, and enabling CPU-native codegen:

```bash
RUSTFLAGS="-C target-cpu=native" python -m pip install --no-binary howzat howzat
```

Prebuilt wheels are compiled for a portable baseline CPU (not `target-cpu=native`).
