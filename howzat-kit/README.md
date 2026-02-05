# howzat-kit

`howzat-kit` is a small Rust library providing:
- parsing backend specifications from their string encoding, and
- running the chosen backend synchronously on `f64` or exact `i64` vertex data.

This is a convenience frontend for `howzat`, shared by `kompute-hirsch` (`hirsch sandbox bench`) and
the FFI crates.

## Backend Specs

Backend specs are strings like:
- `howzat-dd[purify[snap]]:f64[eps[1e-12]]` (default)
- `howzat-dd:f64`
- `cddlib:gmprational`
- `lrslib+hlbl`

Syntax:

`KIND[OPTIONS][:SPEC]`

Backend options (`[OPTIONS]`) are comma-separated and may include:
- `adj[dense]` / `adj[sparse]`: force dense bitset graphs vs adjacency lists (if supported natively)
- `purify[...]`: configure `howzat-dd` purifiers, e.g. `purify[snap]` or `purify[upsnap[gmprat]]`

Supported `KIND[:SPEC]` forms:
- `cddlib[:f64|gmpfloat|gmprational]`
- `cddlib+hlbl[:f64|gmpfloat|gmprational]`
- `howzat-dd[:PIPELINE]`
- `howzat-lrs[:rug|dashu]`
- `lrslib+hlbl[:gmpint]`
- `ppl+hlbl[:gmpint]`

`howzat-dd` `PIPELINE` is a `-`-separated list of steps like:
- `NUM` (compute), e.g. `f64`, `gmprat`, `dashurat`
- `repair[NUM]` or `resolve[NUM]` (check)

Example: `howzat-dd[purify[snap],adj[dense]]:f64`.

When `adj[...]` is omitted and the backend supports both adjacency representations, `howzat-kit`
chooses dense unless the dense bitset representation would exceed `128MiB` (for either the vertex
graph, or the facet graph upper bound), in which case it chooses sparse.

For CLI-style parsing (supporting `^` / `%` prefixes), use `BackendArg`:

```rust
use howzat_kit::BackendArg;

let arg: BackendArg = "^howzat-dd[purify[snap]]:f64[eps[1e-12]]".parse().unwrap();
assert!(arg.authoritative);
```

For parsing only the backend itself, use `Backend`:

```rust
use howzat_kit::Backend;

let backend = Backend::parse("howzat-dd[purify[snap]]:f64[eps[1e-12]]").unwrap();
```

## Running

The core entrypoint is `Backend::solve_row_major`, which accepts a contiguous row-major buffer of
`f64` coefficients (plus a `Representation`) and returns either dense or sparse adjacency graphs
(`BackendRunAny`).

```rust
use howzat_kit::{Backend, BackendRunAny, BackendRunConfig, Representation};

let backend = Backend::parse("howzat-dd[purify[snap]]:f64[eps[1e-12]]").unwrap();
let config = BackendRunConfig::default();

let coords = [
    0.0, 0.0,
    1.0, 0.0,
    0.0, 1.0,
];
let run = backend
    .solve_row_major(Representation::EuclideanVertices, &coords, 3, 2, &config)
    .unwrap();
match run {
    BackendRunAny::Dense(run) => println!("{:?}", run.stats),
    BackendRunAny::Sparse(run) => println!("{:?}", run.stats),
}
```

For exact integer inputs, use `Backend::solve_row_major_exact` or `Backend::solve_exact`.
