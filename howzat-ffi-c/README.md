# howzat-ffi-c

User-friendly C bindings for the Rust crate `howzat` via `howzat-kit`.

## Build

```bash
cd polytopia
cargo build -p howzat-ffi-c --release
```

Artifacts:
- `polytopia/target/release/libhowzat_ffi_c.so`
- `polytopia/target/release/libhowzat_ffi_c.a`
- `polytopia/howzat-ffi-c/include/howzat_ffi.h`

## Usage

The API mirrors `howzat-ffi-py`:

- `howzat_solve(...)` uses the default backend (`howzat-dd[purify[snap]]:f64[eps[1e-12]]`).
- `howzat_solve_exact(...)` uses the default exact backend (`howzat-dd:gmprat`).
- `howzat_backend_new(spec)` creates an explicit backend handle.
- `howzat_backend_solve(...)` / `howzat_backend_solve_exact(...)` run it.

Input is passed as a row-major 2D buffer with shape `(rows, cols)`:
- `howzat_solve`: `double*`
- `howzat_solve_exact`: `int64_t*`

The `howzat_representation_t` argument selects how to interpret the input:
- `HOWZAT_REPR_EUCLIDEAN_VERTICES`: V-rep vertex rows, shape `(n, d)` (`rows=n`, `cols=d`)
- `HOWZAT_REPR_HOMOGENEOUS_GENERATORS`: V-rep homogeneous generator rows `(b, x...)`, shape `(n, d+1)` (`rows=n`, `cols=d+1`)
- `HOWZAT_REPR_INEQUALITY`: H-rep inequality rows `(b, a...)`, shape `(m, d+1)` (`rows=m`, `cols=d+1`)

Backends are selected via the same string syntax as `howzat-kit`; adjacency representation can be
requested with `adj[dense]` or `adj[sparse]` in the backend options (e.g. `howzat-dd[adj[sparse]]:f64`).

### Coefficients

Each result includes both the V-rep and H-rep coefficient matrices (row-major, flattened):
- `howzat_result_generators_*` returns the generator matrix (shape `(n, d+1)`)
- `howzat_result_inequalities_*` returns the inequality matrix (shape `(m, d+1)`)
- use `howzat_result_generators_shape` / `howzat_result_inequalities_shape` to query `(rows, cols)`

Use `howzat_result_coeff_kind(res)` to determine whether coefficients are returned as `double` or GMP rationals:
- `HOWZAT_COEFF_F64`: use `howzat_result_generators_f64` / `howzat_result_inequalities_f64`
- `HOWZAT_COEFF_GMPRAT`:
  - `howzat_result_*_gmprat`: returns a row-major slice of pointers to GMP rationals (`mpq_srcptr`)
  - `howzat_result_*_i64`: returns `int64_t` if all coefficients are integral and fit, else an empty slice
  - `howzat_result_*_str`: returns `const char*` strings derived from the GMP rationals

### Example

```c
#include "howzat_ffi.h"

#include <stdio.h>

int main(void) {
  howzat_error_t *err = NULL;

  const double coords[] = {
      0.0, 0.0,
      1.0, 0.0,
      0.0, 1.0,
  };
  howzat_result_t *res = howzat_solve(coords, 3, 2, HOWZAT_REPR_EUCLIDEAN_VERTICES, &err);
  if (!res) {
    fprintf(stderr, "howzat error: %s\n", howzat_error_message(err));
    howzat_error_free(err);
    return 1;
  }

  printf("spec=%s facets=%zu ridges=%zu\n",
         howzat_result_spec(res),
         (size_t)howzat_result_facets(res),
         (size_t)howzat_result_ridges(res));

  howzat_result_free(res);
  return 0;
}
```

Pointers returned by `howzat_result_*` and `howzat_backend_*` accessors remain valid until the
corresponding owner (`howzat_result_t` / `howzat_backend_t`) is freed.

## Header generation

Regenerate the header with:

```bash
cd polytopia/howzat-ffi-c
cbindgen --config cbindgen.toml --output include/howzat_ffi.h
```
