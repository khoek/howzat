# cddlib-sys

Raw FFI bindings to [cddlib](https://github.com/cddlib/cddlib) for convex
polyhedra via the double-description method. Builds a static cddlib (and
uses `gmp-mpfr-sys` for the GMP-backed backends.

By default, `gmp-mpfr-sys` builds GMP from source and links it into the build
(no system GMP dependency). If you need ABI-level interop with a system GMP
(e.g. embedding into a host that already links GMP), enable `use-system-gmp`
and provide the GMP development headers + libraries (e.g. `gmp.h` + `libgmp`).

## Features

| Feature | Description |
|---------|-------------|
| `f64` | Build the f64 backend |
| `gmp` | Build the GMPFLOAT backend |
| `gmprational` | Build the GMPRATIONAL backend |
| `tools` | Build cddlib CLI tools alongside the library |
| `use-system-gmp` | Use system GMP via `gmp-mpfr-sys/use-system-libs` |

All numeric backends are enabled by default. Use `--no-default-features` to
select a subset.

## Vendored versions

- cddlib `0.94n` (`vendor/cddlib-0.94n.tar.gz`)

## Modules

Bindings are exposed under backend-specific modules: `cddlib_sys::f64`,
`cddlib_sys::gmpfloat`, and `cddlib_sys::gmprational`.

## License

GPL-2.0-or-later (inherited from cddlib). See `LICENSE` for details.
