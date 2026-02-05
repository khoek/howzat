# ppl-sys

Vendored build + raw FFI bindings for the Parma Polyhedra Library (PPL).

Builds a static PPL from vendored sources. The default `gmp` coefficient backend
uses `gmp-mpfr-sys` for GMP/GMPXX.

Building requires `autoreconf` (autoconf + automake + libtool). Building the
`gmp` backend also requires a C++ toolchain.

By default, `gmp-mpfr-sys` builds GMP from source and links it into the build
(no system GMP dependency). If you need ABI-level interop with a system GMP
(e.g. embedding into a host that already links GMP), enable `use-system-gmp`
and provide the GMP development headers + libraries (e.g. `gmp.h`, `gmpxx.h`,
`libgmp`, `libgmpxx`).

## Features

| Feature | Description |
|---------|-------------|
| `gmp` | Build PPL with GMP coefficients (default) |
| `i64` | Build PPL with native int64 coefficients |
| `pic` | Build PPL with `-fPIC` |
| `use-system-gmp` | Use system GMP via `gmp-mpfr-sys/use-system-libs` |

Exactly one coefficient backend must be enabled: `gmp` (default) or `i64`.

## Vendored versions

- PPL `92d0704d3309d55f39a647595f8383b86fcd57e1` (`vendor/ppl-92d0704d3309d55f39a647595f8383b86fcd57e1.tar.gz`)

## License

AGPL-3.0-only for the Rust crate code; the vendored upstream PPL sources are
GPL-3.0-or-later.
