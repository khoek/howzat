# lrslib-sys

Raw FFI bindings to [lrslib](https://cgm.cs.mcgill.ca/~avis/C/lrs.html), David
Avis' lexicographic reverse-search library for convex polyhedra. Builds a
static lrslib from vendored sources---no network access required.

## Features

| Feature | Description |
|---------|-------------|
| - | Fixed-width `LRSLONG` arithmetic (128-bit when supported)---faster but can overflow |
| `gmp` | Arbitrary precision via GMP (`gmp-mpfr-sys`) |
| `use-system-gmp` | Use system GMP via `gmp-mpfr-sys/use-system-libs` |

## Vendored versions

- lrslib `0.73a` (`vendor/lrslib-073a.tar.gz`)
- lrsarith `011` (bundled inside lrslib; built from `lrsarith-011/`)

## License

GPL-2.0-or-later (inherited from lrslib). See `LICENSE` for details.
