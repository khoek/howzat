# cddlib adjacency performance notes

## Summary

In some workloads, the dominant cost in the cddlib pipeline is building the ray/facet adjacency
(`dd_CopyAdjacency`), not numeric arithmetic. This can make the `f64` backend much slower than
expected relative to GMP variants, despite faster scalar operations.

## What’s happening

In cddlib `0.94n`, `dd_CopyAdjacency` (in `lib-src/cddio.c`) computes adjacency by iterating over
all **ordered** ray pairs `(i, j)` and calling `dd_CheckAdjacency` for each pair. Since adjacency is
symmetric, this performs roughly twice the necessary checks.

The expensive part is the adjacency predicate itself (`dd_CheckAdjacency` in `lib-src/cddcore.c`),
which is largely combinatorial/set-based and does not benefit much from switching numeric types.

## Optimization idea (upstream)

Compute adjacency only once per **unordered** pair (`j > i`), then insert both directions into the
adjacency sets:

- Iterate `RayPtr2` starting from `RayPtr1->Next` (and keep `pos2 = pos1 + 1`).
- When adjacent, insert both edges:
  - `set_addelem(F->set[pos1-1], pos2);`
  - `set_addelem(F->set[pos2-1], pos1);`

This preserves the same undirected adjacency relation while avoiding redundant predicate calls.

## Local observation

On the `hirsch sandbox bench` “drum” instance (`n=11 v=12`), timing breakdown showed
`f_adj` (facet adjacency construction) dominating `cddlib:f64` wall time.
