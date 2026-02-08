# Howzat DD Exact-Mode Performance Report

Date: 2026-02-08  
Scope: `howzat-dd:gmprat` vs `ppl+hlbl:gmpint` parity work under `release-lto`  
Primary repo for measurements: `kompute-hirsch`  
Primary implementation repo: `polytopia/howzat`

## 1) Objective and Constraints

This work targeted the performance gap between:

- `howzat-dd:gmprat` (exact DD backend in howzat)
- `ppl+hlbl:gmpint` (PPL-backed backend)

Constraints used throughout:

- Evaluate in `release-lto` only.
- Maintain correctness (`fails=errors=mismatches=issues=0`).
- Prefer exact-path (`IntUmpire`) optimizations.
- Generic-path changes allowed only if free wins (no known regressions on other paths).

## 2) Measurement Protocol (Final Trusted Dataset)

Final clean rerun artifact:

- `/tmp/bench_stage_matrix_5x_20260207_212904/raw_results.tsv`

Benchmark matrix:

- Mode: adjacency
- Deterministic corpus: `--use file:data` (2478 drums)
- Random sweep: `n=4..11`, `v=n+1`, seed `20260207`
- Repeats (5x budget):
  - n4=450000
  - n5=190000
  - n6=72500
  - n7=26500
  - n8=9500
  - n9=3250
  - n10=1000
  - n11=275

Stage order rerun sequentially in one clean chain:

1. `BASE_PRE_P2`
2. `P2`
3. `P2+F1`
4. `P2+F1+F2`
5. `P2+F1+F2+F3`
6. `P2+F1+F2+F3+F4`
7. `P2+F1+F2+F3+F4+P1`

All 126 result rows are correctness-clean.

## 3) PPL Source Review (Algorithmic Baseline)

Reviewed source anchors:

- `target/build-deps/ppl-sys/x86-64-unknown-linux-gnu/92d0704d3309d55f39a647595f8383b86fcd57e1-gmp/PPL-92d0704d3309d55f39a647595f8383b86fcd57e1/src/Polyhedron_conversion_templates.hh`
- `target/build-deps/ppl-sys/x86-64-unknown-linux-gnu/92d0704d3309d55f39a647595f8383b86fcd57e1-gmp/PPL-92d0704d3309d55f39a647595f8383b86fcd57e1/src/Polyhedron_minimize_templates.hh`
- `target/build-deps/ppl-sys/x86-64-unknown-linux-gnu/92d0704d3309d55f39a647595f8383b86fcd57e1-gmp/PPL-92d0704d3309d55f39a647595f8383b86fcd57e1/src/Bit_Row.cc`
- `target/build-deps/ppl-sys/x86-64-unknown-linux-gnu/92d0704d3309d55f39a647595f8383b86fcd57e1-gmp/PPL-92d0704d3309d55f39a647595f8383b86fcd57e1/src/Bit_Row_defs.hh`

Key PPL mechanics relevant to this effort:

- Saturation-row propagation for child generation via `Bit_Row new_satrow(sat[i], sat[j])`.
- Quick non-adjacency and quick adjacency tests before full subset test.
- Full combinatorial adjacency test only when quick tests are inconclusive.
- Exact arithmetic child generation uses normalized scalar products (`normalize2`, `strong_normalize`).
- Constraint-local scalar-product bookkeeping (`scalar_prod`) rather than eager global reclassification.
- Conversion/minimization interface transposes saturation matrices before `simplify()` because conversion and simplify use opposite row/column orientations.

Retained, source-validated notes from earlier planning (kept here for audit continuity):

- `scalar_prod[i]` is defined as product of current constraint `source_k` with generator `dest_rows[i]`; zero means saturation.
- Quick non-adjacency lower bound derives from extremal-ray face dimension argument:
  - `min_saturators = source_num_columns - num_lines_or_equalities - 2`.
- Quick adjacency criterion is explicit:
  - adjacency if `max(sat_num_ones[i], sat_num_ones[j]) + 1 == new_satrow_ones`.
- Full adjacency fallback is a subset check over other generators:
  - if any `subset_or_equal(sat[l], new_satrow)` (`l != i,j`) then pair is non-adjacent.
- Child rays are only materialized after passing quick/full filters and then attached with their saturation row.
- Child arithmetic path uses normalized coefficients (`normalize2`) then `strong_normalize`.
- New child scalar product at current constraint is set to zero by construction.
- In minimization, PPL explicitly transposes saturation information between conversion and simplify phases.

Representative anchors in PPL conversion file:

- Quick-test flags/comments: `Polyhedron_conversion_templates.hh:38`, `:45`
- Current-constraint scalar-product semantics: `Polyhedron_conversion_templates.hh:433`
- First non-saturating row discovery loop: `Polyhedron_conversion_templates.hh:446`
- Child saturation row creation: `:777` to `:780`
- Quick non-adjacency bound derivation: `Polyhedron_conversion_templates.hh:747`
- `min_saturators` definition: `Polyhedron_conversion_templates.hh:755`
- Quick non-adjacency/adjacency logic: `:785` to `:812`
- Full subset fallback: `:816` to `:824`
- Child-row exact normalization path: `:871` to `:886`
- Child scalar product set to zero: `Polyhedron_conversion_templates.hh:893`

Representative anchors in PPL minimization file:

- `tmp_sat` transpose before simplify: `Polyhedron_minimize_templates.hh:208`
- `sat` transpose/simplify/transpose-back path: `Polyhedron_minimize_templates.hh:442`

## 4) Patch Catalogue With PPL Analogue Mapping

Status categories:

- **KEPT**: in final master stack
- **REVERTED**: tested and dropped
- **DEFERRED**: not part of final path

### P2 — seed child zero-set from parents

- Commit: `a292e1d`
- Files: `polytopia/howzat/src/dd/umpire/int.rs`
- Intent:
  - Build child zero-set from parent intersection + entering row id.
  - Avoid re-discovering obviously saturated rows from scratch.
- PPL analogue: **DIRECT**.
  - Matches PPL’s parent-derived `new_satrow` model in conversion.
  - Source evidence: `Polyhedron_conversion_templates.hh:777`.
- Outcome: **observable improvement**.

### F1 — per-ray row-sign cache (stage isolated for benchmarking)

- Benchmark stage commit: `d046aa1` (constructed to isolate F1 effect)
- Underlying implementation carrier commit: `d5b6f67` (also contains F2)
- Files: `polytopia/howzat/src/dd/umpire/int.rs`
- Intent:
  - Cache row signs on rays to avoid repeated dot/sign recomputation.
- PPL analogue: **PARTIAL**.
  - Similar spirit to scalar-product bookkeeping, but not the same structure as PPL’s per-constraint `scalar_prod` flow.
  - Source evidence for the PPL side: `Polyhedron_conversion_templates.hh:433`.
- Outcome: **noise / mixed** on clean rerun.

### F2 — inherit child row signs when mathematically certain

- Commit: `d5b6f67`
- Files: `polytopia/howzat/src/dd/umpire/int.rs`
- Intent:
  - Propagate parent-derived sign certainty into child rays.
  - Compute exact dot only when sign remains unknown.
- PPL analogue: **STRONG PARTIAL**.
  - Not identical API-wise, but same architectural direction as PPL’s saturation/scalar-product propagation and deferred exact work.
  - Source evidence: `Polyhedron_conversion_templates.hh:777`, `Polyhedron_conversion_templates.hh:893`.
- Outcome: **large observable improvement**.

### F3 — first-infeasible frontier cursor with epoching

- Commit: `0b8bc0b`
- Files:
  - `polytopia/howzat/src/dd/state.rs`
  - `polytopia/howzat/src/dd/umpire/int.rs`
  - `polytopia/howzat/src/dd/umpire/mod.rs`
  - `polytopia/howzat/src/dd/umpire/multi_precision.rs`
  - `polytopia/howzat/src/dd/umpire/single_precision.rs`
- Intent:
  - Reduce repeated full scans while maintaining first-infeasible semantics across row-order epochs.
- PPL analogue: **NONE DIRECT**.
  - PPL’s conversion pipeline is organized differently; no clear one-to-one frontier cursor construct.
- Outcome: **in the noise**.

### F4 — broaden quick adjacency/non-adjacency checks

- Commit: `4eb37eb`
- Files: `polytopia/howzat/src/dd/engine.rs`
- Intent:
  - Apply PPL-style quick adjacency/non-adjacency predicates earlier to prune expensive exact subset checks.
- PPL analogue: **DIRECT**.
  - Mirrors PPL quick-test pattern in conversion.
  - Source evidence: `Polyhedron_conversion_templates.hh:755`, `Polyhedron_conversion_templates.hh:810`, `Polyhedron_conversion_templates.hh:822`.
- Outcome: **small regression** in clean rerun aggregate, despite being source-aligned.

### P1 — skip non-saturation reclassification after first infeasible (generated rays)

- Commit: `5d40266`
- Files: `polytopia/howzat/src/dd/umpire/int.rs`
- Intent:
  - Narrow eager work in generated-ray classification once infeasibility is established, while preserving saturation-row behavior.
- PPL analogue: **PARTIAL**.
  - Aligns with PPL’s non-global reclassification tendency, but not a direct transliteration.
  - Source evidence for non-global current-constraint bookkeeping: `Polyhedron_conversion_templates.hh:433`, `Polyhedron_conversion_templates.hh:893`.
- Correctness note from earlier failed attempt:
  - lazy classification cannot be detached from saturation/adjacency invariants; PPL protects this with quick/full adjacency gating before child materialization (`Polyhedron_conversion_templates.hh:785`, `Polyhedron_conversion_templates.hh:816`).
- Outcome: **near-noise absolute; regression in parity ratio**.

### P3 — earlier incremental infeasibility maintenance attempt

- Status: **REVERTED**
- PPL analogue: **NONE DIRECT**
- Outcome: regression in earlier controlled tests; superseded by F3 design.

### P4 — normalization-path candidate changes

- Status: **REVERTED**
- PPL analogue: **PARTIAL** (normalization exists in PPL, but attempted mapping regressed here)
- Outcome: regression; dropped.

### P5 — generic incidence-index/candidate-intersection free-win attempt

- Status: **REVERTED**
- PPL analogue: **NONE DIRECT** (no convincing equivalent hot-path structure found)
- Outcome: mixed/noisy, no robust gain; dropped.

### F5 — instrumentation-only fallback

- Status: **DEFERRED**
- PPL analogue: N/A
- Outcome: not required for final clean matrix.

## 5) Final Clean Benchmark Tables (Authoritative)

### 5.1 Deterministic (`data/*`) summary

| Stage | ppl_s | howzat_s | ppl/howzat | howzat speedup vs pre-P2 |
|---|---:|---:|---:|---:|
| BASE_PRE_P2 | 54.370 | 83.214 | 0.653x | 1.000x |
| P2 | 54.079 | 81.347 | 0.665x | 1.023x |
| P2+F1 | 54.766 | 80.415 | 0.681x | 1.035x |
| P2+F1+F2 | 54.228 | 50.515 | 1.074x | 1.647x |
| P2+F1+F2+F3 | 54.129 | 51.078 | 1.060x | 1.629x |
| P2+F1+F2+F3+F4 | 54.598 | 52.474 | 1.040x | 1.586x |
| P2+F1+F2+F3+F4+P1 | 53.735 | 52.246 | 1.029x | 1.593x |

### 5.2 Random aggregate (`n=4..11`) summary

| Stage | ppl_s (sum n=4..11) | howzat_s (sum n=4..11) | ppl/howzat | howzat speedup vs pre-P2 |
|---|---:|---:|---:|---:|
| BASE_PRE_P2 | 1146.304 | 1356.607 | 0.845x | 1.000x |
| P2 | 1146.557 | 1212.074 | 0.946x | 1.119x |
| P2+F1 | 1139.518 | 1211.858 | 0.940x | 1.119x |
| P2+F1+F2 | 1138.112 | 1086.730 | 1.047x | 1.248x |
| P2+F1+F2+F3 | 1139.275 | 1086.929 | 1.048x | 1.248x |
| P2+F1+F2+F3+F4 | 1148.795 | 1097.464 | 1.047x | 1.236x |
| P2+F1+F2+F3+F4+P1 | 1136.775 | 1100.871 | 1.033x | 1.232x |

### 5.3 Random per-`n` parity (`ppl/howzat`)

| n | BASE_PRE_P2 | P2 | P2+F1 | P2+F1+F2 | P2+F1+F2+F3 | P2+F1+F2+F3+F4 | P2+F1+F2+F3+F4+P1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.657x | 0.697x | 0.697x | 0.743x | 0.735x | 0.742x | 0.729x |
| 5 | 0.715x | 0.787x | 0.777x | 0.846x | 0.846x | 0.853x | 0.852x |
| 6 | 0.760x | 0.845x | 0.834x | 0.917x | 0.924x | 0.942x | 0.924x |
| 7 | 0.782x | 0.895x | 0.899x | 1.020x | 1.017x | 1.012x | 1.000x |
| 8 | 0.837x | 0.947x | 0.951x | 1.098x | 1.101x | 1.094x | 1.085x |
| 9 | 0.931x | 1.057x | 1.047x | 1.219x | 1.217x | 1.213x | 1.183x |
| 10 | 1.032x | 1.192x | 1.180x | 1.333x | 1.341x | 1.322x | 1.298x |
| 11 | 1.193x | 1.344x | 1.329x | 1.470x | 1.478x | 1.427x | 1.414x |

## 6) Improvement / Noise / Regression Classification

Criteria used:

- Absolute howzat time change (deterministic + random aggregate)
- Consistency over random per-`n` pairs (8 points)
- Parity ratio movement (`ppl/howzat`)

### Transition verdicts

1. `BASE_PRE_P2 -> P2`: **observable improvement**
   - howzat absolute: deterministic `-2.24%`, random aggregate `-10.65%`
   - random consistency: `8/8` improved absolute, `8/8` improved ratio

2. `P2 -> P2+F1`: **in the noise / mixed**
   - howzat absolute: deterministic `-1.15%`, random aggregate `-0.02%`
   - random consistency: `3/8` improved absolute, `3/8` improved ratio

3. `P2+F1 -> P2+F1+F2`: **observable improvement (major)**
   - howzat absolute: deterministic `-37.18%`, random aggregate `-10.33%`
   - random consistency: `8/8` improved absolute, `8/8` improved ratio

4. `P2+F1+F2 -> P2+F1+F2+F3`: **in the noise**
   - howzat absolute: deterministic `+1.12%`, random aggregate `+0.02%`
   - random consistency: `4/8` improved absolute, ratio `5/8` improved

5. `P2+F1+F2+F3 -> P2+F1+F2+F3+F4`: **regression (small)**
   - howzat absolute: deterministic `+2.73%`, random aggregate `+0.97%`
   - random consistency: `3/8` improved absolute, `3/8` improved ratio

6. `P2+F1+F2+F3+F4 -> P2+F1+F2+F3+F4+P1`: **mixed, slightly regressive overall**
   - howzat absolute: deterministic `-0.43%`, random aggregate `+0.31%`
   - random consistency: `2/8` improved absolute, ratio `0/8` improved

## 7) Final Auditor Conclusions

- The dominant performance gains came from **P2** and especially **F2**.
- **F1** and **F3** are effectively neutral at this measurement depth.
- **F4** and **P1** both retain source-motivated logic, but in this clean dataset they reduce net gain versus the `P2+F1+F2(+F3)` peak.
- Final head (`P2+F1+F2+F3+F4+P1`) remains substantially faster than pre-P2 baseline:
  - deterministic howzat speedup: `1.593x`
  - random aggregate howzat speedup: `1.232x`
  - random parity changed from `0.845x` (slower than ppl) to `1.033x` (slightly faster than ppl)

## 8) Reproducibility

Primary script used for the trusted rerun:

- `/tmp/run_stage_matrix_5x_with_p1.sh`

Raw outputs:

- `/tmp/bench_stage_matrix_5x_20260207_212904/raw_results.tsv`

Rerun notes:

- The rerun was executed sequentially in one chain, stage by stage, with rebuild each stage in `release-lto`.
- Deterministic mixed-input reporting correctly emits `n=? v=?` in RESULT rows.
