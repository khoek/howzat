#![cfg(feature = "rug")]

#[path = "support/common.rs"]
mod common;

use calculo::num::{DynamicEpsilon, F64Em12Epsilon, Num, RugRat};
use howzat::dd::SinglePrecisionUmpire;
use howzat::dd::umpire::policies;
use howzat::dd::{ConeOptions, EnumerationMode};
use howzat::matrix::LpMatrix as Matrix;
use howzat::polyhedron::{DdConfig, PolyhedronOptions, PolyhedronOutput};
use hullabaloo::types::{Generator, RepresentationKind};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::collections::BTreeSet;
use std::path::PathBuf;

fn build_random_cone_options(seed: u64) -> ConeOptions {
    let mut builder = ConeOptions::builder();
    let _ = seed;
    builder.enumeration_mode(EnumerationMode::Exact);
    builder.finish().expect("cone options")
}

fn upper_triangular_invertible_8(rng: &mut impl Rng) -> [[i64; 8]; 8] {
    let mut a = [[0i64; 8]; 8];
    for (i, row) in a.iter_mut().enumerate() {
        row[i] = if rng.random_bool(0.5) { 1 } else { -1 };
        for cell in row.iter_mut().skip(i + 1) {
            *cell = rng.random_range(-2i64..=2i64);
        }
    }
    a
}

fn moment_curve_point(t: i64) -> [i128; 8] {
    let mut out = [0i128; 8];
    let mut acc = t as i128;
    for slot in &mut out {
        *slot = acc;
        acc *= t as i128;
    }
    out
}

fn apply_affine(a: &[[i64; 8]; 8], b: &[i64; 8], x: &[i128; 8]) -> [i128; 8] {
    let mut out = [0i128; 8];
    for i in 0..8 {
        let mut acc = b[i] as i128;
        for j in 0..8 {
            acc += (a[i][j] as i128) * x[j];
        }
        out[i] = acc;
    }
    out
}

fn generator_matrix_from_vertices(vertices: &[[i128; 8]]) -> Matrix<f64, Generator> {
    let mut rows = Vec::with_capacity(vertices.len());
    for v in vertices {
        let mut row = Vec::with_capacity(9);
        row.push(1.0);
        row.extend(v.iter().map(|x| *x as f64));
        rows.push(row);
    }
    Matrix::<f64, Generator>::from_rows(rows)
}

fn normalized_hrep(
    matrix: &Matrix<f64, hullabaloo::types::Inequality>,
) -> Matrix<f64, hullabaloo::types::Inequality> {
    let eps = F64Em12Epsilon;
    matrix.normalized_sorted_unique(&eps).0
}

fn canonical_facet_count_rug(output: &Matrix<RugRat, hullabaloo::types::Inequality>) -> usize {
    let eps = DynamicEpsilon::<RugRat>::new(RugRat::zero());
    output.normalized_sorted_unique(&eps).0.row_count()
}

#[test]
#[ignore = "utility: run with HOWZAT_WRITE_UMPIRE_FIXTURES=1 to search and write counterexamples"]
fn find_single_precision_counterexample_d8_v18() {
    let wants_write = std::env::var("HOWZAT_WRITE_UMPIRE_FIXTURES")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let eps = F64Em12Epsilon;
    let poly_options = PolyhedronOptions::default();

    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join("umpire_regressions");
    if wants_write {
        std::fs::create_dir_all(&out_dir).expect("create output dir");
    }

    for seed in 0u64..10_000u64 {
        let mut rng = StdRng::seed_from_u64(seed);

        // Cyclic polytope vertices (all points are vertices; no point lies in the hull of the rest).
        let mut ts = BTreeSet::new();
        while ts.len() < 18 {
            ts.insert(rng.random_range(2i64..=50i64));
        }
        let vertices_raw: Vec<[i128; 8]> = ts.into_iter().map(moment_curve_point).collect();

        // Random affine scramble (invertible linear part).
        let a = upper_triangular_invertible_8(&mut rng);
        let mut b = [0i64; 8];
        for bi in &mut b {
            *bi = rng.random_range(-50i64..=50i64);
        }
        let vertices: Vec<[i128; 8]> = vertices_raw
            .iter()
            .map(|v| apply_affine(&a, &b, v))
            .collect();

        let matrix = generator_matrix_from_vertices(&vertices);
        assert_eq!(matrix.representation(), RepresentationKind::Generator);

        let cone_options = build_random_cone_options(seed);
        let single_umpire =
            SinglePrecisionUmpire::with_halfspace_policy(eps, policies::RandomRow::new(seed));

        let single = PolyhedronOutput::<f64, Generator>::from_matrix_dd(
            matrix.clone(),
            DdConfig {
                cone: cone_options.clone(),
                poly: poly_options.clone(),
            },
            single_umpire,
        )
        .expect("single-precision dd");

        let single_out = normalized_hrep(single.output_required());
        let single_rows = single_out.row_count();

        // Only accept a fixture if an exact RugRat run produces strictly more (canonicalized)
        // inequalities than the f64 single-precision run. This captures a stable “facet missing”
        // discrepancy without assuming a closed-form facet count.
        let matrix_exact = {
            let mut rows = Vec::with_capacity(matrix.row_count());
            for row in matrix.rows() {
                rows.push(
                    row.iter()
                        .map(|v| RugRat::try_from_f64(*v).expect("input must be finite"))
                        .collect::<Vec<_>>(),
                );
            }
            Matrix::<RugRat, Generator>::from_rows(rows)
        };
        let exact = PolyhedronOutput::<RugRat, Generator>::from_matrix_dd_int(
            matrix_exact,
            DdConfig {
                cone: ConeOptions::default(),
                poly: poly_options.clone(),
            },
        )
        .expect("exact dd");
        let exact_rows = canonical_facet_count_rug(exact.output_required());
        if exact_rows == 0 || single_rows >= exact_rows {
            continue;
        }

        eprintln!(
            "found facet-missing instance seed={seed} single_rows={single_rows} exact_rows={exact_rows}"
        );

        if wants_write {
            let filename = format!("d8_v18_seed{seed}.ext");
            let path = out_dir.join(filename);
            let mut content = String::new();
            content.push_str("* howzat umpire regression fixture\n");
            content.push_str(&format!("* seed={seed}\n"));
            content.push_str(&common::format_matrix(&matrix));
            std::fs::write(&path, content).expect("write fixture");
            eprintln!("wrote {}", path.display());
            return;
        }

        panic!(
            "found divergent instance seed={seed}; re-run with HOWZAT_WRITE_UMPIRE_FIXTURES=1 to write fixture"
        );
    }

    panic!("no divergent instance found in search range");
}
