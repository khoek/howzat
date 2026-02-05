use anyhow::Result;
use calculo::num::{InexactNum, Num, Rat};
use cddlib_rs::{Matrix as CddMatrix, NumberType, Polyhedron as CddPolyhedron};
use howzat::dd::DefaultNormalizer;
use howzat::dd::IntUmpire;
use howzat::dd::SinglePrecisionUmpire;
use howzat::dd::umpire::policies;
use howzat::dd::{BasisInitialization, ConeOptions, EnumerationMode};
use howzat::matrix::LpMatrixBuilder;
use howzat::polyhedron::PolyhedronOptions;
use hullabaloo::types::{Generator, IncidenceOutput, RowOrder, RowSet};
use rand::{Rng, SeedableRng};
use rstest::rstest;

const SEED_BASE: u64 = 0xDEADBEEF;

fn random_poly_points() -> Vec<Vec<f64>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xA1CE_FEED);
    let mut points = Vec::new();

    while points.len() < 5 {
        let candidate: Vec<f64> = (0..4).map(|_| rng.random_range(-1.0..1.0)).collect();
        if !candidate_redundant(&points, &candidate) {
            points.push(candidate);
        }
    }

    points
}

fn candidate_redundant(existing: &[Vec<f64>], candidate: &[f64]) -> bool {
    if existing.is_empty() {
        return false;
    }

    let mut verts = existing.to_vec();
    verts.push(candidate.to_vec());

    let matrix: CddMatrix<f64, Generator> = CddMatrix::from_vertex_rows(&verts)
        .expect("failed to build generator matrix for redundancy check");

    matrix
        .is_row_redundant(existing.len())
        .expect("redundancy check failed")
}

fn build_generator_matrix<N: Num>(
    points: &[Vec<f64>],
) -> Result<howzat::matrix::LpMatrix<N, Generator>> {
    let rows: Vec<Vec<N>> = points
        .iter()
        .map(|coords| {
            let mut row = Vec::with_capacity(coords.len() + 1);
            row.push(N::one());
            for &val in coords {
                row.push(
                    N::try_from_f64(val)
                        .ok_or_else(|| anyhow::anyhow!("non-finite coordinate {val}"))?,
                );
            }
            Ok(row)
        })
        .collect::<Result<_, anyhow::Error>>()?;

    Ok(LpMatrixBuilder::from_rows(rows).build())
}

fn facets_from_cdd(points: &[Vec<f64>]) -> Result<Vec<RowSet>> {
    let cols = points.first().map_or(0, |p| p.len() + 1);
    let mut matrix: CddMatrix<f64, Generator> =
        CddMatrix::new(points.len(), cols, NumberType::Real)?;
    for (r, coords) in points.iter().enumerate() {
        matrix.set_real(r, 0, 1.0);
        for (c, &v) in coords.iter().enumerate() {
            matrix.set_real(r, c + 1, v);
        }
        matrix.set_generator_type(r, true);
    }
    let poly = CddPolyhedron::from_matrix(&matrix)?;
    let incidence = poly.incidence()?.to_adjacency_lists();
    Ok(incidence
        .into_iter()
        .map(|verts| {
            let mut set = RowSet::new(points.len());
            for v in verts {
                set.insert(v);
            }
            set
        })
        .collect())
}

fn howzat_facets_with_inexact<N: InexactNum + DefaultNormalizer, H: policies::HalfspacePolicy<N>>(
    points: &[Vec<f64>],
    options: &ConeOptions,
    halfspace: H,
) -> Result<(Vec<RowSet>, usize, usize)> {
    let eps = N::default_eps();
    let umpire = SinglePrecisionUmpire::with_halfspace_policy(eps, halfspace);
    let matrix = build_generator_matrix::<N>(points)?;
    let poly_options = PolyhedronOptions {
        output_incidence: IncidenceOutput::Set,
        input_incidence: IncidenceOutput::Set,
        ..PolyhedronOptions::default()
    };
    let poly = howzat::polyhedron::PolyhedronOutput::<N, Generator>::from_matrix_dd(
        matrix,
        howzat::polyhedron::DdConfig {
            cone: options.clone(),
            poly: poly_options,
        },
        umpire,
    )?;
    let redundant = poly.redundant_rows_required().cardinality();
    let dominant = poly.dominant_rows_required().cardinality();
    let output = poly.output();
    let linearity = output.linearity().clone();
    let mut active = vec![true; output.row_count()];
    for idx in linearity.iter() {
        active[idx.as_index()] = false;
    }
    let incidence = poly.incidence_required();
    let mut faces = Vec::new();
    for idx in 0..incidence.family_size() {
        assert!(idx < active.len(), "incidence family index out of range");
        if !active[idx] {
            continue;
        }
        let face = incidence
            .set(idx)
            .expect("incidence set index out of range");
        if faces.iter().any(|f| f == face) {
            continue;
        }
        faces.push(face.clone());
    }
    Ok((faces, redundant, dominant))
}

fn howzat_facets_with_int<N: Rat, H: policies::HalfspacePolicy<N>>(
    points: &[Vec<f64>],
    options: &ConeOptions,
    halfspace: H,
) -> Result<(Vec<RowSet>, usize, usize)> {
    let eps = N::default_eps();
    let umpire = IntUmpire::with_halfspace_policy(eps, halfspace);
    let matrix = build_generator_matrix::<N>(points)?;
    let poly_options = PolyhedronOptions {
        output_incidence: IncidenceOutput::Set,
        input_incidence: IncidenceOutput::Set,
        ..PolyhedronOptions::default()
    };
    let poly =
        howzat::polyhedron::PolyhedronOutput::<N, Generator>::from_matrix_dd_int_with_umpire(
            matrix,
            howzat::polyhedron::DdConfig {
                cone: options.clone(),
                poly: poly_options,
            },
            umpire,
        )?;
    let redundant = poly.redundant_rows_required().cardinality();
    let dominant = poly.dominant_rows_required().cardinality();
    let output = poly.output();
    let linearity = output.linearity().clone();
    let mut active = vec![true; output.row_count()];
    for idx in linearity.iter() {
        active[idx.as_index()] = false;
    }
    let incidence = poly.incidence_required();
    let mut faces = Vec::new();
    for idx in 0..incidence.family_size() {
        assert!(idx < active.len(), "incidence family index out of range");
        if !active[idx] {
            continue;
        }
        let face = incidence
            .set(idx)
            .expect("incidence set index out of range");
        if faces.iter().any(|f| f == face) {
            continue;
        }
        faces.push(face.clone());
    }
    Ok((faces, redundant, dominant))
}

fn facets_match_with_inexact<N: InexactNum + DefaultNormalizer, H: policies::HalfspacePolicy<N>>(
    points: &[Vec<f64>],
    options: &ConeOptions,
    halfspace: H,
) -> Result<()> {
    let expected = facets_from_cdd(points)?;
    let (actual, redundant, dominant) =
        howzat_facets_with_inexact::<N, H>(points, options, halfspace)?;
    assert!(
        redundant == 0 && dominant == 0,
        "unexpected redundant/dominant rows: redundant={redundant} dominant={dominant}"
    );
    for face in expected.iter() {
        assert!(
            actual.iter().any(|f| f == face),
            "missing facet {face:?} for options {:?}",
            options
        );
    }
    for face in actual.iter() {
        assert!(
            expected.iter().any(|f| f == face),
            "extra facet {face:?} for options {:?}",
            options
        );
    }
    Ok(())
}

fn facets_match_with_int<N: Rat, H: policies::HalfspacePolicy<N>>(
    points: &[Vec<f64>],
    options: &ConeOptions,
    halfspace: H,
) -> Result<()> {
    let expected = facets_from_cdd(points)?;
    let (actual, redundant, dominant) =
        howzat_facets_with_int::<N, H>(points, options, halfspace)?;
    assert!(
        redundant == 0 && dominant == 0,
        "unexpected redundant/dominant rows: redundant={redundant} dominant={dominant}"
    );
    for face in expected.iter() {
        assert!(
            actual.iter().any(|f| f == face),
            "missing facet {face:?} for options {:?}",
            options
        );
    }
    for face in actual.iter() {
        assert!(
            expected.iter().any(|f| f == face),
            "extra facet {face:?} for options {:?}",
            options
        );
    }
    Ok(())
}

fn facets_match_for_order_inexact<N: InexactNum + DefaultNormalizer>(
    points: &[Vec<f64>],
    options: &ConeOptions,
    order: RowOrder,
    seed: u64,
) -> Result<()> {
    match order {
        RowOrder::MaxIndex => {
            facets_match_with_inexact::<N, policies::MaxIndex>(points, options, policies::MaxIndex)
        }
        RowOrder::MinIndex => {
            facets_match_with_inexact::<N, policies::MinIndex>(points, options, policies::MinIndex)
        }
        RowOrder::LexMin => {
            facets_match_with_inexact::<N, policies::LexMin>(points, options, policies::LexMin)
        }
        RowOrder::LexMax => {
            facets_match_with_inexact::<N, policies::LexMax>(points, options, policies::LexMax)
        }
        RowOrder::RandomRow => facets_match_with_inexact::<N, policies::RandomRow>(
            points,
            options,
            policies::RandomRow::new(seed),
        ),
        RowOrder::MinCutoff => facets_match_with_inexact::<N, policies::MinCutoff>(
            points,
            options,
            policies::MinCutoff::default(),
        ),
        RowOrder::MaxCutoff => facets_match_with_inexact::<N, policies::MaxCutoff>(
            points,
            options,
            policies::MaxCutoff::default(),
        ),
        RowOrder::MixCutoff => facets_match_with_inexact::<N, policies::MixCutoff>(
            points,
            options,
            policies::MixCutoff::default(),
        ),
    }
}

fn facets_match_for_order_int<N: Rat>(
    points: &[Vec<f64>],
    options: &ConeOptions,
    order: RowOrder,
    seed: u64,
) -> Result<()> {
    match order {
        RowOrder::MaxIndex => {
            facets_match_with_int::<N, policies::MaxIndex>(points, options, policies::MaxIndex)
        }
        RowOrder::MinIndex => {
            facets_match_with_int::<N, policies::MinIndex>(points, options, policies::MinIndex)
        }
        RowOrder::LexMin => {
            facets_match_with_int::<N, policies::LexMin>(points, options, policies::LexMin)
        }
        RowOrder::LexMax => {
            facets_match_with_int::<N, policies::LexMax>(points, options, policies::LexMax)
        }
        RowOrder::RandomRow => facets_match_with_int::<N, policies::RandomRow>(
            points,
            options,
            policies::RandomRow::new(seed),
        ),
        RowOrder::MinCutoff => facets_match_with_int::<N, policies::MinCutoff>(
            points,
            options,
            policies::MinCutoff::default(),
        ),
        RowOrder::MaxCutoff => facets_match_with_int::<N, policies::MaxCutoff>(
            points,
            options,
            policies::MaxCutoff::default(),
        ),
        RowOrder::MixCutoff => facets_match_with_int::<N, policies::MixCutoff>(
            points,
            options,
            policies::MixCutoff::default(),
        ),
    }
}

fn combo_seed(order: RowOrder, mode: EnumerationMode, basis: BasisInitialization) -> u64 {
    let o = order as u64;
    let m = mode as u64;
    let b = basis as u64;
    ((o * 4 + m) * 2) + b
}

#[rstest]
fn polytope_options_match_cdd(
    #[values(
        RowOrder::MaxIndex,
        RowOrder::MinIndex,
        RowOrder::MinCutoff,
        RowOrder::MaxCutoff,
        RowOrder::MixCutoff,
        RowOrder::LexMin,
        RowOrder::LexMax,
        RowOrder::RandomRow
    )]
    order: RowOrder,
    #[values(
        EnumerationMode::Exact,
        EnumerationMode::AssumeNondegenerate,
        EnumerationMode::Relaxed,
        EnumerationMode::RelaxedNondegenerate
    )]
    mode: EnumerationMode,
    #[values(BasisInitialization::Top, BasisInitialization::Bot)] basis: BasisInitialization,
) -> Result<()> {
    let mut builder = ConeOptions::builder();
    builder.enumeration_mode(mode);
    builder.basis_initialization(basis);
    let options = builder.finish()?;

    let seed = SEED_BASE ^ combo_seed(order, mode, basis);
    facets_match_for_order_inexact::<f64>(&random_poly_points(), &options, order, seed)
}

#[rstest]
fn polytope_ddnum_options_match(
    #[values(
        RowOrder::MaxIndex,
        RowOrder::MinIndex,
        RowOrder::MinCutoff,
        RowOrder::MaxCutoff,
        RowOrder::MixCutoff,
        RowOrder::LexMin,
        RowOrder::LexMax,
        RowOrder::RandomRow
    )]
    order: RowOrder,
    #[values(
        EnumerationMode::Exact,
        EnumerationMode::AssumeNondegenerate,
        EnumerationMode::Relaxed,
        EnumerationMode::RelaxedNondegenerate
    )]
    mode: EnumerationMode,
    #[values(BasisInitialization::Top, BasisInitialization::Bot)] basis: BasisInitialization,
    #[values("f64", "rug-rat", "rug-float", "dashu-rat", "dashu-float")] dd: &str,
) -> Result<()> {
    let mut builder = ConeOptions::builder();
    builder.enumeration_mode(mode);
    builder.basis_initialization(basis);
    let options = builder.finish()?;
    let seed = SEED_BASE ^ combo_seed(order, mode, basis);

    match dd {
        "f64" => {
            facets_match_for_order_inexact::<f64>(&random_poly_points(), &options, order, seed)?
        }
        "rug-rat" => facets_match_for_order_int::<calculo::num::RugRat>(
            &random_poly_points(),
            &options,
            order,
            seed,
        )?,
        "rug-float" => facets_match_for_order_inexact::<calculo::num::RugFloat<128>>(
            &random_poly_points(),
            &options,
            order,
            seed,
        )?,
        "dashu-rat" => facets_match_for_order_int::<calculo::num::DashuRat>(
            &random_poly_points(),
            &options,
            order,
            seed,
        )?,
        "dashu-float" => facets_match_for_order_inexact::<calculo::num::DashuFloat<128>>(
            &random_poly_points(),
            &options,
            order,
            seed,
        )?,
        _ => unreachable!("unexpected ddnum variant"),
    }
    Ok(())
}
