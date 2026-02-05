use std::time::{Duration, Instant};

use anyhow::{anyhow, ensure};
use calculo::num::{Num, RugRat};
use hullabaloo::adjacency::{AdjacencyBuilder, AdjacencyStore};
use hullabaloo::set_family::ListFamily;
use hullabaloo::types::RepresentationKind;

use crate::vertices::{VerticesF64, VerticesI64};

use super::{
    Backend, BackendGeometry, BackendRun, BackendTiming, CoefficientMode, InputGeometry,
    AnyPolytopeCoefficients, CoefficientMatrix, LrslibTimingDetail, RowMajorMatrix, Stats,
    TimingDetail,
};

pub(super) fn run_lrslib_hlbl_backend<V, Inc, Adj>(
    spec: Backend,
    vertices: &V,
    output_incidence: bool,
    output_adjacency: bool,
    coeff_mode: CoefficientMode,
    timing: bool,
) -> Result<BackendRun<Inc, Adj>, anyhow::Error>
where
    V: VerticesF64 + ?Sized,
    Inc: From<ListFamily>,
    Adj: AdjacencyStore,
{
    let start_total = Instant::now();

    let dim = vertices.dim();
    ensure!(
        dim > 0,
        "lrslib+hlbl: vertices must have positive dimension"
    );
    let vertex_count = vertices.vertex_count();
    ensure!(vertex_count > 0, "lrslib+hlbl: need at least one vertex");

    let start = Instant::now();
    let mut matrix = lrslib_rs::Matrix::new(vertex_count, dim + 1, RepresentationKind::Generator)
        .map_err(|e| anyhow!("lrslib+hlbl: build matrix failed: {e}"))?;
    for (r, coords) in vertices.rows().enumerate() {
        ensure!(
            coords.len() == dim,
            "lrslib+hlbl: ragged vertex rows (expected {dim}, got {})",
            coords.len()
        );
        matrix.set(r, 0, 1.0);
        for (c, &x) in coords.iter().enumerate() {
            matrix.set(r, c + 1, x);
        }
    }
    let poly = lrslib_rs::Polyhedron::from_matrix(matrix);
    let time_build = start.elapsed();

    let start = Instant::now();
    let solved = poly
        .solve()
        .map_err(|e| anyhow!("lrslib+hlbl: solve failed: {e}"))?;
    let time_incidence = start.elapsed();

    let (input, output, incidence, input_incidence) = solved.into_parts();
    let coefficients = if coeff_mode == CoefficientMode::Off {
        None
    } else {
        let gen_cols = dim.saturating_add(1);
        let mut gen_data = Vec::with_capacity(vertex_count.saturating_mul(gen_cols));
        for coords in vertices.rows() {
            gen_data.push(1.0);
            gen_data.extend(coords);
        }

        let ineq_rows = output.rows();
        let ineq_cols = output.cols();
        let mut ineq_data = Vec::with_capacity(ineq_rows.saturating_mul(ineq_cols));
        for r in 0..ineq_rows {
            for c in 0..ineq_cols {
                ineq_data.push(output.get(r, c));
            }
        }

        Some(AnyPolytopeCoefficients {
            generators: CoefficientMatrix::F64(RowMajorMatrix {
                rows: vertex_count,
                cols: gen_cols,
                data: gen_data,
            }),
            inequalities: CoefficientMatrix::F64(RowMajorMatrix {
                rows: ineq_rows,
                cols: ineq_cols,
                data: ineq_data,
            }),
        })
    };

    if !output_incidence {
        let facet_rows = output.to_rows_vec();
        let duration = start_total.elapsed();
        let detail = if timing {
            Some(TimingDetail::Lrslib(LrslibTimingDetail {
                build: time_build,
                incidence: time_incidence,
                vertex_adjacency: Duration::ZERO,
                facet_adjacency: Duration::ZERO,
                post_inc: Duration::ZERO,
                post_v_adj: Duration::ZERO,
                post_f_adj: Duration::ZERO,
            }))
        } else {
            None
        };
        return Ok(BackendRun {
            spec,
            stats: Stats {
                dimension: dim,
                vertices: vertex_count,
                facets: facet_rows.len(),
                ridges: 0,
            },
            timing: BackendTiming {
                total: duration,
                fast: None,
                resolve: None,
                exact: None,
            },
            facets: Some(facet_rows),
            coefficients,
            geometry: BackendGeometry::Input(InputGeometry {
                vertex_adjacency: Adj::Builder::new(0).finish(),
                facets_to_vertices: ListFamily::from_sorted_sets(Vec::new(), vertex_count).into(),
                facet_adjacency: Adj::Builder::new(0).finish(),
            }),
            fails: 0,
            fallbacks: 0,
            error: None,
            detail,
        });
    }

    let mut time_input_adj = Duration::ZERO;
    let vertex_adjacency = if output_adjacency {
        let start = Instant::now();
        let vertex_adjacency = hullabaloo::adjacency::adjacency_from_incidence_with::<Adj::Builder>(
            input_incidence.sets(),
            output.rows(),
            input.cols(),
            hullabaloo::adjacency::IncidenceAdjacencyOptions::default(),
        );
        time_input_adj = start.elapsed();
        vertex_adjacency
    } else {
        Adj::Builder::new(0).finish()
    };

    let mut time_facet_adj = Duration::ZERO;
    let facet_adjacency = if output_adjacency {
        let start = Instant::now();
        let facet_adjacency = hullabaloo::adjacency::adjacency_from_incidence_with::<Adj::Builder>(
            incidence.sets(),
            input.rows(),
            input.cols(),
            hullabaloo::adjacency::IncidenceAdjacencyOptions::default(),
        );
        time_facet_adj = start.elapsed();
        facet_adjacency
    } else {
        Adj::Builder::new(0).finish()
    };

    let ridges = if output_adjacency {
        (0..facet_adjacency.node_count())
            .map(|i| facet_adjacency.degree(i))
            .sum::<usize>()
            / 2
    } else {
        0
    };

    let stats = Stats {
        dimension: dim,
        vertices: vertex_count,
        facets: incidence.len(),
        ridges,
    };

    let start = timing.then(Instant::now);
    let facets_to_vertices: Inc = incidence.into();
    let time_post_inc = start.map_or(Duration::ZERO, |start| start.elapsed());

    let time_post_v_adj = Duration::ZERO;
    let time_post_f_adj = Duration::ZERO;

    let duration = start_total.elapsed();
    let detail = if timing {
        Some(TimingDetail::Lrslib(LrslibTimingDetail {
            build: time_build,
            incidence: time_incidence,
            vertex_adjacency: time_input_adj,
            facet_adjacency: time_facet_adj,
            post_inc: time_post_inc,
            post_v_adj: time_post_v_adj,
            post_f_adj: time_post_f_adj,
        }))
    } else {
        None
    };
    Ok(BackendRun {
        spec,
        stats,
        timing: BackendTiming {
            total: duration,
            fast: None,
            resolve: None,
            exact: None,
        },
        facets: None,
        coefficients,
        geometry: BackendGeometry::Input(InputGeometry {
            vertex_adjacency,
            facets_to_vertices,
            facet_adjacency,
        }),
        fails: 0,
        fallbacks: 0,
        error: None,
        detail,
    })
}

pub(super) fn run_lrslib_hlbl_backend_i64<V, Inc, Adj>(
    spec: Backend,
    vertices: &V,
    output_incidence: bool,
    output_adjacency: bool,
    coeff_mode: CoefficientMode,
    timing: bool,
) -> Result<BackendRun<Inc, Adj>, anyhow::Error>
where
    V: VerticesI64 + ?Sized,
    Inc: From<ListFamily>,
    Adj: AdjacencyStore,
{
    let start_total = Instant::now();

    let dim = vertices.dim();
    ensure!(
        dim > 0,
        "lrslib+hlbl: vertices must have positive dimension"
    );
    let vertex_count = vertices.vertex_count();
    ensure!(vertex_count > 0, "lrslib+hlbl: need at least one vertex");

    let start = Instant::now();
    let mut row_slices: Vec<&[i64]> = Vec::with_capacity(vertex_count);
    for coords in vertices.rows() {
        ensure!(
            coords.len() == dim,
            "lrslib+hlbl: ragged vertex rows (expected {dim}, got {})",
            coords.len()
        );
        row_slices.push(coords);
    }
    let time_build = start.elapsed();

    let start = Instant::now();
    let output_coefficients = coeff_mode == CoefficientMode::Exact || !output_incidence;
    let (facet_coeffs, facet_to_vertex) = if output_coefficients {
        let (rows, incidence) =
            lrslib_rs::facets_with_incidence_from_integer_vertices_exact(&row_slices)
                .map_err(|e| anyhow!("lrslib+hlbl: solve failed: {e}"))?;
        (Some(rows), incidence)
    } else {
        let incidence = lrslib_rs::facet_incidence_from_integer_vertices(&row_slices)
            .map_err(|e| anyhow!("lrslib+hlbl: solve failed: {e}"))?;
        (None, incidence)
    };
    let time_incidence = start.elapsed();

    let facet_count = facet_to_vertex.len();
    let coefficients = if output_coefficients {
        fn i64_to_rug_rat(value: i64) -> RugRat {
            let mut out = RugRat::from_u64(value.unsigned_abs());
            if value.is_negative() {
                out = -out;
            }
            out
        }

        fn int_to_rug_rat(value: rug::Integer) -> RugRat {
            RugRat(rug::Rational::from(value))
        }

        let cols = dim.saturating_add(1);
        let mut gen_data: Vec<RugRat> = Vec::with_capacity(vertex_count.saturating_mul(cols));
        for coords in vertices.rows() {
            gen_data.push(RugRat::one());
            for &x in coords {
                gen_data.push(i64_to_rug_rat(x));
            }
        }

        let facet_coeffs = facet_coeffs.expect("coefficients requested");
        let mut ineq_data: Vec<RugRat> = Vec::new();
        ineq_data.reserve(facet_coeffs.len().saturating_mul(cols));
        for row in facet_coeffs {
            for coeff in row {
                ineq_data.push(int_to_rug_rat(coeff));
            }
        }

        Some(AnyPolytopeCoefficients {
            generators: CoefficientMatrix::RugRat(RowMajorMatrix {
                rows: vertex_count,
                cols,
                data: gen_data,
            }),
            inequalities: CoefficientMatrix::RugRat(RowMajorMatrix {
                rows: facet_count,
                cols,
                data: ineq_data,
            }),
        })
    } else {
        None
    };

    if !output_incidence {
        let duration = start_total.elapsed();
        let detail = if timing {
            Some(TimingDetail::Lrslib(LrslibTimingDetail {
                build: time_build,
                incidence: time_incidence,
                vertex_adjacency: Duration::ZERO,
                facet_adjacency: Duration::ZERO,
                post_inc: Duration::ZERO,
                post_v_adj: Duration::ZERO,
                post_f_adj: Duration::ZERO,
            }))
        } else {
            None
        };
        return Ok(BackendRun {
            spec,
            stats: Stats {
                dimension: dim,
                vertices: vertex_count,
                facets: facet_count,
                ridges: 0,
            },
            timing: BackendTiming {
                total: duration,
                fast: None,
                resolve: None,
                exact: None,
            },
            facets: None,
            coefficients,
            geometry: BackendGeometry::Input(InputGeometry {
                vertex_adjacency: Adj::Builder::new(0).finish(),
                facets_to_vertices: ListFamily::from_sorted_sets(Vec::new(), vertex_count).into(),
                facet_adjacency: Adj::Builder::new(0).finish(),
            }),
            fails: 0,
            fallbacks: 0,
            error: None,
            detail,
        });
    }

    let adj_dim = dim.saturating_add(1);

    let incidence = ListFamily::from_sorted_sets(facet_to_vertex, vertex_count);

    let mut time_input_adj = Duration::ZERO;
    let vertex_adjacency = if output_adjacency {
        let start = Instant::now();
        let input_incidence = incidence.transpose();
        let vertex_adjacency = hullabaloo::adjacency::adjacency_from_incidence_with::<Adj::Builder>(
            input_incidence.sets(),
            facet_count,
            adj_dim,
            hullabaloo::adjacency::IncidenceAdjacencyOptions::default(),
        );
        time_input_adj = start.elapsed();
        vertex_adjacency
    } else {
        Adj::Builder::new(0).finish()
    };

    let mut time_facet_adj = Duration::ZERO;
    let facet_adjacency = if output_adjacency {
        let start = Instant::now();
        let facet_adjacency = hullabaloo::adjacency::adjacency_from_incidence_with::<Adj::Builder>(
            incidence.sets(),
            vertex_count,
            adj_dim,
            hullabaloo::adjacency::IncidenceAdjacencyOptions::default(),
        );
        time_facet_adj = start.elapsed();
        facet_adjacency
    } else {
        Adj::Builder::new(0).finish()
    };

    let ridges = if output_adjacency {
        (0..facet_adjacency.node_count())
            .map(|i| facet_adjacency.degree(i))
            .sum::<usize>()
            / 2
    } else {
        0
    };

    let stats = Stats {
        dimension: dim,
        vertices: vertex_count,
        facets: facet_count,
        ridges,
    };

    let start = timing.then(Instant::now);
    let facets_to_vertices: Inc = incidence.into();
    let time_post_inc = start.map_or(Duration::ZERO, |start| start.elapsed());

    let time_post_v_adj = Duration::ZERO;
    let time_post_f_adj = Duration::ZERO;

    let duration = start_total.elapsed();
    let detail = if timing {
        Some(TimingDetail::Lrslib(LrslibTimingDetail {
            build: time_build,
            incidence: time_incidence,
            vertex_adjacency: time_input_adj,
            facet_adjacency: time_facet_adj,
            post_inc: time_post_inc,
            post_v_adj: time_post_v_adj,
            post_f_adj: time_post_f_adj,
        }))
    } else {
        None
    };
    Ok(BackendRun {
        spec,
        stats,
        timing: BackendTiming {
            total: duration,
            fast: None,
            resolve: None,
            exact: None,
        },
        facets: None,
        coefficients,
        geometry: BackendGeometry::Input(InputGeometry {
            vertex_adjacency,
            facets_to_vertices,
            facet_adjacency,
        }),
        fails: 0,
        fallbacks: 0,
        error: None,
        detail,
    })
}
