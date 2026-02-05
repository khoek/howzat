use std::time::{Duration, Instant};

use anyhow::anyhow;
use calculo::num::{Num, RugRat};
use hullabaloo::adjacency::{AdjacencyBuilder, AdjacencyStore};
use hullabaloo::set_family::ListFamily;

use crate::vertices::{VerticesF64, VerticesI64};

use super::{
    Backend, BackendGeometry, BackendRun, BackendTiming, CoefficientMatrix, CoefficientMode,
    AnyPolytopeCoefficients, InputGeometry, PplTimingDetail, RowMajorMatrix, Stats, TimingDetail,
};

pub(super) fn run_ppl_hlbl_backend<V, Inc, Adj>(
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

    let start = Instant::now();
    let dim = vertices.dim();
    let vertex_count = vertices.vertex_count();
    let gen_cols = dim.saturating_add(1);
    let mut gen_data =
        matches!(coeff_mode, CoefficientMode::F64).then(|| Vec::with_capacity(vertex_count.saturating_mul(gen_cols)));

    let mut vertex_rows = Vec::with_capacity(vertex_count);
    for coords in vertices.rows() {
        vertex_rows.push(coords.to_vec());
        if let Some(data) = gen_data.as_mut() {
            data.push(1.0);
            data.extend_from_slice(coords);
        }
    }

    let poly = ppl_rs::Polyhedron::from_vertices(vertex_rows)
        .map_err(|e| anyhow!("ppl+hlbl: build polyhedron failed: {e}"))?;
    let time_build = start.elapsed();

    let start = Instant::now();
    let solved = poly
        .solve()
        .map_err(|e| anyhow!("ppl+hlbl: solve failed: {e}"))?;
    let time_incidence = start.elapsed();

    let (dim, vertex_count, facet_count, inequalities, incidence, input_incidence) =
        solved.into_parts();
    let adj_dim = dim.saturating_add(1);

    let coefficients = if let Some(gen_data) = gen_data {
        let ineq_cols = inequalities.col_count();
        let mut ineq_data = Vec::with_capacity(inequalities.row_count().saturating_mul(ineq_cols));
        for row in inequalities.rows() {
            for v in row {
                ineq_data.push(v.to_f64());
            }
        }

        Some(AnyPolytopeCoefficients {
            generators: CoefficientMatrix::F64(RowMajorMatrix {
                rows: vertex_count,
                cols: gen_cols,
                data: gen_data,
            }),
            inequalities: CoefficientMatrix::F64(RowMajorMatrix {
                rows: inequalities.row_count(),
                cols: ineq_cols,
                data: ineq_data,
            }),
        })
    } else {
        None
    };

    if !output_incidence {
        let rows = inequalities.row_count();
        let cols = inequalities.col_count();
        let mut facet_rows: Vec<Vec<f64>> = Vec::with_capacity(rows);
        for row in inequalities.rows() {
            let mut out: Vec<f64> = Vec::with_capacity(cols);
            for v in row.iter() {
                out.push(v.to_f64());
            }
            facet_rows.push(out);
        }

        let duration = start_total.elapsed();
        let detail = if timing {
            Some(TimingDetail::Ppl(PplTimingDetail {
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
        Some(TimingDetail::Ppl(PplTimingDetail {
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

pub(super) fn run_ppl_hlbl_backend_i64<V, Inc, Adj>(
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

    let start = Instant::now();
    let vertex_count = vertices.vertex_count();
    let mut vertex_rows: Vec<&[i64]> = Vec::with_capacity(vertex_count);
    for coords in vertices.rows() {
        vertex_rows.push(coords);
    }
    let poly = ppl_rs::Polyhedron::from_integer_rows(&vertex_rows)
        .map_err(|e| anyhow!("ppl+hlbl: build polyhedron failed: {e}"))?;
    let time_build = start.elapsed();

    let start = Instant::now();
    let solved = poly
        .solve()
        .map_err(|e| anyhow!("ppl+hlbl: solve failed: {e}"))?;
    let time_incidence = start.elapsed();

    let (dim, vertex_count, facet_count, inequalities, incidence, input_incidence) =
        solved.into_parts();
    let adj_dim = dim.saturating_add(1);

    let coefficients = if matches!(coeff_mode, CoefficientMode::Exact) || !output_incidence {
        fn i64_to_rug_rat(value: i64) -> RugRat {
            let mut out = RugRat::from_u64(value.unsigned_abs());
            if value.is_negative() {
                out = -out;
            }
            out
        }

        fn int_to_rug_rat(value: &rug::Integer) -> RugRat {
            RugRat(rug::Rational::from((value.clone(), rug::Integer::from(1))))
        }

        let gen_cols = dim.saturating_add(1);
        let mut gen_data: Vec<RugRat> = Vec::with_capacity(vertex_count.saturating_mul(gen_cols));
        for coords in vertices.rows() {
            gen_data.push(RugRat::one());
            for &value in coords {
                gen_data.push(i64_to_rug_rat(value));
            }
        }

        let ineq_cols = inequalities.col_count();
        let mut ineq_data: Vec<RugRat> =
            Vec::with_capacity(inequalities.row_count().saturating_mul(ineq_cols));
        for row in inequalities.rows() {
            for v in row {
                ineq_data.push(int_to_rug_rat(v));
            }
        }

        Some(AnyPolytopeCoefficients {
            generators: CoefficientMatrix::RugRat(RowMajorMatrix {
                rows: vertex_count,
                cols: gen_cols,
                data: gen_data,
            }),
            inequalities: CoefficientMatrix::RugRat(RowMajorMatrix {
                rows: inequalities.row_count(),
                cols: ineq_cols,
                data: ineq_data,
            }),
        })
    } else {
        None
    };

    if !output_incidence {
        let duration = start_total.elapsed();
        let detail = if timing {
            Some(TimingDetail::Ppl(PplTimingDetail {
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

    let mut time_input_adj = Duration::ZERO;
    let vertex_adjacency = if output_adjacency {
        let start = Instant::now();
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
        Some(TimingDetail::Ppl(PplTimingDetail {
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
