use std::{
    os::raw::c_long,
    time::{Duration, Instant},
};

use anyhow::{anyhow, ensure};
use calculo::num::{Num, RugRat};
use hullabaloo::AdjacencyList;
use hullabaloo::set_family::ListFamily;
use hullabaloo::types::RepresentationKind;

use crate::inequalities::{InequalitiesF64, InequalitiesI64};
use crate::vertices::{VerticesF64, VerticesI64};

use super::{
    Backend, BackendGeometry, BackendRun, BackendTiming, BaselineGeometry, CddlibTimingDetail,
    AnyPolytopeCoefficients, CoefficientMatrix, CoefficientMode, InputGeometry, RowMajorMatrix,
    Stats, TimingDetail,
};

fn parse_cddlib_rug_rat(value: impl ToString) -> Result<RugRat, anyhow::Error> {
    let parsed: rug::Rational = value
        .to_string()
        .parse()
        .map_err(|e| anyhow!("failed to parse cddlib rational: {e}"))?;
    Ok(RugRat(parsed))
}

pub(super) fn run_cddlib_backend<N, V>(
    spec: Backend,
    vertices: &V,
    number_type: cddlib_rs::NumberType,
    use_hull_facet_graph: bool,
    output_incidence: bool,
    output_adjacency: bool,
    coeff_mode: CoefficientMode,
    timing: bool,
) -> Result<BackendRun<ListFamily, AdjacencyList>, anyhow::Error>
where
    N: cddlib_rs::CddNumber,
    V: VerticesF64 + ?Sized,
{
    let start_total = Instant::now();
    let input_vertex_count = vertices.vertex_count();

    let start = Instant::now();
    ensure!(input_vertex_count > 0, "cddlib needs at least one vertex");
    let dim = vertices.dim();
    ensure!(dim > 0, "cddlib vertices must have positive dimension");
    let cols = dim
        .checked_add(1)
        .ok_or_else(|| anyhow!("cddlib vertex dimension too large"))?;

    let poly = {
        let mut matrix: cddlib_rs::Matrix<N, cddlib_rs::Generator> =
            cddlib_rs::Matrix::new(input_vertex_count, cols, number_type)?;
        for (row, coords) in vertices.rows().enumerate() {
            ensure!(
                coords.len() == dim,
                "cddlib vertices must have consistent dimension (expected {dim}, got {} at row {row})",
                coords.len()
            );
            matrix.set_generator_type(row, true);
            for (col, &v) in coords.iter().enumerate() {
                ensure!(v.is_finite(), "non-finite vertex coordinate {v}");
                matrix.set_real(row, col + 1, v);
            }
        }
        cddlib_rs::Polyhedron::from_matrix(&matrix)?
    };
    let facets = poly.facets()?;
    let dim = facets.cols().saturating_sub(1);
    let num_facets = facets.rows();
    if !output_incidence {
        let cols = facets.cols();
        let mut facet_rows: Vec<Vec<f64>> = Vec::with_capacity(num_facets);
        let mut ineq_data = matches!(coeff_mode, CoefficientMode::F64)
            .then(|| Vec::with_capacity(num_facets.saturating_mul(cols)));
        for r in 0..num_facets {
            let mut row: Vec<f64> = Vec::with_capacity(cols);
            for c in 0..cols {
                let v = facets.get_real(r, c);
                row.push(v);
                if let Some(data) = ineq_data.as_mut() {
                    data.push(v);
                }
            }
            facet_rows.push(row);
        }

        let coefficients = if let Some(ineq_data) = ineq_data {
            let generators = poly.generators()?;
            let gen_rows = generators.rows();
            let gen_cols = generators.cols();
            let mut gen_data = Vec::with_capacity(gen_rows.saturating_mul(gen_cols));
            for r in 0..gen_rows {
                for c in 0..gen_cols {
                    gen_data.push(generators.get_real(r, c));
                }
            }

            Some(AnyPolytopeCoefficients {
                generators: CoefficientMatrix::F64(RowMajorMatrix {
                    rows: gen_rows,
                    cols: gen_cols,
                    data: gen_data,
                }),
                inequalities: CoefficientMatrix::F64(RowMajorMatrix {
                    rows: num_facets,
                    cols,
                    data: ineq_data,
                }),
            })
        } else {
            None
        };
        let time_build = start.elapsed();
        let duration = start_total.elapsed();
        let detail = if timing {
            Some(TimingDetail::Cddlib(CddlibTimingDetail {
                build: time_build,
                incidence: Duration::ZERO,
                vertex_adjacency: Duration::ZERO,
                facet_adjacency: Duration::ZERO,
                vertex_positions: Duration::ZERO,
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
                vertices: input_vertex_count,
                facets: num_facets,
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
                vertex_adjacency: AdjacencyList::empty(),
                facets_to_vertices: ListFamily::from_sorted_sets(Vec::new(), input_vertex_count),
                facet_adjacency: AdjacencyList::empty(),
            }),
            fails: 0,
            fallbacks: 0,
            error: None,
            detail,
        });
    }

    let generators = poly.generators()?;
    let vertex_count = generators.rows();
    let time_build = start.elapsed();

    let start = Instant::now();
    let incidence = poly.incidence()?.to_adjacency_lists();
    let time_incidence = start.elapsed();

    let mut time_vertex_graph = Duration::ZERO;
    let mut vertex_graph = Vec::new();
    let mut time_facet_graph = Duration::ZERO;
    let mut facet_graph = Vec::new();
    let mut facet_adjacency_prebuilt = None;
    if output_adjacency {
        let start = Instant::now();
        vertex_graph = poly.input_adjacency()?.to_adjacency_lists();
        time_vertex_graph = start.elapsed();

        let start = Instant::now();
        if use_hull_facet_graph {
            facet_adjacency_prebuilt = Some(hullabaloo::adjacency::adjacency_from_incidence(
                &incidence,
                vertex_count,
                facets.cols(),
                hullabaloo::adjacency::IncidenceAdjacencyOptions::default(),
            ));
        } else {
            #[allow(deprecated)]
            {
                facet_graph = poly.adjacency()?.to_adjacency_lists();
            }
        }
        time_facet_graph = start.elapsed();
    }

    let start = Instant::now();
    ensure!(
        generators.representation() == RepresentationKind::Generator,
        "expected generator representation when extracting vertices"
    );
    let cols = generators.cols();
    ensure!(cols >= 2, "generator matrix has too few columns");
    let gen_rows = generators.rows();
    let mut gen_data =
        matches!(coeff_mode, CoefficientMode::F64).then(|| Vec::with_capacity(gen_rows.saturating_mul(cols)));
    let mut vertex_positions_data = Vec::with_capacity(gen_rows.saturating_mul(cols.saturating_sub(1)));
    for row in 0..gen_rows {
        let generator_type = generators.get_real(row, 0);
        if let Some(data) = gen_data.as_mut() {
            data.push(generator_type);
        }
        ensure!(
            (generator_type - 1.0).abs() <= 1e-9,
            "generator row {row} is not a vertex (type={generator_type})"
        );
        for col in 1..cols {
            let v = generators.get_real(row, col);
            vertex_positions_data.push(v);
            if let Some(data) = gen_data.as_mut() {
                data.push(v);
            }
        }
    }
    let time_vertex_positions = start.elapsed();

    let vertex_positions = CoefficientMatrix::F64(RowMajorMatrix {
        rows: gen_rows,
        cols: cols.saturating_sub(1),
        data: vertex_positions_data,
    });

    let coefficients = if let Some(gen_data) = gen_data {
        let ineq_cols = facets.cols();
        let mut ineq_data = Vec::with_capacity(num_facets.saturating_mul(ineq_cols));
        for r in 0..num_facets {
            for c in 0..ineq_cols {
                ineq_data.push(facets.get_real(r, c));
            }
        }

        Some(AnyPolytopeCoefficients {
            generators: CoefficientMatrix::F64(RowMajorMatrix {
                rows: gen_rows,
                cols,
                data: gen_data,
            }),
            inequalities: CoefficientMatrix::F64(RowMajorMatrix {
                rows: num_facets,
                cols: ineq_cols,
                data: ineq_data,
            }),
        })
    } else {
        None
    };

    let start = timing.then(Instant::now);
    let facets_to_vertices = ListFamily::from_sorted_sets(incidence, vertex_positions.rows());
    let time_post_inc = start.map_or(Duration::ZERO, |start| start.elapsed());

    let start = timing.then(Instant::now);
    let vertex_adjacency = if output_adjacency {
        AdjacencyList::from_unsorted_adjacency_lists(vertex_graph)
    } else {
        AdjacencyList::empty()
    };
    let time_post_v_adj = start.map_or(Duration::ZERO, |start| start.elapsed());

    let start = timing.then(Instant::now);
    let facet_adjacency = if output_adjacency {
        if let Some(prebuilt) = facet_adjacency_prebuilt {
            prebuilt
        } else {
            AdjacencyList::from_unsorted_adjacency_lists(facet_graph)
        }
    } else {
        AdjacencyList::empty()
    };
    let time_post_f_adj = start.map_or(Duration::ZERO, |start| start.elapsed());

    let ridges = if output_adjacency {
        facet_adjacency
            .adjacency_lists()
            .iter()
            .map(|n| n.len())
            .sum::<usize>()
            / 2
    } else {
        0
    };
    let stats = Stats {
        dimension: dim,
        vertices: vertex_positions.rows(),
        facets: num_facets,
        ridges,
    };

    let duration = start_total.elapsed();
    let detail = if timing {
        Some(TimingDetail::Cddlib(CddlibTimingDetail {
            build: time_build,
            incidence: time_incidence,
            vertex_adjacency: time_vertex_graph,
            facet_adjacency: time_facet_graph,
            vertex_positions: time_vertex_positions,
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
        geometry: BackendGeometry::Baseline(BaselineGeometry {
            vertex_positions,
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

pub(super) fn run_cddlib_backend_i64<N, V>(
    spec: Backend,
    vertices: &V,
    number_type: cddlib_rs::NumberType,
    use_hull_facet_graph: bool,
    output_incidence: bool,
    output_adjacency: bool,
    coeff_mode: CoefficientMode,
    timing: bool,
) -> Result<BackendRun<ListFamily, AdjacencyList>, anyhow::Error>
where
    N: cddlib_rs::CddNumber + ToString,
    V: VerticesI64 + ?Sized,
{
    let start_total = Instant::now();
    let input_vertex_count = vertices.vertex_count();

    let start = Instant::now();
    ensure!(input_vertex_count > 0, "cddlib needs at least one vertex");
    let dim = vertices.dim();
    ensure!(dim > 0, "cddlib vertices must have positive dimension");
    let cols = dim
        .checked_add(1)
        .ok_or_else(|| anyhow!("cddlib vertex dimension too large"))?;

    let poly = {
        let mut matrix: cddlib_rs::Matrix<N, cddlib_rs::Generator> =
            cddlib_rs::Matrix::new(input_vertex_count, cols, number_type)?;
        for (row, coords) in vertices.rows().enumerate() {
            ensure!(
                coords.len() == dim,
                "cddlib vertices must have consistent dimension (expected {dim}, got {} at row {row})",
                coords.len()
            );
            matrix.set_generator_type(row, true);
            for (col, &v) in coords.iter().enumerate() {
                let v: c_long = v
                    .try_into()
                    .map_err(|_| anyhow!("vertex coordinate {v} too large for cddlib"))?;
                matrix.set_int(row, col + 1, v);
            }
        }
        cddlib_rs::Polyhedron::from_matrix(&matrix)?
    };
    let facets = poly.facets()?;
    let dim = facets.cols().saturating_sub(1);
    let num_facets = facets.rows();
    if !output_incidence {
        let generators = poly.generators()?;
        let vertex_count = generators.rows();

        let gen_cols = generators.cols();
        let mut gen_data: Vec<RugRat> = Vec::with_capacity(vertex_count.saturating_mul(gen_cols));
        for r in 0..vertex_count {
            for c in 0..gen_cols {
                gen_data.push(parse_cddlib_rug_rat(generators.get(r, c))?);
            }
        }

        let ineq_cols = facets.cols();
        let mut ineq_data: Vec<RugRat> = Vec::with_capacity(num_facets.saturating_mul(ineq_cols));
        for r in 0..num_facets {
            for c in 0..ineq_cols {
                ineq_data.push(parse_cddlib_rug_rat(facets.get(r, c))?);
            }
        }

        let coefficients = Some(AnyPolytopeCoefficients {
            generators: CoefficientMatrix::RugRat(RowMajorMatrix {
                rows: vertex_count,
                cols: gen_cols,
                data: gen_data,
            }),
            inequalities: CoefficientMatrix::RugRat(RowMajorMatrix {
                rows: num_facets,
                cols: ineq_cols,
                data: ineq_data,
            }),
        });
        let time_build = start.elapsed();
        let duration = start_total.elapsed();
        let detail = if timing {
            Some(TimingDetail::Cddlib(CddlibTimingDetail {
                build: time_build,
                incidence: Duration::ZERO,
                vertex_adjacency: Duration::ZERO,
                facet_adjacency: Duration::ZERO,
                vertex_positions: Duration::ZERO,
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
                vertices: input_vertex_count,
                facets: num_facets,
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
                vertex_adjacency: AdjacencyList::empty(),
                facets_to_vertices: ListFamily::from_sorted_sets(Vec::new(), vertex_count),
                facet_adjacency: AdjacencyList::empty(),
            }),
            fails: 0,
            fallbacks: 0,
            error: None,
            detail,
        });
    }

    let generators = poly.generators()?;
    let vertex_count = generators.rows();
    let time_build = start.elapsed();

    let start = Instant::now();
    let incidence = poly.incidence()?.to_adjacency_lists();
    let time_incidence = start.elapsed();

    let mut time_vertex_graph = Duration::ZERO;
    let mut vertex_graph = Vec::new();
    let mut time_facet_graph = Duration::ZERO;
    let mut facet_graph = Vec::new();
    let mut facet_adjacency_prebuilt = None;
    if output_adjacency {
        let start = Instant::now();
        vertex_graph = poly.input_adjacency()?.to_adjacency_lists();
        time_vertex_graph = start.elapsed();

        let start = Instant::now();
        if use_hull_facet_graph {
            facet_adjacency_prebuilt = Some(hullabaloo::adjacency::adjacency_from_incidence(
                &incidence,
                vertex_count,
                facets.cols(),
                hullabaloo::adjacency::IncidenceAdjacencyOptions::default(),
            ));
        } else {
            #[allow(deprecated)]
            {
                facet_graph = poly.adjacency()?.to_adjacency_lists();
            }
        }
        time_facet_graph = start.elapsed();
    }

    ensure!(
        generators.representation() == RepresentationKind::Generator,
        "expected generator representation when extracting vertices"
    );
    let cols = generators.cols();
    ensure!(cols >= 2, "generator matrix has too few columns");
    let coefficients = if matches!(coeff_mode, CoefficientMode::Exact) {
        let gen_cols = cols;
        let mut gen_data: Vec<RugRat> = Vec::with_capacity(vertex_count.saturating_mul(gen_cols));
        for r in 0..vertex_count {
            for c in 0..gen_cols {
                gen_data.push(parse_cddlib_rug_rat(generators.get(r, c))?);
            }
        }

        let ineq_cols = facets.cols();
        let mut ineq_data: Vec<RugRat> = Vec::with_capacity(num_facets.saturating_mul(ineq_cols));
        for r in 0..num_facets {
            for c in 0..ineq_cols {
                ineq_data.push(parse_cddlib_rug_rat(facets.get(r, c))?);
            }
        }

        Some(AnyPolytopeCoefficients {
            generators: CoefficientMatrix::RugRat(RowMajorMatrix {
                rows: vertex_count,
                cols: gen_cols,
                data: gen_data,
            }),
            inequalities: CoefficientMatrix::RugRat(RowMajorMatrix {
                rows: num_facets,
                cols: ineq_cols,
                data: ineq_data,
            }),
        })
    } else {
        None
    };

    let start = timing.then(Instant::now);
    let facets_to_vertices = ListFamily::from_sorted_sets(incidence, vertex_count);
    let time_post_inc = start.map_or(Duration::ZERO, |start| start.elapsed());

    let start = timing.then(Instant::now);
    let vertex_adjacency = if output_adjacency {
        AdjacencyList::from_unsorted_adjacency_lists(vertex_graph)
    } else {
        AdjacencyList::empty()
    };
    let time_post_v_adj = start.map_or(Duration::ZERO, |start| start.elapsed());

    let start = timing.then(Instant::now);
    let facet_adjacency = if output_adjacency {
        if let Some(prebuilt) = facet_adjacency_prebuilt {
            prebuilt
        } else {
            AdjacencyList::from_unsorted_adjacency_lists(facet_graph)
        }
    } else {
        AdjacencyList::empty()
    };
    let time_post_f_adj = start.map_or(Duration::ZERO, |start| start.elapsed());

    let ridges = if output_adjacency {
        facet_adjacency
            .adjacency_lists()
            .iter()
            .map(|n| n.len())
            .sum::<usize>()
            / 2
    } else {
        0
    };
    let stats = Stats {
        dimension: dim,
        vertices: vertex_count,
        facets: num_facets,
        ridges,
    };

    let duration = start_total.elapsed();
    let detail = if timing {
        Some(TimingDetail::Cddlib(CddlibTimingDetail {
            build: time_build,
            incidence: time_incidence,
            vertex_adjacency: time_vertex_graph,
            facet_adjacency: time_facet_graph,
            vertex_positions: Duration::ZERO,
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

pub(super) fn run_cddlib_backend_inequalities<N, I>(
    spec: Backend,
    inequalities: &I,
    number_type: cddlib_rs::NumberType,
    use_hull_facet_graph: bool,
    output_incidence: bool,
    output_adjacency: bool,
    coeff_mode: CoefficientMode,
    timing: bool,
) -> Result<BackendRun<ListFamily, AdjacencyList>, anyhow::Error>
where
    N: cddlib_rs::CddNumber,
    I: InequalitiesF64 + ?Sized,
{
    let start_total = Instant::now();
    let input_facet_count = inequalities.facet_count();

    let start = Instant::now();
    ensure!(
        input_facet_count > 0,
        "cddlib needs at least one inequality"
    );
    let dim = inequalities.dim();
    ensure!(dim > 0, "cddlib inequalities must have positive dimension");
    let cols = dim
        .checked_add(1)
        .ok_or_else(|| anyhow!("cddlib inequality dimension too large"))?;

    let poly = {
        let mut matrix: cddlib_rs::Matrix<N, cddlib_rs::Inequality> =
            cddlib_rs::Matrix::new(input_facet_count, cols, number_type)?;
        for (row, coeffs) in inequalities.rows().enumerate() {
            ensure!(
                coeffs.len() == cols,
                "cddlib inequalities must have consistent width (expected {cols}, got {} at row {row})",
                coeffs.len()
            );
            for (col, &v) in coeffs.iter().enumerate() {
                ensure!(v.is_finite(), "non-finite inequality coefficient {v}");
                matrix.set_real(row, col, v);
            }
        }
        cddlib_rs::Polyhedron::from_matrix(&matrix)?
    };

    let generators = poly.generators()?;
    let vertex_count = generators.rows();
    let ineq_data = if matches!(coeff_mode, CoefficientMode::F64) {
        let mut data = Vec::with_capacity(input_facet_count.saturating_mul(cols));
        for row in inequalities.rows() {
            data.extend_from_slice(row);
        }
        Some(data)
    } else {
        None
    };

    if !output_incidence {
        let coefficients = ineq_data.map(|ineq_data| {
            let gen_rows = generators.rows();
            let gen_cols = generators.cols();
            let mut gen_data = Vec::with_capacity(gen_rows.saturating_mul(gen_cols));
            for r in 0..gen_rows {
                for c in 0..gen_cols {
                    gen_data.push(generators.get_real(r, c));
                }
            }

            AnyPolytopeCoefficients {
                generators: CoefficientMatrix::F64(RowMajorMatrix {
                    rows: gen_rows,
                    cols: gen_cols,
                    data: gen_data,
                }),
                inequalities: CoefficientMatrix::F64(RowMajorMatrix {
                    rows: input_facet_count,
                    cols,
                    data: ineq_data,
                }),
            }
        });
        let time_build = start.elapsed();
        let duration = start_total.elapsed();
        let detail = if timing {
            Some(TimingDetail::Cddlib(CddlibTimingDetail {
                build: time_build,
                incidence: Duration::ZERO,
                vertex_adjacency: Duration::ZERO,
                facet_adjacency: Duration::ZERO,
                vertex_positions: Duration::ZERO,
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
                facets: input_facet_count,
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
                vertex_adjacency: AdjacencyList::empty(),
                facets_to_vertices: ListFamily::from_sorted_sets(Vec::new(), vertex_count),
                facet_adjacency: AdjacencyList::empty(),
            }),
            fails: 0,
            fallbacks: 0,
            error: None,
            detail,
        });
    }

    let time_build = start.elapsed();

    let start = Instant::now();
    let gen_to_facet = poly.incidence()?.to_adjacency_lists();
    let time_incidence = start.elapsed();

    let mut facet_to_gen: Vec<Vec<usize>> = vec![Vec::new(); input_facet_count];
    for (gen_idx, facets) in gen_to_facet.iter().enumerate() {
        for &facet_idx in facets {
            if let Some(target) = facet_to_gen.get_mut(facet_idx) {
                target.push(gen_idx);
            }
        }
    }
    for face in &mut facet_to_gen {
        face.sort_unstable();
        face.dedup();
    }

    let mut time_vertex_graph = Duration::ZERO;
    let mut vertex_graph = Vec::new();
    let mut time_facet_graph = Duration::ZERO;
    let mut facet_graph = Vec::new();
    let mut facet_adjacency_prebuilt = None;
    if output_adjacency {
        let start = Instant::now();
        #[allow(deprecated)]
        {
            vertex_graph = poly.adjacency()?.to_adjacency_lists();
        }
        time_vertex_graph = start.elapsed();

        let start = Instant::now();
        if use_hull_facet_graph {
            facet_adjacency_prebuilt = Some(hullabaloo::adjacency::adjacency_from_incidence(
                &facet_to_gen,
                vertex_count,
                cols,
                hullabaloo::adjacency::IncidenceAdjacencyOptions::default(),
            ));
        } else {
            facet_graph = poly.input_adjacency()?.to_adjacency_lists();
        }
        time_facet_graph = start.elapsed();
    }

    let start = Instant::now();
    ensure!(
        generators.representation() == RepresentationKind::Generator,
        "expected generator representation when extracting vertices"
    );
    let cols = generators.cols();
    ensure!(cols >= 2, "generator matrix has too few columns");
    let mut gen_data =
        matches!(coeff_mode, CoefficientMode::F64).then(|| Vec::with_capacity(vertex_count.saturating_mul(cols)));
    let mut vertex_positions_data =
        Vec::with_capacity(vertex_count.saturating_mul(cols.saturating_sub(1)));
    for row in 0..vertex_count {
        let generator_type = generators.get_real(row, 0);
        if let Some(data) = gen_data.as_mut() {
            data.push(generator_type);
        }
        ensure!(
            (generator_type - 1.0).abs() <= 1e-9,
            "generator row {row} is not a vertex (type={generator_type})"
        );
        for col in 1..cols {
            let v = generators.get_real(row, col);
            vertex_positions_data.push(v);
            if let Some(data) = gen_data.as_mut() {
                data.push(v);
            }
        }
    }
    let time_vertex_positions = start.elapsed();

    let vertex_positions = CoefficientMatrix::F64(RowMajorMatrix {
        rows: vertex_count,
        cols: cols.saturating_sub(1),
        data: vertex_positions_data,
    });

    let coefficients = if let (Some(gen_data), Some(ineq_data)) = (gen_data, ineq_data) {
        Some(AnyPolytopeCoefficients {
            generators: CoefficientMatrix::F64(RowMajorMatrix {
                rows: vertex_count,
                cols,
                data: gen_data,
            }),
            inequalities: CoefficientMatrix::F64(RowMajorMatrix {
                rows: input_facet_count,
                cols,
                data: ineq_data,
            }),
        })
    } else {
        None
    };

    let start = timing.then(Instant::now);
    let facets_to_vertices = ListFamily::from_sorted_sets(facet_to_gen, vertex_count);
    let time_post_inc = start.map_or(Duration::ZERO, |start| start.elapsed());

    let start = timing.then(Instant::now);
    let vertex_adjacency = if output_adjacency {
        AdjacencyList::from_unsorted_adjacency_lists(vertex_graph)
    } else {
        AdjacencyList::empty()
    };
    let time_post_v_adj = start.map_or(Duration::ZERO, |start| start.elapsed());

    let start = timing.then(Instant::now);
    let facet_adjacency = if output_adjacency {
        if let Some(prebuilt) = facet_adjacency_prebuilt {
            prebuilt
        } else {
            AdjacencyList::from_unsorted_adjacency_lists(facet_graph)
        }
    } else {
        AdjacencyList::empty()
    };
    let time_post_f_adj = start.map_or(Duration::ZERO, |start| start.elapsed());

    let ridges = if output_adjacency {
        facet_adjacency
            .adjacency_lists()
            .iter()
            .map(|n| n.len())
            .sum::<usize>()
            / 2
    } else {
        0
    };

    let stats = Stats {
        dimension: dim,
        vertices: vertex_count,
        facets: facets_to_vertices.len(),
        ridges,
    };

    let duration = start_total.elapsed();
    let detail = if timing {
        Some(TimingDetail::Cddlib(CddlibTimingDetail {
            build: time_build,
            incidence: time_incidence,
            vertex_adjacency: time_vertex_graph,
            facet_adjacency: time_facet_graph,
            vertex_positions: time_vertex_positions,
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
        geometry: BackendGeometry::Baseline(BaselineGeometry {
            vertex_positions,
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

pub(super) fn run_cddlib_backend_inequalities_i64<N, I>(
    spec: Backend,
    inequalities: &I,
    number_type: cddlib_rs::NumberType,
    use_hull_facet_graph: bool,
    output_incidence: bool,
    output_adjacency: bool,
    coeff_mode: CoefficientMode,
    timing: bool,
) -> Result<BackendRun<ListFamily, AdjacencyList>, anyhow::Error>
where
    N: cddlib_rs::CddNumber + ToString,
    I: InequalitiesI64 + ?Sized,
{
    let start_total = Instant::now();
    let input_facet_count = inequalities.facet_count();

    let start = Instant::now();
    ensure!(
        input_facet_count > 0,
        "cddlib needs at least one inequality"
    );
    let dim = inequalities.dim();
    ensure!(dim > 0, "cddlib inequalities must have positive dimension");
    let cols = dim
        .checked_add(1)
        .ok_or_else(|| anyhow!("cddlib inequality dimension too large"))?;

    let poly = {
        let mut matrix: cddlib_rs::Matrix<N, cddlib_rs::Inequality> =
            cddlib_rs::Matrix::new(input_facet_count, cols, number_type)?;
        for (row, coeffs) in inequalities.rows().enumerate() {
            ensure!(
                coeffs.len() == cols,
                "cddlib inequalities must have consistent width (expected {cols}, got {} at row {row})",
                coeffs.len()
            );
            for (col, &v) in coeffs.iter().enumerate() {
                let v: c_long = v
                    .try_into()
                    .map_err(|_| anyhow!("inequality coefficient {v} too large for cddlib"))?;
                matrix.set_int(row, col, v);
            }
        }
        cddlib_rs::Polyhedron::from_matrix(&matrix)?
    };

    let generators = poly.generators()?;
    let vertex_count = generators.rows();
    let time_build = start.elapsed();

    let coefficients = if matches!(coeff_mode, CoefficientMode::Exact) || !output_incidence {
        fn i64_to_rug_rat(value: i64) -> RugRat {
            let mut out = RugRat::from_u64(value.unsigned_abs());
            if value.is_negative() {
                out = -out;
            }
            out
        }

        let gen_cols = generators.cols();
        let mut gen_data: Vec<RugRat> = Vec::with_capacity(vertex_count.saturating_mul(gen_cols));
        for r in 0..vertex_count {
            for c in 0..gen_cols {
                gen_data.push(parse_cddlib_rug_rat(generators.get(r, c))?);
            }
        }

        let mut ineq_data: Vec<RugRat> = Vec::with_capacity(input_facet_count.saturating_mul(cols));
        for row in inequalities.rows() {
            for &value in row {
                ineq_data.push(i64_to_rug_rat(value));
            }
        }

        Some(AnyPolytopeCoefficients {
            generators: CoefficientMatrix::RugRat(RowMajorMatrix {
                rows: vertex_count,
                cols: gen_cols,
                data: gen_data,
            }),
            inequalities: CoefficientMatrix::RugRat(RowMajorMatrix {
                rows: input_facet_count,
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
            Some(TimingDetail::Cddlib(CddlibTimingDetail {
                build: time_build,
                incidence: Duration::ZERO,
                vertex_adjacency: Duration::ZERO,
                facet_adjacency: Duration::ZERO,
                vertex_positions: Duration::ZERO,
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
                vertices: generators.rows(),
                facets: input_facet_count,
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
                vertex_adjacency: AdjacencyList::empty(),
                facets_to_vertices: ListFamily::from_sorted_sets(Vec::new(), vertex_count),
                facet_adjacency: AdjacencyList::empty(),
            }),
            fails: 0,
            fallbacks: 0,
            error: None,
            detail,
        });
    }

    let start = Instant::now();
    let gen_to_facet = poly.incidence()?.to_adjacency_lists();
    let time_incidence = start.elapsed();

    let mut facet_to_gen: Vec<Vec<usize>> = vec![Vec::new(); input_facet_count];
    for (gen_idx, facets) in gen_to_facet.iter().enumerate() {
        for &facet_idx in facets {
            if let Some(target) = facet_to_gen.get_mut(facet_idx) {
                target.push(gen_idx);
            }
        }
    }
    for face in &mut facet_to_gen {
        face.sort_unstable();
        face.dedup();
    }

    let mut time_vertex_graph = Duration::ZERO;
    let mut vertex_graph = Vec::new();
    let mut time_facet_graph = Duration::ZERO;
    let mut facet_graph = Vec::new();
    let mut facet_adjacency_prebuilt = None;
    if output_adjacency {
        let start = Instant::now();
        #[allow(deprecated)]
        {
            vertex_graph = poly.adjacency()?.to_adjacency_lists();
        }
        time_vertex_graph = start.elapsed();

        let start = Instant::now();
        if use_hull_facet_graph {
            facet_adjacency_prebuilt = Some(hullabaloo::adjacency::adjacency_from_incidence(
                &facet_to_gen,
                vertex_count,
                cols,
                hullabaloo::adjacency::IncidenceAdjacencyOptions::default(),
            ));
        } else {
            facet_graph = poly.input_adjacency()?.to_adjacency_lists();
        }
        time_facet_graph = start.elapsed();
    }

    let vertex_adjacency = if output_adjacency {
        AdjacencyList::from_unsorted_adjacency_lists(vertex_graph)
    } else {
        AdjacencyList::empty()
    };

    let facet_adjacency = if output_adjacency {
        if let Some(prebuilt) = facet_adjacency_prebuilt {
            prebuilt
        } else {
            AdjacencyList::from_unsorted_adjacency_lists(facet_graph)
        }
    } else {
        AdjacencyList::empty()
    };

    let ridges = if output_adjacency {
        facet_adjacency
            .adjacency_lists()
            .iter()
            .map(|n| n.len())
            .sum::<usize>()
            / 2
    } else {
        0
    };

    let facets_to_vertices = ListFamily::from_sorted_sets(facet_to_gen, vertex_count);

    let duration = start_total.elapsed();
    let detail = if timing {
        Some(TimingDetail::Cddlib(CddlibTimingDetail {
            build: time_build,
            incidence: time_incidence,
            vertex_adjacency: time_vertex_graph,
            facet_adjacency: time_facet_graph,
            vertex_positions: Duration::ZERO,
            post_inc: Duration::ZERO,
            post_v_adj: Duration::ZERO,
            post_f_adj: Duration::ZERO,
        }))
    } else {
        None
    };

    Ok(BackendRun {
        spec,
        stats: Stats {
            dimension: dim,
            vertices: vertex_count,
            facets: facets_to_vertices.len(),
            ridges,
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

pub(super) fn is_cddlib_error_code(err: &anyhow::Error, code: cddlib_rs::CddErrorCode) -> bool {
    use cddlib_rs::{CddError, CddWrapperError};

    err.chain().any(|cause| {
        if let Some(wrapper) = cause.downcast_ref::<CddWrapperError>() {
            matches!(
                wrapper,
                CddWrapperError::Cdd(cddlib_rs::CddError::Cdd(raw)) if *raw == code
            )
        } else if let Some(cdd) = cause.downcast_ref::<CddError>() {
            matches!(cdd, CddError::Cdd(raw) if *raw == code)
        } else {
            false
        }
    })
}

pub(super) fn backend_error_run_sparse(
    spec: Backend,
    dimension: usize,
    vertices: usize,
    duration: Duration,
    error: String,
) -> BackendRun<ListFamily, AdjacencyList> {
    BackendRun {
        spec,
        stats: Stats {
            dimension,
            vertices,
            facets: 0,
            ridges: 0,
        },
        timing: BackendTiming {
            total: duration,
            fast: None,
            resolve: None,
            exact: None,
        },
        facets: None,
        coefficients: None,
        geometry: BackendGeometry::Input(InputGeometry {
            vertex_adjacency: AdjacencyList::empty(),
            facets_to_vertices: ListFamily::from_sorted_sets(Vec::new(), vertices),
            facet_adjacency: AdjacencyList::empty(),
        }),
        fails: 1,
        fallbacks: 0,
        error: Some(error),
        detail: None,
    }
}
