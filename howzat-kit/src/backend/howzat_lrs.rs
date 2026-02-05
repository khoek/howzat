use std::time::Instant;

use anyhow::anyhow;
use calculo::num::{DashuRat, Num, RugRat};
use hullabaloo::adjacency::{AdjacencyBuilder, AdjacencyStore};
use hullabaloo::set_family::SetFamily;
use hullabaloo::types::{AdjacencyOutput, IncidenceOutput};

use crate::inequalities::HowzatInequalities;
use crate::vertices::{HowzatVertices, VerticesF64};

use super::howzat_common::{
    extract_howzat_coefficients, HowzatExtractTimingDetail, HowzatGeometrySummary,
    summarize_howzat_geometry,
};
use super::{
    Backend, BackendGeometry, BackendRun, BackendSpec, BackendTiming, CoefficientMode,
    HowzatLrsTimingDetail, InputGeometry, Stats, TimingDetail,
};

pub(super) fn run_howzat_lrs_backend<
    V: VerticesF64 + HowzatVertices + ?Sized,
    Inc: From<SetFamily>,
    Adj: AdjacencyStore,
>(
    spec: Backend,
    vertices: &V,
    output_incidence: bool,
    output_adjacency: bool,
    howzat_output_adjacency: AdjacencyOutput,
    coeff_mode: CoefficientMode,
    timing: bool,
) -> Result<BackendRun<Inc, Adj>, anyhow::Error> {
    let start_total = Instant::now();

    let cert_options = howzat::polyhedron::PolyhedronOptions {
        output_incidence: IncidenceOutput::Set,
        output_adjacency: AdjacencyOutput::Off,
        input_incidence: IncidenceOutput::Off,
        input_adjacency: AdjacencyOutput::Off,
        save_basis_and_tableau: false,
        save_repair_hints: false,
        profile_adjacency: false,
    };

    let exact_options = howzat::polyhedron::PolyhedronOptions {
        output_incidence: if output_incidence {
            IncidenceOutput::Set
        } else {
            IncidenceOutput::Off
        },
        output_adjacency: howzat_output_adjacency,
        input_incidence: IncidenceOutput::Off,
        input_adjacency: AdjacencyOutput::Off,
        save_basis_and_tableau: false,
        save_repair_hints: false,
        profile_adjacency: false,
    };

    let eps = f64::default_eps();
    let start_fast = Instant::now();
    let start_matrix = Instant::now();
    let matrix = vertices.build_howzat_generator_matrix::<f64>()?;
    let time_matrix = start_matrix.elapsed();

    type HowzatPolyF64 = howzat::polyhedron::PolyhedronOutput<f64, hullabaloo::types::Generator>;
    let start_lrs = Instant::now();
    let poly = match &spec.0 {
        BackendSpec::HowzatLrsRug => HowzatPolyF64::from_matrix_lrs_as_exact::<RugRat, f64>(
            matrix,
            howzat::polyhedron::LrsConfig {
                poly: cert_options,
                lrs: howzat::lrs::Options::default(),
            },
            &eps,
        )
        .map_err(|e| anyhow!("howzat-lrs conversion failed: {e:?}"))?,
        BackendSpec::HowzatLrsDashu => HowzatPolyF64::from_matrix_lrs_as_exact::<DashuRat, f64>(
            matrix,
            howzat::polyhedron::LrsConfig {
                poly: cert_options,
                lrs: howzat::lrs::Options::default(),
            },
            &eps,
        )
        .map_err(|e| anyhow!("howzat-lrs conversion failed: {e:?}"))?,
        _ => {
            return Err(anyhow!(
                "internal: howzat-lrs called with non-howzat-lrs backend"
            ));
        }
    };
    let time_lrs = start_lrs.elapsed();
    let time_fast = start_fast.elapsed();

    let start_exact = Instant::now();
    let cert = poly
        .certificate()
        .map_err(|e| anyhow!("howzat-lrs missing certificate: {e:?}"))?;

    let (geometry, extract_detail, time_exact, facets, coefficients) = match &spec.0 {
        BackendSpec::HowzatLrsRug => {
            let exact_eps = RugRat::default_eps();
            let exact_poly = cert
                .resolve_as::<RugRat>(
                    exact_options,
                    howzat::polyhedron::ResolveOptions::default(),
                    &exact_eps,
                )
                .map_err(|e| anyhow!("howzat-lrs exact resolution failed: {e:?}"))?;
            let time_exact = start_exact.elapsed();

            if !output_incidence {
                let output = exact_poly.output();
                let rows = output.row_count();
                let cols = output.col_count();
                let linearity = output.linearity();
                let facet_count = rows.saturating_sub(linearity.cardinality());
                let mut facet_row_indices = Vec::with_capacity(facet_count);
                for r in 0..rows {
                    if !linearity.contains(r) {
                        facet_row_indices.push(r);
                    }
                }
                let mut facet_rows: Vec<Vec<f64>> = Vec::with_capacity(facet_count);
                for r in facet_row_indices.iter().copied() {
                    let mut row: Vec<f64> = Vec::with_capacity(cols);
                    for x in output.row(r).unwrap().iter() {
                        row.push(x.to_f64());
                    }
                    facet_rows.push(row);
                }
                let coefficients = if coeff_mode == CoefficientMode::Off {
                    None
                } else {
                    extract_howzat_coefficients(&exact_poly, &facet_row_indices, coeff_mode)?
                };
                let stats = Stats {
                    dimension: exact_poly.dimension(),
                    vertices: vertices.vertex_count(),
                    facets: facet_count,
                    ridges: 0,
                };
                (
                    HowzatGeometrySummary {
                        stats,
                        vertex_adjacency: Adj::Builder::new(0).finish(),
                        facets_to_vertices: SetFamily::new(0, vertices.vertex_count()).into(),
                        facet_adjacency: Adj::Builder::new(0).finish(),
                        facet_row_indices: None,
                    },
                    HowzatExtractTimingDetail::default(),
                    time_exact,
                    Some(facet_rows),
                    coefficients,
                )
            } else {
                let (geometry, extract_detail) = summarize_howzat_geometry::<Inc, Adj, _, _>(
                    &exact_poly,
                    output_adjacency,
                    timing,
                    coeff_mode != CoefficientMode::Off,
                )?;
                let coefficients = if coeff_mode == CoefficientMode::Off {
                    None
                } else {
                    let Some(facet_row_indices) = geometry.facet_row_indices.as_deref() else {
                        return Err(anyhow!("internal: facet_row_indices missing"));
                    };
                    extract_howzat_coefficients(&exact_poly, facet_row_indices, coeff_mode)?
                };
                (geometry, extract_detail, time_exact, None, coefficients)
            }
        }
        BackendSpec::HowzatLrsDashu => {
            let exact_eps = DashuRat::default_eps();
            let exact_poly = cert
                .resolve_as::<DashuRat>(
                    exact_options,
                    howzat::polyhedron::ResolveOptions::default(),
                    &exact_eps,
                )
                .map_err(|e| anyhow!("howzat-lrs exact resolution failed: {e:?}"))?;
            let time_exact = start_exact.elapsed();

            if !output_incidence {
                let output = exact_poly.output();
                let rows = output.row_count();
                let cols = output.col_count();
                let linearity = output.linearity();
                let facet_count = rows.saturating_sub(linearity.cardinality());
                let mut facet_row_indices = Vec::with_capacity(facet_count);
                for r in 0..rows {
                    if !linearity.contains(r) {
                        facet_row_indices.push(r);
                    }
                }
                let mut facet_rows: Vec<Vec<f64>> = Vec::with_capacity(facet_count);
                for r in facet_row_indices.iter().copied() {
                    let mut row: Vec<f64> = Vec::with_capacity(cols);
                    for x in output.row(r).unwrap().iter() {
                        row.push(x.to_f64());
                    }
                    facet_rows.push(row);
                }
                let coefficients = if coeff_mode == CoefficientMode::Off {
                    None
                } else {
                    extract_howzat_coefficients(&exact_poly, &facet_row_indices, coeff_mode)?
                };
                let stats = Stats {
                    dimension: exact_poly.dimension(),
                    vertices: vertices.vertex_count(),
                    facets: facet_count,
                    ridges: 0,
                };
                (
                    HowzatGeometrySummary {
                        stats,
                        vertex_adjacency: Adj::Builder::new(0).finish(),
                        facets_to_vertices: SetFamily::new(0, vertices.vertex_count()).into(),
                        facet_adjacency: Adj::Builder::new(0).finish(),
                        facet_row_indices: None,
                    },
                    HowzatExtractTimingDetail::default(),
                    time_exact,
                    Some(facet_rows),
                    coefficients,
                )
            } else {
                let (geometry, extract_detail) = summarize_howzat_geometry::<Inc, Adj, _, _>(
                    &exact_poly,
                    output_adjacency,
                    timing,
                    coeff_mode != CoefficientMode::Off,
                )?;
                let coefficients = if coeff_mode == CoefficientMode::Off {
                    None
                } else {
                    let Some(facet_row_indices) = geometry.facet_row_indices.as_deref() else {
                        return Err(anyhow!("internal: facet_row_indices missing"));
                    };
                    extract_howzat_coefficients(&exact_poly, facet_row_indices, coeff_mode)?
                };
                (geometry, extract_detail, time_exact, None, coefficients)
            }
        }
        _ => {
            return Err(anyhow!(
                "internal: howzat-lrs called with non-howzat-lrs backend"
            ));
        }
    };

    let total = start_total.elapsed();
    let exact_total = total.saturating_sub(time_fast);

    let detail = if timing {
        Some(TimingDetail::HowzatLrs(HowzatLrsTimingDetail {
            matrix: time_matrix,
            lrs: time_lrs,
            cert: time_exact,
            incidence: extract_detail.incidence,
            vertex_adjacency: extract_detail.vertex_adjacency,
            facet_adjacency: extract_detail.facet_adjacency,
        }))
    } else {
        None
    };

    Ok(BackendRun {
        spec,
        stats: geometry.stats,
        timing: BackendTiming {
            total,
            fast: Some(time_fast),
            resolve: None,
            exact: Some(exact_total),
        },
        facets,
        coefficients,
        geometry: BackendGeometry::Input(InputGeometry {
            vertex_adjacency: geometry.vertex_adjacency,
            facets_to_vertices: geometry.facets_to_vertices,
            facet_adjacency: geometry.facet_adjacency,
        }),
        fails: 0,
        fallbacks: 0,
        error: None,
        detail,
    })
}

pub(super) fn run_howzat_lrs_backend_from_inequalities<
    I: HowzatInequalities + ?Sized,
    Inc: From<SetFamily>,
    Adj: AdjacencyStore,
>(
    spec: Backend,
    inequalities: &I,
    output_incidence: bool,
    output_adjacency: bool,
    howzat_output_adjacency: AdjacencyOutput,
    coeff_mode: CoefficientMode,
    timing: bool,
) -> Result<BackendRun<Inc, Adj>, anyhow::Error> {
    let start_total = Instant::now();

    let cert_options = howzat::polyhedron::PolyhedronOptions {
        output_incidence: IncidenceOutput::Set,
        output_adjacency: AdjacencyOutput::Off,
        input_incidence: IncidenceOutput::Off,
        input_adjacency: AdjacencyOutput::Off,
        save_basis_and_tableau: false,
        save_repair_hints: false,
        profile_adjacency: false,
    };

    let exact_options = howzat::polyhedron::PolyhedronOptions {
        output_incidence: if output_incidence {
            IncidenceOutput::Set
        } else {
            IncidenceOutput::Off
        },
        output_adjacency: howzat_output_adjacency,
        input_incidence: IncidenceOutput::Off,
        input_adjacency: AdjacencyOutput::Off,
        save_basis_and_tableau: false,
        save_repair_hints: false,
        profile_adjacency: false,
    };

    let eps = f64::default_eps();
    let start_fast = Instant::now();
    let start_matrix = Instant::now();
    let matrix = inequalities.build_howzat_inequality_matrix::<f64>()?;
    let time_matrix = start_matrix.elapsed();

    type HowzatPolyF64 = howzat::polyhedron::PolyhedronOutput<f64, hullabaloo::types::Inequality>;
    let start_lrs = Instant::now();
    let poly = match &spec.0 {
        BackendSpec::HowzatLrsRug => HowzatPolyF64::from_matrix_lrs_as_exact::<RugRat, f64>(
            matrix,
            howzat::polyhedron::LrsConfig {
                poly: cert_options,
                lrs: howzat::lrs::Options::default(),
            },
            &eps,
        )
        .map_err(|e| anyhow!("howzat-lrs conversion failed: {e:?}"))?,
        BackendSpec::HowzatLrsDashu => HowzatPolyF64::from_matrix_lrs_as_exact::<DashuRat, f64>(
            matrix,
            howzat::polyhedron::LrsConfig {
                poly: cert_options,
                lrs: howzat::lrs::Options::default(),
            },
            &eps,
        )
        .map_err(|e| anyhow!("howzat-lrs conversion failed: {e:?}"))?,
        _ => {
            return Err(anyhow!(
                "internal: howzat-lrs called with non-howzat-lrs backend"
            ));
        }
    };
    let time_lrs = start_lrs.elapsed();
    let time_fast = start_fast.elapsed();

    let start_exact = Instant::now();
    let cert = poly
        .certificate()
        .map_err(|e| anyhow!("howzat-lrs missing certificate: {e:?}"))?;

    let (geometry, extract_detail, time_exact, coefficients) = match &spec.0 {
        BackendSpec::HowzatLrsRug => {
            let exact_eps = RugRat::default_eps();
            let exact_poly = cert
                .resolve_as::<RugRat>(
                    exact_options,
                    howzat::polyhedron::ResolveOptions::default(),
                    &exact_eps,
                )
                .map_err(|e| anyhow!("howzat-lrs exact resolution failed: {e:?}"))?;
            let time_exact = start_exact.elapsed();

            if !output_incidence {
                let vertices = exact_poly.output().row_count();
                let facets = exact_poly.input().row_count();
                let coefficients = if coeff_mode == CoefficientMode::Off {
                    None
                } else {
                    let facet_row_indices: Vec<usize> = (0..facets).collect();
                    extract_howzat_coefficients(&exact_poly, &facet_row_indices, coeff_mode)?
                };
                let stats = Stats {
                    dimension: exact_poly.dimension(),
                    vertices,
                    facets,
                    ridges: 0,
                };
                (
                    HowzatGeometrySummary {
                        stats,
                        vertex_adjacency: Adj::Builder::new(0).finish(),
                        facets_to_vertices: SetFamily::new(0, vertices).into(),
                        facet_adjacency: Adj::Builder::new(0).finish(),
                        facet_row_indices: None,
                    },
                    HowzatExtractTimingDetail::default(),
                    time_exact,
                    coefficients,
                )
            } else {
                let (geometry, extract_detail) = summarize_howzat_geometry::<Inc, Adj, _, _>(
                    &exact_poly,
                    output_adjacency,
                    timing,
                    coeff_mode != CoefficientMode::Off,
                )?;
                let coefficients = if coeff_mode == CoefficientMode::Off {
                    None
                } else {
                    let Some(facet_row_indices) = geometry.facet_row_indices.as_deref() else {
                        return Err(anyhow!("internal: facet_row_indices missing"));
                    };
                    extract_howzat_coefficients(&exact_poly, facet_row_indices, coeff_mode)?
                };
                (geometry, extract_detail, time_exact, coefficients)
            }
        }
        BackendSpec::HowzatLrsDashu => {
            let exact_eps = DashuRat::default_eps();
            let exact_poly = cert
                .resolve_as::<DashuRat>(
                    exact_options,
                    howzat::polyhedron::ResolveOptions::default(),
                    &exact_eps,
                )
                .map_err(|e| anyhow!("howzat-lrs exact resolution failed: {e:?}"))?;
            let time_exact = start_exact.elapsed();

            if !output_incidence {
                let vertices = exact_poly.output().row_count();
                let facets = exact_poly.input().row_count();
                let coefficients = if coeff_mode == CoefficientMode::Off {
                    None
                } else {
                    let facet_row_indices: Vec<usize> = (0..facets).collect();
                    extract_howzat_coefficients(&exact_poly, &facet_row_indices, coeff_mode)?
                };
                let stats = Stats {
                    dimension: exact_poly.dimension(),
                    vertices,
                    facets,
                    ridges: 0,
                };
                (
                    HowzatGeometrySummary {
                        stats,
                        vertex_adjacency: Adj::Builder::new(0).finish(),
                        facets_to_vertices: SetFamily::new(0, vertices).into(),
                        facet_adjacency: Adj::Builder::new(0).finish(),
                        facet_row_indices: None,
                    },
                    HowzatExtractTimingDetail::default(),
                    time_exact,
                    coefficients,
                )
            } else {
                let (geometry, extract_detail) = summarize_howzat_geometry::<Inc, Adj, _, _>(
                    &exact_poly,
                    output_adjacency,
                    timing,
                    coeff_mode != CoefficientMode::Off,
                )?;
                let coefficients = if coeff_mode == CoefficientMode::Off {
                    None
                } else {
                    let Some(facet_row_indices) = geometry.facet_row_indices.as_deref() else {
                        return Err(anyhow!("internal: facet_row_indices missing"));
                    };
                    extract_howzat_coefficients(&exact_poly, facet_row_indices, coeff_mode)?
                };
                (geometry, extract_detail, time_exact, coefficients)
            }
        }
        _ => {
            return Err(anyhow!(
                "internal: howzat-lrs called with non-howzat-lrs backend"
            ));
        }
    };

    let total = start_total.elapsed();
    let exact_total = total.saturating_sub(time_fast);

    let detail = if timing {
        Some(TimingDetail::HowzatLrs(HowzatLrsTimingDetail {
            matrix: time_matrix,
            lrs: time_lrs,
            cert: time_exact,
            incidence: extract_detail.incidence,
            vertex_adjacency: extract_detail.vertex_adjacency,
            facet_adjacency: extract_detail.facet_adjacency,
        }))
    } else {
        None
    };

    Ok(BackendRun {
        spec,
        stats: geometry.stats,
        timing: BackendTiming {
            total,
            fast: Some(time_fast),
            resolve: Some(exact_total),
            exact: Some(time_exact),
        },
        facets: None,
        coefficients,
        geometry: BackendGeometry::Input(InputGeometry {
            vertex_adjacency: geometry.vertex_adjacency,
            facets_to_vertices: geometry.facets_to_vertices,
            facet_adjacency: geometry.facet_adjacency,
        }),
        fails: 0,
        fallbacks: 0,
        error: None,
        detail,
    })
}
